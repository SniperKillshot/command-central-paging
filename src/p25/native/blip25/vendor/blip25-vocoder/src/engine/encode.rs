//! Encoder layers: analysis front end, parameter estimation, payload packing.
#![allow(dead_code, non_snake_case, non_upper_case_globals, clippy::all)]

use crate::engine::decode::*;
use crate::engine::dsp::*;

/// Snapshot a slice into caller-supplied stack storage and hand back a view of
/// exactly the copied length.
///
/// The original aliases `dst == src` wherever this appears — a renormalise or
/// a scale done in place — so the source has to be read from a copy. Every
/// such buffer is bounded by the codec's fixed geometry (129 spectrum bins,
/// 56 harmonics, a 383-word analysis ring), so the copy is a stack array.
#[inline(always)]
fn snap<'a>(buf: &'a mut [i16], src: &[i16]) -> &'a [i16] {
    let n = src.len();
    buf[..n].copy_from_slice(src);
    &buf[..n]
}

/// [`snap`] for a 32-bit vector.
#[inline(always)]
fn snap32<'a>(buf: &'a mut [i32], src: &[i32]) -> &'a [i32] {
    let n = src.len();
    buf[..n].copy_from_slice(src);
    &buf[..n]
}

/// [`normalise_to_exp`] with `dst == src`, which is how the original calls it.
///
/// `shift_copy16` is an ascending element-wise map, so running it over one
/// buffer is bit-identical to running it from a copy — and it saves the copy,
/// which for the analysis window is half a kilobyte of stack.
#[inline]
fn normalise_to_exp_inplace(v: &mut [i16], n: u16, exp: &mut i16) {
    let e = norm_exp_absmax(v, n) as u16;
    let sh = e.wrapping_neg() as i16;
    if sh != 0 && (n as i16) > 0 {
        let b = if sh < 0 {
            if sh < -0x1e {
                0x1fu32
            } else {
                (sh.wrapping_neg() as u8) as u32 & 0x1f
            }
        } else {
            0
        };
        for x in v[..n as usize].iter_mut() {
            let w = (*x as i32) << 16;
            let w = if sh < 0 {
                w >> b
            } else {
                w.wrapping_shl(sh as u32 & 0x1f)
            };
            *x = (w >> 16) as i16;
        }
    }
    *exp = exp.wrapping_add(e as i16);
}

// ======================= from bL2b1 =======================

// ---------------------------------------------------------------------------
// shared primitives
// ---------------------------------------------------------------------------

/// Hardware `bsr` semantics: the index of the highest set bit, and — for a
/// zero source — the destination register *unchanged*.
///
/// Every call site in this batch feeds `bsr` with the register that already
/// holds the magnitude, so a zero magnitude leaves the answer at `0`. That is
/// only reachable for an input of `-1` (magnitude `~(-1) == 0`), where it makes
/// the block exponent come out **30, not 31** — see SPEC §4.1. The Ghidra
/// rendering as a search loop initialised to `0x1f` reports `-1` and is wrong.
#[inline]
fn bsr_keep(v: u32) -> i32 {
    if v == 0 {
        0
    } else {
        31 - v.leading_zeros() as i32
    }
}

/// Block exponent `0x1e - bsr(|x|)` with the `bsr`-preserves-destination rule.
/// The caller must have excluded `x == 0` (every site here does, explicitly).
#[inline]
fn norm_exp(x: i32) -> i32 {
    let m = if x < 0 { !(x as u32) } else { x as u32 };
    0x1e - bsr_keep(m)
}

/// The library's open-coded variable shift: a non-negative count shifts left
/// (wrapping, count masked to 5 bits), a count in `-30 ..= -1` shifts right
/// arithmetically, and anything below `-30` is a full sign fill.
#[inline]
fn shift32_bL2b1(v: i32, s: i16) -> i32 {
    if s >= 0 {
        ((v as u32) << (s as u32 & 0x1f)) as i32
    } else if s < -0x1e {
        v >> 31
    } else {
        v >> ((s as i32).unsigned_abs() & 0x1f)
    }
}

#[inline]
fn high16_bL2b1(v: i32) -> i16 {
    (v >> 16) as i16
}

#[inline]
fn up16_bL2b1(v: i16) -> i32 {
    (v as i32) << 16
}

/// MSVC's `__allshl` (`0x10305930`). Reproduced locally rather than reaching
/// into a sibling batch; only the low half is ever consumed (SPEC §4.2).
fn allshl_bL2b1(v: i64, count: u8) -> i64 {
    let hi = (v >> 32) as u32;
    let lo = v as u32;
    if count >= 0x40 {
        return 0;
    }
    if count >= 0x20 {
        return (((lo << (count & 0x1f)) as u64) << 32) as i64;
    }
    let n = count & 0x1f;
    let (new_hi, new_lo) = if n == 0 {
        (hi, lo)
    } else {
        ((hi << n) | (lo >> (32 - n)), lo << n)
    };
    (((new_hi as u64) << 32) | new_lo as u64) as i64
}

/// The 32×16 multiply of SPEC §4.2: a Q31 accumulator scaled by a Q15 word
/// without forming a 64-bit product. The high half is pre-doubled, which is why
/// a decompiler prints it as two multiplies with different immediates.
#[inline]
fn mul32x16_bL2b1(v: i32, w: i32) -> i32 {
    let lo = ((v as u16 as i32).wrapping_mul(w)) >> 15;
    let hi = ((v >> 16) as i16 as i32).wrapping_mul(w).wrapping_mul(2);
    lo.wrapping_add(hi)
}

// ---------------------------------------------------------------------------
// 1. atan2 in block floating point
// ---------------------------------------------------------------------------

/// `0x1030ad30` — four-quadrant arctangent of `y/x`, block floating point.
///
/// `y` carries exponent `y_exp` and `x` carries `x_exp`; the result is returned
/// as a mantissa with `*out_exp` set, in the same units as the rest of the
/// phase machinery (`0x6488` is π/4 in Q15, so π/2 is `0x6488` at exponent 1
/// and π is `0x6488` at exponent 2).
///
/// Structure: classify the quadrant from the two signs, pick the additive
/// offset (0, +π or −π), normalise both operands, divide with [`div_norm`],
/// take the principal arctangent with [`atan_bfp`], align it to the offset's
/// exponent and add. The sum is then renormalised, which is the one place in
/// this function where the `−1` block-exponent trap of SPEC §4.1 bites: the sum
/// `atan + offset` really can be `-1`, and the original's `bsr` then leaves a
/// shift of **30**.
///
/// Degenerate axes are answered directly and never reach the divide:
/// `x == 0` gives ±π/2 (or 0 for `y == 0` too), and `y == 0` with `x < 0`
/// gives π.
///
/// Returns the value left in `eax`; callers use the low 16 bits.
pub fn atan2_bfp(y: i16, y_exp: i16, x: i16, x_exp: i16, out_exp: &mut i16) -> i32 {
    let mut quad = 0u32;
    if x < 0 {
        quad |= 2;
    }
    if y < 0 {
        quad |= 1;
    }

    if x == 0 {
        if quad & 1 != 0 {
            *out_exp = 1;
            return 0xffff9b78u32 as i32; // −π/2
        }
        if y != 0 {
            *out_exp = 1;
            return 0x6488; // +π/2
        }
        *out_exp = 0;
        return 0;
    }
    if y == 0 {
        if quad & 2 != 0 {
            *out_exp = 2;
            return 0x6488; // π
        }
        *out_exp = 0;
        return 0;
    }

    // Quadrant offset and the exponent it is expressed at.
    let (off, off_exp): (i16, i32) = match quad {
        2 => (0x6488, 2),
        3 => (-0x6488, 2),
        _ => (0, 0),
    };

    // Normalise the denominator; its shift is folded into x_exp.
    let xw = up16_bL2b1(x);
    let sh_x = (norm_exp(xw) as u16) as u32;
    let x_exp = (x_exp as i32).wrapping_sub(sh_x as i32);
    let xm = high16_bL2b1(((xw as u32) << (sh_x & 0x1f)) as i32);

    // Normalise the numerator; its shift is folded into y_exp.
    let yw = up16_bL2b1(y);
    let sh_y = (norm_exp(yw) as u16) as u32;
    let num = ((yw as u32) << (sh_y & 0x1f)) as i32;

    let mut e: i16 = 0;
    let q = div_norm(
        num,
        (y_exp as i32).wrapping_sub(sh_y as i32) as i16,
        xm,
        x_exp as i16,
        &mut e,
    );
    let a = atan_bfp(q, e, &mut e);

    // Align the arctangent to the offset's exponent (one extra bit of headroom
    // for the sum) and add.
    let cnt = e.wrapping_sub(off_exp as i16).wrapping_sub(1);
    let v = shift32_bL2b1(up16_bL2b1(a), cnt).wrapping_add(up16_bL2b1(off) >> 1);

    let sh = if v == 0 { 0i32 } else { norm_exp(v) } as u16;
    *out_exp = ((off_exp as i16).wrapping_sub(sh as i16)).wrapping_add(1);
    (((v as u32) << (sh as u32 & 0x1f)) as i32 >> 16) as u16 as i32
}

// ---------------------------------------------------------------------------
// 2. per-harmonic log magnitudes
// ---------------------------------------------------------------------------

/// `0x103108c0` — integrate the magnitude spectrum over harmonic bands and
/// emit each band's log magnitude.
///
/// This is the amplitude-estimation stage. `step` is the fundamental expressed
/// as a Q16 bin increment scaled by 32 (`inc = step·32`, i.e. `step` is Q11 in
/// bins), and the walk starts half a step in, at bin `0.5 + step/2` in Q16.
/// For each of `n` harmonics:
///
/// * the band runs from the current Q16 position to the next, clamped at bin
///   `0x7f`;
/// * the two end bins are **partially** weighted — the first by `1 − frac` via
///   the 32×16 multiply of [`mul32x16_bL2b1`], the last by `frac` — and every whole
///   bin between them is added at full weight;
/// * the total accumulates in 64 bits, is normalised by [`norm64`] and shifted
///   with `__allshl`/`__allshr`, keeping only the low word (SPEC §4.4);
/// * the mantissa is `high16_bL2b1` of that word and the exponent is
///   `spec_exp − shift`.
///
/// The log is taken by [`log2_at_q`] at Q5 and biased by `offset`. A band that
/// normalises to a zero mantissa, or whose exponent has fallen below `−0x3b`,
/// is floored at `-0x7800` — the same floor value that `0x103164b0` clamps to
/// and that `0x1031d6c0` uses as its log-domain origin.
///
/// Domain: `step > 0` and small enough that the walk stays inside the 129-bin
/// spectrum; `spec` is the raw state block words of the row, so it must be
/// indexable to word `2 * 0x7f + 1`.
pub fn harmonic_log_mags(
    out: &mut [i16],
    step: i16,
    n: u16,
    spec: &[i16],
    spec_exp: i16,
    offset: i16,
) {
    let inc = ((step as i32) << 9) >> 4;
    let acc0 = (inc >> 1).wrapping_add(0x8000);
    if (n as i16) <= 0 {
        return;
    }

    let mut count = n as u32;
    let mut pos_hi = high16_bL2b1(acc0) as u16;
    let mut pos_lo = acc0 as u16;
    let mut oi = 0usize;

    loop {
        // The original keeps the Q16 position split across two registers and
        // rejoins it here; per SPEC §4.2 this is an identity, `pos += inc`.
        let base = (up16_bL2b1(pos_hi as i16) as u32) | (pos_lo as u32);
        let acc = base.wrapping_add(inc as u32) as i32;

        let raw_hi = high16_bL2b1(acc) as u16;
        let next_hi = if 0x7f < (raw_hi as i16) { 0x7f } else { raw_hi };

        // Q15 fraction of the *previous* position: the first bin's share is
        // `1 − frac`, formed as `v - v·frac`.
        let w0 = ((pos_lo as u32) >> 1) as i32;
        let span = next_hi.wrapping_sub(pos_hi);
        let idx = (pos_hi as i16) as i32;

        let v0 = rd32(spec, 2 * idx as usize);
        // The `i32::MIN` guard on the negation is unreachable: `w0` is a
        // 15-bit non-negative weight, so the product bottoms out at
        // `2·(-0x8000)·0x7fff = -0x7fff0000`, never at `-0x80000000`.
        let p0 = mul32x16_bL2b1(v0, w0);
        let p0 = if p0 == i32::MIN {
            i32::MAX
        } else {
            p0.wrapping_neg()
        };
        let first = p0.wrapping_add(v0);

        // Q15 fraction of the new position weights the last bin.
        let w1 = ((((acc as u32) >> 1) & 0x7fff) as i16) as i32;
        let v1 = rd32(spec, 2 * (idx + (span as i16) as i32) as usize);
        let last = mul32x16_bL2b1(v1, w1);

        let mut acc64 = (first as i64).wrapping_add(last as i64);
        if (span as i16) > 1 {
            let whole = span.wrapping_sub(1);
            for k in 0..whole as usize {
                acc64 = acc64.wrapping_add(rd32(spec, 2 * (idx as usize + 1 + k)) as i64);
            }
        }

        let s = norm64(acc64 as u32, (acc64 >> 32) as u32) as i16;
        let shifted = if s < 0 {
            allshr(acc64, ((s as i32).wrapping_neg() as u32 & 0xff) as u8)
        } else {
            allshl_bL2b1(acc64, (s as u32 & 0xff) as u8)
        } as i32;

        let d = (spec_exp as u16).wrapping_sub(s as u16);
        let mant = high16_bL2b1(shifted);
        out[oi] = if (d as i16) < -0x3b || mant == 0 {
            -0x7800
        } else {
            (log2_at_q(mant, d.wrapping_add(0x1e) as i16, 5) as i16).wrapping_add(offset)
        };

        oi += 1;
        count -= 1;
        if count == 0 {
            break;
        }
        pos_hi = next_hi;
        pos_lo = acc as u16;
    }
}

// ---------------------------------------------------------------------------
// 3. voicing quantiser
// ---------------------------------------------------------------------------

/// `0x10312af0` — quantise the band voicing decisions.
///
/// `prev_bits` is the previous frame's voicing word; [`gather_dibits`] squeezes
/// its 16 dibits down to 16 bits, which seeds the predictive part of the
/// codebook. [`build_voicing_table`] then materialises every reachable
/// candidate pattern (`0x38` pre-baked ones plus `2^nbits` raw-bit ones) into a
/// 312-dword scratch array, [`voicing_codebook_search`] picks the index that
/// minimises the weighted error against the three band-measure vectors, and
/// [`voicing_pattern`] expands the winner back to a 32-bit dibit word which
/// [`expand_dibits`] splits into the two dwords the caller keeps as the current
/// voicing state.
///
/// The return value is the codebook index — the quantised voicing field that
/// goes into the payload. Callers use the low 16 bits.
///
/// Domain: `3 <= nbits <= 7` (below 3 the table walk does not terminate; above
/// 7 the search overruns).
#[allow(clippy::too_many_arguments)]
pub fn quantise_voicing_bL2b1(
    out0: &mut u32,
    out1: &mut u32,
    prev_bits: i32,
    w_a: &[i16],
    w_b: &[i16],
    w_c: &[i16],
    nbits: i32,
) -> u16 {
    let prev = gather_dibits(prev_bits) as u16 as u32;
    // The original zeroes all 312 dwords before filling the reachable ones.
    // Nothing reads the gap the table builder skips, so the fill is defensive.
    let mut book = [0u32; 312];
    build_voicing_table(&mut book, prev, nbits);
    let idx = voicing_codebook_search(w_a, w_b, w_c, &book, nbits as u8, 0) as u16;
    let pat = voicing_pattern(idx as i16, prev, nbits as i16);
    expand_dibits(out0, out1, pat);
    idx
}

// ---------------------------------------------------------------------------
// 4. log amplitude with the transmission clamp
// ---------------------------------------------------------------------------

/// `0x103164b0` — one block-floating-point magnitude to a clamped log amplitude.
///
/// The scalar counterpart of the per-band conversion inside
/// [`harmonic_log_mags`]: `log2_at_q(mant, exp + 0x1e, 5) + offset`, then
/// clamped to `[-0x1400, 0x7800]`. A zero mantissa or an exponent below `-0x3b`
/// short-circuits to the low clamp. The two limits are the log-domain dynamic
/// range the amplitude quantiser expects — a span of `0x8c00`.
pub fn log_amp_clamped(mant: i16, exp: i32, offset: i16) -> i32 {
    if (exp as i16) >= -0x3b && mant != 0 {
        let l = log2_at_q(mant, (exp as i16).wrapping_add(0x1e), 5) as i16;
        let v = l.wrapping_add(offset);
        if v > 0x7800 {
            return 0x7800;
        }
        if v >= -0x1400 {
            return v as i32;
        }
    }
    -0x1400
}

// ---------------------------------------------------------------------------
// 5. pitch-plausibility gate
// ---------------------------------------------------------------------------

/// `0x1031b280` — reject an implausible pitch candidate, or score it.
///
/// Two energy ratios have to hold before the candidate is scored at all:
///
/// 1. the energy collected by the harmonic-band walk at spacing `pitch`
///    ([`band_energy_walk`] with `num = 0x10000`, i.e. bands offset by
///    `0x10000/pitch` widths) must exceed `0.909` (`0xe8ba` in Q16) of the
///    plain spectral sum from the bin just below the first band
///    (`start − 0.375·width`, the `0x6000` Q15 constant taken at `sh − 7`) up to
///    bin `0x7f`;
/// 2. that same plain sum must exceed `0.1` (`0x1998` in Q16) of the sum taken
///    from half a harmonic spacing plus one bin up to `0x7f`.
///
/// Either failing returns `-1`. Otherwise the walk is repeated with
/// `num = pitch` (offset one whole width) and the result is turned into
/// `1 − ratio` against the second sum by [`one_minus_ratio`] — a Q15 measure of
/// how much of the band's energy is *not* explained by the harmonic comb, which
/// is the shape the voicing measures downstream consume.
///
/// Domain: `pitch` and `sh` have to keep the whole walk inside the 129-bin
/// spectrum — both derived bin indices in `0 ..= 0x7f` and the band width
/// positive. The original indexes unguarded.
pub fn pitch_gate_score(spec: &[i32], pitch: i16, sh: i16) -> i32 {
    let mut e1: i16 = 0;
    let mut start: i32 = 0;
    let m1 = band_energy_walk(&mut e1, spec, 0x10000, pitch, sh, &mut start);

    // Bin just below the first band: 0.375 of a band width back from `start`.
    let back = shift32_bL2b1((pitch as i32).wrapping_mul(0x6000), sh.wrapping_sub(7));
    let idx1 = high16_bL2b1(start.wrapping_sub(back)) as u16;

    let mut e2: i16 = 0;
    let m2 = sum_mant_exp(
        &spec[(idx1 as i16) as usize..],
        (0x80i32 - idx1 as i32) as i16,
        &mut e2,
    );

    let lhs = shift32_bL2b1(up16_bL2b1(m1), e1.wrapping_sub(e2));
    if lhs.wrapping_sub((m2 as i32).wrapping_mul(0xe8ba)) <= 0 {
        return -1;
    }

    // Half a harmonic spacing, plus one bin.
    let idx2 =
        high16_bL2b1(shift32_bL2b1(up16_bL2b1(pitch), sh.wrapping_sub(8)).wrapping_add(0x10000))
            as u16;
    let mut e3: i16 = 0;
    let m3 = sum_mant_exp(
        &spec[(idx2 as i16) as usize..],
        (0x80i32 - idx2 as i32) as i16,
        &mut e3,
    );

    let lhs2 = shift32_bL2b1(up16_bL2b1(m2), e2.wrapping_sub(e3));
    if lhs2.wrapping_sub((m3 as i32).wrapping_mul(0x1998)) <= 0 {
        return -1;
    }

    let mut e4: i16 = 0;
    let mut start2: i32 = 0;
    let m4 = band_energy_walk(&mut e4, spec, pitch as i32, pitch, sh, &mut start2);
    one_minus_ratio(m4, e4, m3, e3) as i32
}

// ---------------------------------------------------------------------------
// 6. background-level tracker over the 16 analysis bands
// ---------------------------------------------------------------------------

/// `0x1031d6c0` — convert the 16 band energies to the log domain and update the
/// running background minimum.
///
/// `state` is addressed by byte offset: 16 band mantissas at `+0x52`, their 16
/// exponents at `+0x72`, and an `i32` at `+0x40` holding the previously latched
/// total.
///
/// For each band `k`:
///
/// * the mantissa is aligned by `exp − 5` and biased by `0x7800 << 16`, giving
///   the same log-domain origin the amplitude clamp of [`log_amp_clamped`]
///   uses; the high word of that is the band's provisional level;
/// * a per-band tilt is subtracted — a ramp starting at `-0x400` and stepping by
///   `-0x40` per band, i.e. a fixed spectral pre-emphasis across the 16 bands;
/// * the tilted value is floored at `0x66`, passed through [`log2_at_q`] at Q5
///   and biased by `-0x21d`, and if that correction is negative it is added to
///   the provisional level with 32-bit saturation. The saturation limit is taken
///   from the **second** operand (the level), the per-call-site variation SPEC
///   §4.1 warns about;
/// * the result is written to `out[k]` and summed.
///
/// The total is then compared against the latched one with a hysteresis of
/// `30·K/64` where `K` is `0x4000`, or `0x2000` once the total exceeds
/// `0x10000` — a *smaller* margin for loud frames. Only a total that falls
/// below the latched value by more than that margin replaces it, and only then
/// does the function return `2`; otherwise `0`. It is therefore a
/// minimum-statistics noise-floor tracker: it follows the level down quickly
/// and never follows it up.
pub fn band_level_floor(out: &mut [i16], state: &mut [i16]) -> i32 {
    const MANT: usize = 0x52 / 2;
    const EXPS: usize = 0x72 / 2;

    let mut total: i32 = 0;
    let mut tilt: i16 = -0x400;

    for k in 0..16usize {
        let mut v = shift32_bL2b1(up16_bL2b1(state[MANT + k]), state[EXPS + k].wrapping_sub(5));
        v = v.wrapping_add(0x78000000u32 as i32);
        let mut level = high16_bL2b1(v);

        let t = high16_bL2b1(v.wrapping_sub(up16_bL2b1(tilt))) as u16;
        let t = if (t as i16) < 0x66 { 0x66 } else { t };

        let l = log2_at_q(t as i16, 5, 5) as i16;
        let corr = high16_bL2b1(up16_bL2b1(l).wrapping_sub(0x21c0001));

        if corr < 0 {
            let a = up16_bL2b1(corr);
            let b = up16_bL2b1(level);
            let s = a.wrapping_add(b);
            let s = if ((a ^ s) & (s ^ b)) < 0 {
                // Limit chosen from the *second* operand. `a` is always
                // negative here (the `corr < 0` guard), so an overflow implies
                // `b < 0` too and the choice of operand is not separately
                // observable at this site — but it is what the original does.
                (0x7fffffffu32.wrapping_add((b < 0) as u32)) as i32
            } else {
                s
            };
            level = high16_bL2b1(s);
        }

        out[k] = level;
        total = total.wrapping_add(level as i32);
        tilt = high16_bL2b1(up16_bL2b1(tilt).wrapping_sub(0x3f0001));
    }

    let margin = if total > 0x10000 { 0x2000i32 } else { 0x4000 };
    let m = (margin as i16) as i32;
    let hyst = (m << 4).wrapping_sub(m); // 15·margin
    let prev = read_i32(state, 0x40);
    let d = total.wrapping_sub(prev).wrapping_shl(6);
    if d.wrapping_add(hyst.wrapping_mul(2)) < 0 {
        write_i32(state, 0x40, total);
        return 2;
    }
    0
}

#[inline]
fn read_i32(state: &[i16], byte_off: usize) -> i32 {
    let i = byte_off / 2;
    (state[i] as u16 as i32) | ((state[i + 1] as i32) << 16)
}

#[inline]
fn write_i32(state: &mut [i16], byte_off: usize, v: i32) {
    let i = byte_off / 2;
    state[i] = v as i16;
    state[i + 1] = (v >> 16) as i16;
}

// ======================= from bL2b2 =======================

// ---------------------------------------------------------------------------
// shared arithmetic
// ---------------------------------------------------------------------------

/// Block-exponent normalisation shift exactly as the original inlines it:
/// `0x1e - bsr(|x|)`, zero for a zero input.
///
/// Every site in this batch preloads `bsr`'s destination with the *complement*
/// of the value before the scan, so for `x == -1` the magnitude `~x` is zero,
/// `bsr` leaves the destination unchanged at 0 and the shift comes out **30**,
/// not 31 (SPEC §4.1). The decompiler renders the scan as a loop seeded with
/// `0x1f` and therefore reports 31; transcribing that is wrong.
fn nrm30(x: i32) -> i16 {
    if x == 0 {
        return 0;
    }
    let m = if x < 0 { !(x as u32) } else { x as u32 };
    let b = if m == 0 {
        0
    } else {
        31 - m.leading_zeros() as i32
    };
    (0x1e - b) as i16
}

/// `x86` variable shift: left for a non-negative count, arithmetic right for a
/// negative one, and a full sign fill below −30. The count is masked to five
/// bits in both directions.
fn shift32_bL2b2(v: i32, s: i16) -> i32 {
    if s >= 0 {
        v.wrapping_shl(s as u32 & 0x1f)
    } else if s < -0x1e {
        v >> 31
    } else {
        v >> ((-s) as u32 & 0x1f)
    }
}

fn high16_bL2b2(x: i32) -> i16 {
    (x >> 16) as i16
}

fn up16_bL2b2(x: i16) -> i32 {
    (x as i32) << 16
}

/// 16-bit `shl` with an x86 count: masked to five bits, so 16..31 shift the
/// whole word out and yield zero. Rust's `<<` would panic there.
fn shl16_bL2b2(x: u16, c: u16) -> u16 {
    let c = (c & 0x1f) as u32;
    if c >= 16 {
        0
    } else {
        x << c
    }
}

/// Little-endian `i32` read out of an `i16` word view of a state block.
fn get32_bL2b2(s: &[i16], w: usize) -> i32 {
    (s[w] as u16 as i32) | ((s[w + 1] as i32) << 16)
}

fn put32_bL2b2(s: &mut [i16], w: usize, v: i32) {
    s[w] = v as i16;
    s[w + 1] = (v >> 16) as i16;
}

// ---------------------------------------------------------------------------
// 0x1030f470 — band voicing metric for one frame
// ---------------------------------------------------------------------------

/// `0x1030f470` — run the per-frame band voicing measurement.
///
/// Three stages, in order:
///
/// 1. [`interp_mid`] resolves the mid-frame value of the confidence/parameter
///    triple held in `mid[0..3]` / `mid[5..8]`, rejecting onsets and offsets.
/// 2. [`ratio_of_block_sums`] divides the sum of the 128 `i32` at
///    `spec[129..257]` (byte `0x204`) by the sum of the 128 at `spec[0..128]`,
///    with the two block exponents taken from the halves of `spec[387]` (bytes
///    `0x60c`, `0x60e`). The quotient is the frame's high-band / full-band
///    energy ratio; its exponent lands in `bands[0x51]` added to `bands[0x50]`.
/// 3. [`band_ratio_weights`] turns that ratio, together with the four
///    eight-band vectors in `bands`, into the per-band metric and weights, and
///    [`band_floor`] finally raises the two output vectors in `out` to a noise
///    floor.
///
/// `bands` is the eight-band working block: words `0..8` and `16..24` are the
/// two energy vectors, `8..16` receives the metric, `40..48` and `56..64` are
/// the two envelopes, `48..56` receives the weights, and words `0x50`/`0x52`
/// hold the two block exponents. `out` is the caller's parameter block; its
/// words `0x0c`, `0x14`, `0x1c` and `0x24` are the four eight-band slots
/// [`band_floor`] works on.
pub fn frame_band_voicing(
    bands: &mut [i16],
    mid: &mut [i16],
    spec: &[i16],
    out: &mut [i16],
    mode: i16,
) {
    {
        let (v, w) = mid.split_at_mut(5);
        interp_mid(w, v);
    }

    let mut exp: i16 = 0;
    // Row 1 is 129 `i32` in, and word 387 is the exponent pair.
    let exps = rd32(spec, 2 * 387);
    let ratio = ratio_of_block_sums(
        &spec[2 * 129..],
        (exps >> 16) as u16 as i32,
        spec,
        exps as i16,
        &mut exp,
    );
    bands[0x51] = bands[0x50].wrapping_add(exp);

    // The six vectors band_ratio_weights sees are disjoint windows of `bands`;
    // copy them out and back rather than fighting the borrow checker.
    let mut a = [0i16; 8];
    let mut c = [0i16; 8];
    let mut d = [0i16; 8];
    let mut f = [0i16; 8];
    a.copy_from_slice(&bands[0..8]);
    c.copy_from_slice(&bands[16..24]);
    d.copy_from_slice(&bands[40..48]);
    f.copy_from_slice(&bands[56..64]);
    let mut metric = [0i16; 8];
    let mut weights = [0i16; 8];
    metric.copy_from_slice(&bands[8..16]);
    weights.copy_from_slice(&bands[48..56]);
    band_ratio_weights(
        &a,
        &mut metric,
        &c,
        &d,
        &mut weights,
        &f,
        bands[0x50] as u16 as i32,
        bands[0x52] as u16 as i32,
        ratio,
    );
    bands[8..16].copy_from_slice(&metric);
    bands[48..56].copy_from_slice(&weights);

    let mut src = [0i16; 8];
    let mut levels = [0i16; 8];
    src.copy_from_slice(&out[0x0c..0x14]);
    levels.copy_from_slice(&out[0x24..0x2c]);
    let mut cur = [0i16; 8];
    let mut alt = [0i16; 8];
    cur.copy_from_slice(&out[0x14..0x1c]);
    alt.copy_from_slice(&out[0x1c..0x24]);
    band_floor(mid, &src, &mut cur, &mut alt, &levels, mode);
    out[0x14..0x1c].copy_from_slice(&cur);
    out[0x1c..0x24].copy_from_slice(&alt);
}

// ---------------------------------------------------------------------------
// 0x10310a80 — eight-band log level map
// ---------------------------------------------------------------------------

/// `0x10310a80` — map eight band energies to a log-domain level, with an
/// optional high-band tilt.
///
/// For each band, `log2_at_q(src[i], exp, 6)` gives `log2` of the energy at
/// Q6; the level is `(-0x4a3 - log2) · 2^16`, i.e. an offset and a negation so
/// that a *smaller* energy maps to a *larger* number. When `tilt` is non-zero
/// the four upper bands (index > 3) are pulled down by a further `0x200` at
/// Q16 — a fixed high-frequency de-emphasis. Negative results clamp to zero,
/// so the map is one-sided, and the answer is finally scaled by 32
/// ([`shl_sat`] by 5) and taken as its high word.
///
/// A zero energy short-circuits to `0x7fff` rather than taking the logarithm of
/// zero.
pub fn log_band_levels(out: &mut [i16], src: &[i16], exp: i16, tilt: i16) {
    for i in 0..8usize {
        let x = src[i];
        if x == 0 {
            out[i] = 0x7fff;
            continue;
        }
        let l = log2_at_q(x, exp, 6) as i16;
        let mut acc = (-0x4a3i32).wrapping_sub(l as i32).wrapping_shl(16);
        if tilt != 0 && (i as i16) > 3 {
            acc = acc.wrapping_sub(0x0200_0000);
        }
        if acc < 0 {
            acc = 0;
        }
        out[i] = high16_bL2b2(shl_sat(acc, 5));
    }
}

// ---------------------------------------------------------------------------
// 0x103136e0 — eight-band spectral match
// ---------------------------------------------------------------------------

/// `0x103136e0` — per-band match between a reference spectrum and the measured
/// one.
///
/// Both spectra are complex, stored as interleaved `(re, im)` `i16` pairs, so
/// one bin is four bytes. The half-spectrum is cut into eight bands at bins
/// `0, 13, 29, 45, 61, 77, 93, 109, 127` — the first is 13 bins wide, the next
/// six 16 bins, and the last runs to bin 127. Per band:
///
/// * [`sq_diff_norm`] gives `Σ|ref − meas|²` as a block-floating value — the
///   match error;
/// * [`dot_norm64`] over the same `2·bins` words gives `Σ|meas|²` — the band
///   energy — whose exponent is offset by `2·meas_exp` because the dot product
///   squares the mantissas.
///
/// The eight errors and the eight energies are each brought onto one shared
/// exponent by [`align_block_exp`]. A three-tap moving sum (band `i−1`, `i`,
/// `i+1`, truncated at the ends) is then taken over both, and the ratio
/// `error / energy` is written to `out[0x0c + i]` with its own exponent; a band
/// whose smoothed energy normalises to zero yields `0x4000 @ 2^1` instead of
/// dividing. Those eight ratios are finally aligned onto one exponent, which is
/// the return value, and the energy exponent is left in `out[9]`.
///
/// The smoothing is what makes this a *voicing* measurement rather than a bare
/// residual: a band is judged against its neighbours, so a single mismatched
/// bin cannot declare a band unvoiced on its own.
pub fn band_spectral_match(
    meas: &[i16],
    meas_exp: i16,
    reference: &[i16],
    ref_exp: i16,
    out: &mut [i16],
) -> u16 {
    let mut err = [0i16; 8];
    let mut err_exp = [0i16; 8];
    let mut energy = [0i16; 8];
    let mut energy_exp = [0i16; 8];

    let mut lo: u32 = 0;
    let mut hi: u32 = 0xd;
    for i in 0..8usize {
        if i == 7 {
            hi = 0x7f;
        }
        let base = ((lo as i16) as i32 * 4) as usize; // byte offset
        let word = base / 2;
        let n = hi.wrapping_sub(lo) as i16;

        err[i] = sq_diff_norm(
            &mut err_exp[i],
            &reference[word..],
            ref_exp,
            &meas[word..],
            meas_exp,
            n,
        );

        let mut e: i16 = 0;
        let d = dot_norm64(
            &mut e,
            &meas[word..],
            &meas[word..],
            (n as u16).wrapping_mul(2),
        );
        energy_exp[i] = e.wrapping_add(meas_exp.wrapping_mul(2));
        energy[i] = high16_bL2b2(d);

        lo = hi & 0xffff;
        hi = hi.wrapping_add(0x10);
    }

    let src = err;
    let ec = align_block_exp(&mut err, &src, &err_exp, 8) as u16;
    let src = energy;
    let en = align_block_exp(&mut energy, &src, &energy_exp, 8) as u16;
    let (err_aligned, energy_aligned) = (err, energy);

    let mut ratio_exp = [0i16; 8];
    for i in 0..8usize {
        let mut se = err_aligned[i] as i32;
        let mut sn = energy_aligned[i] as i32;
        if i > 0 {
            se = se.wrapping_add(err_aligned[i - 1] as i32);
            sn = sn.wrapping_add(energy_aligned[i - 1] as i32);
        }
        if i < 7 {
            se = se.wrapping_add(err_aligned[i + 1] as i32);
            sn = sn.wrapping_add(energy_aligned[i + 1] as i32);
        }
        let ne = nrm30(se) as u16;
        let nn = nrm30(sn) as u16;
        let den = sn.wrapping_shl(nn as u32 & 0x1f);
        if den == 0 {
            out[0x0c + i] = 0x4000;
            ratio_exp[i] = 1;
        } else {
            let mut e: i16 = 0;
            out[0x0c + i] = div_norm(
                se.wrapping_shl(ne as u32 & 0x1f),
                ec.wrapping_sub(ne) as i16,
                high16_bL2b2(den),
                en.wrapping_sub(nn) as i16,
                &mut e,
            );
            ratio_exp[i] = e;
        }
    }

    let mut ratios = [0i16; 8];
    ratios.copy_from_slice(&out[0x0c..0x14]);
    let src = ratios;
    let r = align_block_exp(&mut ratios, &src, &ratio_exp, 8) as u16;
    out[0x0c..0x14].copy_from_slice(&ratios);
    out[9] = en as i16;
    r
}

// ---------------------------------------------------------------------------
// 0x10316b10 — two-band selection mask
// ---------------------------------------------------------------------------

/// `0x10316b10` — pick two of `groups` harmonic groups and return them as a
/// bit mask.
///
/// The harmonic count `L` comes from [`frame_size_class`] feeding
/// [`band_count_for_mode`] (plus one). The first selection walks the partition
/// table at `0x10381920` — six rows of eight nibble-sized group widths, one row
/// per `groups` value in `3..=8`, each row summing to exactly 8 — starting from
/// the offset `round(L · 16 · cfg[6] / 8)` and stopping at the group where the
/// running total reaches 8. That is: the eight harmonic bands are distributed
/// over `groups` slots, and the walk finds which slot a given fractional
/// position falls in.
///
/// The second selection is arithmetic: `groups - (L + 2) / 3`, clamped at zero.
///
/// Both shifts are 16-bit, so a count of 16 or more shifts the bit out entirely
/// rather than wrapping.
///
/// Domain: `3 <= groups <= 8`, otherwise the table index leaves the six rows.
/// Note that [`band_count_for_mode`]'s own table path indexes the cosine table
/// from word 300 and, for the frame classes this function can reach, runs past
/// its 512 words into the adjacent image data — so `cos_tbl` must be a window
/// wide enough to cover `300 + tbl_base + 2·cfg[1]`.
pub fn two_group_mask(
    cfg: &[i16],
    groups: i16,
    tbl_base: i16,
    cos_tbl: &[i16],
    parts: &[i16],
) -> u16 {
    let cls = frame_size_class(cfg[1]);
    let l = (band_count_for_mode(cls as i32, cfg[1], tbl_base, cos_tbl) as u16).wrapping_add(1);

    let start = high16_bL2b2(
        ((((l as u32) << 4) as u16 as i16 as i32)
            .wrapping_mul(cfg[6] as i32)
            .wrapping_mul(2)
            >> 4)
            .wrapping_add(0x4000),
    );

    let mut pick: i16 = 0;
    let mut fell_through = true;
    if 0 < groups {
        let mut acc = start;
        let mut i: i16 = 0;
        loop {
            let idx = (i as i32) + (groups as i32) * 8 - 0x18;
            acc = acc.wrapping_add(parts[idx as usize]);
            if acc >= 8 {
                if i < groups {
                    pick = i;
                    fell_through = false;
                }
                break;
            }
            i = i.wrapping_add(1);
            if i >= groups {
                break;
            }
        }
        if fell_through {
            pick = groups.wrapping_sub(1);
        }
    } else {
        pick = groups.wrapping_sub(1);
    }

    let q = div32((l as i16 as i32).wrapping_add(2), 3) as i16;
    let d = (groups as i32).wrapping_sub(q as i32) as i16;
    let c = if d < 0 { 0u16 } else { d as u16 };

    shl16_bL2b2(1, c) | shl16_bL2b2(1, pick as u16)
}

// ---------------------------------------------------------------------------
// 0x1031b7b0 — harmonic least-squares pitch refinement
// ---------------------------------------------------------------------------

/// `0x1031b7b0` — refine a fundamental by weighted least squares over its
/// harmonics.
///
/// The seed is the largest value of `corr[lo..n]` ([`argmax`]) refined to
/// sub-bin resolution by the three-point parabolic fit of [`parabolic_peak`];
/// normalising it gives the fundamental as `(mantissa, 15 - shift)`.
///
/// Harmonics `k = 1, 2, …` are then walked. [`band_edges`] projects the current
/// fundamental estimate to harmonic `k` and returns a four-word window
/// `[lo0, hi0, lo1, hi1]`; it reports 1 — and the walk stops — once the
/// predicted harmonic runs past bin `n`. Inside the window:
///
/// * from `k = 2` on, the local peak is re-found by [`argmax`] +
///   [`parabolic_peak`] over `corr[lo1..=hi1]`, so each harmonic contributes
///   its *own* measured position rather than the predicted one;
/// * [`sum_i16`] over `corr[lo0..=hi0]` and over `corr[lo1..=hi1]` feeds
///   [`ratio_percent`], giving a per-harmonic confidence in percent;
/// * [`update_ratio_accum`] folds `(k, confidence, measured position)` into a
///   running `Σ k·w·p / Σ k²·w` — the closed-form weighted least-squares
///   estimate of the fundamental — and writes the new estimate straight back
///   into the accumulator's first two words, so the next harmonic is projected
///   from the refined value.
///
/// At most 15 harmonics are used. The answer is returned as the mantissa
/// requantised by the accumulated exponent less 6.
///
/// The exponent handed to [`band_edges`] is assembled from two adjacent stack
/// words and its high half is *uninitialised* on the first pass; [`band_edges`]
/// only ever reads the low 16 bits, so the value is harmless. Verified by
/// mutation: substituting the high half changes nothing.
pub fn refine_pitch_by_harmonics(corr: &[i16], lo: i16, n: i16) -> i16 {
    // `argmax` runs over `corr + 2*lo`, so its index is relative to `lo`.
    let peak = argmax(&corr[lo as usize..], n.wrapping_sub(lo)).wrapping_add(lo);
    let v = parabolic_peak(peak, corr, n);
    let s = nrm30(v);

    // st[0..10]: fundamental mantissa/exponent, per-harmonic confidence pair,
    // per-harmonic position pair, and the two accumulators.
    let mut st = [0i16; 10];
    st[0] = high16_bL2b2(v.wrapping_shl(s as u32 & 0x1f));
    st[1] = 0xf - s;
    st[4] = st[0];
    st[5] = st[1];

    let mut edges = [0i16; 4];
    let mut k: i16 = 1;
    loop {
        let exp = ((st[2] as u16 as u32) << 16 | st[1] as u16 as u32) as i32;
        if band_edges(&mut edges, st[0], exp, k, n) == 1 {
            break;
        }
        let width1 = edges[3].wrapping_sub(edges[2]).wrapping_add(1);
        if k > 1 {
            let p = argmax(&corr[edges[2] as usize..], width1).wrapping_add(edges[2]);
            let v = parabolic_peak(p, corr, n);
            let s = nrm30(v);
            st[5] = 0xf - s;
            st[4] = high16_bL2b2(v.wrapping_shl(s as u32 & 0x1f));
        }
        let width0 = edges[1].wrapping_sub(edges[0]).wrapping_add(1);
        let s0 = sum_i16(&corr[edges[0] as usize..], width0);
        let s1 = sum_i16(&corr[edges[2] as usize..], width1);
        let mut e: i16 = 0;
        st[2] = ratio_percent(s1 as u32, s0, &mut e);
        st[3] = e;
        update_ratio_accum(k, &mut st);

        k = k.wrapping_add(1);
        if k >= 0x10 {
            break;
        }
    }

    high16_bL2b2(shift32_bL2b2(up16_bL2b2(st[0]), st[1].wrapping_sub(6)))
}

// ---------------------------------------------------------------------------
// 0x1031e1d0 — sixteen-band voicing and level tracker
// ---------------------------------------------------------------------------

/// `0x1031e1d0` — the encoder's sixteen-band voicing / noise tracker.
///
/// This is the per-frame update that turns the sixteen sub-band energies into
/// the voicing decision inputs, the background-noise estimate and the frame
/// SNR. It is a large sequencer, so the shape first:
///
/// **Per-frame set-up.** `0x5b95 / f0` (a [`div_norm`] of the constant
/// `0x5b950000` by the normalised fundamental) is the frame's harmonic density
/// term, carried into every band. [`band_gain_envelope`] builds the sixteen
/// per-band gains from the retained level pair at state `0x52` / `0x72`. A
/// tracking step `0.2 · (lvl - ref)` is derived from the two level words at
/// state `0x48` / `0x50` and clamped into `[0x100, 0x200]`; a level below
/// `-0x1900` (very quiet) forces the maximum step.
///
/// **Per band `i` of 16.** The new energy `new[i]` and the retained energy
/// `state[i]` are normalised and summed ([`add_norm`]), and
/// [`log_scaled_product`] takes the base-2 logarithm of that sum scaled by the
/// density term — this is the band's log energy, the quantity everything
/// downstream works in. Which tracker updates the band then depends on the
/// hangover counter at state `0x138`:
///
/// * counter below 8 — still in a transient — [`update_pitch_track`] folds the
///   observation into the running mean;
/// * otherwise, if the log energy exceeds the retained level plus the tracking
///   step ([`compare_scaled`]), [`blend_toward`] snaps the level upward;
/// * otherwise the band's own hangover word at state `0xd2` selects between
///   [`track_level_ar`] (attack/decay) and a hard reset to `0x7fff @ 2^5`.
///
/// [`accumulate_weighted_diff`] then accumulates `|retained - observed|`
/// weighted by the band energy over the low six bands — the weighted spectral
/// distortion — and [`noise_track_update`] folds the band into the
/// background-noise running/peak pair, gated by the band gain envelope.
///
/// **Outputs.** Band pairs are combined ([`add_norm`]) and written as eight
/// words from state `0x102`; band 0's raw level goes to state `0x132`; the
/// 64-bit sum of all sixteen log energies drives the long-term level at state
/// `0x13a` through a `0.602 / 0.8` one-pole (whose weights do not sum to one —
/// the tracker has a net gain of ≈1.4); state `0xf4` receives the weighted
/// distortion, `*peak` the scaled noise peak, `0xf2` the input exponent, and
/// the return value is the noise floor estimate.
///
/// Domain: the divides inside want non-degenerate inputs — a zero `f0` or a
/// zero smoothing denominator faults in [`div_norm`].
#[allow(clippy::too_many_arguments)]
pub fn band_voicing_tracker(
    state: &mut [i16],
    new: &[i32],
    new_exp: i16,
    f0: i16,
    passthru: i16,
    peak: &mut i16,
    gain: i16,
) -> u16 {
    let mut acc_m: i16 = 0; // local_54
    let mut acc_e: i16 = 0; // local_80
    let mut noise_m: i16 = 0; // local_5c
    let mut noise_e: i16 = 0; // local_64
    let mut cur_m: i16 = 0; // local_28
    let mut cur_e: i16 = 0; // local_30
    let mut peak_e: i16 = 0; // local_68
    let mut dist_m: i16 = 0; // local_44
    let mut dist_e: i16 = 0; // local_40
    let mut wsum_m: i16 = 32000; // local_50
    let mut wsum_e: i16 = 10; // local_4c
    let mut prev_m: i16 = 0; // local_74
    let mut prev_e: i16 = 0; // local_7c
    let mut out_at: usize = 0x81; // local_58 = state + 0x102

    *peak = 0;

    // Harmonic density term: 0x5b95 / f0.
    let fv = up16_bL2b2(f0);
    let nf = nrm30(fv) as u16;
    let mut dens_e: i16 = 0;
    let dens = div_norm(
        0x5b95_0000u32 as i32,
        0,
        high16_bL2b2(fv.wrapping_shl(nf as u32 & 0x1f)),
        (0xfu32.wrapping_sub(nf as u32)) as i16,
        &mut dens_e,
    ) as u16;

    let mut env = [0i16; 16];
    {
        let mut m = [0i16; 16];
        let mut e = [0i16; 16];
        m.copy_from_slice(&state[0x29..0x39]);
        e.copy_from_slice(&state[0x39..0x49]);
        band_gain_envelope(&m, &e, &mut env);
    }

    // Tracking step, clamped into [0x100, 0x200].
    let lvl = state[0x24];
    let step: i32 = if lvl < -0x18ff {
        0x200
    } else {
        let d = (lvl as i32).wrapping_sub(state[0x28] as i32) as i16;
        let t = (((d as i32).wrapping_mul(0x3332)) >> 16) as i16;
        if t > 0x200 {
            0x200
        } else if t <= 0x100 {
            0x100
        } else {
            t as u16 as i32
        }
    };

    let mut sum: i64 = 0;
    for band in 0..16i16 {
        let i = band as usize;
        let f2 = state[0x79] as u16; // state + 0xf2

        let a = new[i];
        let b = get32_bL2b2(state, 2 * i);
        let na = nrm30(a) as u16;
        let nb = nrm30(b) as u16;
        let mut e_sum: i16 = 0;
        let m_sum = add_norm(
            high16_bL2b2(a.wrapping_shl(na as u32 & 0x1f)),
            (new_exp as u16).wrapping_sub(na) as i16,
            high16_bL2b2(b.wrapping_shl(nb as u32 & 0x1f)),
            f2.wrapping_sub(nb) as i16,
            &mut e_sum,
        ) as u16;
        e_sum = e_sum.wrapping_sub(1);

        let mut lac = state[0x29 + i] as u16; // retained level mantissa
        let mut la0 = state[0x39 + i] as u16; // retained level exponent
        let mut l94 = state[0x49 + i] as u16;
        let mut la4 = state[0x59 + i] as u16;

        let mut d_mant: i16 = 0;
        let mut d_exp: i16 = 0;
        let mut log_e: i16 = 0;
        let log_m = log_scaled_product(
            &mut d_mant,
            &mut d_exp,
            &mut log_e,
            band,
            m_sum as i16,
            e_sum,
            dens as i16,
            dens_e,
        ) as u16 as i16;

        let mut thr_e: i16 = 0;
        let thr = add_norm(lac as i16, la0 as i16, step as i16, 7, &mut thr_e) as u16;

        let hang = state[0x9c]; // state + 0x138
        let ret: u16;
        if hang < 8 {
            let mut c = l94 as i16;
            let mut ag = la4 as i16;
            let mut v = lac as i16;
            let mut e = la0;
            ret = update_pitch_track(
                &mut v, &mut e, &mut c, &mut ag, log_m, log_e, passthru, hang,
            ) as u16;
            lac = v as u16;
            la0 = e;
            l94 = c as u16;
            la4 = ag as u16;
        } else if compare_scaled(log_m, log_e, thr as i16, thr_e) {
            let mut oa = l94 as i16;
            let mut ob = la4 as i16;
            let mut v = lac as i16;
            let mut e = la0;
            ret = blend_toward(
                &mut v,
                &mut e,
                &mut oa,
                &mut ob,
                log_m,
                log_e as u16,
                passthru as u16,
            );
            lac = v as u16;
            la0 = e;
            l94 = oa as u16;
            la4 = ob as u16;
        } else {
            let age = state[0x69 + i]; // state + 0xd2
            if age > 0 {
                let mut m = l94 as i16;
                let mut e = la4 as i16;
                ret = track_level_ar(&mut m, &mut e, log_m, log_e, age as i32, step as i16) as u16;
                l94 = m as u16;
                la4 = e as u16;
            } else {
                lac = l94;
                la0 = la4;
                ret = passthru as u16;
                l94 = 0x7fff;
                la4 = 5;
            }
        }

        state[0x69 + i] = ret as i16;
        state[0x49 + i] = l94 as i16;
        state[0x59 + i] = la4 as i16;
        state[0x29 + i] = lac as i16;
        state[0x39 + i] = la0 as i16;
        put32_bL2b2(state, 2 * i, new[i]);

        let mut diff_m: i16 = 0;
        let mut diff_e: i16 = 0;
        accumulate_weighted_diff(
            &mut dist_m,
            &mut dist_e,
            &mut wsum_m,
            &mut wsum_e,
            &mut diff_m,
            &mut diff_e,
            lac as i16,
            la0 as i16,
            log_m,
            log_e,
            m_sum as i16,
            e_sum,
            band,
        );

        noise_track_update(
            &mut acc_m,
            &mut acc_e,
            &mut noise_m,
            &mut noise_e,
            m_sum as i16,
            e_sum,
            log_m,
            log_e,
            lac as i16,
            la0 as i16,
            &mut cur_m,
            &mut cur_e,
            peak,
            &mut peak_e,
            band,
            gain,
            env[i],
            state[0x24],
            state[0x28],
        );

        let dm = diff_m;
        let de = diff_e;
        if band & 1 == 0 {
            prev_m = dm;
            prev_e = de;
        } else {
            let mut e = prev_e;
            let r = add_norm(prev_m, prev_e, dm, de, &mut e);
            prev_m = r;
            prev_e = e;
            state[out_at] = high16_bL2b2(shift32_bL2b2(up16_bL2b2(r), e.wrapping_sub(7)));
            out_at += 1;
        }

        if band == 0 {
            state[0x99] = high16_bL2b2(shift32_bL2b2(up16_bL2b2(dm), de.wrapping_sub(5)));
        }

        sum = sum.wrapping_add(shift32_bL2b2(up16_bL2b2(log_m), log_e.wrapping_sub(7)) as i64);
    }

    // Long-term level, a one-pole whose weights deliberately do not sum to 1:
    // 0x9a1c/2^16 = 0.602 on the frame mean, 0xcccc/2^16 = 0.8 on the retained
    // value, so the tracker has a net gain of about 1.4.
    let mean = (((sum >> 4) as i32).wrapping_add(0x0700_0000) >> 16) as i16;
    state[0x9d] = high16_bL2b2(
        (mean as i32)
            .wrapping_mul(0x9a1c)
            .wrapping_add((state[0x9d] as i32).wrapping_mul(0xcccc)),
    );

    let mut e88: i16 = 0;
    let q = div_norm(up16_bL2b2(dist_m), dist_e, wsum_m, wsum_e, &mut e88);
    state[0x7a] = high16_bL2b2(shl_sat(up16_bL2b2(q), e88.wrapping_sub(2)));

    let v = shl_sat((*peak as i32).wrapping_mul(0x3332), peak_e.wrapping_sub(3));
    let r = v.wrapping_add(0x8000);
    let r = if ((r ^ v) & r) < 0 {
        0x7fff_ffffi32.wrapping_add(if v < 0 { 1 } else { 0 })
    } else {
        r
    };
    *peak = high16_bL2b2(r);

    let (fm, fe) = if noise_e < 3 {
        (0x4000i16, 3i16)
    } else {
        (noise_m, noise_e)
    };
    let mut e80 = acc_e;
    let s = div_norm(up16_bL2b2(acc_m), acc_e, fm, fe, &mut e80);
    let v = shl_sat(up16_bL2b2(s), e80.wrapping_sub(3));
    let r = v.wrapping_add(0x8000);
    let r = if ((r ^ v) & r) < 0 {
        0x7fff_ffffi32.wrapping_add(if v < 0 { 1 } else { 0 })
    } else {
        r
    };
    state[0x79] = new_exp;
    (r >> 16) as u16
}

// ======================= from bL2b3 =======================

// ---------------------------------------------------------------------------
// shared primitives
// ---------------------------------------------------------------------------

/// The block-exponent bit-scan as the original *inlines* it (SPEC §4.1).
///
/// `bsr` leaves its destination unchanged when the source is zero, and at every
/// one of the eight sites in this batch the destination already holds that same
/// zero source — so the scan yields **0**, not 31, and the shift comes out
/// `0x1e`. The decompiler renders these as a search loop primed with `0x1f`,
/// which would give `-1`; transcribing that is wrong. The case is reachable:
/// `x == -1` makes the magnitude `~x == 0`, and three of the sites here scan a
/// *sum*, which really can be `-1`.
///
/// The count is zero-extended to 16 bits by the following `movzx`, which is why
/// this returns `u16`.
#[inline]
fn norm_shift_bL2b3(v: i32) -> u16 {
    let m = if v < 0 { !(v as u32) } else { v as u32 };
    let b = if m == 0 { 0 } else { 31 - m.leading_zeros() };
    0x1eu32.wrapping_sub(b) as u16
}

/// Normalise a 32-bit accumulator to a Q15 mantissa and the exponent the
/// original stores alongside it — the *negated* shift, so callers add it.
/// A zero accumulator normalises to `(0, 0)` rather than being scanned.
#[inline]
fn norm_pair(x: i32) -> (i16, i16) {
    let n = if x == 0 { 0 } else { norm_shift_bL2b3(x) };
    let m = (((x as u32) << (n as u32 & 0x1f)) as i32 >> 16) as i16;
    (m, (n as i16).wrapping_neg())
}

/// The library's open-coded variable shift: non-negative counts shift left
/// (wrapping), counts in `-30..0` shift right arithmetically, and anything
/// below that is a full sign fill.
#[inline]
fn shift32_bL2b3(v: i32, s: i16) -> i32 {
    if s >= 0 {
        ((v as u32) << (s as u32 & 0x1f)) as i32
    } else if s < -0x1e {
        v >> 31
    } else {
        v >> ((-s) as u32 & 0x1f)
    }
}

#[inline]
fn high16_bL2b3(x: i32) -> i16 {
    (x >> 16) as i16
}

// ---------------------------------------------------------------------------
// Ported functions.
// ---------------------------------------------------------------------------

/// `0x1030f630` — conditional log-domain gain boost.
///
/// Gates on two tests and, when both pass, multiplies the caller's Q15 gain by
/// `2^(headroom · rate)`.
///
/// 1. The block-floating level `(level, level_exp)` is brought to a fixed point
///    five binades below its own exponent — `level << 16` shifted by
///    `level_exp - 5`, then taken as its high word — and must exceed `floor`.
/// 2. The headroom `hi - lo - bias`, formed in Q16 so the subtraction cannot
///    lose the low bits, must be positive. Its high word is clamped to `0x200`,
///    capping the boost this call may apply.
///
/// The clamped headroom times `rate` (doubled, the standard Q15×Q15 → Q31
/// idiom) is normalised and fed to [`pow2_scaled`] at input exponent
/// `8 - shift` and output exponent 3, so the exponential is evaluated on the
/// mantissa with the normalisation folded into its scale. The result multiplies
/// the gain, and the product is shifted back by 3 with saturation
/// ([`shl_sat`]) before its high word is stored.
///
/// Returns 0 when the gain was updated and 1 when either gate rejected — an
/// "applied?" flag, not an error code.
pub fn boost_gain_if_above_floor(
    gain: &mut i16,
    lo: i16,
    level: i16,
    level_exp: i16,
    hi: i16,
    floor: i16,
    rate: i16,
    bias: i16,
) -> u32 {
    let at_scale = high16_bL2b3(shift32_bL2b3(
        (level as i32) << 16,
        level_exp.wrapping_sub(5),
    ));
    if at_scale <= floor {
        return 1;
    }
    let headroom = ((hi as i32) << 16)
        .wrapping_sub((lo as i32) << 16)
        .wrapping_sub((bias as i32) << 16);
    if headroom <= 0 {
        return 1;
    }
    let mut h = high16_bL2b3(headroom);
    if h > 0x200 {
        h = 0x200;
    }

    let p = (h as i32).wrapping_mul(rate as i32).wrapping_mul(2);
    let sh = if p == 0 { 0 } else { norm_shift_bL2b3(p) };
    let mant = (((p as u32) << (sh as u32 & 0x1f)) as i32 >> 16) as i16;
    let g = pow2_scaled(mant, 8i16.wrapping_sub(sh as i16), 3) as i16;

    let prod = (g as i32).wrapping_mul(*gain as i32).wrapping_mul(2);
    *gain = high16_bL2b3(shl_sat(prod, 3));
    0
}

/// `0x10310c50` — per-slot spectral weights and a spectral-tilt bit.
///
/// One analysis slot's eight sub-band magnitudes are turned into eight
/// inverse-energy weights, plus a single flag that says whether the slot's
/// energy leans towards the high bands.
///
/// `slot` indexes an eight-word-per-slot matrix; `stride` is the neighbour
/// distance, so `slot ± stride` are the adjacent slots. `active_mask` carries
/// one bit per slot (bit `slot + 2`), and the whole computation is skipped —
/// weights and flag zeroed — when the current slot's bit is clear.
///
/// Data flow when the slot is active:
///
/// - `mags[slot]`, floored at 1, is copied to `clamped` (this happens
///   unconditionally, before the gate).
/// - The same eight words go through the 8-point harmonic cosine transform
///   [`harmonic_dft`], keeping two coefficients. The flag is
///   `2·c1 + c0 < 0` — c0 is the slot's mean level and c1 its first cosine
///   moment, so the test is a spectral tilt with the tilt weighted twice.
/// - The eight per-band accumulators come from the slot history `bands`: with a
///   neighbour available ([`band_min_of_max`], the `min(max(prev, next), cur)`
///   floor) or, for slot 0 or an inactive neighbour, from [`voiced_scale8`].
/// - [`log_weight8`] converts those to `-log2` weights. Its per-band shift
///   vector is **four zeros followed by four copies of the tilt flag**, built
///   on the stack as one zeroed `i64` and two copies of `flag | flag << 16`.
///   So a high-leaning slot pre-scales the upper four bands by 2.
///
/// The activity mask is read as a 16-bit register, so the shift count is masked
/// to 5 bits and a count of 16..31 is a full sign fill of the *low half*; the
/// caller's high half is never consulted (SPEC §4.2).
///
/// Domain: `slot - stride` and `slot + stride` must be valid slot indices.
#[allow(clippy::too_many_arguments)]
pub fn slot_band_weights(
    out: &mut [i16],
    clamped: &mut [i16],
    bands: &[i16],
    mags: &[i16],
    active_mask: i32,
    slot: i32,
    stride: i32,
    flag: &mut u16,
    cos_tbl: &[i16],
) {
    let s = slot as i16 as i32;
    let d = stride as i16 as i32;
    let m = &mags[(s as usize) * 8..][..8];

    for j in 0..8 {
        let v = m[j];
        clamped[j] = if v < 1 { 1 } else { v };
    }

    let mask16 = active_mask as i16 as i32;
    let sh_cur = ((slot as u32).wrapping_add(2) as u8 & 0x1f) as u32;
    if (mask16 >> sh_cur) & 1 == 0 {
        out[..8].fill(0);
        *flag = 0;
        return;
    }

    copy16_batch3(out, m, 8);
    harmonic_dft(out, 8, 2, cos_tbl);

    let t = ((out[0] as i32) << 16) >> 1;
    let t = if t == i32::MIN {
        i32::MAX
    } else {
        t.wrapping_neg()
    };
    let f = u16::from(((out[1] as i32) << 16) < t);
    *flag = f;
    let shifts = [0i16, 0, 0, 0, f as i16, f as i16, f as i16, f as i16];

    let sh_prev = (((slot as u32).wrapping_sub(stride as u32) as u8).wrapping_add(2) & 0x1f) as u32;
    let has_prev = (slot as i16) != 0 && (mask16 >> sh_prev) & 1 != 0;

    let mut acc = [0i32; 8];
    let cur = &bands[(s as usize) * 8..];
    let next = &bands[((s + d) as usize) * 8..];
    if has_prev {
        let prev = &bands[((s - d) as usize) * 8..];
        band_min_of_max(
            &mut acc,
            cur,
            prev,
            next,
            active_mask as i16,
            stride as i8,
            slot as i8,
        );
    } else {
        voiced_scale8(
            &mut acc,
            cur,
            next,
            active_mask as i16,
            stride as i16,
            slot as i16,
        );
    }

    log_weight8(out, &acc, 0, &shifts);
}

/// `0x10313cb0` — magnitude-squared spectrum to a log-magnitude complex vector.
///
/// Each 32-bit bin is individually normalised, its mantissa passed to
/// [`log2_at_q`] at Q5 with the exponent `exp - shift`, and the result written
/// as the **real** part of an interleaved (re, im) pair whose imaginary part is
/// zeroed. The output is therefore laid out ready for a complex transform of
/// the log spectrum.
///
/// A bin whose mantissa collapses to zero, or whose exponent falls below
/// `-0x1f`, is floored at `0x8000` (the most negative Q15 value) — the "log of
/// zero" sentinel, not a wrap.
///
/// `n` is signed: a non-positive count writes nothing. Returns a constant 4,
/// which is the block exponent the caller attaches to the Q5 logs.
pub fn log_spectrum_pairs(dst: &mut [i16], src: &[i16], exp: i32, n: u16) -> u32 {
    if (n as i16) > 0 {
        for i in 0..n as usize {
            let v = rd32(src, 2 * i);
            let sh = if v == 0 { 0 } else { norm_shift_bL2b3(v) };
            let mant = (((v as u32) << (sh as u32 & 0x1f)) as i32 >> 16) as i16;
            let e = exp.wrapping_sub(sh as i32) as i16;
            dst[2 * i] = if mant == 0 || e <= -0x20 {
                0x8000u16 as i16
            } else {
                log2_at_q(mant, e, 5) as i16
            };
            dst[2 * i + 1] = 0;
        }
    }
    4
}

/// `0x103173a0` — normalise a 32-bit block to Q15 words and fold the shift into
/// a running exponent.
///
/// [`i32_block_exp`] gives the block's exponent (already negated, so it is what
/// the caller adds), [`round_block_i32`] shifts every word by that amount with
/// saturation and round-half-up, and `exp` accumulates it. This is SPEC §4.4's
/// normalise-and-shift step applied to a whole vector rather than one
/// accumulator.
///
/// The original passes the block exponent **zero-extended** from 16 bits; the
/// callee truncates the difference back to 16 bits, so the sign is recovered
/// and only the low half is ever significant.
pub fn block_normalise_to_q15(dst: &mut [i16], src: &[i32], n: i16, exp: &mut i16) {
    let be = i32_block_exp(src, n) as i16;
    round_block_i32(dst, (be as u16) as i32, src, 0, n as u16);
    *exp = exp.wrapping_add(be);
}

/// `0x1031c420` — the encoder's pitch-doubling test.
///
/// Two candidate fundamentals are scored by the same ratio and the better one
/// is returned. For a candidate the numerator is [`sum_bands`] — the energy
/// captured by bands spaced at the candidate's harmonic step — and the
/// denominator is [`sum_i16`] over the tail of the 32-word energy vector from
/// that candidate's start bin, i.e. the total energy the bands are competing
/// for. Both are normalised to (mantissa, exponent) pairs.
///
/// Candidate 1 is `step` at band exponent 6, starting at bin `start1`.
/// Candidate 2 is the **same `step` at exponent 5** — half the band spacing,
/// therefore half the fundamental and twice the pitch period — starting at bin
/// `start2`. Candidate 2 is only considered at all when `step > 0x0ccd`; below
/// that the fundamental is already low enough that halving it is not a
/// plausible alternative.
///
/// The choice is a cross-multiplied ratio comparison,
/// `num2·den1 > num1·den2` via [`mant_exp_gt`], evaluated entirely in block
/// floating point: each product is doubled, normalised, and carries the sum of
/// its two operands' exponents less its own normalisation shift. The winner's
/// numerator and denominator are written out as mantissa/exponent pairs for the
/// caller to divide.
///
/// Domain: `start1`/`start2` in `0..=0x20`; `v` must hold at least 32 words.
#[allow(clippy::too_many_arguments)]
pub fn pick_pitch_by_band_ratio(
    num: &mut i16,
    num_exp: &mut i16,
    den: &mut i16,
    den_exp: &mut i16,
    v: &[i16],
    step: i32,
    start1: i32,
    start2: i32,
) {
    let n1 = 0x20i32.wrapping_sub(start1) as i16;
    let (den1, den1_e) = norm_pair(sum_i16(&v[(start1 as i16) as usize..], n1));
    let (num1, num1_e) = norm_pair(sum_bands(v, step as i16, 6, 0x40));

    if (step as i16) > 0x0ccd {
        let n2 = 0x20i32.wrapping_sub(start2) as i16;
        let (den2, den2_e) = norm_pair(sum_i16(&v[start2 as usize..], n2));
        let (num2, num2_e) = norm_pair(sum_bands(v, step as i16, 5, 0x40));

        let p2 = (num2 as i32).wrapping_mul(den1 as i32).wrapping_mul(2);
        let p1 = (den2 as i32).wrapping_mul(num1 as i32).wrapping_mul(2);
        let s2 = if p2 == 0 { 0 } else { norm_shift_bL2b3(p2) } as i16;
        let s1 = if p1 == 0 { 0 } else { norm_shift_bL2b3(p1) } as i16;
        let m2 = (((p2 as u32) << (s2 as u32 & 0x1f)) as i32 >> 16) as i16;
        let m1 = (((p1 as u32) << (s1 as u32 & 0x1f)) as i32 >> 16) as i16;
        let e2 = num2_e.wrapping_sub(s2).wrapping_add(den1_e);
        let e1 = den2_e.wrapping_sub(s1).wrapping_add(num1_e);

        if mant_exp_gt(m2, e2, m1, e1) {
            *num = num2;
            *num_exp = num2_e;
            *den = den2;
            *den_exp = den2_e;
            return;
        }
    }

    *num = num1;
    *num_exp = num1_e;
    *den = den1;
    *den_exp = den1_e;
}

// ======================= from bL2b4 =======================

// ---------------------------------------------------------------------------
// shared primitives (private copies of the idioms the originals inline)
// ---------------------------------------------------------------------------

/// The library's open-coded variable shift of an `i32`: a non-negative count
/// shifts left (wrapping), a count in `-30 ..= -1` shifts right arithmetically,
/// and anything at or below `-31` is a full sign fill. The count is taken from
/// `cl`, so only its low five bits reach the shifter.
#[inline]
fn shift32_bL2b4(v: i32, s: i16) -> i32 {
    if s >= 0 {
        ((v as u32) << (s as u16 as u32 & 0x1f)) as i32
    } else if s < -0x1e {
        v >> 31
    } else {
        v >> ((s.wrapping_neg()) as u16 as u32 & 0x1f)
    }
}

/// MSVC's `__allshl` (`0x10305930`). Not in the ported map — no layer-0 batch
/// reached it — so it is reproduced here rather than depending on a sibling
/// batch's file. Only `cl` is significant, matching the register ABI.
fn allshl_bL2b4(v: i64, count: u8) -> i64 {
    let hi = (v >> 32) as u32;
    let lo = v as u32;
    if count >= 0x40 {
        return 0;
    }
    if count >= 0x20 {
        let n = count & 0x1f;
        return (((lo << n) as u64) << 32) as i64;
    }
    let n = count & 0x1f;
    let (new_hi, new_lo) = if n == 0 {
        (hi, lo)
    } else {
        ((hi << n) | (lo >> (32 - n)), lo << n)
    };
    (((new_hi as u64) << 32) | new_lo as u64) as i64
}

/// A 16-bit left shift as the hardware performs it: the count is masked to five
/// bits and a count of 16..31 shifts the whole word out.
#[inline]
fn shl16_bL2b4(v: u16, c: u32) -> u16 {
    let c = c & 0x1f;
    if c >= 16 {
        0
    } else {
        v << c
    }
}

/// A 32-bit arithmetic right shift with the x86 5-bit count mask.
#[inline]
fn sar32(v: i32, c: u32) -> i32 {
    v >> (c & 0x1f)
}

// ---------------------------------------------------------------------------
// 0x10310460 — tone / signalling frame packer
// ---------------------------------------------------------------------------

/// `0x10310460` — pack a tone (signalling) frame into the channel bit array.
///
/// A tone frame replaces the usual speech parameters with a fixed layout:
///
/// | field | bits | contents |
/// |---|---|---|
/// | marker | 6 | `0x3f` — an all-ones escape the decoder recognises |
/// | amplitude, high | 6 | bits 6..1 of a 7-bit level index |
/// | tone id | 8 × *n* | the tone identifier, repeated |
/// | amplitude, low | 1 | bit 0 of the level index |
///
/// `n = (total_bits - 13) >> 3` is taken from word 2 of the bit-allocation
/// config block that `init` builds at `state+0x7a0` (FINDINGS §4): 4 repeats
/// for AMBE's 49 bits, 9 for IMBE's 88. The repeat count is what makes one
/// routine serve both codecs — the amplitude and marker fields cost a fixed 13
/// bits and the tone id fills whatever is left, in whole bytes.
///
/// The level is quantised from a Q15 log amplitude by
/// `clamp((a + 0x800) * 0x220 >> 17, 0, 0x7f)`, i.e. a 7-bit uniform quantiser
/// with a step of `2^17/0x220 = 241.4` over the useful range and an offset of
/// `-0x800`. Which amplitude is quantised depends on the tone id: ids below
/// `0x80` are single tones and take `amp[1]`, while `id >= 0x80` marks a dual
/// tone and takes the mean `(amp[0] + amp[1]) >> 1`, clamped to
/// `[-0x5000, 0x6fff]` first.
///
/// Splitting the level index 6 + 1 around the repeated id is not decorative:
/// it puts the marker, the coarse level and the id in the first bits of the
/// frame so a truncated payload still carries them.
pub fn pack_tone_frame(tone: i16, amp: &[i16], cfg: &[i16], cursor: &mut usize, dst: &mut [i16]) {
    emit_bits(cursor, dst, 6, 0x3f);

    let level = if tone < 0x80 {
        amp[1]
    } else {
        // Both operands are promoted by << 16 and halved before the add, so the
        // sum cannot overflow; the result is (amp[0] + amp[1]) >> 1.
        let s = (((amp[1] as i32) << 16) >> 1).wrapping_add(((amp[0] as i32) << 16) >> 1);
        let m = (s >> 16) as i16;
        // This clamp is invisible downstream: the quantiser below already maps
        // everything from about -0x800 down to index 0 and everything from
        // 0x6fdf up to index 0x7f, so both limits sit inside a saturated
        // region. Transcribed because it is what the original does, not
        // because a differential test can see it.
        if m > 0x6fff {
            0x6fff
        } else if m < -0x5000 {
            -0x5000
        } else {
            m
        }
    };

    let q = ((level as i32).wrapping_add(0x800)).wrapping_mul(0x220) >> 17;
    let mut idx = q as i16;
    if idx > 0x7f {
        idx = 0x7f;
    } else if idx < 0 {
        idx = 0;
    }
    emit_bits(cursor, dst, 6, idx >> 1);

    let mut n = (((cfg[2] as i32) - 0xd) >> 3) as i16;
    while n > 0 {
        emit_bits(cursor, dst, 8, tone);
        n = n.wrapping_sub(1);
    }
    emit_bits(cursor, dst, 1, idx & 1);
}

// ---------------------------------------------------------------------------
// 0x10312a10 — voicing quantiser front door
// ---------------------------------------------------------------------------

/// `0x10312a10` — quantise the band voicing decisions, or force all-unvoiced.
///
/// `enable == 1` runs the voicing codebook search (`0x10312700`) over the
/// weight vectors `w_a`, `w_b`, `w_c` against the packed-dibit book at
/// `0x10381720`, then expands the winning codeword into the two 32-bit
/// nibble-per-band masks the rest of the encoder uses. The returned index is
/// the *sub-sampled* one: the book is 128 entries and an `nbits`-bit field
/// selects one of `2^nbits` of them at stride `2^(7-nbits)`, so the table is
/// indexed by `idx << (7 - nbits)` while the transmitted field is `idx`.
///
/// Any other `enable` short-circuits: both masks are cleared and the index of
/// book entry 64 in the same sub-sampled space, `0x40 >> (7 - nbits)`, is
/// returned. Entry 64 is the all-unvoiced pattern (`0x55555555`), so this is
/// the "declare the frame unvoiced without searching" path — used for tone,
/// silence and repeat frames.
///
/// The 16-bit `idx << (7 - nbits)` is a hardware shift: a count of 16 or more
/// clears the word, and the product is sign-extended before indexing.
pub fn quantise_voicing_bL2b4(
    enable: i16,
    mask0: &mut u32,
    mask1: &mut u32,
    w_a: &[i16],
    w_b: &[i16],
    w_c: &[i16],
    book: &[u32],
    nbits: i32,
) -> u16 {
    let sh = (7i32).wrapping_sub(nbits) as u32;
    if enable == 1 {
        let idx = voicing_codebook_search(w_a, w_b, w_c, book, nbits as u8, 1) as u16;
        let slot = shl16_bL2b4(idx, sh) as i16 as i32;
        expand_dibits(mask0, mask1, book[slot as usize]);
        idx
    } else {
        *mask1 = 0;
        *mask0 = 0;
        sar32(0x40, sh) as i16 as u16
    }
}

// ---------------------------------------------------------------------------
// 0x10316290 — two-band energy pair at a common exponent
// ---------------------------------------------------------------------------

/// `0x10316290` — sum two five-bin runs of an energy spectrum and return both
/// at one shared block exponent.
///
/// Each run is accumulated with [`accumulate_exp`] (`0x10317140`), which sums
/// five `i32` in 64 bits, normalises, and folds the normalisation into a copy
/// of the caller's exponent `exp`. The two results generally land on different
/// exponents; the larger is chosen as the common one and both mantissas are
/// shifted to it, then rounded down to `i16` by taking the high half.
///
/// This is the band-pair comparison the voicing decision needs: a voicing test
/// only means anything when the two energies share a scale.
///
/// The exponent out-parameters alias 32-bit stack slots that the caller
/// pre-loads with the zero-extended `exp`, and the callee writes only their low
/// half — so the shift counts are differences of *zero-extended* `u16`
/// exponents, not sign-extended ones (SPEC §4.2). The returned value is the
/// common exponent; the original leaves stale high bits in `eax`, which the
/// caller masks off.
pub fn band_pair_at_common_exp(
    out_a: &mut i16,
    out_b: &mut i16,
    energy: &[i32],
    exp: i16,
    idx_a: i16,
    idx_b: i16,
) -> u16 {
    let mut ea = exp;
    let va = accumulate_exp(&energy[idx_a as usize..], 5, &mut ea) as i32;
    let mut eb = exp;
    let vb = accumulate_exp(&energy[idx_b as usize..], 5, &mut eb) as i32;

    let ua = ea as u16 as i32;
    let ub = eb as u16 as i32;
    let common = if ea > eb { ua } else { ub };

    *out_a = (shift32_bL2b4(va, (ua - common) as i16) >> 16) as i16;
    *out_b = (shift32_bL2b4(vb, (ub - common) as i16) >> 16) as i16;
    common as u16
}

// ---------------------------------------------------------------------------
// 0x1031ab80 — sub-band spectral analysis driver
// ---------------------------------------------------------------------------

/// `0x1031ab80` — the encoder's sub-band analysis loop: 32-point transform per
/// 10 ms hop, magnitude, square root, and interpolation onto the output grid.
///
/// For each of `frames` hops (two 16-bin rows per hop, i.e. one row per 8-sample
/// half hop) the routine:
///
/// 1. windows 32 input samples with the symmetric 32-wide taper at
///    `0x1037fed4` ([`apply_sym_window`]) into a 32-word buffer, the second
///    half written forward from word 16 and the first half backward from
///    word 15 — the standard FFT input rotation;
/// 2. takes the 32-point real DFT in place ([`dft32_real`]);
/// 3. forms `2·(re² + im²)` for the 16 retained bins ([`cplx_mag_sq_x2`]);
/// 4. accumulates that power spectrum into the caller's long-term accumulator
///    at the caller's exponent ([`acc_add_shifted`]);
/// 5. writes bin 0 as a floored-at-zero DC term and bins 1..15 as
///    `sqrt(power)` rescaled to the caller's exponent ([`sqrt_to_exp`]);
/// 6. advances the input by 4 samples — the original's `add edi, 8` is a
///    *byte* pointer bump, so consecutive rows overlap by 28 of their 32
///    samples and one hop covers 8 new samples.
///
/// The rows land in `work` after a 0x50-word history prefix, which is seeded
/// from `hist` on entry — verbatim if the caller's exponent is unchanged since
/// last call, otherwise rescaled by [`scale_block_exp`]. That prefix is what
/// lets the 7-tap interpolation filter reach backwards.
///
/// Every time the hop counter reaches the phase marker, [`interp_filter`]
/// (`0x1031adf0`, taps at `0x1037fef4`) resamples the accumulated rows into
/// `out` — 16 output bands, row stride 11 — and the working buffer is either
/// slid down by 0x20 words (the common case, keeping the history prefix
/// aligned) or, on the final hop, the tail is copied back into `hist` for the
/// next call. `*hist_exp` is left holding `exp` so the next call knows whether
/// a rescale is needed.
///
/// The exponent passed to `acc_add_shifted` is `2·exp`, because the power
/// spectrum is the square of a value carried at `exp`.
///
/// Domain: `frames >= 1` (a non-positive count only stores `*hist_exp = exp`
/// and returns); `input` must hold `0x20 + 4·(2·frames - 1)` words; `work`
/// must hold `0x50 + 0x20·frames` words; `out` must hold
/// `16·11` words from `off = 0`.
#[allow(clippy::too_many_arguments)]
pub fn subband_analysis(
    hist: &mut [i16],
    hist_exp: &mut i16,
    out: &mut [i16],
    work: &mut [i16],
    exp: i16,
    acc: &mut [i32],
    acc_exp: i16,
    input: &[i16],
    frames: i16,
    taper: &[i16],
    interp_coef: &[i16],
) {
    let last = (frames as i32) - 1;
    // The phase marker starts at 0 when there is more than one hop, otherwise
    // at the final hop index — a single-hop call flushes immediately.
    let mut marker: i32 = if last > 0 {
        0
    } else {
        ((frames as i32) - 1) as u16 as i32
    };

    if (exp as u16) == (*hist_exp as u16) {
        copy16_batch3(work, hist, 0x50);
    } else {
        scale_block_exp(work, hist, 0x50, *hist_exp as u16 as i32, exp as i32);
    }

    let pow_exp = (exp.wrapping_mul(2)) as u16 as i16;

    if frames < 1 {
        *hist_exp = exp;
        return;
    }

    let mut spec = [0i16; 32];
    let mut power = [0i32; 16];

    let mut col: i32 = 0; // local_a0 — output column for interp_filter
    let mut wp: usize = 0x50; // running write cursor into `work`
    let mut hop: i32 = 0; // local_94 — absolute hop index
    let mut hop16: i32 = 0; // local_88 — the same, compared as i16
    let mut in_off: usize = 0;

    loop {
        for _ in 0..2 {
            apply_sym_window(&mut spec, &input[in_off..], 0x20, 0x10, taper, 0x20);
            dft32_real(&mut spec);
            let dc = spec[0];
            cplx_mag_sq_x2(&mut power, &spec, 0x10);
            in_off += 4;
            acc_add_shifted(&power, pow_exp, acc, acc_exp, 0x10);

            work[wp] = if dc > 0 { dc } else { 0 };
            for j in 0..15usize {
                work[wp + 1 + j] = sqrt_to_exp(power[1 + j], pow_exp, exp) as i16;
            }
            wp += 0x10;
        }

        // `marker` starts at 0 and both it and `hop16` advance by one per
        // iteration, so with the argument shape the callers use they never
        // diverge and this test is always true — the interpolation runs once
        // per hop with a width of one column. The comparison is kept because
        // the original makes it.
        if (hop16 as i16) == (marker as i16) {
            interp_filter(out, interp_coef, work, (hop16 - col + 1) as i16, col as i16);
            if hop == last {
                let k = ((hop - (col as i16 as i32) + 1) * 0x20) as usize;
                copy16_batch3(hist, &work[k..], 0x50);
            } else {
                work.copy_within(0x20..0x20 + 0x50, 0);
                wp -= 0x20;
            }
            col += 1;
            marker += 1;
            if (marker as i16 as i32) > last {
                marker = ((frames as i32) - 1) as u16 as i32;
            }
        }

        hop += 1;
        hop16 += 1;
        if (hop16 as i16) >= frames {
            break;
        }
    }

    *hist_exp = exp;
}

// ---------------------------------------------------------------------------
// 0x1031c660 — per-band voicing measure
// ---------------------------------------------------------------------------

/// `0x1031c660` — accumulate the eight-band voicing measure `1 - E_err/E_sig`.
///
/// The spectrum is divided into eight bands whose edges are laid down by a
/// running Q16 accumulator: band *k* ends at bin `round(k · step)`, clamped to
/// `max_bin`, and *starts* at `round((k - 0.5) · step)` — the bands therefore
/// overlap by half a step, so a harmonic sitting on an edge contributes to both
/// neighbours instead of falling between them. `step` comes from
/// `div_norm(0x20000000, -3, spacing, -3)` renormalised to Q16, i.e. it is
/// `2^29 / spacing` in block-floating point: `spacing` is the harmonic spacing
/// and `step` the number of FFT bins per band.
///
/// For each band both `err` and `sig` are summed over the band's bins in 64
/// bits, normalised with [`norm64`] and brought to a mantissa/exponent pair by
/// `__allshl`/`__allshr` (SPEC §4.4), then [`one_minus_ratio`] forms
/// `1 - Σerr / Σsig` in Q15, floored at zero. The result is **minimised** into
/// `measure[k]`, so repeated calls over successive analysis windows keep the
/// worst (least voiced) evidence for each band.
///
/// `err` is the synthesis-error spectrum and `sig` the signal power spectrum;
/// a band where the single-harmonic model explains the energy well gives a
/// small ratio and a measure near `0x7fff`. That measure is what the voicing
/// codebook search (`0x10312700`) later quantises.
///
/// Domain: `spacing != 0` (a zero divisor faults inside `div_norm`), and the
/// band edges must stay inside `err` / `sig` — the routine clamps the upper
/// edge to `max_bin` but not the lower one.
#[allow(clippy::too_many_arguments)]
pub fn band_voicing_measure_bL2b4(
    measure: &mut [i16],
    sig: &[i32],
    sig_exp: i32,
    err: &[i32],
    err_exp: i16,
    spacing: i16,
    max_bin: u16,
) {
    let mut e: i16 = 0;
    let m = div_norm(0x2000_0000, -3, spacing, -3, &mut e);
    let half = shift32_bL2b4((m as i32) << 16, e.wrapping_sub(0xf));
    let step = half.wrapping_mul(2);

    let mut start: u16 = 0;
    let mut edge: i32 = 0;

    for b in 0..8usize {
        edge = edge.wrapping_add(step);
        let rounded = (edge.wrapping_add(0x8000) >> 16) as i16;
        let stop = if rounded > max_bin as i16 {
            max_bin as i32
        } else {
            rounded as u16 as i32
        };

        let base = (start as i16) as i32 * 4;
        let span = (stop - start as i32) as u16;

        // The numerator side (`err`) is accumulated first.
        let mut acc: i64 = 0;
        if span < 0x8000 {
            let o = (base / 4) as usize;
            for k in 0..=span as usize {
                acc = acc.wrapping_add(err[o + k] as i64);
            }
        }
        let n1 = norm64(acc as u32, (acc >> 32) as u32) as i16;
        let s1 = if n1 >= 0 {
            allshl_bL2b4(acc, n1 as u8)
        } else {
            allshr(acc, (n1 as i32).wrapping_neg() as u8)
        };
        let num_exp = (err_exp as i32 - n1 as u16 as i32) as u16 as i16;
        let num = ((s1 as i32) >> 16) as i16;

        let mut acc: i64 = 0;
        if span < 0x8000 {
            let o = (base / 4) as usize;
            for k in 0..=span as usize {
                acc = acc.wrapping_add(sig[o + k] as i64);
            }
        }
        let n2 = norm64(acc as u32, (acc >> 32) as u32) as i16;
        let s2 = if n2 >= 0 {
            allshl_bL2b4(acc, n2 as u8)
        } else {
            allshr(acc, (n2 as i32).wrapping_neg() as u8)
        };
        let den = ((s2 as i32) >> 16) as i16;
        let den_exp = (sig_exp - n2 as u16 as i32) as i16;

        let v = one_minus_ratio(num, num_exp, den, den_exp) as i16;
        if v < measure[b] {
            measure[b] = v;
        }

        start = ((edge.wrapping_sub(half).wrapping_add(0x8000)) >> 16) as u16;
    }
}

// ======================= from bL3b1 =======================

// ---------------------------------------------------------------------------
// local primitives (each module in this port keeps its own copy — SPEC §4)
// ---------------------------------------------------------------------------

/// Block exponent of a non-zero magnitude: `0x1e - bsr(x)`, i.e. the left shift
/// that normalises `x` to Q30 (SPEC §4). Zero is handled by the callers, which
/// short-circuit the `bsr` — see the note on the `-1` trap in SPEC §4.1: every
/// use here is on an already-absolute value, so the trap is not reachable.
#[inline]
fn norm_shift_abs(x: u32) -> u16 {
    debug_assert!(x != 0);
    let bsr = 31 - x.leading_zeros();
    (0x1eu32.wrapping_sub(bsr)) as u16
}

/// x86 variable arithmetic shift of an `i32`: positive shifts left, negative
/// right, count masked to five bits, below `-30` a full sign fill.
#[inline]
fn shift_i32(v: i32, sh: i16) -> i32 {
    if sh >= 0 {
        ((v as u32) << (sh as u32 & 0x1f)) as i32
    } else if sh < -0x1e {
        v >> 31
    } else {
        v >> ((sh.wrapping_neg() as u32) & 0x1f)
    }
}

/// Write a 32-bit immediate into two consecutive `i16` state words. The
/// original stores these as unaligned `mov dword ptr`, so the low half lands in
/// the first word.
#[inline]
fn wr32_bL3b1(st: &mut [i16], w: usize, v: u32) {
    st[w] = v as u16 as i16;
    st[w + 1] = (v >> 16) as u16 as i16;
}

// ---------------------------------------------------------------------------
// 0x10309ea0 — enc_init
// ---------------------------------------------------------------------------

/// `0x10309ea0` — `enc_init(state, mode, reset)`: the encoder's power-on state.
///
/// The mirror of `dec_init` (`0x10309cd0`). `reset` guards everything except
/// the final call: with `reset == 0` only the bit allocation and AGC
/// configuration are (re)installed, so the analysis history is preserved and
/// the codec can be switched between AMBE and IMBE mid-stream. `mode` is the
/// usual 1 = AMBE half-rate / 0 = IMBE selector and is only read by
/// [`init_bit_alloc_and_agc`].
///
/// The reset path is a map of the encoder state block. Byte offsets, with the
/// word index in brackets:
///
/// | region | contents |
/// |---|---|
/// | `0x000 .. 0x16e0` | zeroed (`0xb70` words) — everything below is seeded on top |
/// | `0x1170` \[`0x8b8`\] | 7 words of `-15`, a set of block exponents |
/// | `0x630` \[`0x318`\] | `0x2aab, 0x147b, 0x147b` — a level tracker's initial value and two copies of its bound |
/// | `0x63a` \[`0x31d`\] | 5 words of `0x1000` (0.125 Q15) |
/// | `0x644` \[`0x322`\] | 5 words of `0x7fff` |
/// | `0x64e` \[`0x327`\] | five 8-word rows of `3` at a stride of 8 words |
/// | `0x69e` \[`0x34f`\] | five 8-word rows of `1`, same stride |
/// | `0x6ee` \[`0x377`\] | 5 exponents of `-30` (`0xffe2`), one per row |
/// | `0x70a` \[`0x385`\] | 3 exponents of `-30` |
/// | `0x710` \[`0x388`\] | three 8-word rows of `0x7fff`, stride 8 words |
/// | `0x978`.. | the pitch tracker's seeds (see below) |
/// | `0xa78` \[`0x53c`\] | 16 zeroed words with 16 exponents of `-30` at `0xa98` |
/// | `0x122e` \[`0x917`\] | `0x2000` |
/// | `0x1372` \[`0x9b9`\] | starting gain for frame class `0x20` at `-10` dB |
/// | `0x137e` \[`0x9bf`\] | 6 words of `0xb001` |
/// | `0x16a2` \[`0xb51`\] | the AGC parameter set ([`set_agc_defaults`]) |
///
/// The pitch/level tracker block at `0x978 ..0xa78` is the interesting part: a
/// descending ramp of candidate pitch words `0xa401, 0xa001, 0x9c01, 0x9801,
/// 0x9401, 0x9001` at `0x98a`, ten more at `0x8c01` from `0x996`, then four
/// 16-word histories — `5` (`0x9aa`), `0x7fff` (`0x9ca`), `5` (`0x9ea`) and
/// `300` (`0xa0a`). `300` recurs at `0x986` and `0x984+2`, and `20000` at
/// `0xa74`; these are the level-tracker's decay limits.
///
/// `0x1030cdb0` ([`frame_size_class`]) is called with `0x20` and
/// its result **discarded** — `eax` is reloaded with `0x20` on the next
/// instruction. It is a pure function, so the call is dead and is not
/// reproduced here.
///
/// `state` must be at least `0xb80` words (`0x1700` bytes).
pub fn enc_init(st: &mut [i16], mode: i16, reset: i16) {
    if reset != 0 {
        fill16(st, 0, 0xb70);
        fill16(&mut st[0x8b8..], -15, 7);

        wr32_bL3b1(st, 0x318, 0x147b_2aab);
        st[0x31a] = 0x147b;

        for i in 0..5 {
            st[0x31d + i] = 0x1000;
            st[0x322 + i] = 0x7fff;
        }
        for i in 0..5 {
            fill16(&mut st[0x34f + 8 * i..], 1, 8);
            st[0x377 + i] = -30;
            fill16(&mut st[0x327 + 8 * i..], 3, 8);
        }
        for i in 0..3 {
            st[0x385 + i] = -30;
            fill16(&mut st[0x388 + 8 * i..], 0x7fff, 8);
        }

        wr32_bL3b1(st, 0x4c5, 0xa001_a401);
        wr32_bL3b1(st, 0x4c7, 0x9801_9c01);
        wr32_bL3b1(st, 0x4c9, 0x9001_9401);
        fill16(&mut st[0x4cb..], 0x8c01u16 as i16, 10);
        fill16(&mut st[0x4d5..], 5, 0x10);
        fill16(&mut st[0x4e5..], 0x7fff, 0x10);
        fill16(&mut st[0x4f5..], 5, 0x10);
        fill16(&mut st[0x505..], 300, 0x10);

        wr32_bL3b1(st, 0x4c3, 0xd301_012c);
        st[0x4c0] = 0xe201u16 as i16;
        st[0x53b] = -16;
        wr32_bL3b1(st, 0x4bc, 0x000f_a000);
        wr32_bL3b1(st, 0x539, 0x4e20_e200);
        fill16(&mut st[0x53c..], 0, 0x10);
        fill16(&mut st[0x54c..], -30, 0x10);
        wr32_bL3b1(st, 0x55c, 0);
        wr32_bL3b1(st, 0x4c2, 0x012c_0000);
        st[0x917] = 0x2000;

        st[0x9ba] = 0x20;
        gain_for_rate(&mut st[0x9b9], 0x20, -10);
        st[0x9b8] = 0;
        wr32_bL3b1(st, 0x9bf, 0xb001_b001);
        wr32_bL3b1(st, 0x9c1, 0xb001_b001);
        wr32_bL3b1(st, 0x9c3, 0xb001_b001);

        set_agc_defaults(st, -22, 20, -10);
    }

    // Unconditional: the bit-allocation vector at 0x138a and the AGC config
    // block at 0xabc.
    let (lo, hi) = st.split_at_mut(0x9c5);
    let alloc: &mut [i16; 13] = (&mut hi[..13]).try_into().unwrap();
    init_bit_alloc_and_agc(mode, alloc, &mut lo[0x55e..]);
}

// ---------------------------------------------------------------------------
// 0x1030d890 — the analysis front end
// ---------------------------------------------------------------------------

/// `0x1030d890` — window a segment of the input, transform it, and optionally
/// take its magnitude-squared spectrum.
///
/// This is the encoder's spectral analysis stage, and it is where the
/// `state+0x22c` magnitude-squared array documented in FINDINGS §7 comes from.
/// Four steps:
///
/// 1. `count` input words are copied into a 256-word scratch buffer and
///    renormalised in place ([`normalise_to_exp`], `0x103175d0`), which yields
///    a block exponent `s`. **The scratch buffer is the reason `count` is
///    capped at 256** — the window is 199 taps (SPEC §6), well inside it.
/// 2. The scratch is multiplied by the symmetric window `win` and laid into
///    `fft_buf` split around the buffer's ends ([`apply_sym_window`],
///    `0x1030d960`), zero-padding the middle to the transform length
///    `N = 1 << log2n`. That split is the circular time-alignment which puts
///    the window's centre at sample 0, so the transform's phase reference is
///    the middle of the analysis window rather than its edge.
/// 3. A real forward FFT ([`rfft_forward`], `0x1031a2f0`) at exponent
///    `exp_in + s` — the caller's exponent plus the credit earned in step 1.
/// 4. If `mag` is supplied, `N/2 + extra` bins are converted to `2(re²+im²)`
///    ([`cplx_mag_sq_x2`], `0x10317610`) and the returned exponent is
///    **doubled**, because squaring doubles the exponent.
///
/// Returns the spectrum's block exponent (`i16`; the original leaves junk in
/// the high half of `eax`, `mov ax, si`).
///
/// Domain: `0 < count <= 0x100`, `4 <= log2n <= 8`; `input` at least `count`
/// words, `win` at least `count/2 + 1`, `fft_buf` at least `N + 2`, `mag` at
/// least `N/2 + extra`. `out_off` must be 0 for an odd `count` or `count/2` for
/// an even one — any other value makes the two window halves and the zero-fill
/// overrun the transform buffer, in the original as much as here.
#[allow(clippy::too_many_arguments)]
pub fn analysis_fft(
    mag: Option<&mut [i32]>,
    input: &[i16],
    win: &[i16],
    exp_in: i32,
    count: i16,
    out_off: i16,
    fft_buf: &mut [i16],
    log2n: i16,
    extra: i16,
) -> i16 {
    let mut scratch = [0i16; 256];
    let mut exp: i16 = 0;

    copy16_batch3(&mut scratch, input, count);
    normalise_to_exp_inplace(&mut scratch, count as u16, &mut exp);

    let n = (1u16 << (log2n as u16 & 0x1f)) as i16;
    apply_sym_window(fft_buf, &scratch, n, out_off, win, count);

    let e = rfft_forward(
        fft_buf,
        exp_in.wrapping_add(exp as u16 as i32),
        log2n,
        extra,
    ) as i16;

    match mag {
        None => e,
        Some(m) => {
            cplx_mag_sq_x2(m, fft_buf, (n >> 1).wrapping_add(extra));
            e.wrapping_mul(2)
        }
    }
}

// ---------------------------------------------------------------------------
// 0x10312fc0 — IMBE channel de-interleave
// ---------------------------------------------------------------------------

/// `0x10312fc0` — de-interleave an 88-bit IMBE channel frame into parameter
/// order.
///
/// A three-instruction wrapper: `interleave_imbe_frame(chan, nharm, hbits,
/// alloc, store = false)` (`0x10312fe0`, SPEC §3.2). The `store` flag is the
/// direction switch inside [`copy16_batch4`]; hard-wiring it to `false`
/// makes this the **unpack** direction — the frame image is read into the
/// routine's scratch and the parameter fields are written back out over `chan`.
/// The encoder's packing side calls `0x10312fe0` directly with `store = true`.
pub fn imbe_frame_deinterleave(chan: &mut [i16], nharm: i16, hbits: &[i16], alloc: &[i16]) {
    interleave_imbe_frame(chan, nharm, hbits, alloc, false);
}

// ---------------------------------------------------------------------------
// 0x10313d40 — vector log2
// ---------------------------------------------------------------------------

/// `0x10313d40` — take the base-2 logarithm of a 32-bit energy vector and
/// return it as Q15 words on one shared block exponent.
///
/// The whole vector arrives at a single block exponent `exp`. The output
/// exponent is chosen as `(exp + 3) >> 2` — a quarter of the input exponent,
/// rounded up — before any element is converted, so every `log2` result is
/// produced pre-aligned to it by [`log2_at_exp`] (`0x1030b0b0`) rather than
/// being rescaled afterwards. Once the vector is filled it is renormalised as a
/// block ([`normalise_to_exp`], `0x103175d0`) and the shift credited to that
/// exponent, which is returned.
///
/// The `>> 2` is not arbitrary: a log2 of a value spanning `2^exp` needs only
/// about `exp` units of range, so a quarter of the input exponent leaves the
/// log comfortably inside a Q15 word while keeping the *whole vector* on one
/// exponent — the block-floating-point discipline of SPEC §4.
///
/// Domain: `n >= 0`. The conversion loop is guarded (`0 < (short)n`, SPEC §4.1)
/// but the renormalisation that follows is not, so a negative count runs the
/// original off the end of `dst`; no caller supplies one.
pub fn log2_vector(dst: &mut [i16], src: &[i16], exp: i16, n: i16) -> u16 {
    let mut target = (((exp as i32).wrapping_add(3)) >> 2) as i16;

    if n > 0 {
        for i in 0..n as u16 as usize {
            dst[i] = log2_at_exp(rd32(src, 2 * i) as u32, exp, target) as i16;
        }
    }

    normalise_to_exp_inplace(dst, n as u16, &mut target);
    target as u16
}

// ---------------------------------------------------------------------------
// 0x10316bf0 — signalling-tone detector
// ---------------------------------------------------------------------------

/// The `0.875·prev + 0.125·(x + 4)` one-pole used by the tone detector to
/// smooth its two levels across consecutive frames.
///
/// `0xdffe` is an **unsigned** Q16 fraction (57342 ≈ 0.875, SPEC §4.1), and the
/// `+ 4` on the new sample is a half-LSB rounding offset for the `<< 13`
/// (= `0.125` in Q16) that follows it.
#[inline]
fn tone_smooth(prev: i16, x: i16) -> i16 {
    let a = (prev as i32).wrapping_mul(0xdffe);
    let b = ((x as i32).wrapping_add(4)).wrapping_shl(13);
    (a.wrapping_add(b) >> 16) as i16
}

/// Clamp a log-domain level into the transmission range `[-0x1400, 0x7800]`,
/// the same span [`log_amp_clamped`] (`0x103164b0`) uses.
#[inline]
fn clamp_level(v: i16) -> i16 {
    if v > 0x7800 {
        0x7800
    } else if v >= -0x1400 {
        v
    } else {
        -0x1400
    }
}

/// `0x10316bf0` — decide whether this frame is a single signalling tone, and if
/// so report which one.
///
/// This is the encoder's counterpart of the decoder's tone path (SPEC §7.1's
/// "signalling tone" frame class). It runs on the magnitude-squared spectrum
/// produced by [`analysis_fft`] and answers one question: *is essentially all
/// the energy in this frame concentrated in a single spectral peak?*
///
/// The algorithm, in order:
///
/// 1. **Gate on the previous frame's disposition.** `st[4]` (byte `+8`) is the
///    last frame's class code. `0xff` means "was a tone" and always proceeds;
///    otherwise only a speech-range code (`5 ..= 0x7a`) is considered, and
///    anything else returns 0 immediately.
/// 2. **Find the peak.** [`argmax_i32`] over bins `5 ..123` (`0x76` of them),
///    biased back by 5, gives the peak bin — the returned tone identity. Bins
///    below 5 are excluded, i.e. below roughly 156 Hz.
/// 3. **Peak level.** The five bins centred on the peak are summed and
///    normalised ([`accumulate_exp`], `0x10317140`) and converted to a log
///    level with [`log2_at_q`] plus the caller's `offset`, clamped by
///    [`clamp_level`].
/// 4. **Floor level.** `ref_level << 16` (the caller's frame-energy reference)
///    minus the peak energy rescaled to `ref_exp` is the energy *not* in the
///    peak. Its absolute value is normalised and converted to a log level the
///    same way. So the two levels are "energy in the peak" and "everything
///    else".
/// 5. **Smoothing.** If the previous frame's peak bin was within one bin of
///    this one, both levels are blended with the retained values at `st[0xb]`
///    and `st[0xc]` by [`tone_smooth`] — one bin of tolerance, so a genuine
///    steady tone accumulates confidence over frames while a moving spectral
///    peak never does.
/// 6. **Threshold on the difference.** `delta = peak_level - floor_level`, in
///    the log domain, i.e. the peak-to-residual ratio in dB. It must clear
///    `0x1a93`; a peak at or below bin `0x12` must clear `0x2138` and one at or
///    below bin `0xa` must clear `0x2689` — low-frequency peaks are held to a
///    stricter test because speech has its own low-frequency concentration.
///    When `enable` is set, the previous frame was already a tone and the peak
///    is above bin `0x58`, the bar is raised to `0x2a85` instead: *leaving* a
///    high-frequency tone is made harder than entering one, which is
///    hysteresis in the wrong direction for entry and the right one for hold.
/// 7. **Shape test.** Finally the same five bins are renormalised
///    ([`block_normalise_to_q15`], `0x103173a0`) and put through
///    [`spread_test`] (`0x10315b00`), a second-moment-versus-mean-squared test
///    that rejects a five-bin window whose energy is spread rather than
///    concentrated. Only if that passes is the frame declared a tone.
///
/// On success `*peak_out` is the peak bin, `result[0]` is the marker
/// `0xb000` and `result[1]` the peak's clamped log amplitude, the two smoothed
/// levels are retained in `st`, and 1 is returned. Every rejection returns 0
/// and leaves `st` untouched.
///
/// Domain: `spec` at least 125 words; `st` at least 13 words (bytes `+8`,
/// `+0x16`, `+0x18`).
#[allow(clippy::too_many_arguments)]
pub fn detect_signalling_tone(
    peak_out: &mut i16,
    result: &mut [i16],
    st: &mut [i16],
    spec: &[i32],
    exp: u16,
    offset: i16,
    ref_level: i16,
    ref_exp: i32,
    enable: i16,
) -> i32 {
    let cls = st[4];
    if cls != 0x00ff && (cls < 5 || cls > 0x7a) {
        return 0;
    }

    // 2. peak bin
    let peak = argmax_i32(&spec[5..], 0x76).wrapping_add(5);
    let win = &spec[(peak as i32 - 2) as usize..][..5];

    // 3. peak level
    let mut pexp: i16 = exp as i16;
    let acc = accumulate_exp(win, 5, &mut pexp);
    let mant = ((acc as i32) >> 16) as i16;

    let mut lvl_peak: i16 = -0x1400;
    if pexp >= -0x3b && mant != 0 {
        let l = log2_at_q(mant, pexp.wrapping_add(0x1e), 5) as i16;
        lvl_peak = clamp_level(l.wrapping_add(offset));
    }

    // 4. floor level — the energy outside the peak
    let d = (pexp as u16 as i32).wrapping_sub(ref_exp);
    let scaled = shift_i32((mant as i32) << 16, d as i16);
    let mut diff = ((ref_level as i32).wrapping_shl(16)).wrapping_sub(scaled);

    let sh: u16 = if diff < 0 {
        if diff == i32::MIN {
            diff = 0x7fff_ffff;
            norm_shift_abs(0x7fff_ffff)
        } else {
            diff = diff.wrapping_neg();
            norm_shift_abs(diff as u32)
        }
    } else if diff == 0 {
        0
    } else {
        norm_shift_abs(diff as u32)
    };

    let fexp = ref_exp.wrapping_sub(sh as i32) as u16 as i16;
    let fmant = (((diff as u32) << (sh as u32 & 0x1f)) as i32 >> 16) as i16;

    let mut lvl_floor: i16 = -0x1400;
    if fexp >= -0x3b && fmant != 0 {
        let l = log2_at_q(fmant, fexp.wrapping_add(0x1e), 5) as i16;
        lvl_floor = clamp_level(l.wrapping_add(offset));
    }

    // 5. smooth against the retained levels when the peak has not moved
    let cls = st[4];
    let (mut lp, mut lf) = (lvl_peak, lvl_floor);
    if cls >= 5 {
        let dd = peak.wrapping_sub(cls);
        let near = if dd == -0x8000 {
            false
        } else {
            (dd as i32).abs() < 2
        };
        if near {
            lp = tone_smooth(st[0xb], lp);
            lf = tone_smooth(st[0xc], lf);
        }
    }

    // 6. peak-to-residual thresholds
    let delta = lp.wrapping_sub(lf);
    if enable != 0 && cls == 0x00ff && peak >= 0x58 && delta < 0x2a85 {
        return 0;
    }
    if delta < 0x1a93 {
        return 0;
    }
    if peak <= 0x12 && delta < 0x2138 {
        return 0;
    }
    if peak <= 0xa && delta < 0x2689 {
        return 0;
    }

    // 7. is the peak actually peaky?
    let mut shape = [0i16; 6];
    let mut sexp: i16 = exp as i16;
    block_normalise_to_q15(&mut shape, win, 5, &mut sexp);
    if !spread_test(mant, pexp as u16 as i32, &shape, sexp as u16 as i32, 5) {
        return 0;
    }

    *peak_out = peak;
    result[0] = 0xb000u16 as i16;
    result[1] = log_amp_clamped(mant, pexp as u16 as i32, offset) as i16;
    st[0xc] = lf;
    st[0xb] = lp;
    1
}

// ---------------------------------------------------------------------------
// 0x1031c960 — eight-band voicing decision
// ---------------------------------------------------------------------------

/// `0x1031c960` — score the eight analysis bands for voicing and return the
/// voiced/unvoiced bitmap.
///
/// The encoder's V/UV decision. `spec` is the frame's band-energy vector laid
/// out as eight contiguous 32-word blocks (`0x40` bytes apart), one per band;
/// `band_exp` carries each band's incoming exponent.
///
/// **History shift first.** Before anything is computed, three parallel
/// histories are advanced. Each is a row of five 8-word slots (one word for the
/// scalar history) and each is shifted by **two** slots, not one:
///
/// | history | slots (`i16` word index) | moves |
/// |---|---|---|
/// | per-band voicing measure | `0x00, 0x08, 0x10, 0x18, 0x20` | `0x10→0x20`, `0x08→0x18`, `0x00→0x10` |
/// | per-band denominator | `0x28, 0x30, 0x38, 0x40, 0x48` | `0x38→0x48`, `0x30→0x40`, `0x28→0x38` |
/// | shared exponent | `0x50 .. 0x54` (1 word each) | `0x52→0x54`, `0x51→0x53`, `0x50→0x52` |
///
/// The stride-two shift is the tell: slots 0 and 1 are the two half-frames of
/// the current 20 ms frame, so the row holds two-and-a-half frames' worth and
/// advancing it by a whole frame moves each entry two slots. This function
/// writes slot 0; slot 1 is filled by the other half-frame's path. The history
/// is read by the pitch/voicing tracker above, not here.
///
/// **Per band**, [`pick_pitch_by_band_ratio`] (`0x1031c420`) scores the band's
/// 32 energy words against two candidate harmonic spacings — `step` and
/// `step/2`, the pitch-doubling test — and returns the winner as two
/// block-floating pairs: `num` (energy captured by harmonically-spaced bands)
/// and `den` (total energy competing for it). [`one_minus_ratio`]
/// (`0x1031c390`) turns that into `1 - num/den` in Q15, so **a small value
/// means the band's energy sits on the harmonic grid**, i.e. it is voiced. The
/// threshold is `0x199a` = 0.2: below it the band's bit is set.
///
/// The bitmap is built MSB-first (`mask = mask*2 + bit`), so band 0 lands in
/// bit 7 of the returned 8-bit value.
///
/// **Denominator bookkeeping.** Each band's `den` mantissa is stored at
/// `st[0x28 + i]` with exponent `band_exp[i] + den_exp`, and the eight are then
/// brought onto one shared exponent by [`align_block_exp`] (`0x10314530`),
/// which is written to `st[0x50]`. If every aligned mantissa comes out zero —
/// a silent frame — the vector is forced to eight ones at exponent `-30`, the
/// smallest representable non-zero level, so that downstream divisions never
/// see a zero denominator.
///
/// The return is the 8-bit bitmap in `ax`; the original leaves an address or
/// `0xffff` in the high half of `eax`, which no caller reads.
///
/// Domain: `step > 0` (the band start indices are formed by an arithmetic
/// shift and go out of range for a negative step); `spec` at least 256 words;
/// `st` at least `0x55` words; `band_exp` 8 words, read-only.
pub fn band_voicing_decision(st: &mut [i16], step: i16, spec: &[i16], band_exp: &[i16]) -> i16 {
    // three-deep history shift, oldest first
    for k in 0..3usize {
        let d = 0x20 - 8 * k; // 0x20, 0x18, 0x10
        st.copy_within(d - 0x10..d - 8, d);
        let d2 = 0x48 - 8 * k; // 0x48, 0x40, 0x38
        st.copy_within(d2 - 0x10..d2 - 8, d2);
        let p = 0x54 - k; // 0x54, 0x53, 0x52
        st[p] = st[p - 2];
    }

    let iv = (step as i32) << 16;
    let start1 = ((((iv >> 10).wrapping_add(0x10000)) >> 16) as u16) as i32;
    let start2 = ((((iv >> 11).wrapping_add(0x10000)) >> 16) as u16) as i32;

    let mut mask: i16 = 0;
    let mut exps = [0i16; 8];

    for i in 0..8usize {
        let (mut num, mut num_e, mut den, mut den_e) = (0i16, 0i16, 0i16, 0i16);
        pick_pitch_by_band_ratio(
            &mut num,
            &mut num_e,
            &mut den,
            &mut den_e,
            &spec[i * 0x20..],
            step as i32,
            start1,
            start2,
        );
        st[0x28 + i] = den;
        exps[i] = band_exp[i].wrapping_add(den_e);

        let r = one_minus_ratio(num, num_e, den, den_e) as i16;
        mask = mask.wrapping_mul(2);
        st[i] = r;
        if r < 0x199a {
            mask = mask.wrapping_add(1);
        }
    }

    let mants: [i16; 8] = st[0x28..0x30].try_into().unwrap();
    let e = align_block_exp(&mut st[0x28..0x30], &mants, &exps, 8);
    st[0x50] = e;

    if st[0x28..0x30].iter().fold(0i16, |a, &b| a | b) == 0 {
        for i in 0..8 {
            st[0x28 + i] = 1;
        }
        st[0x50] = -30;
    }

    mask
}

// ======================= from bL3b2 =======================

// ---------------------------------------------------------------------------
// shared primitives (see SPEC §4)
// ---------------------------------------------------------------------------

/// High word of a 32-bit accumulator — the Q15 result of a Q31 value.
#[inline]
fn high16_bL3b2(x: i32) -> i16 {
    (x >> 16) as i16
}

/// Index of the highest set bit, or 31 for zero — the open-coded `bsr` whose
/// destination is left unchanged (here: the 31 the search starts at) when the
/// source is zero.
#[inline]
fn top_bit_bL3b2(v: u32) -> i32 {
    if v == 0 {
        31
    } else {
        31 - v.leading_zeros() as i32
    }
}

/// The library's variable shift: a non-negative count shifts left (wrapping),
/// a count in `-30..0` shifts right arithmetically, anything below that is a
/// full sign fill. x86 masks the count to 5 bits (SPEC §4).
#[inline]
fn shift32_bL3b2(v: i32, s: i16) -> i32 {
    if s >= 0 {
        ((v as u32) << (s as u32 & 0x1f)) as i32
    } else if s < -0x1e {
        v >> 31
    } else {
        v >> ((-(s as i32)) as u32 & 0x1f)
    }
}

/// MSVC's `__allshl` (`0x10305930`); `__allshr` is `batch1::allshr`. Both keep
/// only the low half at every call site in this codec (SPEC §4.2).
fn allshl_bL3b2(v: i64, count: u8) -> i64 {
    let hi = (v >> 32) as u32;
    let lo = v as u32;
    if count >= 0x40 {
        return 0;
    }
    if count >= 0x20 {
        let n = count & 0x1f;
        return (((lo << n) as u64) << 32) as i64;
    }
    let n = count & 0x1f;
    let (new_hi, new_lo) = if n == 0 {
        (hi, lo)
    } else {
        ((hi << n) | (lo >> (32 - n)), lo << n)
    };
    (((new_hi as u64) << 32) | new_lo as u64) as i64
}

/// The SPEC §4.4 normalise-and-shift idiom: normalise a 64-bit accumulator,
/// shift it there keeping only the low 32 bits, and report both the mantissa
/// (`high16_bL3b2` of that word) and the shift.
#[inline]
fn norm_shift_mant(acc: i64) -> (i16, i16) {
    let s = norm64(acc as u32, (acc >> 32) as u32) as i16;
    let r = if s < 0 {
        allshr(acc, ((s as i32).wrapping_neg() as u32 & 0xff) as u8)
    } else {
        allshl_bL3b2(acc, (s as u32 & 0xff) as u8)
    };
    (high16_bL3b2(r as i32), s)
}

// ---------------------------------------------------------------------------
// 0x1030dc80 — three-band gain boost
// ---------------------------------------------------------------------------

/// `0x1030dc80` — apply the conditional log-domain gain boost to three
/// low-frequency bands at once.
///
/// Three back-to-back calls to [`boost_gain_if_above_floor`] (`0x1030f630`),
/// one per band, each with its own ceiling, level floor and boost rate:
///
/// | band | ceiling | floor | rate |
/// |---|---|---|---|
/// | 0 | `0x0480` | `-0x4800` | `0x42b8` |
/// | 1 | `0x0400` | `-0x5000` | `0x32b8` |
/// | 2 | `0x0380` | `-0x5400` | `0x22b8` |
///
/// Higher bands get a lower ceiling, a lower level floor and a smaller rate —
/// the boost is progressively gentler with frequency.
///
/// `bias` is the *low byte* of `count` sign-extended and scaled by 256: the
/// original does `shl 24 / sar 16`, so a voicing-activity count of `n` becomes
/// a headroom penalty of `n · 256` in the Q16 subtraction `hi - lo - bias`.
/// The same bias applies to all three bands.
///
/// `lo` is indexed with stride 1, `level` and `level_exp` with stride 2 (they
/// are two-word records), and `gains` with stride 1.
pub fn boost_three_band_gains(
    gains: &mut [i16],
    lo: &[i16],
    level: &[i16],
    level_exp: &[i16],
    count: i16,
) {
    // shl 0x18 / sar 0x10: sign-extend the low byte, then scale by 256.
    let bias = (count as i8 as i16).wrapping_mul(0x100);
    const HI: [i16; 3] = [0x480, 0x400, 0x380];
    const FLOOR: [i16; 3] = [-0x4800, -0x5000, -0x5400];
    const RATE: [i16; 3] = [0x42b8, 0x32b8, 0x22b8];
    for b in 0..3usize {
        boost_gain_if_above_floor(
            &mut gains[b],
            lo[b],
            level[2 * b],
            level_exp[2 * b],
            HI[b],
            FLOOR[b],
            RATE[b],
            bias,
        );
    }
}

// ---------------------------------------------------------------------------
// 0x103138e0 — rotate a complex spectrum by a fixed phasor
// ---------------------------------------------------------------------------

/// `0x103138e0` — multiply an interleaved complex spectrum by one fixed
/// complex phasor, then renormalise it.
///
/// `buf` holds `n` interleaved `(re, im)` `i16` pairs at block exponent
/// `spec_exp`; `phasor` is a single `(re, im)` pair at exponent `phasor_exp`.
/// Each bin becomes
///
/// ```text
/// re' = (pr·re - pi·im) >> 16
/// im' = (re·pi + pr·im) >> 16
/// ```
///
/// Note the `>> 16`, not the usual doubled `>> 15`: the products are *not*
/// renormalised, which is why the output exponent gains the extra `+1` in
/// `spec_exp + phasor_exp + 1`. The whole `2n` words are then block-normalised
/// by [`normalise_to_exp`] (`0x103175d0`) and the final exponent returned.
///
/// The original takes separate source and destination pointers but its only
/// caller (`0x10313ec0`) passes the same buffer for both, and the loop reads
/// each pair before writing it, so the in-place form here is exact.
///
/// In the encoder this applies the frame's global phase correction — the
/// phasor is built by `0x10313ec0` from an `atan2` of the accumulated
/// cross-spectrum — to the whole analysis spectrum before the band matcher
/// (`0x103136e0`) scores it.
pub fn rotate_spectrum_by_phasor(
    buf: &mut [i16],
    phasor: &[i16],
    phasor_exp: i32,
    spec_exp: i32,
    n: i16,
) -> u16 {
    let mut exp = spec_exp.wrapping_add(phasor_exp).wrapping_add(1) as u16 as i16;
    let pr = phasor[0] as i32;
    let pi = phasor[1] as i32;
    if n > 0 {
        for i in 0..(n as u16 as usize) {
            let re = buf[2 * i] as i32;
            let im = buf[2 * i + 1] as i32;
            let a = pr.wrapping_mul(re).wrapping_sub(pi.wrapping_mul(im));
            let b = re.wrapping_mul(pi).wrapping_add(pr.wrapping_mul(im));
            buf[2 * i] = high16_bL3b2(a);
            buf[2 * i + 1] = high16_bL3b2(b);
        }
    }
    let count = (n as u16).wrapping_mul(2) as usize;
    let mut sbuf = [0i16; 0x102];
    let src = snap(&mut sbuf, &buf[..count]);
    normalise_to_exp(buf, src, count as u16, &mut exp);
    exp as u16
}

// ---------------------------------------------------------------------------
// 0x10313db0 — minimum-phase phasor from a log-magnitude spectrum
// ---------------------------------------------------------------------------

/// `0x10313db0` — turn a real log-magnitude spectrum into a unit phasor
/// spectrum by cepstral folding: the classic minimum-phase / Hilbert-transform
/// construction, done entirely in fixed point.
///
/// With `n = 0x100` and `half = 0x80` (the only call, from `0x10314070`):
///
/// 1. **Inverse real FFT** ([`inverse_real_fft`], `0x1031a330`, `log2n = 8`) of
///    the 129-point real log spectrum — this is the *cepstrum*.
/// 2. **Fold to a causal sequence**: halve `c[0]` and `c[half]`, then scale the
///    whole `half + 1` window by `0x70f7` (0.8826 Q15) and renormalise. Halving
///    the two self-conjugate taps is exactly the `2·u[n]` weighting of the
///    discrete Hilbert transform; the `0x70f7` factor is the fixed headroom the
///    subsequent forward transform needs.
/// 3. **Lifter**: zero everything from `half + 1` to `n - 1`, discarding the
///    anticausal half of the cepstrum.
/// 4. **Forward real FFT** ([`rfft_forward`], `0x1031a2f0`) — the imaginary
///    part of the result is the minimum-phase phase spectrum.
/// 5. **Phase → phasor** ([`cos_pairs`], `0x10314380`): each odd word (the
///    phase) is replaced by the interleaved pair `(cos φ, sin φ)` read from the
///    512-entry table at `0x1037f978`.
///
/// `buf` therefore leaves as `half` interleaved unit `(cos, sin)` pairs — the
/// phase reference the encoder's spectral matcher uses to make a
/// magnitude-only spectrum comparable to a complex one.
///
/// The exponent bookkeeping is done in a 32-bit slot of which only the low 16
/// bits are written by the callees, so a `-2` that borrows leaves `0xffff` in
/// the high half; that word is what the forward transform receives, and it is
/// reproduced here rather than cleaned up.
///
/// Returns 0 unconditionally (the original's `return 0`).
pub fn minimum_phase_phasor(
    buf: &mut [i16],
    exp_in: i32,
    log2n: i32,
    n: i16,
    cos_tbl: &[i16],
) -> i32 {
    let half = ((n / 2) as u16) as i16; // (short)n / 2, kept as a 16-bit index
    let keep = (half as u16).wrapping_add(1); // half + 1 words survive the lifter

    let r = inverse_real_fft(buf, exp_in, log2n, 1);
    buf[0] >>= 1;
    let mut e32: u32 = r as u32 & 0xffff;
    buf[half as usize] >>= 1;

    let mut sbuf = [0i16; 0x102];
    let src = snap(&mut sbuf, &buf[..keep as usize]);
    scale_vec(buf, src, keep as i16, 0x70f7);

    e32 = e32.wrapping_sub(2);
    let mut e16 = e32 as u16 as i16;
    let src = snap(&mut sbuf, &buf[..keep as usize]);
    normalise_to_exp(buf, src, keep, &mut e16);
    e32 = (e32 & 0xffff_0000) | (e16 as u16 as u32);

    fill16(
        &mut buf[keep as i16 as usize..],
        0,
        (n as i32).wrapping_sub(keep as i32) as i16,
    );

    let f = rfft_forward(buf, e32 as i32, log2n as i16, 1) as u16 as i16;
    let mut pbuf = [0i16; 0x102];
    let phase = snap(&mut pbuf, &buf[1..1 + 2 * half as usize]);
    cos_pairs(buf, phase, f, half as u16, cos_tbl);
    0
}

// ---------------------------------------------------------------------------
// 0x1031cb10 — MDCT-style sub-band filter bank
// ---------------------------------------------------------------------------

/// `0x1031cb10` — the encoder's critically-sampled analysis filter bank: a
/// time-domain-aliasing fold followed by a length-`n` DCT-IV, block-floating
/// throughout.
///
/// `x` holds `2n` windowed input samples at block exponent `*exp`; `window`
/// holds `2n` Q15 taps. The result is `n` sub-band amplitudes in `out`, all
/// brought to one common block exponent, which the return value carries.
///
/// **Stage 1 — TDAC fold.** After renormalising the `2n` input words, pairs are
/// folded to `n` values:
///
/// ```text
/// x[i]      = (x[i]·w[i]      - x[n-1-i]·w[n-1-i]) >> 16
/// x[n-1-i]  = (x[n+i]·w[n-1-i] + x[2n-1-i]·w[i])   >> 16      i = 0 .. n/2-1
/// ```
///
/// which is the standard MDCT windowing-and-folding butterfly. The products
/// accumulate in 64 bits and are taken `>> 17` of the doubled product, i.e.
/// `>> 16` of the plain one.
///
/// **Stage 2 — DCT-IV.** For each output bin `k` the routine accumulates
///
/// ```text
/// acc(k) = Σ_j  s_a(k)·x[j]·T[a_k + j·(2k+1)]  +  s_d(k)·x[n-1-j]·T[a_k + 120 + j·(2k+1)]
/// ```
///
/// over `j = 0 .. n/2-1`, where `T` is the 160-entry Q15 table at `0x1037ff40`
/// — `T[m] = round(32767·cos(2π(m + ½)/160))` — indexed modulo 160, the start
/// phase is `a_k = (20 + 41k) mod 160` and the per-sample advance is `2k + 1`.
/// The `+120` offset is the three-quarter-period lookup that supplies the
/// second (sine) leg. The two legs' signs rotate with `k mod 4` as
/// `(+,−), (+,+), (−,+), (−,−)`, i.e. the quadrant rotation of a complex
/// exponential.
///
/// Each accumulator is normalised by the SPEC §4.4 idiom, its mantissa scaled
/// by `1/n` (obtained once as `div_norm(0x40000000, 1, …)`, `0x1030a660`), and
/// its own exponent recorded.
///
/// **Stage 3 — common exponent.** The largest exponent over the *non-zero*
/// bins wins (strict `<`, so the first maximum is kept), every bin is shifted
/// to it, and the caller receives `(exp_after_stage_2) + 1 + recip_exp +
/// max_exp`. If every bin is zero the routine returns `-0x20` and leaves `out`
/// untouched — the "silent band" sentinel.
///
/// Domain: `n` even and `2 ≤ n ≤ 40` (the phase increment is reduced modulo 160
/// by a single conditional subtraction, which only covers `n ≤ 40`; the sole
/// caller, `0x1030f710`, uses `n = 40` with an 80-word window).
pub fn subband_filterbank(
    out: &mut [i16],
    x: &mut [i16],
    exp: i32,
    window: &[i16],
    n: i32,
    tbl: &[i16],
) -> i32 {
    let ns = n as i16;
    let halfn = (ns >> 1) as u16; // n/2 pairs in the fold and in each bin sum
    let mut e = exp as u16 as i16;

    // --- stage 1: renormalise, then the TDAC fold -------------------------
    {
        let mut sbuf = [0i16; 0x102];
        let src = snap(&mut sbuf, &x[..2 * n as usize]);
        normalise_to_exp(x, src, (n as u16).wrapping_mul(2), &mut e);
    }
    if (halfn as i16) > 0 {
        for i in 0..halfn as usize {
            let ni = ns as usize;
            let a = (x[i] as i64) * (window[i] as i64) * 2;
            let b = (x[ni - 1 - i] as i64) * (window[ni - 1 - i] as i64) * 2;
            let lo = ((a - b) >> 17) as i16;
            let c = (window[ni - 1 - i] as i64) * (x[ni + i] as i64) * 2;
            let d = (x[2 * ni - 1 - i] as i64) * (window[i] as i64) * 2;
            let hi = ((c + d) >> 17) as i16;
            x[i] = lo;
            x[ni - 1 - i] = hi;
        }
    }
    {
        let mut sbuf = [0i16; 0x102];
        let src = snap(&mut sbuf, &x[..n as usize]);
        normalise_to_exp(x, src, n as u16, &mut e);
    }

    // --- the 1/n scale factor --------------------------------------------
    let nn = (n as i16) as i32;
    let v = (nn as u32) << 16;
    let sh = if v == 0 {
        0u32
    } else {
        let m = if (v as i32) < 0 { !v } else { v };
        (0x1eu32.wrapping_sub(top_bit_bL3b2(m) as u32)) & 0xffff
    };
    let mut recip_exp: i16 = 0;
    let recip = div_norm(
        0x4000_0000,
        1,
        (((v << (sh & 0x1f)) as i32) >> 16) as i16,
        (0xfu32.wrapping_sub(sh)) as u16 as i16,
        &mut recip_exp,
    );

    // --- stage 2: DCT-IV --------------------------------------------------
    let mut bin_exp = [0i16; 64];
    let mut nbins = 0usize;
    if ns > 0 {
        // Two alternating phase registers, exactly as the original's unrolled
        // group of four: the *next* bin's start phase is computed before the
        // current bin's inner loop runs.
        let mut inc: i16 = 0x29;
        let mut k: i16 = 0;
        let mut cur: i16 = 0x14;
        'outer: loop {
            for pos in 0..4i16 {
                let next = {
                    let t = cur.wrapping_add(0x29);
                    if t > 0x9f {
                        cur.wrapping_sub(0x77)
                    } else {
                        t
                    }
                };
                let step = inc.wrapping_add(2 * pos);
                let mut a = cur;
                let mut acc: i64 = 0;
                if (halfn as i16) > 0 {
                    for j in 0..halfn as usize {
                        let a2 = {
                            let t = a.wrapping_add(0x78);
                            if t > 0x9f {
                                a.wrapping_sub(0x28)
                            } else {
                                t
                            }
                        };
                        let nxt = {
                            let t = a2.wrapping_add(step);
                            if t > 0x9f {
                                t.wrapping_sub(0xa0)
                            } else {
                                t
                            }
                        };
                        let asc = (x[j] as i64) * (tbl[a as usize] as i64) * 2;
                        let desc = (x[ns as usize - 1 - j] as i64) * (tbl[a2 as usize] as i64) * 2;
                        acc = acc.wrapping_add(match pos {
                            0 => asc - desc,
                            1 => asc + desc,
                            2 => desc - asc,
                            _ => -(asc + desc),
                        });
                        a = nxt;
                    }
                }
                let (mant, s) = norm_shift_mant(acc);
                out[nbins] = high16_bL3b2((mant as i32).wrapping_mul(recip as i32).wrapping_mul(2));
                bin_exp[nbins] = s.wrapping_neg();
                nbins += 1;
                cur = next;
                if ns <= k.wrapping_add(pos).wrapping_add(1) {
                    break 'outer;
                }
            }
            inc = inc.wrapping_add(8);
            k = k.wrapping_add(4);
            if k >= ns {
                break;
            }
        }
    }

    // --- stage 3: bring every bin to the largest live exponent ------------
    let mut max_exp: u32 = 0xffff_ffe0;
    if ns > 0 {
        for i in 0..ns as u16 as usize {
            if out[i] != 0 && (max_exp as i16) < bin_exp[i] {
                max_exp = bin_exp[i] as u16 as u32;
            }
        }
        if (max_exp as i16) > -0x20 {
            for i in 0..ns as u16 as usize {
                let v = (out[i] as i32) << 16;
                let s = bin_exp[i].wrapping_sub(max_exp as u16 as i16);
                out[i] = high16_bL3b2(shift32_bL3b2(v, s));
            }
            return (e as i32)
                .wrapping_add(1)
                .wrapping_add(recip_exp as i32)
                .wrapping_add(max_exp as i32);
        }
    }
    -0x20
}

// ---------------------------------------------------------------------------
// 0x1031af90 — sub-band analysis for one frame's worth of new audio
// ---------------------------------------------------------------------------

/// `0x1031af90` — advance the encoder's sub-band analysis by however many
/// 8-sample hops the newly-arrived audio provides, and age the band trackers by
/// the same amount.
///
/// The word at `state + 0x6dc` is a **sample-phase accumulator**: `advance`
/// samples are added to it, the quotient by 8 is the number of complete
/// analysis hops now available, and the remainder is written back so the phase
/// carries into the next frame. Everything else scales with that hop count:
///
/// * `0x1c + 8·hops` words are copied out of the analysis history ring, ending
///   `origin - remainder` words from the ring base — i.e. the newest samples,
///   backed off by the leftover phase — and block-normalised from exponent 0.
/// * The six **band hold counters** at `state + 0x630 … 0x63a` are decremented
///   by `hops`, floored at zero, and written one word lower (`0x62e … 0x638`):
///   the array shifts down by one word per call as well as counting down. The
///   new top counter is seeded to `0x3a - hops`.
/// * The six **band peak values** at `state + 0x620 … 0x62a` are refreshed from
///   that counter array: a band whose counter *rose* over its neighbour keeps
///   its successor's value, otherwise it is reset to `-0x8000`. This is the
///   hold-and-decay peak tracker the pitch/voicing decision reads.
/// * `*rows` receives `2 · hops` — the number of 16-bin spectral rows the
///   sub-band analysis is about to produce.
/// * `acc` (16 `i32`) is zeroed, then [`subband_analysis`] (`0x1031ab80`) runs
///   the hops: 32-point transforms, magnitude-squared accumulation into `acc`
///   at exponent `2·exp + 7`, square roots into the 16 × 11 output grid `out`,
///   with the 0x50-word history at `state + 0x63c` carrying the filter's
///   overlap between calls and its exponent at `state + 0x62c`.
///
/// Returns the hop count.
///
/// Domain: the original's stack scratch is 116 words of input and 112 of
/// working buffer, so `hops` must satisfy `8·hops + 0x1c <= 116`, i.e.
/// `hops <= 11`; the output grid has 11 columns, one per hop, which is the same
/// bound. `pcm` is indexed from the pointer the caller passes (the 383-word
/// analysis ring, with `origin = 0x17f`).
#[allow(clippy::too_many_arguments)]
pub fn subband_analysis_frame(
    state: &mut [i16],
    out: &mut [i16],
    acc: &mut [i32],
    rows: &mut i16,
    pcm: &[i16],
    origin: i16,
    advance: i16,
    taper: &[i16],
    interp_coef: &[i16],
) -> u16 {
    const PHASE: usize = 0x6dc / 2;
    const PEAK: usize = 0x620 / 2;
    const HIST_EXP: usize = 0x62c / 2;
    const COUNT: usize = 0x630 / 2;
    const HIST: usize = 0x63c / 2;

    state[PHASE] = state[PHASE].wrapping_add(advance);
    let hops_i16 = state[PHASE] >> 3;
    let hops = hops_i16 as u16 as u32;
    let rem = state[PHASE].wrapping_add((hops_i16 as u16).wrapping_mul(0xfff8) as i16);
    let nwords = (hops.wrapping_mul(8).wrapping_add(0x1c)) as u16;
    state[PHASE] = rem;

    // 116 words: the original's stack buffer. Only `nwords` of it is ever read.
    let mut inbuf = [0i16; 116];
    let start = (origin as i32) - (rem as i32) - (nwords as i16 as i32);
    copy16_batch3(&mut inbuf, &pcm[start as usize..], nwords as i16);

    // Age the six band hold counters, shifting them down one word.
    for i in 0..6usize {
        let v = (state[COUNT + i] as i32) - (hops_i16 as i32);
        state[COUNT - 1 + i] = if v > 0 { v as i16 } else { 0 };
    }
    state[0x63a / 2] = (0x3au32.wrapping_sub(hops)) as u16 as i16;

    // Refresh the six band peaks from the counters.
    for i in 0..6usize {
        let hi = state[PEAK + i + 8] as i32; // +0x10 bytes
        let lo = state[PEAK + i + 7] as i32; // +0x0e bytes
        state[PEAK + i] = if hi - lo > 0 {
            state[PEAK + i + 1]
        } else {
            -0x8000i32 as i16
        };
    }

    *rows = (hops.wrapping_mul(2)) as u16 as i16;
    // The original memsets 0x20 *words* — exactly the 16 `i32` accumulators.
    acc[..0x10].fill(0);

    let mut exp: i16 = 0;
    {
        let mut sbuf = [0i16; 0x102];
        let src = snap(&mut sbuf, &inbuf[..nwords as usize]);
        normalise_to_exp(&mut inbuf, src, nwords, &mut exp);
    }
    let acc_exp = ((exp as u16 as i32) * 2 + 7) as i16;

    let mut work = [0i16; 112];
    let (head, tail) = state.split_at_mut(HIST);
    let mut hist_exp = head[HIST_EXP];
    subband_analysis(
        tail,
        &mut hist_exp,
        out,
        &mut work,
        exp,
        acc,
        acc_exp,
        &inbuf,
        hops_i16,
        taper,
        interp_coef,
    );
    head[HIST_EXP] = hist_exp;

    hops as u16
}

// ---------------------------------------------------------------------------
// 0x1030bda0 — per-frame parameter quantisation and packing hub
// ---------------------------------------------------------------------------

/// `0x1030bda0` — turn one analysed frame into payload fields: pick the bit
/// allocation, derive the pitch parameters, quantise the band voicing, and emit
/// the first two fields of the channel stream.
///
/// This is the encoder's counterpart to the decoder's parameter decode, and it
/// makes the shape of the payload explicit:
///
/// 1. **Bit allocation.** `cfg[0]` is the mode word: 0 selects
///    [`imbe_bit_alloc`] (`0x1030c300`), anything else [`ambe_bit_alloc`]
///    (`0x1030bc70`) at sub-mode 0. The *previous* total `cfg[2]` — read before
///    the allocation is rebuilt — is the number of payload words zeroed in
///    `bits`, so a mode change clears exactly the old frame's footprint. The
///    write cursor is then reset to the start of `bits`.
///
/// 2. **Frame class.** `frame_class` is stored in both parameter blocks and
///    `cur[1]` is invalidated to `0xff`. Only class 3 (speech) runs
///    [`pitch_config_init`] (`0x1030ce40`) over the two pitch peaks; if the
///    tone flag is set and the channel unit length is exactly `0x20`, the pitch
///    lag `cur[6]` is forced to `0x563b` — the signalling-tone lag.
///
/// 3. **Pitch.** [`lag_to_pitch_index`] (`0x10311170`) maps the lag to the
///    8-bit pitch index that goes on the wire, and [`set_pitch_params`]
///    (`0x10310f80`) expands it back into the fundamental `cur[6]` and the
///    harmonic count `cur[2]`. `nbits = min(triads_capped(L), 8)` — the number
///    of voicing bands, `ceil(L/3)` capped at 8 — sizes everything downstream.
///
/// 4. **Voicing.** Two paths.
///    * *All-voiced / two-group*: reached only for a class-3 frame whose
///      `cur[1]` survived `pitch_config_init`. With no tone flag, or a channel
///      unit longer than `0x7a`, the mask is the union of two
///      [`two_group_mask`] searches (`0x10316b10`) with table bases 0 and 1;
///      otherwise it is the all-ones mask `(1 << nbits) - 1`. The code is
///      `mask + 0x38`, i.e. it sits *above* the 56-entry voicing codebook, and
///      the pattern is expanded straight through [`voicing_pattern`]
///      (`0x103124b0`) seeded by the previous frame's dibits
///      ([`gather_dibits`]).
///    * *Quantised*: otherwise [`quantise_voicing`] (`0x10312af0`) searches the
///      codebook against the three 16-word band-measure vectors and returns the
///      index.
///    Either way [`expand_dibits`] leaves the chosen dibit pattern as two
///    dwords at word 4 of both parameter blocks.
///
/// 5. **Split.** A code of `0x38` or below is a *codebook* index: the pitch
///    field becomes `0x19` for exactly `0x38` and `code + 0x80` otherwise, the
///    pitch parameters are recomputed from it, and the voicing field emitted is
///    zero. A code above `0x38` is a *raw* mask: `0x38` is subtracted and, when
///    the band count exceeds 8, [`extract_bit_pattern`] (`0x10312b90`)
///    re-encodes it against `L`.
///
/// 6. **Emit.** `cfg[5]` bits of the pitch field, then `triads_capped(L)` bits
///    of the voicing field, MSB first ([`emit_bits`], `0x10310310`). `cfg[5]`
///    is the second allocation entry — 7 for AMBE, 8 for IMBE — which is why
///    the pitch field leads the AMBE frame image (SPEC §3.1).
///
/// Returns the voicing field.
///
/// `cursor` is the original's `i16**` write pointer, expressed as a word index
/// into `bits`. `pblk[4..6]` is the previous frame's packed voicing dword and
/// `weights` the three 16-word band-measure vectors at byte offsets 0, 0x20 and
/// 0x40.
#[allow(clippy::too_many_arguments)]
pub fn encode_frame_params(
    bits: &mut [i16],
    cursor: &mut usize,
    cur: &mut [i16],
    prev: &mut [i16],
    pblk: &[i16],
    rate: i16,
    peaks: &[i16],
    tone_flag: i16,
    frame_class: i16,
    cfg: &mut [i16],
    weights: &[i16],
    tbl: &[i16],
    parts: &[i16],
) -> u16 {
    let prev_total = cfg[2];
    {
        let t: &mut [i16; 13] = (&mut cfg[..13]).try_into().unwrap();
        if t[0] == 0 {
            imbe_bit_alloc(t);
        } else {
            ambe_bit_alloc(t, 0);
        }
    }
    fill16(bits, 0, prev_total);
    *cursor = 0; // store32: the cursor is seeded with `bits` itself

    prev[0] = frame_class;
    cur[0] = frame_class;
    cur[1] = 0xff;
    if frame_class == 3 {
        pitch_config_init(cur, rate, peaks[0], peaks[1], 1, tbl);
        if tone_flag != 0 && rate == 0x20 {
            cur[6] = 0x563b;
        }
    }

    // `val` and `code` live in the argument slots the original reuses.
    let mut val = lag_to_pitch_index(cur[6]) as u16 as u32;
    set_pitch_params(cur, val as u16 as i16);
    let t = triads_capped(cur[2]);
    let nbits: u32 = if t > 8 { 8 } else { t as u16 as u32 };

    let mut code: u32;
    if frame_class == 3 && cur[1] != 0xff {
        let mask: u32 = if tone_flag == 0 || rate > 0x7a {
            let m0 = two_group_mask(cur, nbits as u16 as i16, 0, tbl, parts);
            let m1 = two_group_mask(cur, nbits as u16 as i16, 1, tbl, parts);
            (m0 | m1) as u32
        } else {
            // shl ax,cl / dec ax — a 16-bit shift with the count masked to 5.
            let sh = nbits & 0x1f;
            (if sh >= 16 { 0u16 } else { 1u16 << sh }).wrapping_sub(1) as u32
        };
        code = mask.wrapping_add(0x38);
        let dib = gather_dibits(prev_voicing(pblk)) as u16 as u32;
        let pat = voicing_pattern(code as u16 as i16, dib, nbits as u16 as i16);
        let mut a = 0u32;
        let mut b = 0u32;
        expand_dibits(&mut a, &mut b, pat);
        put_u32(prev, 4, a);
        put_u32(cur, 4, b);
    } else {
        let mut a = 0u32;
        let mut b = 0u32;
        code = quantise_voicing_bL2b1(
            &mut a,
            &mut b,
            prev_voicing(pblk),
            &weights[0..],
            &weights[0x10..],
            &weights[0x20..],
            nbits as i32,
        ) as u32;
        put_u32(prev, 4, a);
        put_u32(cur, 4, b);
    }

    let nb: i16;
    if (code as u16 as i16) <= 0x38 {
        val = if (code as u16 as i16) == 0x38 {
            0x19
        } else {
            code.wrapping_add(0x80) & 0xffff
        };
        code = 0;
        set_pitch_params(cur, val as u16 as i16);
        nb = triads_capped(cur[2]);
    } else {
        code = code.wrapping_sub(0x38);
        nb = triads_capped(cur[2]);
        if nb > 8 {
            code = extract_bit_pattern(code as u16 as i16, nb as u16, cur[2]) as u32;
        }
    }

    emit_bits(cursor, bits, cfg[5], val as u16 as i16);
    emit_bits(cursor, bits, nb, code as u16 as i16);
    code as u16
}

/// The previous frame's packed voicing dword, at byte offset 8 of the
/// parameter block.
#[inline]
fn prev_voicing(pblk: &[i16]) -> i32 {
    ((pblk[5] as u16 as u32) << 16 | pblk[4] as u16 as u32) as i32
}

#[inline]
fn put_u32(dst: &mut [i16], word: usize, v: u32) {
    dst[word] = v as u16 as i16;
    dst[word + 1] = (v >> 16) as u16 as i16;
}

// ======================= from bL3b3 =======================

// ---------------------------------------------------------------------------
// shared primitives (same shapes as every other batch's private helpers)
// ---------------------------------------------------------------------------

/// High word of a 32-bit accumulator.
#[inline]
fn high16_bL3b3(x: i32) -> i16 {
    (x >> 16) as i16
}

/// Index of the highest set bit; 31 for zero, matching the open-coded `bsr`
/// search loop the compiler emits (SPEC §4.1).
#[inline]
fn top_bit_bL3b3(v: u32) -> i32 {
    if v == 0 {
        31
    } else {
        31 - v.leading_zeros() as i32
    }
}

/// Block exponent `0x1e - bsr(|x|)`, with the call sites' `x == 0 → 0`
/// short-circuit folded in (SPEC §4).
#[inline]
fn norm_shift_bL3b3(x: i32) -> u16 {
    if x == 0 {
        return 0;
    }
    let m = if x < 0 { !(x as u32) } else { x as u32 };
    (0x1e - top_bit_bL3b3(m)) as u16
}

/// The library's open-coded variable shift: non-negative counts shift left
/// (wrapping), `-30..0` shift right arithmetically, below that is a sign fill.
#[inline]
fn shift32_bL3b3(v: i32, s: i16) -> i32 {
    if s >= 0 {
        ((v as u32) << (s as u32 & 0x1f)) as i32
    } else if s < -0x1e {
        v >> 31
    } else {
        v >> ((-s) as u32 & 0x1f)
    }
}

/// `|x|` with the x86 `neg` behaviour the original guards explicitly:
/// `0x80000000` saturates to `0x7fffffff`.
#[inline]
fn abs_sat(x: i32) -> i32 {
    if x < 0 {
        if x == i32::MIN {
            i32::MAX
        } else {
            x.wrapping_neg()
        }
    } else {
        x
    }
}

// ---------------------------------------------------------------------------
// 0x1030c3f0 — one analysis slot: weights, levels and pitch
// ---------------------------------------------------------------------------

/// The pointer block the slot driver at `0x1030c3f0` is handed. Byte offsets
/// into the original structure are given for traceability; the engine builds it
/// once per frame and walks the slots.
pub struct SlotCtx<'a> {
    /// `+0x00` — the sub-band history matrix, eight words per slot.
    pub bands: &'a [i16],
    /// `+0x04` — the per-slot sub-band magnitudes, eight words per slot.
    pub mags: &'a [i16],
    /// `+0x08` — the five pitch candidates.
    pub pitch_cand: &'a [i16],
    /// `+0x0c` — the five matching pitch errors.
    pub pitch_err: &'a [i16],
    /// `+0x10` — the pitch the previous frame settled on.
    pub pitch_target: i16,
    /// `+0x28` — per-slot band energies, eight words per slot.
    pub energies: &'a [i16],
    /// `+0x2c` — the block exponent of each slot's energies.
    pub energy_exp: &'a [i16],
    /// `+0x30` — one activity bit per slot, read as a 16-bit register.
    pub active_mask: i32,
}

/// `0x1030c3f0` — drive one analysis slot: spectral weights, band levels and
/// the pitch choice.
///
/// `slot = (base + half) & 0xffff` selects the slot; `half` is the half-frame
/// index, so consecutive calls walk the analysis history. Three stages, in
/// order:
///
/// 1. [`slot_band_weights`] (`0x10310c50`) turns the slot's magnitudes into
///    eight `-log2` weights in `weights`, floors the same magnitudes into
///    `clamped`, and produces a spectral-tilt bit. The neighbour stride it is
///    given is `1` for slot ≤ 1 and `2` above — i.e. the very first slots
///    compare against their immediate neighbour and later ones skip one.
/// 2. [`log_band_levels`] (`0x10310a80`) maps the slot's eight band energies,
///    at that slot's own block exponent, to `levels`. The tilt argument is a
///    hard zero here: the high-band de-emphasis is *not* used on this path.
/// 3. [`pitch_select`] (`0x10310540`) picks among the five pitch candidates,
///    using `slot` as the candidate index and `1 - half` as the fallback, and
///    the doubled result is stored to `out[6]` (byte `+0x0c` of the caller's
///    frame block).
///
/// The tilt bit `slot_band_weights` returns is written back into the *caller's
/// own argument slot* for `half` and then discarded — `pitch_select` still sees
/// the original `half`, because the compiler had already copied it to a
/// register. That is SPEC §4.2's "a reused stack slot rendered as a write to a
/// parameter"; it is not observable and this port does not model it.
///
/// The original passes a ninth argument to `0x10310c50` (the difference of two
/// state words at `+0x44`/`+0x46`) that the callee never reads. It is dropped.
///
/// Domain: `slot` in `0..` such that `slot ± stride` index `bands`, and
/// `slot` in `0..5` so `pitch_select`'s candidate index is in range.
pub fn slot_analysis(
    out: &mut [i16],
    weights: &mut [i16],
    clamped: &mut [i16],
    levels: &mut [i16],
    half: i32,
    base: i32,
    ctx: &SlotCtx<'_>,
    cos_tbl: &[i16],
) {
    let slot_u = (base.wrapping_add(half) as u32 & 0xffff) as i32;
    let slot = slot_u as i16;

    let mut tilt: u16 = 0;
    slot_band_weights(
        weights,
        clamped,
        ctx.bands,
        ctx.mags,
        ctx.active_mask,
        slot_u,
        i32::from(slot > 1) + 1,
        &mut tilt,
        cos_tbl,
    );

    let si = slot as usize;
    log_band_levels(levels, &ctx.energies[si * 8..], ctx.energy_exp[si], 0);

    let p = pitch_select(
        ctx.pitch_cand,
        ctx.pitch_err,
        ctx.pitch_target,
        slot,
        (1i32.wrapping_sub(half)) as i16,
    );
    out[6] = p as i16;
}

// ---------------------------------------------------------------------------
// 0x10310750 — one slot's harmonic amplitude vector
// ---------------------------------------------------------------------------

/// `0x10310750` — harmonic log magnitudes for one analysis slot, biased and
/// peak-smeared.
///
/// The amplitude-estimation stage seen from one slot up:
///
/// 1. [`harmonic_log_mags`] (`0x103108c0`) integrates that slot's 129-bin
///    magnitude spectrum — `spec` is a matrix of `0x204` bytes (129 `i32`) per
///    slot — over the harmonic bands implied by `step`, and writes `n` log
///    magnitudes to `out`. The slot's own block exponent comes from `exps`.
/// 2. [`bias_clamp`] (`0x103106f0`) adds a fixed `+170` to every log magnitude
///    with saturation and clamps the result to `0x77ff`.
/// 3. When bit 0 of `flags` is set, [`smooth_peak`] (`0x103107b0`) smears the
///    fundamental into the lowest bins of `out` and updates the leaky peak
///    tracker held at `state[0x40]` (byte `+0x80`).
///
/// So the peak smear is optional per call while the bias/clamp is not: the
/// caller turns it on only for the slot it is going to transmit.
pub fn slot_harmonic_levels(
    out: &mut [i16],
    step: i16,
    n: i16,
    flags: i32,
    state: &mut [i16],
    offset: i16,
    spec: &[i16],
    exps: &[i16],
    slot: i16,
) {
    let si = slot as i32 as usize;
    harmonic_log_mags(out, step, n as u16, &spec[si * ROW_W..], exps[si], offset);
    bias_clamp(out, n);
    if flags & 1 != 0 {
        smooth_peak(out, &mut state[0x40], step);
    }
}

// ---------------------------------------------------------------------------
// 0x10313ad0 — window an interleaved complex vector and renormalise
// ---------------------------------------------------------------------------

/// `0x10313ad0` — multiply an interleaved (re, im) vector by a real window and
/// renormalise the block.
///
/// `src` holds `n` complex pairs; `win[i]` scales pair `i`'s real and imaginary
/// parts alike, as `hi16(x · w · 2)` — the standard Q15×Q15 product of
/// SPEC §4. The result is written to `dst` and then renormalised in place by
/// [`normalise_to_exp`] (`0x103175d0`) over `2·n` words, so the whole complex
/// block shares one exponent.
///
/// The incoming exponent is `(exp + exp_add) & 0xffff`; the return value is
/// that exponent after the renormalisation credit. The original returns it in
/// `ax` only, leaving the high half of `eax` as whatever the callee left, so
/// only 16 bits are meaningful.
///
/// `n` is signed: a non-positive count skips the windowing but still
/// renormalises (over a count of `2n`, which is also non-positive, so nothing
/// moves and the exponent gains the empty-vector shift).
pub fn window_complex_block(
    dst: &mut [i16],
    src: &[i16],
    exp: i32,
    win: &[i16],
    exp_add: i32,
    n: u16,
) -> i16 {
    let mut e = (exp_add.wrapping_add(exp) as u32 & 0xffff) as u16 as i16;

    if (n as i16) > 0 {
        for i in 0..n as usize {
            let w = win[i] as i32;
            dst[2 * i] = high16_bL3b3((src[2 * i] as i32).wrapping_mul(w).wrapping_mul(2));
            dst[2 * i + 1] = high16_bL3b3((src[2 * i + 1] as i32).wrapping_mul(w).wrapping_mul(2));
        }
    }

    // The original normalises in place (dst and src are the same pointer).
    let mut sbuf = [0i16; 0x104];
    let scratch = snap(&mut sbuf, &dst[..2 * n as usize]);
    normalise_to_exp(dst, scratch, n.wrapping_mul(2), &mut e);
    e
}

// ---------------------------------------------------------------------------
// 0x10315650 — signalling-tone detector
// ---------------------------------------------------------------------------

/// `0x10381c08` — the four candidate tone bands, as (first bin, last bin)
/// pairs: (10, 13), (13, 14), (14, 19) and (10, 16). At the 256-point FFT's
/// 31.25 Hz resolution those cover 312–406, 406–437, 437–594 and 312–500 Hz.
pub const TONE_BANDS: [i16; 8] = [10, 13, 13, 14, 14, 19, 10, 16];

/// `0x10315650` — decide whether the frame is one of the four signalling tones,
/// and if so report its identity and level.
///
/// The decoder's frame dispositions include a *signalling tone* class; this is
/// the encoder side that raises it. Returns 1 when a tone is accepted, 0 when
/// it is rejected — and on rejection nothing is written, not even the tracker.
///
/// The bands come from [`TONE_BANDS`]; the tone identity is `0xa0 + index`.
/// If the state already holds an identity in `0xa0..=0xa3` (word `state[4]`,
/// byte `+0x08`) that band is re-tested, giving hysteresis; otherwise
/// [`pick_best_band`] (`0x10315cc0`) picks the strongest of the four.
///
/// Four independent rejection tests, in order:
///
/// 1. **Balance.** Three-bin energies at the band's first and last bin are
///    brought to a common exponent; each must be within a factor of four of the
///    other (`a/8 · 2^16 < b · 2^15`, i.e. `a/4 < b`). A tone is a *pair* of
///    partials of comparable size, so a single dominant peak is not one.
/// 2. **Level.** The whole band (from one bin below the first, for
///    `last - first + 5` bins) is accumulated and converted to a log amplitude
///    by [`log_amp_clamped`] (`0x103164b0`). A second log amplitude measures
///    how far the band sits *below* the caller's reference level
///    `ref_level @ ref_exp` — the frame energy. The tone is rejected unless
///    `band_log - 0x23e0 >= deficit_log`, so it must stand out from the frame
///    by a fixed margin.
/// 3. **Continuity.** When the identity matches the one already latched, both
///    logs are smoothed as `0.875·previous + 0.125·(new + 4)` before that
///    comparison — an eight-frame one-pole, so a tone has to persist.
/// 4. **Spread.** For a narrow band (`last - first < 4`) the five bins centred
///    on `(first + last + 2)/2` are renormalised ([`block_normalise_to_q15`],
///    `0x103173a0`) and put through [`spread_test`] (`0x10315b00`); a spectrum
///    that is not concentrated is rejected. Wide bands skip this.
///
/// On acceptance `*id` gets the identity, both words of `mag` get
/// `hi16((band_log << 16) - 0x3ff0001)` — the log level shifted into the
/// transmitted amplitude's origin — and the two smoothing memories at
/// `state[0xb]` / `state[0xc]` (bytes `+0x16` / `+0x18`) are updated.
///
/// `exp` and `ref_exp` are block exponents; `offset` is the log-domain origin
/// forwarded to [`log_amp_clamped`]. `energy` must be indexable to bin 21 (the
/// widest band plus `pick_best_band`'s three-bin overhang).
pub fn tone_detect(
    id: &mut i16,
    mag: &mut [i16],
    state: &mut [i16],
    energy: &[i32],
    exp: i16,
    offset: i16,
    ref_level: i16,
    ref_exp: i32,
    tbl: &[i16],
) -> u16 {
    let latched = state[4];
    let band = if latched < 0xa0 || latched >= 0xa4 {
        let mut best = 0i16;
        pick_best_band(&mut best, 4, energy, tbl, 3);
        best
    } else {
        latched.wrapping_sub(0xa0)
    };
    let ti = (band as usize & 0xffff) * 2;

    // --- 1. the two three-bin edge energies, at a common exponent ----------
    let e_in = exp as u16 as i32;
    let mut e_lo = e_in as i16;
    let mut e_hi = e_in as i16;
    let lo = tbl[ti] as i32;
    let hi = tbl[ti + 1] as i32;
    let a = accumulate_exp(&energy[lo as usize..], 3, &mut e_lo) as i32;
    let bb = accumulate_exp(&energy[hi as usize..], 3, &mut e_hi) as i32;

    let (ea, eb) = (e_lo as u16 as i32, e_hi as u16 as i32);
    let emax = if e_lo <= e_hi { eb } else { ea };
    let ma = (shift32_bL3b3(a, (ea - emax) as i16) >> 16) as i16 as i32;
    let mb = (shift32_bL3b3(bb, (eb - emax) as i16) >> 16) as i16 as i32;

    if ((ma << 16) >> 3) > (mb << 15) {
        return 0;
    }
    if ((mb << 16) >> 3) > (ma << 15) {
        return 0;
    }

    // --- 2. the band level and the deficit against the reference ----------
    let mut e_band = e_in as i16;
    let cnt = (tbl[ti + 1] as u16)
        .wrapping_sub(tbl[ti] as u16)
        .wrapping_add(5) as i16;
    let c = accumulate_exp(&energy[(lo - 1) as usize..], cnt, &mut e_band) as i32;
    let mant = ((c >> 16) as u16) as i16;

    let ident = (band as i32).wrapping_add(0xa0) as i16;
    let mut m1 = log_amp_clamped(mant, e_band as u16 as i32, offset) as i16;

    let sh = (e_band as u16 as i32).wrapping_sub(ref_exp);
    let scaled = shift32_bL3b3((mant as i32) << 16, sh as i16);
    let d = abs_sat(((ref_level as i32) << 16).wrapping_sub(scaled));
    let n = if d == 0 { 0u16 } else { norm_shift_bL3b3(d) };
    let mut m2 = log_amp_clamped(
        (((d as u32) << (n as u32 & 0x1f)) as i32 >> 16) as i16,
        ref_exp.wrapping_sub(n as i32),
        offset,
    ) as i16;

    // --- 3. continuity smoothing (0xdffe / 0x2000 are unsigned Q16) -------
    if ident == state[4] {
        m1 = high16_bL3b3(
            (state[0xb] as i32)
                .wrapping_mul(0xdffe)
                .wrapping_add(((m1 as i32).wrapping_add(4)).wrapping_shl(13)),
        );
        m2 = high16_bL3b3(
            (state[0xc] as i32)
                .wrapping_mul(0xdffe)
                .wrapping_add(((m2 as i32).wrapping_add(4)).wrapping_shl(13)),
        );
    }
    if ((m1 as i32).wrapping_sub(0x23e0).wrapping_shl(16))
        .wrapping_sub((m2 as i32).wrapping_shl(16))
        < 0
    {
        return 0;
    }

    // --- 4. spread test, narrow bands only --------------------------------
    if hi - lo < 4 {
        let mut e_sp = e_in as i16;
        let mid = (((lo + 2 + hi) / 2) as i16) as i32;
        let mut buf = [0i16; 6];
        block_normalise_to_q15(&mut buf, &energy[(mid - 2) as usize..], 5, &mut e_sp);
        if spread_test(mant, e_band as u16 as i32, &buf, e_sp as u16 as i32, 5) {
            return 0;
        }
    }

    *id = ident;
    let v = high16_bL3b3(((m1 as i32).wrapping_shl(16)).wrapping_sub(0x03ff_0001));
    mag[0] = v;
    mag[1] = v;
    state[0xb] = m1;
    state[0xc] = m2;
    1
}

// ---------------------------------------------------------------------------
// 0x1031b980 — pitch halving decision
// ---------------------------------------------------------------------------

/// Normalisation shift written the way the compiler emits it here: an
/// open-coded `bsr` whose destination is left **unchanged** when the source is
/// zero. The source is `~x` for a negative `x`, so `x == -1` gives `bsr` a zero
/// operand and the shift comes out **30**, not 31 (SPEC §4.1). Callers
/// short-circuit `x == 0` before this is reached.
#[inline]
fn norm_shift_bsr(x: i32) -> u16 {
    let m = if x < 0 { !(x as u32) } else { x as u32 };
    let bsr = if m == 0 {
        0 // `bsr` left its destination at the zero it was loaded with
    } else {
        31 - m.leading_zeros() as i32
    };
    (0x1e - bsr) as u16
}

/// `0x1031b980` — decide whether the pitch estimate is an octave too high, and
/// halve it if so.
///
/// The classic MBE pitch-doubling error: a periodic waveform correlates just as
/// well at twice the true period, so the tracker can lock an octave high. This
/// function compares the harmonic energy collected at the candidate spacing
/// against the energy at *half* that spacing and, weighing several corroborating
/// signals, either keeps the candidate or returns `round(pitch/2)`.
///
/// - `pitch` is the candidate, `tail` the per-band energy vector the harmonic
///   sums are taken over, `spec` the magnitude-squared spectrum.
/// - `prev_a`, `prev_b`, `prev_c`, `prev_d` (`p5`, `p8`, `p9`, `p10`) are recent
///   pitch values the candidate is checked for consistency against;
///   `voiced_run` (`p7`) is the count of consecutive voiced frames, `p6` and
///   `p11` are level/confidence words, and `p12` enables the `p9` test.
/// - `*ratio` receives [`unvoiced_ratio`] (`0x1031b180`) evaluated at whichever
///   spacing was chosen — the voicing measure the caller goes on to use.
///
/// Flow:
///
/// 1. A confident candidate (`pitch > 0x199a`) that [`pitch_gate_score`]
///    (`0x1031b280`) does not reject is kept outright.
/// 2. [`sum_bands`] (`0x10320130`) at exponent 6 gives the energy at the
///    candidate spacing, at exponent 5 the energy at half spacing. Both are
///    normalised; the comparison is always `E_half · k` against `E_cand/2`.
/// 3. `k = 0x8a60/0x8000 = 1.081` guards the first (strict) test, `0xe8ba`
///    (0.909) the main one, `0xb6dc` (0.714) the low-confidence one and `1.0`
///    the final one — so how much more energy the half-spacing must show
///    depends on how much the other evidence supports halving.
/// 4. The pitch-consistency tests all have the shape "is `pitch/2` closer to
///    the remembered pitch than `pitch` is", with the `0xe666` (0.9),
///    `0xe8ba` (0.909), `0xd99a` (0.85) and `0xde9c` (0.87) ratio windows
///    deciding whether the remembered value is close enough to be evidence.
/// 5. `0x599a0000 - p6·0x2666` is a level-dependent threshold: a louder `p6`
///    lowers the energy the half spacing must reach.
///
/// The returned value is only meaningful in its low 16 bits — the keep path
/// returns `pitch` zero-extended, the halve path a sign-extended `pitch/2`.
///
/// The original stores its shift difference into the stack slot of `pitch`; per
/// SPEC §4.2 that is a reused slot, not a write to the caller's argument.
pub fn pitch_halve_decision(
    ratio: &mut i16,
    pitch: i32,
    tail: &[i16],
    spec: &[i32],
    prev_a: i16,
    level: i16,
    voiced_run: i16,
    prev_b: i16,
    prev_c: i16,
    prev_d: i16,
    conf: i16,
    enable: i16,
) -> u16 {
    let sv = pitch as i16;

    if sv > 0x199a {
        let gate = pitch_gate_score(spec, sv, -3) as i16;
        if gate >= 0 {
            *ratio = gate;
            return pitch as u16;
        }
    }

    // Energy at the candidate spacing.
    let s1 = sum_bands(tail, sv, 6, 0x40);
    let sh1 = if s1 == 0 { 0u16 } else { norm_shift_bsr(s1) };
    let v1 = ((s1 as u32) << (sh1 as u32 & 0x1f)) as i32;
    let u2 = sh1.wrapping_neg(); // zero-extended -shift
    let m1 = ((v1 >> 16) as u16) as i16;
    let score = unvoiced_ratio(m1, u2 as i16, tail, sv, 6) as i16;

    if sv <= 0xccc {
        *ratio = score;
        return pitch as u16;
    }

    // Energy at half the candidate spacing.
    let s2 = sum_bands(tail, sv, 5, 0x40);
    let sh2 = if s2 == 0 { 0u16 } else { norm_shift_bsr(s2) };
    let v2 = ((s2 as u32) << (sh2 as u32 & 0x1f)) as i32;
    let u3 = sh2.wrapping_neg();
    let m2 = ((v2 >> 16) as u16) as i16;

    let iv8 = (u3 as i32) - (u2 as i32) - 1;
    let half = ((m1 as i32) << 16) >> 1; // E_cand / 2

    // The result of the whole decision: `true` selects the halved pitch.
    let halve = 'decide: {
        // Strict first test. The comparison is `(i16)(-sh2) <= (i16)(-sh1) + 1`
        // — the `cwde` that sign-extends the left-hand side is easy to miss in
        // the disassembly, and reading it as the zero-extended word makes this
        // branch unreachable.
        if ((u3 as i16) as i32) <= ((u2 as i16).wrapping_add(1) as i32)
            && shl_sat((m2 as i32).wrapping_mul(0x8a60), iv8 as i16) > half
        {
            let mut fall_through = true;
            if voiced_run > 2 {
                let c1 = high16_bL3b3(
                    ((sv as i32) << 16).wrapping_sub((prev_d as i32).wrapping_mul(0xe666)),
                );
                if c1 >= 0 {
                    let c2 = high16_bL3b3(
                        (sv as i32)
                            .wrapping_mul(0xe8ba)
                            .wrapping_sub((prev_d as i32) << 16),
                    );
                    if c2 <= 0 {
                        fall_through = false; // straight to the main test
                    }
                }
            }
            if fall_through {
                let d1 = high16_bL3b3(
                    ((sv as i32) << 16).wrapping_sub((prev_b as i32).wrapping_mul(0xd99a)),
                );
                if d1 < 0 {
                    break 'decide true;
                }
                let d2 = high16_bL3b3(
                    (sv as i32)
                        .wrapping_mul(0xde9c)
                        .wrapping_sub((prev_b as i32) << 16),
                );
                if d2 > 0 {
                    break 'decide true;
                }
            }
        }

        // Main test at 0.909. Every branch that does not accept falls through
        // to the final test below — none of them returns "keep" directly.
        if shl_sat((m2 as i32).wrapping_mul(0xe8ba), iv8 as i16) > half {
            if (conf < score || conf < 0x199a) && enable != 0 {
                if closer_halved(sv, prev_c) {
                    break 'decide true;
                }
            } else if level < 0xccd {
                // Low confidence: 0.714 is the easiest threshold to clear.
                if shl_sat((m2 as i32).wrapping_mul(0xb6dc), iv8 as i16) > half {
                    break 'decide true;
                }
            } else {
                // Loud frame: a level-dependent threshold on the raw ratio.
                let t = shl_sat((m2 as i32) << 16, iv8 as i16);
                let thr =
                    high16_bL3b3(0x599a_0000i32.wrapping_sub((level as i32).wrapping_mul(0x2666)))
                        as i32
                        * (m1 as i32)
                        * 2;
                if t > thr && closer_halved(sv, prev_a) {
                    break 'decide true;
                }
            }
        }

        // Final test at 1.0, only for a run of voiced frames.
        if voiced_run < 3 {
            break 'decide false;
        }
        if shl_sat((m2 as i32) << 16, iv8 as i16) < half {
            break 'decide false;
        }
        closer_halved(sv, prev_c)
    };

    if halve {
        *ratio = unvoiced_ratio(m2, u3 as i16, tail, sv, 5) as i16;
        return (((((sv as i32) << 16) >> 1).wrapping_add(0x8000)) >> 16) as u16;
    }
    *ratio = score;
    pitch as u16
}

/// The pitch-consistency comparison this decision repeats four times: is the
/// *halved* candidate closer to `prev` than the candidate itself?
///
/// Both differences are formed at Q16 and taken through the original's
/// explicit `0x80000000 → 0x7fffffff` absolute value.
#[inline]
fn closer_halved(sv: i16, prev: i16) -> bool {
    let p = (prev as i32) << 16;
    let a = abs_sat((((sv as i32) << 16) >> 1).wrapping_sub(p));
    let b = abs_sat(((sv as i32) << 16).wrapping_sub(p));
    a < b
}

// ---------------------------------------------------------------------------
// 0x1031d810 — per-frame voicing / level tracker sequencer
// ---------------------------------------------------------------------------

/// `0x1031d810` — advance the encoder's frame-level trackers by one frame.
///
/// This is the hub that owns the encoder's slow state: the background-noise
/// estimate, the speech level, the voicing history and the frame counter. It
/// is called once per 20 ms frame with the sixteen sub-band energies of that
/// frame.
///
/// 1. **Age the history.** Three eight-word blocks at `state[0x81]`,
///    `state[0x89]` and `state[0x91]` (bytes `0x102`, `0x112`, `0x122`) are a
///    three-frame ring of per-band voicing decisions; they are shifted one
///    frame older, oldest first. The five words ending at `state[0x80]` and the
///    three at `state[0x99..=0x9b]` are shifted the same way — a five-frame and
///    a three-frame scalar history.
/// 2. **Frame energy.** [`log_mean_energy`] (`0x1031e8c0`) reduces the sixteen
///    band energies to one log level, stored at `state[0x25]` (byte `0x4a`).
/// 3. **Gain schedule.** When the long-term level `state[0x24]` is below
///    `-0x1400` the tracker is in a quiet passage and the gain handed to the
///    band tracker is `2^((level + 0x1400)·0.332 / 2^7) · 0.5437`, floored at
///    an argument of `-0x352`; otherwise it is the fixed `0x459a`. So the
///    per-band adaptation is slowed down in quiet frames and runs at a constant
///    rate once the level is up. The two arms **agree exactly at the
///    threshold**: at `level == -0x1400` the exponent argument is zero, so
///    `pow2_scaled` returns Q15 one and `hi16(0x8000 · 0x8b34)` is `0x459a`.
///    The schedule is continuous and `0x459a` is not an independent constant —
///    it is `0x8b34` (0.5437) evaluated at unity gain.
/// 4. **Band tracker.** [`band_voicing_tracker`] (`0x1031e1d0`) does the actual
///    per-band work and returns the frame's level estimate, with a scaled noise
///    peak through its out-parameter.
/// 5. **Level smoothing and classification.** [`smooth_level`] (`0x1031ea20`)
///    updates the attack/release pair from the tracker's level; the larger of
///    the level and the noise peak then goes through [`level_penalty`]
///    (`0x1031e080`), whose return value is this frame's classification bit.
///    A zero bit — the frame counts as background — also runs
///    [`update_smoothed_level`] (`0x1031ebf0`).
/// 6. **Bookkeeping.** The startup counter at `state[0x9c]` saturates at 8 (the
///    band tracker uses `< 8` to mean "still initialising"), and the
///    classification bit is shifted into the history word `state[0x22]`.
///
/// `bands` is the sixteen 32-bit band energies at block exponent `exp`; `den`
/// is the divisor that turns their sum into a mean and doubles as the band
/// tracker's fundamental term, so it must be neither `0` nor `-1`.
///
/// The original passes the address of its own `state` argument as the band
/// tracker's out-parameter and then reads the low half back as the noise peak;
/// per SPEC §4.2 that is a reused stack slot, modelled here as a local.
pub fn frame_tracker_step(state: &mut [i16], bands: &[i32], exp: i32, den: i16) {
    // 1. age the histories
    for k in 0..5usize {
        state[0x80 - k] = state[0x7f - k];
    }
    let mut blk = 0x91usize;
    let mut sc = 0x9busize;
    for _ in 0..2 {
        let (lo, hi) = state.split_at_mut(blk);
        copy16_batch3(hi, &lo[blk - 8..], 8);
        state[sc] = state[sc - 1];
        blk -= 8;
        sc -= 1;
    }

    // 2. frame energy
    state[0x25] = log_mean_energy(bands, exp, den) as i16;

    // 3. gain schedule
    let gain = if state[0x24] < -0x1400 {
        let d = (state[0x24] as i32).wrapping_add(0x1400) as i16;
        let s = high16_bL3b3((d as i32).wrapping_mul(0x550a));
        let arg = if s < -0x352 { -0x352 } else { s };
        let p = pow2_scaled(arg, 7, 0) as i16;
        high16_bL3b3((p as i32).wrapping_mul(0x8b34)) as u16 as i16
    } else {
        0x459a
    };

    // 4. band tracker
    let mut peak: i16 = 0;
    let passthru = state[0x27];
    let level_u = band_voicing_tracker(state, bands, exp as i16, den, passthru, &mut peak, gain);
    let level_s = level_u as i16;

    // 5. level smoothing and classification
    smooth_level(state, level_s);
    let mut level = if level_s < peak { peak } else { level_s };
    let pen = level_penalty(state, &mut level) as u16;
    state[0x7b] = level;
    if pen == 0 {
        update_smoothed_level(state);
    }

    // 6. bookkeeping
    if state[0x9c] < 8 {
        state[0x9c] = state[0x9c].wrapping_add(1);
    }
    state[0x22] = state[0x22].wrapping_mul(2) | (pen as i16);
}

// ======================= from bL3b4 =======================

// ---------------------------------------------------------------------------
// shared primitives (private copies of the idioms the originals inline)
// ---------------------------------------------------------------------------

/// The library's open-coded variable shift of an `i32`: a non-negative count
/// shifts left (wrapping), a count in `-30 ..= -1` shifts right arithmetically,
/// and anything at or below `-31` is a full sign fill (SPEC §4).
#[inline]
fn shift32_bL3b4(v: i32, s: i16) -> i32 {
    if s >= 0 {
        ((v as u32) << (s as u16 as u32 & 0x1f)) as i32
    } else if s < -0x1e {
        v >> 31
    } else {
        v >> (s.wrapping_neg() as u16 as u32 & 0x1f)
    }
}

/// `shl ax,cl` — a 16-bit left shift whose count is masked to five bits, so a
/// count of 16..31 shifts the whole word out and yields zero.
#[inline]
fn shl16_bL3b4(v: i16, count: i32) -> i16 {
    let c = (count as u32) & 0x1f;
    if c >= 16 {
        0
    } else {
        ((v as u16) << c) as i16
    }
}

/// MSVC's `__allshl` (`0x10305930`). Not in the ported map — no layer-0 batch
/// reached it — so it is reproduced here rather than depending on a sibling
/// batch's file. Only `cl` is significant, matching the register ABI.
fn allshl_bL3b4(v: i64, count: u8) -> i64 {
    let hi = (v >> 32) as u32;
    let lo = v as u32;
    if count >= 0x40 {
        return 0;
    }
    if count >= 0x20 {
        return (((lo << (count & 0x1f)) as u64) << 32) as i64;
    }
    let n = count & 0x1f;
    let (nh, nl) = if n == 0 {
        (hi, lo)
    } else {
        ((hi << n) | (lo >> (32 - n)), lo << n)
    };
    (((nh as u64) << 32) | nl as u64) as i64
}

// ---------------------------------------------------------------------------
// 0x1030d810 — windowed power spectrum
// ---------------------------------------------------------------------------

/// `0x1030d810` — window a signal segment, transform it, and optionally form
/// its power spectrum. This is the encoder's spectral analysis front door.
///
/// Three stages, exactly as the MBE analysis model prescribes:
///
/// 1. [`apply_sym_window`] (`0x1030d960`) multiplies `win_count` input samples
///    by a symmetric window stored as its right half, writing the halves to
///    opposite ends of `fft` and zero-filling the gap. `fft` is `n = 1 << log2n`
///    words long (plus the two the real transform needs for the Nyquist bin),
///    so the window is zero-padded up to the transform length.
/// 2. [`rfft_forward`] (`0x1031a2f0`) takes the real transform in place. Its
///    block-exponent input is the **sum** of the two exponents the caller
///    carries, `exp_sig + exp_win`, and its return is the running scaling
///    credit (SPEC §4.3 — zero-extended from 16 bits before use).
/// 3. When `mag` is supplied, [`cplx_mag_sq_x2`] (`0x10317610`) squares
///    `n/2 + extra` complex bins into it, and the returned exponent is
///    **doubled** because the magnitudes are squares. `extra` is normally 1,
///    which makes the count `n/2 + 1` — the full set of retained bins of a real
///    transform, DC through Nyquist.
///
/// `extra` does double duty: it is also `rfft_forward`'s `stage_flag`, which
/// decides whether the Nyquist bin is written out past the end of the half
/// spectrum and whether the identically-zero imaginary part of DC is cleared.
/// The two uses agree — asking for the Nyquist bin to be produced is the same
/// thing as asking for one more magnitude.
///
/// Passing `None` for `mag` skips stage 3 and returns the *undoubled* exponent,
/// which is how the caller obtains a windowed complex spectrum rather than a
/// power spectrum.
///
/// Domain: `4 <= log2n <= 8` (the inner complex FFT wants `log2n - 1` in
/// `[2, 7]`), and `out_off` must be 0 or at least `win_count / 2`.
#[allow(clippy::too_many_arguments)]
pub fn windowed_power_spectrum(
    mag: Option<&mut [i32]>,
    input: &[i16],
    exp_sig: i32,
    win: &[i16],
    exp_win: i32,
    win_count: i16,
    out_off: i16,
    fft: &mut [i16],
    log2n: i16,
    extra: i16,
) -> i16 {
    let n = shl16_bL3b4(1, log2n as i32);
    apply_sym_window(fft, input, n, out_off, win, win_count);
    let exp = rfft_forward(fft, exp_sig.wrapping_add(exp_win), log2n, extra) as u16 as i32;
    match mag {
        None => exp as i16,
        Some(m) => {
            let count = (n >> 1).wrapping_add(extra);
            cplx_mag_sq_x2(m, fft, count);
            exp.wrapping_mul(2) as i16
        }
    }
}

// ---------------------------------------------------------------------------
// 0x10311070 — pitch quantiser
// ---------------------------------------------------------------------------

/// `0x10311070` — quantise the fundamental to a channel index, then re-derive
/// the fundamental and harmonic count from that index.
///
/// `w0` is read as the analysed fundamental (Q15 radians/sample) and, on
/// return, is **overwritten with the dequantised value**; `l` receives the
/// matching harmonic count. The returned index is what the packer transmits.
/// Quantising and dequantising in one call is what keeps the encoder's
/// harmonic analysis aligned with what the decoder will reconstruct.
///
/// The quantiser is uniform in `log2(w0)`:
///
/// ```text
/// x   = -log2(w0 · 2^4)              (0x1030b100, then negated)
/// x   = hi16(x << 12) - 0x44fd       (to Q12, biased by the low pitch limit)
/// idx = x · 2N / 0x54a4              clamped to [0, N-1]
/// w0' = 0x1030e60( -0x44fd - (2·idx + 1)·0x2a52 / 2N )
/// ```
///
/// `0x2a52` is exactly half of `0x54a4`, so the reconstruction lands on the
/// **centre** of cell `idx` while the forward map floors — SPEC §4's
/// quantise-floors / reconstruct-to-centre rule, here in its clearest form.
/// `0x44fd` is the log-domain offset of the lowest representable pitch.
///
/// `levels_shift` sets the codebook size: `N = 0x3c << (levels_shift - 6)`, so
/// `levels_shift = 6` gives the 60-level codebook that fits a 6-bit field and
/// each further bit doubles it. The shift is `shl ax,cl` with the count masked
/// to five bits, so `levels_shift - 6 >= 16` collapses `N` to zero and the
/// division by `2N` faults.
///
/// `mode` selects the operation:
/// - `1` — quantise as above and return the index.
/// - `2` — the fixed default: `w0 = 0x4027`, `L = 15`, return
///   `0x3e << (levels_shift - 6)`, i.e. an out-of-range index two levels above
///   the codebook. This is the escape the frame classifier uses for a frame
///   that carries no pitch.
/// - anything else — no effect, return 0.
///
/// `allow_high` is passed through to [`fundamental_from_log2`]: zero clamps the
/// reconstructed fundamental at the `L = 9` end.
///
/// Domain: `mode == 1` requires `w0 != 0` (`log2_q` of zero is meaningless) and
/// `N != 0`.
pub fn quantise_pitch(
    w0: &mut i16,
    l: &mut i16,
    mode: i16,
    levels_shift: i32,
    allow_high: i16,
) -> i16 {
    if mode == 1 {
        let n = shl16_bL3b4(0x3c, levels_shift.wrapping_sub(6));
        let two_n = (n as u16).wrapping_mul(2) as i16;

        let lg = log2_q(*w0, -4);
        let neg = if lg == i32::MIN { 0x7fff_ffff } else { -lg };
        let hi = (((neg as u32) << 12) as i32 >> 16) as i16;
        let d = hi.wrapping_sub(0x44fd);

        let mut idx = div32((d as i32).wrapping_mul(two_n as i32), 0x54a4) as i16;
        if idx < 0 {
            idx = 0;
        } else if idx >= n {
            idx = (n as u16).wrapping_sub(1) as i16;
        }

        let cell = (idx as u16).wrapping_mul(2).wrapping_add(1) as i16;
        let recon = div32((cell as i32).wrapping_mul(0x2a52), two_n) as i16;
        let log_pitch = (-0x44fd_0000i32).wrapping_sub((recon as i32) << 16) >> 16;
        fundamental_from_log2(w0, l, log_pitch as i16, allow_high);
        idx
    } else if mode == 2 {
        let n = shl16_bL3b4(0x3e, levels_shift.wrapping_sub(6));
        *w0 = 0x4027;
        *l = 0x000f;
        n
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// 0x10313b60 — complex cross-correlation
// ---------------------------------------------------------------------------

/// `0x10313b60` — multiply one interleaved complex vector by the conjugate of
/// another, then renormalise the result to a block exponent.
///
/// For each of `n` bins, with `a = (re, im)` and `b = (re, im)` read as
/// consecutive Q15 word pairs:
///
/// ```text
/// dst.re = (a.re·b.re + a.im·b.im) >> 16
/// dst.im = (a.im·b.re - a.re·b.im) >> 16
/// ```
///
/// which is `a · conj(b)` — the cross-spectrum whose phase is the phase
/// difference between the two vectors. **The products are not doubled** before
/// the `>> 16` (contrast SPEC §4.1's usual `a·b·2 >> 16`), so the result is one
/// bit down from Q15; the exponent seeded below accounts for that with its
/// `+ 1`.
///
/// The block exponent starts at `exp_a + exp_b + 1` and [`normalise_to_exp`]
/// (`0x103175d0`) then rescales all `2n` words to fill the register and adds
/// its own shift. The final exponent is returned.
///
/// A non-positive `n` skips the multiply loop entirely but **still**
/// renormalises `2n` words — which for `n <= 0` is a no-op inside the callee —
/// and returns `exp_a + exp_b + 1` unchanged.
pub fn cplx_mul_conj_norm(
    dst: &mut [i16],
    a: &[i16],
    exp_a: i16,
    b: &[i16],
    exp_b: i16,
    n: i16,
) -> i16 {
    let mut exp = exp_b.wrapping_add(1).wrapping_add(exp_a);

    if n > 0 {
        for i in 0..(n as u16 as usize) {
            let b_re = b[2 * i] as i32;
            let b_im = b[2 * i + 1] as i32;
            let a_re = a[2 * i] as i32;
            let a_im = a[2 * i + 1] as i32;
            let re = a_re
                .wrapping_mul(b_re)
                .wrapping_add(a_im.wrapping_mul(b_im))
                >> 16;
            let im = a_im
                .wrapping_mul(b_re)
                .wrapping_sub(a_re.wrapping_mul(b_im))
                >> 16;
            dst[2 * i] = re as i16;
            dst[2 * i + 1] = im as i16;
        }
    }

    let count = (n as u16).wrapping_mul(2);
    let mut sbuf = [0i16; 0x102];
    let src = snap(&mut sbuf, &dst[..count as usize]);
    normalise_to_exp(dst, src, count, &mut exp);
    exp
}

// ---------------------------------------------------------------------------
// 0x10315a40 — two-candidate peak gate
// ---------------------------------------------------------------------------

/// `0x10315a40` — run the weighted-slope peak test on two windows of the same
/// 32-bit table and report whether **both** pass.
///
/// Each window is five consecutive `i32` from `table`, taken at `idx_a` and
/// `idx_b`, brought to Q15 words by [`block_normalise_to_q15`] (`0x103173a0`)
/// with a common starting exponent `exp`, and handed to [`peak_test`]
/// (`0x10315970`). Note that the window offset is also the seventh argument of
/// the test itself, so the *position* of the candidate participates in the
/// decision, not just the five samples around it.
///
/// The two windows are normalised independently — each gets its own exponent,
/// both seeded from `exp` — so the test compares each candidate against its own
/// local scale rather than a shared one.
///
/// Returns 1 only if both tests pass, 0 otherwise; the second test is skipped
/// when the first fails.
///
/// Domain: `idx_a` and `idx_b` must satisfy `0 <= idx <= table.len() - 5`.
#[allow(clippy::too_many_arguments)]
pub fn dual_peak_gate(
    gate_a: i16,
    gate_b: i16,
    ref_exp: i16,
    table: &[i32],
    exp: i16,
    idx_a: i16,
    lin_a: i16,
    idx_b: i16,
    lin_b: i16,
    thresh: i16,
) -> u32 {
    let mut buf_a = [0i16; 6];
    let mut exp_a = exp;
    block_normalise_to_q15(&mut buf_a, &table[idx_a as usize..][..5], 5, &mut exp_a);

    let mut buf_b = [0i16; 6];
    let mut exp_b = exp;
    block_normalise_to_q15(&mut buf_b, &table[idx_b as usize..][..5], 5, &mut exp_b);

    if peak_test(&buf_a, exp_a, 5, gate_a, ref_exp, lin_a, idx_a, thresh)
        && peak_test(&buf_b, exp_b, 5, gate_b, ref_exp, lin_b, idx_b, thresh)
    {
        1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// 0x1031c660 — per-band voicing measure (private copy)
// ---------------------------------------------------------------------------

/// `0x1031c660` — accumulate the eight-band voicing measure `1 - Σerr / Σsig`.
///
/// A private copy: this address belongs to a sibling batch that was still in
/// flight, and depending on that batch's file would couple the two.
///
/// The harmonic index range `0 ..= max_idx` is split into eight groups by a
/// running Q16 accumulator whose increment is `2 · 2^29 / spacing` renormalised
/// to Q16, i.e. `L / 8` harmonics per group. Group *k* ends at index
/// `round(k · inc)` clamped to `max_idx`, and starts at `round((k - 0.5) · inc)`
/// — the groups therefore *overlap by half*, so a harmonic near a boundary
/// contributes to both neighbours.
///
/// Both `err` and `sig` are summed over the group in 64 bits, normalised with
/// [`norm64`] and the `__allshl`/`__allshr` pair (SPEC §4.4), and
/// [`one_minus_ratio`] (`0x1031c390`) forms the Q15 result, floored at zero.
/// The result is **minimised** into `measure[k]`, so a caller sweeping several
/// analysis windows keeps the most-voiced evidence for each band.
///
/// Domain: `spacing != 0` (a zero divisor faults inside `div_norm`), and both
/// arrays must extend to `max_idx`.
#[allow(clippy::too_many_arguments)]
fn band_voicing_measure_bL3b4(
    measure: &mut [i16],
    sig: &[i32],
    sig_exp: i32,
    err: &[i32],
    err_exp: i16,
    spacing: i16,
    max_idx: u16,
) {
    let mut e: i16 = 0;
    let m = div_norm(0x2000_0000, -3, spacing, -3, &mut e);
    let half = shift32_bL3b4((m as i32) << 16, e.wrapping_sub(0xf));
    let inc = half.wrapping_mul(2);

    let mut start: u16 = 0;
    let mut edge: i32 = 0;

    for slot in measure.iter_mut().take(8) {
        edge = edge.wrapping_add(inc);
        let rounded = (edge.wrapping_add(0x8000) >> 16) as i16;
        let stop = if rounded > max_idx as i16 {
            max_idx as u32
        } else {
            rounded as u16 as u32
        };
        let span = stop.wrapping_sub(start as u32) as u16;
        // The original forms the byte offset as `movsx eax,si ; shl eax,2`, so
        // the group start is sign-extended from 16 bits before scaling.
        let base = (start as i16) as usize;

        let num = accumulate(err, base, span);
        let n1 = norm64(num as u32, (num >> 32) as u32) as i16;
        let s1 = if n1 >= 0 {
            allshl_bL3b4(num, n1 as u8)
        } else {
            allshr(num, n1.wrapping_neg() as u8)
        };
        let num_m = ((s1 as i32) >> 16) as i16;
        let num_e = (err_exp as i32).wrapping_sub(n1 as u16 as i32) as u16 as i16;

        let den = accumulate(sig, base, span);
        let n2 = norm64(den as u32, (den >> 32) as u32) as i16;
        let s2 = if n2 >= 0 {
            allshl_bL3b4(den, n2 as u8)
        } else {
            allshr(den, n2.wrapping_neg() as u8)
        };
        let den_m = ((s2 as i32) >> 16) as i16;
        let den_e = (sig_exp.wrapping_sub(n2 as u16 as i32)) as i16;

        let v = one_minus_ratio(num_m, num_e, den_m, den_e) as i16;
        if v < *slot {
            *slot = v;
        }

        start = (edge.wrapping_sub(half).wrapping_add(0x8000) >> 16) as u16;
    }
}

/// The 64-bit accumulation loop inside `0x1031c660`: `span + 1` words starting
/// at `base`, skipped entirely when `span` is negative as an `i16`.
fn accumulate(v: &[i32], base: usize, span: u16) -> i64 {
    if (span as i16) < 0 {
        return 0;
    }
    let mut acc: i64 = 0;
    for k in 0..=(span as usize) {
        acc = acc.wrapping_add(v[base + k] as i64);
    }
    acc
}

// ---------------------------------------------------------------------------
// 0x1031c820 — voicing decision driver
// ---------------------------------------------------------------------------

/// `0x1031c820` — the MBE voicing decision, from magnitude spectrum to eight
/// band measures.
///
/// This is the whole voicing analysis in one call, and it makes the algorithm
/// explicit. Two integrations of the same power spectrum are performed with the
/// same harmonic spacing but different band widths:
///
/// - [`band_energy`] (`0x103201d0`) integrates over the **full** harmonic band
///   `[(k - 0.5)·w0, (k + 0.5)·w0]` — the total energy attributable to harmonic
///   *k*. It also determines how many harmonics fit below bin 128 and returns
///   that count.
/// - [`band_energy_sum`] (`0x10320390`) integrates over the **central half**
///   `[(k - 0.25)·w0, (k + 0.25)·w0]` — the energy that a single clean harmonic
///   would account for.
///
/// Each set is then brought to a common block exponent with
/// [`align_to_max_exp`] (`0x1031c8f0`), and [`band_voicing_measure_bL3b4`]
/// (`0x1031c660`) groups the harmonics into eight voicing bands and forms
/// `1 - Σ(half-band) / Σ(full band)` per group. That ratio is the fraction of
/// each band's energy lying *away* from the harmonic peak: near zero for a
/// clean periodic band, large for a noisy one. It is minimised into `measure`,
/// which the caller may therefore call repeatedly to keep the strongest voicing
/// evidence.
///
/// Both energy arrays are addressed one element early, with a zero planted in
/// front of them (the `local_e0 = 0` / `local_1c0 = 0` of the decompilation),
/// so index 0 of a voicing group is a harmless zero rather than an off-by-one
/// read. The port reproduces that by allocating 56 words and handing the
/// integrators the tail.
///
/// `measure` is 8 words and must be pre-loaded by the caller (the minimum is
/// taken against whatever is already there). `step` is the fundamental in units
/// of 1/1024 of an FFT bin, and `spec` is the 129-bin power spectrum.
///
/// Domain: `step` must be strictly positive, and large enough that no more than
/// 55 harmonics fit below bin 128 — i.e. `step >= 2384`, which the codec's
/// `L <= 56` clamp guarantees.
pub fn voicing_measure(measure: &mut [i16], step: i16, spec: &[i32]) {
    // 56 words with index 0 the planted zero; the integrators fill 1..
    let mut full = [0i32; 56];
    let mut half = [0i32; 56];
    let mut exps = [0i16; 55];

    let n = band_energy(&mut full[1..], &mut exps, step, spec);
    let mut sbuf = [0i32; 55];
    let src = snap32(&mut sbuf, &full[1..]);
    let exp_full = align_to_max_exp(&mut full[1..], src, &exps, n as u16) as u16 as i32;

    band_energy_sum(&mut half[1..], &mut exps, step, n as u16, spec);
    let src = snap32(&mut sbuf, &half[1..]);
    let exp_half = align_to_max_exp(&mut half[1..], src, &exps, n as u16) as u16 as i16;

    band_voicing_measure_bL3b4(measure, &full, exp_full, &half, exp_half, step, n as u16);
}

// ======================= from bL4b1 =======================

// ---------------------------------------------------------------------------
// shared primitives (private copies of the idioms the originals inline)
// ---------------------------------------------------------------------------

/// The library's open-coded variable shift of an `i32`: a non-negative count
/// shifts left (wrapping), a count in `-30 ..= -1` shifts right arithmetically,
/// and anything at or below `-31` is a full sign fill (SPEC §4).
#[inline]
fn shift32_bL4b1(v: i32, s: i16) -> i32 {
    if s >= 0 {
        ((v as u32) << (s as u16 as u32 & 0x1f)) as i32
    } else if s < -0x1e {
        v >> 31
    } else {
        v >> (s.wrapping_neg() as u16 as u32 & 0x1f)
    }
}

/// High half of a 32-bit word, i.e. the Q15 mantissa of a Q31 accumulator.
#[inline]
fn high16_bL4b1(v: i32) -> i16 {
    ((v as u32) >> 16) as i16
}

/// MSVC's `__allshl` / `__allshr` pair (`0x10305930` / `0x10320530`) as the
/// callers here use them: a 64-bit value shifted by a small signed count, of
/// which only the low 32 bits are ever consumed (SPEC §4.2).
fn shift64(v: i64, s: i16) -> i64 {
    let c = (s as i32).unsigned_abs() & 0x3f;
    if s < 0 {
        if c >= 64 {
            v >> 63
        } else {
            v >> c
        }
    } else if c >= 64 {
        0
    } else {
        ((v as u64) << c) as i64
    }
}

/// The block-exponent bit scan exactly as the compiler emits it at
/// `0x10315ef0`: a search initialised to 31, so a zero magnitude reports −1.
/// Both reachable degenerate inputs (`0`, and `−1` which cannot occur because
/// the operand is always a multiple of `0x8000`) are guarded by the caller.
#[inline]
fn norm30(x: i32) -> u16 {
    if x == 0 {
        return 0;
    }
    let m = if x < 0 { !(x as u32) } else { x as u32 };
    let mut i = 31i32;
    if m != 0 {
        while (m >> i) == 0 {
            i -= 1;
        }
    }
    (0x1e - i) as u16
}

// ---------------------------------------------------------------------------
// 0x1031bd30 — per-frame pitch tracker
// ---------------------------------------------------------------------------

/// `0x1031bd30` — one frame of the pitch tracker: refine, guard against octave
/// errors, update the level tracker, and publish a smoothed pitch/voicing pair.
///
/// `track` is the seven-word level-tracker state that [`level_track_update`]
/// (`0x1031c0c0`) owns; only words 2, 4, 5 and 6 are read here (`+0x04`,
/// `+0x08`, `+0x0a`, `+0x0c`). `par` is a ten-word parameter block holding two
/// three-deep histories:
///
/// | word | 0 | 1 | 2 | 3 | 4 |
/// |---|---|---|---|---|---|
/// | pitch | current | smoothed | previous | −2 | −3 |
///
/// | word | 5 | 6 | 7 | 8 | 9 |
/// |---|---|---|---|---|---|
/// | voicing | current | smoothed | previous | −2 | −3 |
///
/// Flow:
///
/// 1. Both histories shift down one frame (`[4]←[2]`, `[3]←[1]`, `[2]←[0]` and
///    `[9]←[7]`, `[8]←[6]`, `[7]←[5]`), oldest first so nothing is clobbered.
/// 2. `first` clearing the voiced-run counter (`track[6]`) is what makes the
///    search window widen at the start of an utterance: the lag search floor
///    handed to [`refine_pitch_by_harmonics`] (`0x1031b7b0`) is 4 by default,
///    5 once three voiced frames have accumulated and either the tracked or the
///    previous level is at least `0x2000`, and 6 when **both** are at least
///    `0x2400`. A steadier, louder talker is trusted to have a higher pitch, so
///    the low lags are excluded and octave-low errors become impossible. Note
///    the ordering: the history has already shifted, so the "previous level"
///    this reads at `par[2]` is the `par[0]` the caller passed in.
/// 3. The refined value is clamped to `[0x666, 0x3e00]` — the codec's pitch
///    range in its internal log-lag units.
/// 4. [`pitch_halve_decision`] (`0x1031b980`) applies the octave-error guard
///    and also returns the frame's voicing ratio in `par[5]`.
/// 5. [`level_track_update`] (`0x1031c0c0`) advances the level tracker from the
///    new pitch, voicing and their previous values.
/// 6. The smoothed outputs `par[1]` (pitch) and `par[6]` (voicing) are chosen
///    by a three-way rule on the current and previous voicing:
///    * current voicing above `0x3333` and previous below it → **jump**: adopt
///      the previous values outright, so the tracker re-locks on an onset;
///    * previous voicing high while the current is low, or a previous voicing
///      above `0x199a` that is more than twice the current → **hold**: keep the
///      new values;
///    * otherwise **average** the current and previous of each.
///
/// The return value is the low half of [`pitch_halve_decision`]'s; the average
/// path leaves the sign fill of the previous voicing in the high half, which is
/// a compiler artefact of the halving there and is never read by a caller.
///
/// Domain: `corr` must be long enough for the harmonic walk (the original reads
/// past `n = 0x20`; 1024 words is ample) and `spec` is the 129-bin
/// magnitude-squared spectrum.
#[allow(clippy::too_many_arguments)]
pub fn pitch_track_frame(
    track: &mut [i16],
    par: &mut [i16],
    corr: &[i16],
    spec: &[i32],
    _unused_p5: i32,
    first: i16,
    _unused_p7: i32,
    flag_out: &mut i16,
) -> u32 {
    // Both histories shift down together, oldest slot first.
    for k in (5..=7).rev() {
        par[k - 3] = par[k - 5];
        par[k + 2] = par[k];
    }

    if first == 0 {
        track[6] = 0;
    }

    let mut lo: i16 = 4;
    if track[6] > 2 {
        if track[2] < 0x2400 || par[2] < 0x2400 {
            if track[2] > 0x1fff && par[2] > 0x1fff {
                lo = 5;
            }
        } else {
            lo = 6;
        }
    }

    let mut pitch = refine_pitch_by_harmonics(corr, lo, 0x20);
    if pitch < 0x3e01 {
        if pitch < 0x666 {
            pitch = 0x666;
        }
    } else {
        pitch = 0x3e00;
    }

    let alt = if track[6] < 1 { track[2] } else { par[2] };

    let mut ratio = par[5];
    let ret = pitch_halve_decision(
        &mut ratio,
        pitch as i32,
        corr,
        spec,
        track[4],
        track[5],
        track[6],
        track[2],
        alt,
        par[2],
        par[7],
        first,
    );
    par[5] = ratio;
    par[0] = ret as i16;
    *flag_out = 0;

    level_track_update(track, par[0], par[5], par[2], par[7]);

    let cur_v = par[5];
    let prev_v = par[7];
    if cur_v > 0x3333 && prev_v < 0x3333 {
        par[6] = prev_v;
        par[1] = par[2];
        return ret as u32;
    }
    if (prev_v < 0x3334 || cur_v > 0x3332) && (prev_v < 0x199b || (prev_v >> 1) <= cur_v) {
        par[1] = ((par[2] as i32).wrapping_add(par[0] as i32) >> 1) as i16;
        par[6] = ((cur_v as i32).wrapping_add(prev_v as i32) >> 1) as i16;
        // The high half is the sign fill of `prev_v` that the compiler left in
        // `edx:eax` after the halving; no caller looks at it.
        return (((prev_v >> 15) as u16 as u32) << 16) | ret as u32;
    }
    par[1] = par[0];
    par[6] = cur_v;
    ret as u32
}

// ---------------------------------------------------------------------------
// 0x1030b920 — frame prologue: bit allocation, pitch and voicing fields
// ---------------------------------------------------------------------------

/// The AMBE rate table at `0x1037fd78`, which Ghidra renders as the wide string
/// `L"10/.---"` because its entries fall in the printable ASCII range. In
/// decimal it reads **49, 48, 47, 46, 45, 45, 45** — the payload size in bits
/// of each of the five sub-modes of SPEC §3.1, with the top three rate indices
/// saturating at the smallest. The caller's rate index selects a sub-mode as
/// `49 - table[rate]`, i.e. the sub-mode number *is* the count of bits shed
/// from the 49-bit frame.
pub const AMBE_RATE_TABLE: [u16; 7] = [49, 48, 47, 46, 45, 45, 45];

/// `0x1030b920` — open a channel frame: rebuild the bit allocation, clear the
/// payload, and emit the fundamental and voicing fields.
///
/// This is the first thing the encoder does with a finished parameter block,
/// and it establishes the frame layout for everything that follows.
///
/// 1. The allocation vector `cfg` is rebuilt in place. `cfg[0] == 0` selects
///    [`imbe_bit_alloc`] (`0x1030c300`); anything else selects
///    [`ambe_bit_alloc`] (`0x1030bc70`) at the sub-mode `0x31 - `
///    [`AMBE_RATE_TABLE`]`[rate]`. So the rate index is a *per-frame* input,
///    not a session constant.
/// 2. `payload` is cleared to `0x31 = 49` words — one word per bit — and the
///    write cursor reset to its start.
/// 3. Both the current (`cur`) and previous (`prev`) parameter blocks record
///    the frame class in word 0. **Class 3 is the signalling-tone frame**: word
///    1 takes the tone index and [`pack_tone_frame`] (`0x10310460`) writes the
///    whole payload, and nothing else happens.
/// 4. Otherwise word 1 is set to `0xff` (no tone) and the two fields are
///    packed:
///    * [`quantise_voicing_bL2b4`] (`0x10312a10`) searches the voicing codebook over
///      the three 16-word weight blocks of `weights` and writes the expanded
///      dibit map into word 4 of *both* parameter blocks;
///    * [`split_voiced_freq_code`] (`0x103132b0`) tries to carry a joint
///      frequency refinement in the pitch field. If it declines, the pitch
///      field is [`quantise_pitch`] (`0x10311070`) of the fundamental in word
///      6 with the harmonic count in word 2. If it succeeds it has already
///      written the field, and both blocks are forced to the fixed
///      `w0 = 0x1079`, `L = 0x38` that the joint code implies — i.e. the
///      maximum harmonic count of SPEC §5.
/// 5. The two fields are emitted **`b1` first, then `b0`** (`cfg[5]` bits then
///    `cfg[4]` bits), which is where SPEC §3.1's "`b1` leads, not `b0`" comes
///    from. `b1` is therefore the fundamental/pitch field (7 bits AMBE, 8 bits
///    IMBE) and `b0` the voicing field (5 bits AMBE, 12 bits IMBE).
///
/// `weights` is three consecutive 16-word blocks (the voicing codebook search
/// reads `w`, `w + 0x20` and `w + 0x40` bytes).
///
/// Note that `b0` is 12 bits wide for IMBE, so an IMBE frame of class 1 asks
/// [`quantise_voicing_bL2b4`] for a 12-bit index. The original copes: the codebook
/// stride `1 << (7 - 12)` is a 16-bit `shl` by 27 and comes out **zero**, so
/// all 4096 candidates score book entry 0 and the search returns the first.
/// That combination cannot currently be exercised through the port because
/// `l1_batch2::voicing_codebook_search` forms the same stride with a Rust
/// shift, which overflows for counts above 15 — a defect in that batch, not
/// here.
#[allow(clippy::too_many_arguments)]
pub fn pack_pitch_and_voicing(
    payload: &mut [i16],
    cursor: &mut usize,
    rate: i16,
    cur: &mut [i16],
    prev: &mut [i16],
    _unused_p6: i32,
    tone: i16,
    tone_amp: &[i16],
    class: i16,
    cfg: &mut [i16; 13],
    weights: &[i16],
    book: &[u32],
    allow_high: i16,
) {
    if cfg[0] == 0 {
        imbe_bit_alloc(cfg);
    } else {
        let sub = (0x31u32).wrapping_sub(AMBE_RATE_TABLE[rate as usize] as u32) as i16;
        ambe_bit_alloc(cfg, sub);
    }
    fill16(payload, 0, 0x31);
    *cursor = 0;

    prev[0] = class;
    cur[0] = class;
    if class == 3 {
        cur[1] = tone;
        pack_tone_frame(tone, tone_amp, cfg, cursor, payload);
        return;
    }
    cur[1] = 0xff;

    let mut mask_prev = ((prev[5] as u16 as u32) << 16) | prev[4] as u16 as u32;
    let mut mask_cur = ((cur[5] as u16 as u32) << 16) | cur[4] as u16 as u32;
    let vidx = quantise_voicing_bL2b4(
        class,
        &mut mask_prev,
        &mut mask_cur,
        &weights[..0x10],
        &weights[0x10..0x20],
        &weights[0x20..0x30],
        book,
        cfg[4] as i32,
    );
    prev[4] = mask_prev as u16 as i16;
    prev[5] = (mask_prev >> 16) as u16 as i16;
    cur[4] = mask_cur as u16 as i16;
    cur[5] = (mask_cur >> 16) as u16 as i16;

    let mut code = rate;
    let (fa, fb) = (prev[7], cur[7]);
    let bits = cfg[5] as u32;
    let split = split_voiced_freq_code(&mut code, prev, cur, fa, fb, bits, 1);
    if split == 0 {
        let mut w0 = cur[6];
        let mut l = cur[2];
        code = quantise_pitch(&mut w0, &mut l, class, cfg[5] as i32, allow_high);
        cur[6] = w0;
        cur[2] = l;
    } else {
        prev[6] = 0x1079;
        cur[6] = 0x1079;
        prev[2] = 0x38;
        cur[2] = 0x38;
    }

    emit_bits(cursor, payload, cfg[5], code);
    emit_bits(cursor, payload, cfg[4], vidx as i16);
}

// ---------------------------------------------------------------------------
// 0x1030c480 — per-slot amplitude hub
// ---------------------------------------------------------------------------

/// Everything `0x1030c480` reaches through its context pointer, which the
/// original addresses as raw byte offsets into a caller-owned struct.
pub struct SlotContext {
    /// `+0x14` — the per-slot magnitude-squared spectrum matrix, 129 `i32` per
    /// slot.
    pub spec_off: u32,
    /// `+0x18` — per-slot block exponents.
    pub exps_off: u32,
    /// `+0x1c`, `+0x20` — opaque pointers forwarded to the sink.
    pub sink_a: u32,
    pub sink_b: u32,
    /// `+0x30` — first (16-bit) argument of the sink.
    pub sink_arg0: u16,
    /// `+0x32` — non-zero enables the sink.
    pub sink_enable: i16,
    /// `+0x34`, `+0x38`, `+0x48` — further opaque sink arguments.
    pub sink_c: u32,
    pub sink_d: u32,
    pub sink_e: u32,
    /// `+0x3c` — the sink itself.
    pub sink_fn: u32,
    /// `+0x40` — log-magnitude bias handed to `0x10310750`.
    pub level_bias: i16,
    /// `+0x42` — last (16-bit) argument of the sink.
    pub sink_arg9: u16,
}

/// The ten arguments `0x1030c480` pushes at its indirect sink call, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SinkCall {
    pub args: [u32; 10],
}

/// `0x1030c480` — finish one analysis slot's parameter block and hand it to the
/// registered sink.
///
/// `slot` is the sub-frame index and `base` the frame's slot base; their sum
/// (as a 16-bit value) is the absolute slot number used to index the spectrum
/// matrix and passed on to the sink.
///
/// On `slot == 0` — i.e. once per parameter block, not once per sub-frame — the
/// harmonic amplitude vector at `par[8..]` is produced, in one of two ways:
///
/// * **class 2** (repeat/silence) copies it wholesale from `template[1..]`,
///   `par[2] = L` words. No spectrum is consulted, which is what makes a repeat
///   frame cost nothing.
/// * otherwise [`slot_harmonic_levels`] (`0x10310750`) integrates the slot's
///   129-bin spectrum over the harmonic bands implied by the fundamental in
///   `par[6]`, biases and clamps it, and — because bit 0 of the voicing dibit
///   map at `par[4..6]` is the fundamental's own voicing bit — optionally
///   smears the peak.
///
/// The sink then runs for every class except 3 (tone), provided it is enabled.
/// It is a plain cdecl call of ten arguments, seven of them opaque context
/// words; the port returns them instead of calling, so a caller can dispatch
/// however it likes and a test can compare them word for word.
#[allow(clippy::too_many_arguments)]
pub fn slot_amplitudes_and_dispatch(
    par: &mut [i16],
    state: &mut [i16],
    slot: i16,
    base: i16,
    ctx: &SlotContext,
    template: &[i16],
    spec: &[i16],
    exps: &[i16],
) -> Option<SinkCall> {
    let abs = (base as u16).wrapping_add(slot as u16);

    if slot == 0 {
        if par[0] == 2 {
            let n = par[2];
            copy16_batch3(&mut par[8..], &template[1..], n);
        } else {
            let flags = ((par[5] as u16 as u32) << 16) | par[4] as u16 as u32;
            let (step, n) = (par[6], par[2]);
            let (_, tail) = par.split_at_mut(8);
            slot_harmonic_levels(
                tail,
                step,
                n,
                flags as i32,
                state,
                ctx.level_bias,
                spec,
                exps,
                abs as i16,
            );
        }
    }

    if ctx.sink_enable != 0 && par[0] != 3 {
        return Some(SinkCall {
            args: [
                ctx.sink_arg0 as u32,
                ctx.sink_d,
                0, // filled in by the caller: the address of `par`
                ctx.sink_e,
                ctx.sink_c,
                ctx.sink_a,
                ctx.sink_b,
                slot as u16 as u32,
                abs as u32,
                ctx.sink_arg9 as u32,
            ],
        });
    }
    None
}

// ---------------------------------------------------------------------------
// 0x1030e460 — sub-band segment assembly and power spectrum
// ---------------------------------------------------------------------------

/// The 29-tap half-window at `0x1037ff04` that `0x1030e460` analyses with.
pub const T_SUBBAND_WINDOW: u32 = 0x1037_ff04;

/// `0x1030e460` — assemble one 58-sample analysis segment from the history ring
/// plus the newest samples, put its seven sub-bands on a common block exponent,
/// and take its 64-point power spectrum.
///
/// This is the sub-band front end of the voicing decision. The seven bands come
/// out of the filterbank at seven *different* block exponents; a single
/// transform of the concatenation is only meaningful once they share one, which
/// is what the [`rebase_block_exp`] (`0x103172f0`) loop does.
///
/// 1. `0x3a - new_len` words of history are copied from `hist` ending 9 words
///    before `hist[new_len]`, then `new_len` words of `fresh` are appended, so
///    `work` always holds exactly `0x3a = 58` samples with the newest at the
///    top.
/// 2. `edges[0..7]` are the band start offsets inside that 58-word buffer and
///    `exps[0..7]` their exponents. Bands `0..6` run to the next edge; band 6
///    runs from `edges[6]` to 58. Each is shifted to the common exponent
///    `target`.
/// 3. [`windowed_power_spectrum`] (`0x1030d810`) windows the 58 samples with
///    the 29-tap half-window at [`T_SUBBAND_WINDOW`], writing the halves either
///    side of offset `0x1d`, takes the 64-point real transform in place and
///    squares the 32 retained bins into `mag`.
///
/// The returned exponent is `2 ×` the transform's, because the bins are
/// squared; the original discards it but it is the value the caller's spectrum
/// is scaled by.
#[allow(clippy::too_many_arguments)]
pub fn subband_power_spectrum(
    mag: &mut [i32],
    hist: &[i16],
    fresh: &[i16],
    new_len: i16,
    edges: &[i16],
    exps: &[i16],
    target: i32,
    work: &mut [i16],
    win: &[i16],
) -> i16 {
    let keep = (0x3au32).wrapping_sub(new_len as u16 as u32) as i16;
    // `hist + 2*new_len - 0x12` — the history is read ending nine words before
    // the point the fresh samples take over.
    let start = (new_len as i32).wrapping_sub(9) as usize;
    copy16_batch3(work, &hist[start..], keep);
    copy16_batch3(&mut work[keep as u16 as usize..], fresh, new_len);

    for i in 0..6usize {
        let at = edges[i] as usize;
        let n = edges[i + 1].wrapping_sub(edges[i]) as u16;
        let mut sbuf = [0i16; 0x3a];
        let src = snap(&mut sbuf, &work[at..at + n as usize]);
        rebase_block_exp(&mut work[at..], src, n, exps[i] as i32, target);
    }
    let at = edges[6] as usize;
    let n = (0x3au32).wrapping_sub(edges[6] as u16 as u32) as u16;
    let mut sbuf = [0i16; 0x3a];
    let src = snap(&mut sbuf, &work[at..at + n as usize]);
    rebase_block_exp(&mut work[at..], src, n, exps[6] as i32, target);

    let mut ibuf = [0i16; 0x3a];
    let input = snap(&mut ibuf, &work[..0x3a]);
    windowed_power_spectrum(Some(mag), input, target, win, 0, 0x3a, 0x1d, work, 6, 0)
}

// ---------------------------------------------------------------------------
// 0x10313ec0 — global phase alignment and band scoring
// ---------------------------------------------------------------------------

/// `0x10313ec0` — align the measured spectrum to the reference in phase, then
/// score the eight bands.
///
/// The MBE voicing decision compares a synthetic reference spectrum against the
/// measured one. That comparison is meaningless if the two are offset in phase,
/// so this derives the single complex rotation that best aligns them and
/// applies it before scoring.
///
/// 1. [`complex_correlate`] (`0x10314660`) accumulates the windowed
///    cross-spectrum `Σ ref·conj(meas)` over 127 bins, returning it as a
///    complex `(re, im)` pair at one shared exponent.
/// 2. [`atan2_bfp`] (`0x1030ad30`) turns that pair into an angle. Scaled by
///    `slope` and shifted by `slope_exp - 7`, it is written to `out[0]`: the
///    **linear phase term**, i.e. the sub-sample time offset between the two
///    spectra.
/// 3. The same pair, clamped to symmetric Q15 (`-0x7fff`, SPEC §4), is then
///    normalised to unit magnitude: `|z|²` is accumulated in 64 bits,
///    normalised ([`norm64`], `0x10310150`), square-rooted ([`sqrt_norm`],
///    `0x1030b2d0`), and `0.5 / |z|` obtained by [`div_norm`] (`0x1030a660`).
///    Scaling both components by that gives the unit phasor. A non-positive
///    `|z|²` leaves the pair unscaled at exponent 0.
/// 4. [`rotate_spectrum_by_phasor`] (`0x103138e0`) rotates all 127 measured
///    bins by it, and [`band_spectral_match`] (`0x103136e0`) writes the eight
///    per-band error/energy ratios into `out`.
///
/// So `out[0]` carries the linear phase and `out[0x0c..0x14]` the band scores —
/// two different products of the same correlation.
///
/// Note that the original reads the *third* word of `complex_correlate`'s
/// two-word output when assembling `atan2`'s first argument. That word is
/// uninitialised stack, but it lands only in the high half of a value whose
/// low half is all `atan2_bfp` uses (SPEC §4.2), so it is not load-bearing.
#[allow(clippy::too_many_arguments)]
pub fn align_phase_and_score_bands(
    win: &[i16],
    meas: &[i16],
    meas_exp: i16,
    reference: &mut [i16],
    ref_exp: i16,
    slope: i16,
    slope_exp: i16,
    out: &mut [i16],
) -> u16 {
    let mut z = [0i16; 2];
    let corr_exp = complex_correlate(&mut z, win, meas, meas_exp, reference, ref_exp, 0x7f) as u16;

    let mut ang_exp = 0i16;
    let ang = atan2_bfp(z[1], corr_exp as i16, z[0], corr_exp as i16, &mut ang_exp);

    let scaled = (ang as i16 as i32)
        .wrapping_mul(slope as i32)
        .wrapping_mul(2);
    let sh = ang_exp.wrapping_add(slope_exp).wrapping_sub(7);
    out[0] = if sh < 0 {
        if sh < -0x1e {
            (scaled >> 31) as i16
        } else {
            high16_bL4b1(scaled >> (sh.wrapping_neg() as u16 as u32 & 0x1f))
        }
    } else {
        high16_bL4b1(((scaled as u32) << (sh as u16 as u32 & 0x1f)) as i32)
    };

    if z[0] < -0x7ffe {
        z[0] = -0x7fff;
    }
    if z[1] < -0x7ffe {
        z[1] = -0x7fff;
    }

    let x = z[0] as i32;
    let y = z[1] as i32;
    let acc = (y.wrapping_mul(y).wrapping_mul(2) as i64)
        .wrapping_add(x.wrapping_mul(x).wrapping_mul(2) as i64);
    let n = norm64(acc as u32, (acc >> 32) as u32) as u16;
    let shifted = shift64(acc, n as i16) as i32;

    let phasor_exp = if shifted < 1 {
        0u32
    } else {
        let mut e = (corr_exp.wrapping_mul(2).wrapping_sub(n)) as i16;
        let root = sqrt_norm(shifted, &mut e);
        let mut de = 0i16;
        let inv = div_norm(0x4000_0000, 1, high16_bL4b1(root), e, &mut de);
        z[0] = high16_bL4b1((inv as i32).wrapping_mul(x).wrapping_mul(2));
        z[1] = high16_bL4b1((inv as i32).wrapping_mul(y).wrapping_mul(2));
        corr_exp.wrapping_add(de as u16) as u32
    };

    let rot = rotate_spectrum_by_phasor(reference, &z, phasor_exp as i32, ref_exp as i32, 0x7f);
    band_spectral_match(meas, meas_exp, reference, rot as i16, out)
}

// ---------------------------------------------------------------------------
// 0x10315ef0 — two-peak signalling-tone detector
// ---------------------------------------------------------------------------

/// Base of the nine small tables `0x10315ef0` indexes: a 16-entry tone-class
/// matrix at `0x10381ba8` followed by eight four-entry candidate tables at
/// `0x10381bc8`, `bd0`, `bd8`, `be0`, `be8`, `bf0`, `bf8`, `c00`.
pub const T_TONE_TABLES: u32 = 0x1038_1ba8;

/// Column offsets into [`ToneTables`] for the two candidate groups.
#[derive(Clone, Copy)]
pub struct ToneTables<'a> {
    /// 16 words at `0x10381ba8` — the class matrix, indexed `(i + 4·j) & 0xf`.
    pub class: &'a [i16],
    /// `0x10381bc8` — low-group linear positions.
    pub lin_lo_a: &'a [i16],
    /// `0x10381bd0` — low-group bin offsets.
    pub bin_lo_a: &'a [i16],
    /// `0x10381bd8` — low-group companion linear positions.
    pub lin_lo_b: &'a [i16],
    /// `0x10381be0` — low-group companion bin offsets.
    pub bin_lo_b: &'a [i16],
    /// `0x10381be8` — high-group linear positions.
    pub lin_hi_a: &'a [i16],
    /// `0x10381bf0` — high-group bin offsets.
    pub bin_hi_a: &'a [i16],
    /// `0x10381bf8` — high-group companion linear positions.
    pub lin_hi_b: &'a [i16],
    /// `0x10381c00` — high-group companion bin offsets.
    pub bin_hi_b: &'a [i16],
}

/// `0x10315ef0` — detect a two-tone signalling pair in the energy spectrum.
///
/// The codec carries in-band signalling tones (SPEC §5, `0x1037fea8`) that must
/// be transmitted as *symbols*, not vocoded, because a parametric coder
/// destroys them. This is the detector, and it is a classic dual-tone receiver
/// built out of the same block-floating primitives as the rest of the encoder.
///
/// Two candidate groups are examined, distinguished only by their tables and by
/// the class offset they add:
///
/// | group | bin tables | class offset | peak threshold |
/// |---|---|---|---|
/// | low  | `0x10381bd0` / `0x10381be0` | `+0x80` | `0x51ec` |
/// | high | `0x10381bf0` / `0x10381c00` | `+0x90` | `0x4189` |
///
/// Per group:
///
/// 1. [`argmax_block_sum`] (`0x10316510`) picks, out of four candidate bin
///    offsets in each of the group's two tables, the one whose five-bin energy
///    sum is largest. Ties take the **last** candidate (SPEC §4.1).
/// 2. [`band_pair_at_common_exp`] (`0x10316290`) re-sums those two winners onto
///    one shared exponent, giving the pair `(a, b)`.
/// 3. The class matrix at `0x10381ba8`, indexed `(i + 4·j) & 0xf` by the two
///    winning candidate indices, plus the group's offset, names the tone.
/// 4. [`dual_peak_gate`] (`0x10315a40`) requires *both* winners to look like
///    isolated spectral peaks. That test is **skipped** when the resulting class
///    already equals `expected` — hysteresis, so a tone already being tracked
///    survives a frame in which the peak shape degrades.
///
/// If both groups pass, the louder pair wins: the two sums are halved, each
/// normalised, and compared by [`mant_exp_gt`] (`0x1030a440`). If only one
/// passes it is used; if neither, the frame is not a tone.
///
/// The surviving pair then has to clear two more tests before the detection is
/// reported:
///
/// * **balance** — neither component may exceed the other by more than 10×
///   (`x·0x1000 <= y·0xa000` both ways). A real signalling pair is transmitted
///   at equal levels, so a lopsided pair is speech.
/// * **absolute level** — the mean of the two, shifted to exponent
///   `exp - floor + 1`, must reach `threshold · 0xf810`.
///
/// On success `*class_out` receives the tone class and `amps[0..2]` the two log
/// amplitudes ([`log_amp_clamped`], `0x103164b0`), and 1 is returned.
#[allow(clippy::too_many_arguments)]
pub fn detect_dual_tone(
    class_out: &mut i16,
    amps: &mut [i16],
    expected: u16,
    energy: &[i32],
    exp: i16,
    amp_offset: i16,
    threshold: i16,
    floor: i32,
    t: &ToneTables<'_>,
) -> u32 {
    let mut lo_ok = false;
    let mut hi_ok = false;

    // -- low group ---------------------------------------------------------
    let mut best = 0u32;
    let mut i_lo = 0i16;
    let mut j_lo = 0i16;
    argmax_block_sum(&mut best, &mut i_lo, 4, energy, t.bin_lo_a, 5);
    argmax_block_sum(&mut best, &mut j_lo, 4, energy, t.bin_lo_b, 5);

    let mut a1 = 0i16;
    let mut b1 = 0i16;
    let e1 = band_pair_at_common_exp(
        &mut a1,
        &mut b1,
        energy,
        exp,
        t.bin_lo_a[i_lo as usize],
        t.bin_lo_b[j_lo as usize],
    );
    let c1 = (t.class[((i_lo as i32 + j_lo as i32 * 4) & 0xf) as usize] as u16).wrapping_add(0x80);
    if c1 == expected
        || dual_peak_gate(
            a1,
            b1,
            e1 as i16,
            energy,
            exp,
            t.bin_lo_a[i_lo as usize],
            t.lin_lo_a[i_lo as usize],
            t.bin_lo_b[j_lo as usize],
            t.lin_lo_b[j_lo as usize],
            0x51ec,
        ) as i16
            == 1
    {
        lo_ok = true;
    }

    // -- high group --------------------------------------------------------
    let mut i_hi = 0i16;
    let mut j_hi = 0i16;
    argmax_block_sum(&mut best, &mut i_hi, 4, energy, t.bin_hi_a, 5);
    argmax_block_sum(&mut best, &mut j_hi, 4, energy, t.bin_hi_b, 5);

    let mut a2 = 0i16;
    let mut b2 = 0i16;
    let e2 = band_pair_at_common_exp(
        &mut a2,
        &mut b2,
        energy,
        exp,
        t.bin_hi_a[i_hi as usize],
        t.bin_hi_b[j_hi as usize],
    );
    let c2 = (t.class[((i_hi as i32 + j_hi as i32 * 4) & 0xf) as usize] as u16).wrapping_add(0x90);
    if c2 == expected
        || dual_peak_gate(
            a2,
            b2,
            e2 as i16,
            energy,
            exp,
            t.bin_hi_a[i_hi as usize],
            t.lin_hi_a[i_hi as usize],
            t.bin_hi_b[j_hi as usize],
            t.lin_hi_b[j_hi as usize],
            0x4189,
        ) as i16
            == 1
    {
        hi_ok = true;
    }

    // -- pick the group ----------------------------------------------------
    let mut a = a1;
    let mut b = b1;
    let mut e = e1;
    let mut c = c1;
    let mut adopt_high = false;

    if !lo_ok {
        if !hi_ok {
            return 0;
        }
        adopt_high = true;
    } else if hi_ok {
        let s2 = (((a2 as i32) << 16) >> 1).wrapping_add(((b2 as i32) << 16) >> 1);
        let n2 = norm30(s2);
        let x2 = shift32_bL4b1(s2, n2 as i16) >> 16;
        let e2m = e2.wrapping_sub(n2);

        let s1 = (((a1 as i32) << 16) >> 1).wrapping_add(((b1 as i32) << 16) >> 1);
        let n1 = norm30(s1);
        let x1 = shift32_bL4b1(s1, n1 as i16) >> 16;

        if mant_exp_gt(x2 as i16, e2m as i16, x1 as i16, e1.wrapping_sub(n1) as i16) {
            adopt_high = true;
        }
    }

    if adopt_high {
        a = a2;
        b = b2;
        e = e2;
        c = c2;
    }

    // -- balance and level -------------------------------------------------
    let ia = (a as i32) << 16;
    let ib = (b as i32) << 16;
    if (ia >> 4) <= (b as i32).wrapping_mul(0xa000) && (ib >> 4) <= (a as i32).wrapping_mul(0xa000)
    {
        let sh = ((e as i32).wrapping_sub(floor)).wrapping_add(1);
        let mean = (ib >> 1).wrapping_add(ia >> 1);
        let level = shift32_bL4b1(mean, sh as i16);
        if (threshold as i32).wrapping_mul(0xf810) <= level {
            *class_out = c as i16;
            amps[0] = log_amp_clamped(a, e as i32, amp_offset) as i16;
            amps[1] = log_amp_clamped(b, e as i32, amp_offset) as i16;
            return 1;
        }
    }
    0
}

// ======================= from bL5b1 =======================

// ---------------------------------------------------------------------------
// 0x10315e20 — frame classifier: is this frame a tone, and which one?
// ---------------------------------------------------------------------------

/// `0x10315e20` — classify the frame as one of the signalling tones, or as
/// ordinary speech.
///
/// This is the gate that keeps the vocoder from trying to code in-band
/// signalling with a parametric speech model (SPEC §5, `0x1037fea8`). It runs
/// three detectors over the same magnitude-squared spectrum in strict priority
/// order and stops at the first that fires:
///
/// 1. [`detect_dual_tone`] (`0x10315ef0`) — a *pair* of tones, one from each of
///    two frequency groups. Its hysteresis input is the previously latched
///    class held at `st[4]`.
/// 2. [`detect_signalling_tone`] (`0x10316bf0`) — a single isolated spectral
///    peak.
/// 3. [`tone_detect`] (`0x10315650`) — the four fixed-band tones.
///
/// Only the *first* of these is allowed to consume a frame: the later ones are
/// not consulted once one has claimed it, so a dual-tone reading always beats a
/// single-tone reading of the same spectrum.
///
/// **The common reference the three share** is computed once here.
/// [`accumulate_exp`] (`0x10317140`) sums the first `0x80` bins into a
/// normalised (mantissa, exponent) pair; the mantissa's high word is the
/// frame's energy reference and the exponent replaces the caller's. Two
/// short-circuits follow: a zero mantissa (a silent frame) and an exponent
/// below `-0x1a` (an absurdly small total) both return `0xff`, the
/// "not a tone" code, without running any detector.
///
/// **The exponent is passed on zero-extended.** The original reuses its own
/// `exp` argument slot as the out-parameter, having first written the argument
/// back into it through a `movzx`; the callee then writes only the low 16 bits.
/// The `i32` reference exponent the three detectors see is therefore
/// `exp as u16 as i32` — non-negative even when the exponent itself is
/// negative. This is not cosmetic: it is what the detectors' shift logic
/// actually receives (SPEC §4.2, reused argument slots).
///
/// Note the detectors get the caller's *original* `exp` as their spectral
/// exponent and the *updated* one only as the reference exponent — the two are
/// different arguments carrying different meanings.
///
/// Returns the tone class, or `0xff` for "speech".
#[allow(clippy::too_many_arguments)]
pub fn classify_frame_tone(
    result: &mut [i16],
    energy: &[i32],
    exp: i16,
    offset: i16,
    st: &mut [i16],
    enable: i16,
    dual: &ToneTables<'_>,
    bands: &[i16],
) -> u16 {
    let mut e = exp;
    let acc = accumulate_exp(energy, 0x80, &mut e);
    let ref_level = (acc >> 16) as u16 as i16;
    if ref_level == 0 || e < -0x1a {
        return 0xff;
    }
    // The out-parameter slot was zero-extended before the call, so the caller's
    // sign never reaches the detectors.
    let ref_exp = e as u16 as i32;

    let mut cls: i16 = 0xff;
    if detect_dual_tone(
        &mut cls,
        result,
        st[4] as u16,
        energy,
        exp,
        offset,
        ref_level,
        ref_exp,
        dual,
    ) as u16
        != 1
        && detect_signalling_tone(
            &mut cls, result, st, energy, exp as u16, offset, ref_level, ref_exp, enable,
        ) as u16
            != 1
    {
        tone_detect(
            &mut cls, result, st, energy, exp, offset, ref_level, ref_exp, bands,
        );
    }
    cls as u16
}

// ---------------------------------------------------------------------------
// 0x10314070 — spectral envelope analysis and band-score ageing
// ---------------------------------------------------------------------------

/// `0x10314070` — build the frame's minimum-phase reference spectrum, align the
/// measured spectrum to it, score the eight voicing bands, and age the retained
/// band scores.
///
/// This is the encoder's spectral-envelope stage. `power` is the 129-bin
/// magnitude-squared spectrum at exponent `power_exp` (the original adds
/// `0x1e` to the argument on entry) and `meas` is the same frame's interleaved
/// complex spectrum at `meas_exp` (`+0xf` on entry). What comes out is written
/// into the tracker block `st`.
///
/// **The history shift comes first**, before anything is computed. Five
/// parallel rows are advanced by one frame, oldest slot first so nothing is
/// clobbered:
///
/// | row | slots | meaning |
/// |---|---|---|
/// | `st[0], st[1], st[2]` | 1 word | linear phase offset |
/// | `st[3], st[4], st[5]` | 1 word | (written elsewhere) |
/// | `st[6], st[7], st[8]` | 1 word | (written elsewhere) |
/// | `st[0x0c..], st[0x14..], st[0x1c..]` | 8 words | per-band match scores |
/// | `st[0x24], st[0x25], st[0x26]` | 1 word | the band scores' shared exponent |
///
/// Slot 0 of each row is this frame; slots 1 and 2 are the two previous frames.
///
/// **The reference spectrum is minimum phase.** MBE compares the measured
/// spectrum against what a perfectly periodic excitation through the same
/// spectral envelope would have produced, and that reference has to be
/// *constructed*:
///
/// 1. [`log_spectrum_pairs`] (`0x10313cb0`) turns the 129 power bins into
///    interleaved log-magnitude pairs.
/// 2. [`minimum_phase_phasor`] (`0x10313db0`) takes the 256-point transform of
///    that log spectrum and exponentiates it — the cepstral construction of a
///    minimum-phase signal from its magnitude alone.
/// 3. [`log2_vector`] (`0x10313d40`) separately produces the per-bin log
///    magnitudes, which become the analysis **weight**: the smallest non-zero
///    entry sets the normalisation shift `0xf - blockexp((min - 1) << 16)`, and
///    [`reciprocal_vec`] (`0x10313c10`) inverts the vector at that shift. So
///    quiet bins are weighted *up*, by the reciprocal of their own magnitude.
/// 4. Both the reference and the measured spectrum are then windowed by that
///    weight ([`window_complex_block`], `0x10313ad0`) and multiplied by the
///    conjugate of themselves shifted one bin ([`cplx_mul_conj_norm`],
///    `0x10313b60`) — a one-bin autocorrelation of each spectrum, which is what
///    makes the subsequent comparison insensitive to absolute phase.
///
/// **The weight vector is then reused as the band mask.**
/// [`voiced_band_mask`] (`0x103145a0`) overwrites it with the eight-band
/// voiced/unvoiced expansion driven by `amp`/`lvl`, and the first twelve bins
/// are zeroed outright — below roughly 375 Hz nothing contributes to the
/// voicing comparison. [`align_phase_and_score_bands`] (`0x10313ec0`) then
/// derives the single rotation that best aligns the two, writes the linear
/// phase to `st[0]` and the eight band scores to `st[0x0c..0x14]`, and returns
/// their shared exponent, which lands in `st[0x24]`. The slope constant it
/// carries is `0xae84` at exponent 6.
///
/// **The ageing loop is a per-band min-hold.** It runs only for a frame that
/// looks like a *voiced onset*: `st[1] > 0` (last frame's phase offset
/// positive), `st[0] < 0` (this frame's negative), and the two one-word rows
/// ordered `st[9] >= st[10]`, with `st[6] >= st[7]` breaking the tie. When it
/// runs, every band whose current score is *below* the retained one — compared
/// after both are shifted to the common exponent `max(cur_exp, prev_exp) + 1` —
/// has the retained score replaced by the current one, rescaled to the retained
/// exponent. So the record of each band's best match can only improve within an
/// utterance; it never decays here.
///
/// Domains: `st` at least `0x27` words, `meas` at least `0x102`, `power` at
/// least `0x81`, `amp`/`lvl` 8 each.
#[allow(clippy::too_many_arguments)]
pub fn spectral_envelope_frame(
    st: &mut [i16],
    meas: &mut [i16],
    meas_exp: i32,
    power: &[i16],
    power_exp: i32,
    amp: &[i16],
    lvl: &[i16],
    lvl_exp: i16,
    cos_tbl: &[i16],
) {
    let power_exp = power_exp.wrapping_add(0x1e);
    let meas_exp = meas_exp.wrapping_add(0xf);

    // -- history shift, oldest slot first -----------------------------------
    st[2] = st[1];
    st[5] = st[4];
    st[8] = st[7];
    let mut row = [0i16; 8];
    row.copy_from_slice(&st[0x14..0x1c]);
    copy16_batch3(&mut st[0x1c..], &row, 8);
    st[0x26] = st[0x25];

    st[1] = st[0];
    st[4] = st[3];
    st[7] = st[6];
    let mut row = [0i16; 8];
    row.copy_from_slice(&st[0x0c..0x14]);
    copy16_batch3(&mut st[0x14..], &row, 8);
    st[0x25] = st[0x24];

    // -- minimum-phase reference and the analysis weight ---------------------
    let mut phas = [0i16; 0x102];
    let mut win = [0i16; 0x102];
    let mut sbuf = [0i16; 0x102];

    let e = log_spectrum_pairs(&mut phas, power, power_exp, 0x81);
    let e = minimum_phase_phasor(&mut phas, e as u16 as i32, 8, 0x100, cos_tbl);
    let wexp = log2_vector(&mut win, power, power_exp as i16, 0x81);
    let src = snap(&mut sbuf, &phas);
    let ref_exp = window_complex_block(&mut phas, src, e as u16 as i32, &win, wexp as i32, 0x81);

    // The smallest non-zero log magnitude sets the reciprocal's shift.
    let mut min = 0x7fffi16;
    for &v in win[..0x81].iter() {
        if v != 0 && v < min {
            min = v;
        }
    }
    let shift = if min.wrapping_sub(1) == 0 {
        0u16
    } else {
        // `(min - 1) << 16` can never be 0 or -1 here, so the normalisation is
        // the plain `0x1e - bsr` form (SPEC §4.1 does not bite).
        let x = ((min.wrapping_sub(1) as i32) << 16) as u32;
        let mag = if (x as i32) < 0 { !x } else { x };
        let mut i = 31u32;
        while mag >> i == 0 {
            i -= 1;
        }
        (0x0fu16).wrapping_sub((0x1eu32.wrapping_sub(i)) as u16)
    };

    let src = snap(&mut sbuf, &win);
    let win_exp = reciprocal_vec(&mut win, src, wexp as i32, shift as i16, 0x81);

    let src = snap(&mut sbuf, &phas);
    let ref_exp = cplx_mul_conj_norm(&mut phas, src, ref_exp, &src[2..], ref_exp, 0x80);

    // -- the measured spectrum through the same two stages -------------------
    let src = snap(&mut sbuf, &meas[..0x102]);
    let m = window_complex_block(meas, src, meas_exp, &win, win_exp as u16 as i32, 0x81);
    let src = snap(&mut sbuf, &meas[..0x102]);
    let meas_exp2 = cplx_mul_conj_norm(meas, src, m, &src[2..], m, 0x80);

    // -- the weight buffer becomes the band mask -----------------------------
    voiced_band_mask(&mut win, amp, lvl, lvl_exp);
    fill16(&mut win, 0, 0xc);

    let cur_exp = align_phase_and_score_bands(
        &win,
        &phas,
        ref_exp,
        meas,
        meas_exp2,
        0xae84u16 as i16,
        6,
        st,
    );
    st[0x24] = cur_exp as i16;

    // -- per-band min-hold ---------------------------------------------------
    if st[1] > 0 && st[0] < 0 && st[9] >= st[10] && (st[9] != st[10] || st[6] >= st[7]) {
        let prev_exp = st[0x25];
        let top = if (cur_exp as i16) > prev_exp {
            cur_exp as i16 as i32
        } else {
            prev_exp as i32
        };
        // The common exponent both scores are brought to, one above the larger.
        let base = top.wrapping_add(1) as u16 as i32;
        let dc = (cur_exp as i32).wrapping_sub(base);
        let dp = (prev_exp as u16 as i32).wrapping_sub(base);
        let dd = (cur_exp as i32).wrapping_sub(prev_exp as u16 as i32);

        for k in 0..8usize {
            let cur = (st[0x0c + k] as i32) << 16;
            let a = tishift(cur, dc);
            let b = tishift((st[0x14 + k] as i32) << 16, dp);
            if a.wrapping_sub(b) < 0 {
                st[0x14 + k] = (tishift(cur, dd) >> 16) as i16;
            }
        }
    }
}

/// The TI variable shift as the original inlines it: the sign test is on the
/// **low 16 bits** of the count, the shift amount is masked to 5 bits, and a
/// count at or below `-0x1f` is a full sign fill (SPEC §4).
fn tishift(x: i32, n: i32) -> i32 {
    if (n as i16) < 0 {
        if (n as i16) > -0x1f {
            x >> (n.wrapping_neg() as u32 & 0x1f)
        } else {
            x >> 31
        }
    } else {
        ((x as u32) << (n as u32 & 0x1f)) as i32
    }
}

// ---------------------------------------------------------------------------
// 0x1030eaa0 — per-frame pitch / voicing sequencer
// ---------------------------------------------------------------------------

/// `0x1030eaa0` — run one frame of pitch tracking and the voicing decision,
/// then condition the per-band voicing gains on the frame's noise picture.
///
/// This is the encoder's parameter-estimation hub. It owns three regions of the
/// state block, addressed here in `i16` words:
///
/// | words | bytes | contents |
/// |---|---|---|
/// | `0x30a .. 0x311` | `0x614` | pitch level-tracker state (7 words) |
/// | `0x311 .. 0x31b` | `0x622` | pitch / voicing parameter block (10 words) |
/// | `0x31b .. ` | `0x636` | per-band voicing measures and their histories |
/// | `0x48d, 0x48e, 0x48f` | `0x91a` | three-deep history of the voicing code |
///
/// Order of business:
///
/// 1. [`pitch_track_frame`] (`0x1031bd30`) refines the fundamental from the
///    autocorrelation `corr` and the power spectrum `energy`, guarding against
///    octave errors, and publishes a smoothed pitch/voicing pair. Its
///    "first frame" input is the *previous* frame's voicing code, read before
///    the history is shifted.
/// 2. The voicing-code history shifts (`0x91e ← 0x91c ← 0x91a`) and
///    [`band_voicing_decision`] (`0x1031c960`) produces this frame's code from
///    the band-energy vector, landing in `0x91a`.
/// 3. **Pitch halving.** A fundamental above `0x3759` is halved outright. The
///    quantity at `0x622` is the pitch parameter the tracker just wrote, so
///    this is a hard octave clamp applied after tracking, not during it.
/// 4. If the pitch is above `0x1000` *or* the caller's flag is set,
///    [`voicing_measure`] (`0x1031c820`) re-scores the eight voicing bands at
///    that spacing, and a set flag additionally floors band 1's measure at
///    `0x1999` (0.2).
/// 5. [`subband_energy_update`] (`0x1030e660`) folds the frame's energy into
///    the sixteen sub-band noise trackers held in `aux`.
///
/// **The gain conditioning is driven by a two-frame voicing bit count.**
/// `popcount_low` (`0x1031ecc0`) over the low 8 bits of the current and the
/// two-frames-ago voicing codes gives `n`, the number of voiced bands across
/// the pair — but only if `aux`'s frame-class bits (`byte 0x44 & 0x1c`) say the
/// frame is codable; otherwise `n` is forced to 0. Then:
///
/// * **Noise-floor branch** — taken when both the current and the previous
///   smoothed voicing measure (`0x62c` = `par[5]`, `0x630` = `par[7]`, low
///   means strongly voiced) are above `0x2000`, *or* fewer than four bands are
///   voiced. [`boost_three_band_gains`] (`0x1030dc80`) lifts the three lowest
///   bands, and if the frame is loud (`level·0x550a - 0xd000000` clears the
///   reference at exponent 7), quiet in absolute terms (`0xfa < 0x1c00`),
///   *fully unvoiced in both this frame and two frames back* (both codes have
///   their low seven bits clear) and rising by more than `0x1600` over the
///   frame at `0x13a`, band 0's measure is floored at `0x1c29`. That
///   conjunction is an onset detector: silence, then a sudden loud
///   unvoiced-but-rising frame.
///
/// * **Voiced branch** — taken when either measure is at or below `0x2000` and
///   at least four bands are voiced. Under the same loudness test (with
///   the softer `-0xb000000` bias) and with band 1's measure below `0x199a`,
///   band 0's measure is multiplied by band 1's and then by `0xa000` with a
///   saturating `<< 3` — a net `×5` with saturation, since
///   `0xa000 · 8 / 2^16 = 5`. `0xa000` is an **unsigned** Q16 immediate
///   (SPEC §4.1); reading it as a negative `i16` inverts the sign of the boost.
///
/// `p8`, `p10` and `p11` are pushed by the original but read by nothing.
#[allow(clippy::too_many_arguments)]
pub fn pitch_voicing_frame(
    st: &mut [i16],
    corr: &[i16],
    spec: &[i16],
    band_exp: &[i16],
    aux: &mut [i16],
    energy: &[i32],
    energy_exp: i16,
    flag: &mut i16,
) {
    let first = st[0x48d];

    let v = {
        let (a, b) = st.split_at_mut(0x311);
        pitch_track_frame(&mut a[0x30a..], b, corr, energy, 0, first, 0, flag)
    };

    st[0x48f] = st[0x48e];
    st[0x48e] = st[0x48d];
    st[0x48d] = band_voicing_decision(&mut st[0x31b..], v as u16 as i16, spec, band_exp);

    if st[0x311] > 0x3759 {
        st[0x311] >>= 1;
    }
    if st[0x311] > 0x1000 || *flag != 0 {
        let step = st[0x311];
        voicing_measure(&mut st[0x31b..], step, energy);
        if *flag != 0 && st[0x322] < 0x1999 {
            st[0x322] = 0x1999;
        }
    }

    subband_energy_update(energy, energy_exp, st, aux, *flag);

    // Two-frame voiced-band count, zeroed for a frame class that is not coded.
    let mut n = (popcount_low(st[0x48f] as i32 as u32, 8) as i16)
        .wrapping_add(popcount_low(st[0x48d] as i32 as u32, 8) as i16);
    if aux[0x22] as u16 & 0x1c == 0 {
        n = 0;
    }

    let voiced = st[0x316] <= 0x2000 || st[0x318] <= 0x2000;
    if voiced && n >= 4 {
        // -- voiced branch --------------------------------------------------
        let level = ((aux[0x24] as i32)
            .wrapping_mul(0x550a)
            .wrapping_sub(0x0b00_0000)
            >> 16) as i16;
        if compare_scaled(level, 7, aux[0x29], aux[0x39])
            && compare_scaled(aux[0x2b], aux[0x3b], aux[0x29], aux[0x39])
            && st[0x31c] < 0x199a
        {
            let t = ((st[0x31b] as i32)
                .wrapping_mul(st[0x31c] as i32)
                .wrapping_mul(2)
                >> 16) as i16;
            // 0xa000 is unsigned Q16 (SPEC §4.1): 40960, not -24576.
            st[0x31b] = (shl_sat((t as i32).wrapping_mul(0xa000), 3) >> 16) as i16;
        }
    } else {
        // -- noise-floor branch ----------------------------------------------
        boost_three_band_gains(
            &mut st[0x31b..],
            &aux[0x91..],
            &aux[0x29..],
            &aux[0x39..],
            n,
        );

        let l = ((aux[0x24] as i32)
            .wrapping_mul(0x550a)
            .wrapping_sub(0x0d00_0000)
            >> 16) as i16;
        if compare_scaled(l, 7, aux[0x29], aux[0x39])
            && aux[0x7d] < 0x1c00
            && compare_scaled(aux[0x2b], aux[0x3b], aux[0x29], aux[0x39])
            && (st[0x48d] as u16) as u8 & 0x7f == 0
            && (st[0x48f] as u16) as u8 & 0x7f == 0
            && (aux[0x24] as i32).wrapping_sub(aux[0x9d] as i32) > 0x1600
            && st[0x31b] < 0x1c29
        {
            st[0x31b] = 0x1c29;
        }
    }
}

// ---------------------------------------------------------------------------
// 0x1030e100 — sub-band filterbank spectrum builder
// ---------------------------------------------------------------------------

/// A Q31 accumulator times a Q15 word, as the original inlines it: the low half
/// zero-extended and shifted down 15, plus the sign-extended high half doubled
/// (SPEC §4.1's 32×16 idiom).
fn mul32x16_bL5b1(v: i32, s: i16) -> i32 {
    let lo = (v as u16 as i32).wrapping_mul(s as i32) >> 15;
    let hi = ((v >> 16) as i16 as i32)
        .wrapping_mul(s as i32)
        .wrapping_mul(2);
    lo.wrapping_add(hi)
}

fn get32_bL5b1(b: &[i16], i: usize) -> i32 {
    (b[2 * i] as u16 as i32) | ((b[2 * i + 1] as i32) << 16)
}

fn put32_bL5b1(b: &mut [i16], i: usize, v: i32) {
    b[2 * i] = v as i16;
    b[2 * i + 1] = (v >> 16) as i16;
}

/// `0x1030e100` — build the encoder's eight-band sub-band spectrum and the
/// summed spectrum that goes with it.
///
/// Eight bands × two half-frames = sixteen 58-sample analysis segments, each
/// turned into a 32-bin power spectrum by [`subband_power_spectrum`]
/// (`0x1030e460`), weighted, and accumulated two ways: per band into `spec`,
/// and across everything into `total`.
///
/// **Geometry.** `hist` advances 49 words and `fresh` 11 words per segment, so
/// the sixteen segments tile a 784-word history against 176 fresh samples. That
/// is exactly what `0x1030e460` wants: it keeps `0x3a - 11 = 47` history words
/// starting nine words back and appends the 11 new ones to make 58. `edges` and
/// `exps` are the same seven-entry pair for every segment, and their maximum
/// (`vec_max`, `0x103174e0`) is the common block exponent every segment is
/// rebased to.
///
/// **The weight depends on the band.** [`speech_activity_gate`]
/// (`0x1030dd30`) runs once. If it says the frame is not speech, no weight is
/// applied at all — every segment enters the total unscaled. Otherwise:
///
/// * **Band 0** uses [`log_energy_to_gain`] (`0x1030eed0`), *called separately
///   for each half-frame* with the half index — the automatic-gain term. Band 0
///   alone also has its own per-band spectrum scaled in place by the same gain
///   and its exponent raised by it, so the AGC follows band 0 into `spec` as
///   well as into `total`.
/// * **Bands 1..7** use [`voicing_threshold`] (`0x1030ed30`) indexed by the
///   running segment number `0 ..= 15`, i.e. a per-(band, half-frame)
///   threshold, applied only to the copy that enters `total`.
///
/// **`spec` is used at two widths.** During a band it holds 32 `i32` power
/// bins at byte offset `band · 0x40`; [`block_normalise_to_q15`]
/// (`0x103173a0`) then rewrites the first 64 bytes of that as 32 `i16` and
/// leaves the exponent in `band_exp[band]`. Because the `i32` form is 128 bytes
/// and the stride is 64, each band's upper half is scratch that the next band
/// overwrites — and band 7's upper half is simply left behind. The result is
/// the eight contiguous 32-word blocks that [`band_voicing_decision`]
/// (`0x1031c960`) reads.
///
/// `total` receives the same 32 bins summed over all sixteen segments with
/// [`block_add_sat`] (`0x1030f820`), whose exponent starts at `-30`; the
/// running exponent is truncated to 16 bits after every step and discarded at
/// the end.
///
/// Bins 0 and 1 of every segment are forced to zero before weighting — DC and
/// the first bin carry no voicing information and would otherwise dominate.
///
/// Domains: `spec` at least `0x120` words (576 bytes), `total` and `band_exp`
/// 32 and 8 words, `hist` `16·49` words, `fresh` `16·11`, `new_len` 11 (the
/// value the geometry above is built on).
#[allow(clippy::too_many_arguments)]
pub fn subband_spectrum_frame(
    total: &mut [i16],
    spec: &mut [i16],
    band_exp: &mut [i16],
    hist: &[i16],
    fresh: &[i16],
    new_len: i16,
    exps: &[i16],
    edges: &[i16],
    state: &[i16],
    win: &[i16],
) {
    let target = vec_max(exps, 7) as u16 as i32;

    let mut level = 0i16;
    let mut lexp = 0i16;
    let mut vflag = 0i16;
    let gate = speech_activity_gate(state, &mut level, &mut lexp, &mut vflag) as u16 as i16;
    // The original zero-extends these three out-parameters into 32-bit slots
    // before the call, so the callee's 16-bit writes leave the high halves
    // clear and the two `i32` arguments below are never negative (SPEC §4.2).
    let lexp32 = lexp as u16 as i32;
    let flag32 = vflag as u16 as i32;

    let mut acc = [0i32; 32];
    let mut acc_exp: u16 = 0xffe2; // -30
    let mut seg = 0i32; // running segment index, 0 ..= 15
    let mut hist_at = 0usize;
    let mut fresh_at = 0usize;

    let mut work16 = [0i16; 80];
    let mut half1 = [0i32; 32]; // the second half-frame's spectrum

    for band in 0..8usize {
        let base = band * 16; // `i32` index of this band's block
        let mut seg_exp = [0u16; 2];
        let mut gain_m = 0i16;
        let mut gain_e = 0u16;

        for half in 0..2usize {
            let mut mag = [0i32; 32];
            let e = subband_power_spectrum(
                &mut mag,
                &hist[hist_at..],
                &fresh[fresh_at..],
                new_len,
                edges,
                exps,
                target,
                &mut work16,
                win,
            );
            seg_exp[half] = e as u16;
            mag[0] = 0;
            mag[1] = 0;

            let mut uexp = e as u16 as u32;
            let mut weighted = [0i32; 32];
            if gate == 0 {
                weighted = mag;
            } else {
                let scale;
                let bexp;
                if band == 0 {
                    gain_e = log_energy_to_gain(&mut gain_m, state, half as i16) as u16;
                    scale = gain_m;
                    bexp = gain_e;
                } else {
                    let mut s = 0i16;
                    bexp =
                        voicing_threshold(&mut s, state, level, lexp32, seg as i16, flag32) as u16;
                    scale = s;
                }
                for k in 0..32 {
                    weighted[k] = mul32x16_bL5b1(mag[k], scale);
                }
                uexp = uexp.wrapping_add(bexp as u32);
                if band == 0 {
                    for k in 0..32 {
                        mag[k] = mul32x16_bL5b1(mag[k], gain_m);
                    }
                    seg_exp[half] = (gain_e as i16).wrapping_add(seg_exp[half] as i16) as u16;
                }
            }

            if half == 0 {
                for k in 0..32 {
                    put32_bL5b1(spec, base + k, mag[k]);
                }
            } else {
                half1 = mag;
            }

            let src = acc;
            acc_exp =
                block_add_sat(&mut acc, &src, &weighted, acc_exp as i16, uexp as i16, 0x20) as u16;

            seg += 1;
            hist_at += 49;
            fresh_at += 11;
        }

        // Fold the two half-frames together, then narrow the block to `i16`.
        let mut blk = [0i32; 32];
        for k in 0..32 {
            blk[k] = get32_bL5b1(spec, base + k);
        }
        let src = blk;
        band_exp[band] = block_add_sat(
            &mut blk,
            &src,
            &half1,
            seg_exp[0] as i16,
            seg_exp[1] as i16,
            0x20,
        );
        for k in 0..32 {
            put32_bL5b1(spec, base + k, blk[k]);
        }
        let mut out = [0i16; 32];
        block_normalise_to_q15(&mut out, &blk, 0x20, &mut band_exp[band]);
        spec[band * 32..band * 32 + 32].copy_from_slice(&out);
    }

    let mut e = acc_exp as i16;
    block_normalise_to_q15(total, &acc, 0x20, &mut e);
}

// ======================= from bL6b1 =======================

// ---------------------------------------------------------------------------
// 0x1030c360 — IMBE spectral quantiser and channel interleave
// ---------------------------------------------------------------------------

/// `0x1030c360` — quantise one IMBE frame's spectral envelope into the channel
/// bit stream, interleave the frame, and promote the current parameter block.
///
/// This is the encode-side mirror of the decoder's `0x1030bfd0` tail, and it
/// runs the same three routines in the opposite direction:
///
/// 1. [`unpack_nibbles`] (`0x10311b00`) reads the **per-harmonic bit
///    allocation** for `cur[2]` harmonics out of the packed nibble table at
///    `0x10381428` into a 56-word scratch. The allocation is a function of the
///    harmonic count alone, so the encoder and the decoder derive it
///    independently and never transmit it.
/// 2. [`code_log_mags_with_voicing`] (`0x10311fa0`) with `encode = 1` codes the
///    log-magnitude residual into the payload, bracketed by the unvoiced-band
///    attenuation, and emits the alternating framing bit from `prev[0x44]`.
/// 3. [`imbe_frame_deinterleave`] (`0x10312fc0`) permutes the finished
///    payload from parameter order into channel order (SPEC §3.2). The
///    direction flag is the opposite of the one the decoder uses, so a single
///    routine serves both sides.
///
/// Then `cur[0] = 1` — the frame is unconditionally marked *speech*, because
/// only the speech path reaches this function — and [`copy_params_batch2`]
/// (`0x10311a20`) promotes `cur` into `prev` with `mask` landing in
/// `prev[0x45]`, which is what makes the next frame's prediction work.
///
/// `pos` is the channel bit cursor. In the original it is a `short **` cell
/// that the quantisation hub `0x1030bda0` left pointing into `chan`; here it is
/// an index into `chan`, which is the same thing for every reachable call.
///
/// The original also passes `alloc` to `copy_params_batch2` as a fourth argument that
/// the callee never reads; it is not modelled.
///
/// Domain: `alloc[0] == 0` (an IMBE allocation vector) — the interleave is
/// IMBE-only. `cur[2]` is the harmonic count *L*, 9…56.
#[allow(clippy::too_many_arguments)]
pub fn encode_frame_imbe_spectrum(
    chan: &mut [i16],
    pos: &mut i32,
    cur: &mut [i16],
    prev: &mut [i16],
    mask: i16,
    alloc: &[i16],
    nib_tbl: &[i16],
    gain_book: &[i16],
    weight: &[i16],
    loading: &[i16],
    band_tbl: &[i16],
    cos_tbl: &[i16],
) {
    // The original's 112-byte stack local.
    let mut hbits = [0u16; 56];
    unpack_nibbles(&mut hbits, cur[2], nib_tbl);

    code_log_mags_with_voicing(
        prev, cur, mask, &hbits, chan, pos, 1, gain_book, weight, loading, band_tbl, cos_tbl,
    );

    let mut hbits_i = [0i16; 56];
    for (d, &x) in hbits_i.iter_mut().zip(hbits.iter()) {
        *d = x as i16;
    }
    imbe_frame_deinterleave(chan, cur[2], &hbits_i, alloc);

    cur[0] = 1;
    copy_params_batch2(prev, cur, mask);
}

// ---------------------------------------------------------------------------
// 0x1030f920 — per-half-frame tone classification with hold
// ---------------------------------------------------------------------------

/// Word offsets inside the tone tracker (`state + 0x1370`).
mod trk {
    /// Byte `+0x06` — the hold budget: how many further half-frames may repeat
    /// the latched class.
    pub const HOLD: usize = 3;
    /// Byte `+0x08` — the last class emitted, and the hysteresis input the
    /// three detectors read as `st[4]`.
    pub const LAST: usize = 4;
    /// Byte `+0x0a` — per-half-frame class, indexed by the half.
    pub const CLASS: usize = 5;
    /// Byte `+0x0e` — per-half-frame amplitude pair, indexed by the half.
    pub const AMPS: usize = 7;
}

/// `0x1030f920` — decide whether this half-frame carries a signalling tone, and
/// keep the decision alive across a short dropout.
///
/// The classifier proper is [`classify_frame_tone`] (`0x10315e20`), which runs
/// the dual-tone, single-tone and band-tone detectors over the frame's
/// magnitude-squared spectrum in priority order and answers with a class or
/// `0xff` for "speech". This function is the **latch** around it, and the two
/// results per 20 ms frame — one per half — are what the packing hub later
/// turns into a tone frame.
///
/// * `flags & 0x1000` is the master enable. Cleared, the half is recorded as
///   speech (`0xff`) with a zero amplitude pair and *nothing else is touched* —
///   in particular the hold budget and the latched class survive, so disabling
///   detection mid-utterance does not reset the tracker.
/// * A detector that fires writes the class and its amplitude pair, latches the
///   class, and **recharges the hold budget to 2**.
/// * A detector that declines consults the latch. A latched class of `0x80` or
///   above (the dual-tone range) repeats, spends one unit of budget, and plants
///   the sentinel amplitude pair `0xb001, 0xb001`. Classes `0x13` and `0x3a` —
///   the two single-frequency call-progress tones — repeat the same way but
///   keep the amplitudes they already had.
/// * Anything else, or an exhausted budget, falls through to speech and clears
///   the latch.
///
/// So a tone survives at most two consecutive half-frames of non-detection
/// before the tracker gives up, and only for the classes the codec is willing
/// to extrapolate.
///
/// Domain: `half` is the half-frame index, 0 or 1; `trk` must be at least
/// `0x20` words (what the detectors' shared state block needs) and `energy` at
/// least `0x80` bins.
#[allow(clippy::too_many_arguments)]
pub fn tone_class_half_frame(
    trk: &mut [i16],
    energy: &[i32],
    exp: i16,
    flags: u32,
    half: i16,
    enable: i16,
    dual: &ToneTables<'_>,
    bands: &[i16],
) {
    let i = half as i32 as usize;
    let cls_w = trk::CLASS + i;
    let amp_w = trk::AMPS + 2 * i;

    if flags & 0x1000 == 0 {
        trk[cls_w] = 0xff;
        trk[amp_w] = 0;
        trk[amp_w + 1] = 0;
        return;
    }

    // The original reuses the `flags` argument slot as this two-word scratch,
    // seeding it from the tracker before the call and reading it back after
    // (SPEC §4.2, reused argument slots).
    //
    // The seeding is **not observable**: all three detectors write both words
    // whenever they set a class, and the caller only stores the pair back when
    // a class was set, so no input distinguishes this from starting at zero.
    // Confirmed by mutation. Kept because the original does it, and because a
    // future detector path that declined to write would make it load-bearing.
    let mut amps = [trk[amp_w], trk[amp_w + 1]];
    let cls = classify_frame_tone(&mut amps, energy, exp, 0, trk, enable, dual, bands);

    if cls != 0xff {
        trk[cls_w] = cls as i16;
        trk[amp_w] = amps[0];
        trk[amp_w + 1] = amps[1];
        trk[trk::HOLD] = 2;
        trk[trk::LAST] = trk[cls_w];
        return;
    }

    let last = trk[trk::LAST];
    if last >= 0x80 && trk[trk::HOLD] > 0 {
        trk[cls_w] = last;
        trk[trk::HOLD] = trk[trk::HOLD].wrapping_sub(1);
        trk[amp_w] = 0xb001u16 as i16;
        trk[amp_w + 1] = 0xb001u16 as i16;
        trk[trk::LAST] = trk[cls_w];
        return;
    }
    if (last == 0x13 || last == 0x3a) && trk[trk::HOLD] > 0 {
        trk[cls_w] = last;
        trk[trk::HOLD] = trk[trk::HOLD].wrapping_sub(1);
        trk[trk::LAST] = trk[cls_w];
        return;
    }

    trk[cls_w] = 0xff;
    trk[amp_w] = 0;
    trk[amp_w + 1] = 0;
    trk[trk::LAST] = trk[cls_w];
}

// ---------------------------------------------------------------------------
// 0x1030e5a0 — sub-band analysis for one call's worth of new audio
// ---------------------------------------------------------------------------

/// `0x1030e5a0` — advance the encoder's sub-band front end by the hops the new
/// audio affords, then build the eight-band spectrum from them.
///
/// Four steps, in order:
///
/// 1. [`subband_analysis_frame`] (`0x1031af90`) converts `advance` new samples
///    into `hops` eight-sample analysis hops — the phase accumulator at
///    `sub[0x6dc]` carries the remainder between calls — and produces a
///    16 × 11 grid of sub-band amplitudes plus the 16 `i32` energy
///    accumulators. It also ages the band hold counters and peak trackers that
///    live immediately above the history in `sub[0x620 … 0x63c]`.
/// 2. [`frame_tracker_step`] (`0x1031d810`) folds those 16 energies into the
///    long-term level and gain trackers held in `trk`, with `2 · hops` as the
///    divisor and `2 · sub[0x62c] + 7` as the energy exponent.
/// 3. [`subband_spectrum_frame`] (`0x1030e100`), **only when `active > 0`**,
///    tiles the 784-word band history against the fresh grid into eight bands ×
///    two half-frames and produces the per-band spectra, their exponents, and
///    the summed spectrum. The peak/counter pair the previous step just aged is
///    what it uses as its per-segment block exponents and band edges.
/// 4. [`shift_band_history`] (`0x1030e530`) slides all sixteen 49-word history
///    rows left by `hops` and appends the fresh grid — so step 3 always sees
///    the history *before* this call's samples were folded in, and the next
///    call sees them.
///
/// **The caller aliases three of these buffers**, and an implementation is free
/// to keep them separate. `0x1030e020` passes one 64-byte stack slot as both
/// `acc` and `total`, which is safe because `acc` is consumed in step 2 before
/// `total` is written in step 3; and it places `spec` so that its last 64 bytes
/// *are* that slot, which is safe because those bytes are band 7's `i32`
/// scratch, written before and overwritten by `total` at the very end of
/// step 3. Neither alias is observable from inside this function.
///
/// Domain: `hops` must land in `0 ..= 11` (the analysis stage's 116-word input
/// buffer and the grid's eleven columns), so `sub[0x6dc] + advance <= 95`.
/// `pcm` is the 383-word analysis ring, indexed from its base with
/// `origin = 0x17f`. `sub` needs `0x370` words, of which `0 .. 0x310` is the
/// 16 × 49 band history.
#[allow(clippy::too_many_arguments)]
pub fn subband_analysis_step(
    total: &mut [i16],
    spec: &mut [i16],
    band_exp: &mut [i16],
    acc: &mut [i32],
    sub: &mut [i16],
    active: i16,
    pcm: &[i16],
    advance: i16,
    trk: &mut [i16],
    taper: &[i16],
    interp_coef: &[i16],
    win: &[i16],
) {
    // The original's 352-byte stack local: the 16 x 11 grid of fresh samples.
    let mut fresh = [0i16; 16 * 11];
    let mut rows: i16 = 0;

    let hops = subband_analysis_frame(
        sub,
        &mut fresh,
        acc,
        &mut rows,
        pcm,
        0x17f,
        advance,
        taper,
        interp_coef,
    );

    // `2·exp + 7`, formed in 16 bits and zero-extended.
    let energy_exp = (sub[0x62c / 2].wrapping_mul(2).wrapping_add(7)) as u16 as i32;
    frame_tracker_step(trk, acc, energy_exp, rows);

    if active > 0 {
        // `hist` is the 16 x 49 band history at the base of the sub-block, and
        // the seven-entry exponent and edge vectors sit immediately above it.
        // Nothing in the call writes `sub`, so the three views are plain
        // borrows into it rather than copies.
        subband_spectrum_frame(
            total,
            spec,
            band_exp,
            &sub[..0x310],
            &fresh,
            hops as i16,
            &sub[0x310..0x317],
            &sub[0x317..0x31e],
            trk,
            win,
        );
    }

    shift_band_history(sub, &fresh, hops as i16);
}

// ---------------------------------------------------------------------------
// 0x1030f710 — spectral envelope, band voicing, and the filter-bank taps
// ---------------------------------------------------------------------------

/// Word offsets inside the spectral block the caller passes as `S + 0x18`.
mod sp {
    /// Byte `+0x60c` — this frame's spectrum block exponent (`S+0x624`).
    pub const EXP: usize = 0x60c / 2;
    /// Byte `+0x622` — the mid-frame interpolation triple pair (`S+0x63a`).
    pub const MID: usize = 0x622 / 2;
    /// Byte `+0x636` — the eight-band voicing working block (`S+0x64e`).
    pub const BANDS: usize = 0x636 / 2;
    /// Words of that block the voicing stage touches.
    pub const BANDS_N: usize = 0x53;
    /// Byte `+0x686` — band levels, i.e. the working block at `+0x50`.
    pub const LVL: usize = 0x686 / 2;
    /// Byte `+0x6d6` — their block exponent, i.e. the working block at `+0xa0`.
    pub const LVL_EXP: usize = 0x6d6 / 2;
    /// Byte `+0x6e0` — the spectral-envelope tracker (`S+0x6f8`).
    pub const TRK: usize = 0x6e0 / 2;
    /// Words of that tracker the envelope stage writes: three one-word rows,
    /// three eight-word rows and three exponents. It stops exactly where the
    /// filter-bank history begins.
    pub const TRK_N: usize = 0x27;
    /// Words of it the band-voicing stage addresses. Its noise floor is the
    /// eight-word vector at byte `+0x48`, which runs five words *into* the
    /// filter-bank history — so the floor the voicing decision uses is the
    /// previous call's sub-band amplitudes.
    pub const TRK_V: usize = 0x2c;
    /// Byte `+0x72e` — newest filter-bank output row (`S+0x746`).
    pub const FB: usize = 0x72e / 2;
    /// Byte `+0x90e` — newest filter-bank exponent pair (`S+0x926`).
    pub const FB_EXP: usize = 0x90e / 2;
}

/// Words of the `S + 0x18` spectral block [`analyse_frame_spectrum`] addresses:
/// up to and including the filter-bank exponent pair at byte `+0x916`.
pub const SPECTRAL_BLOCK_WORDS: usize = 0x48d;

/// Words of the `S + 0xb50` sub-band sub-state [`subband_analysis_step`]
/// addresses; the first `0x310` are the 16 x 49 band history.
pub const SUBBAND_STATE_WORDS: usize = 0x370;

/// `0x1030f710` — the encoder's spectral analysis stage: envelope matching,
/// band voicing, and two 40-band filter-bank taps over the new audio.
///
/// **Envelope first.** [`spectral_envelope_frame`] (`0x10314070`) constructs the
/// minimum-phase reference spectrum from the frame's 129 power bins, compares
/// the measured spectrum against it, and writes the linear phase plus the eight
/// per-band match scores into the tracker at `+0x6e0`. Both of its exponent
/// arguments come from the single spectrum exponent word at `+0x60c`: the power
/// spectrum uses it as it stands, the complex spectrum uses it **halved** — an
/// arithmetic 16-bit `>> 1`, then zero-extended, so a negative exponent enters
/// the callee as a large positive one. The two eight-word vectors it reads,
/// `amp` at `+0x636` and `lvl` at `+0x686`, are the first and the sixth vectors
/// of the band voicing working block, and `+0x6d6` is that block's exponent.
///
/// **Then band voicing**, but only when `active > 0` — i.e. once per 20 ms
/// frame, not once per half. [`frame_band_voicing`] (`0x1030f470`) resolves the
/// mid-frame parameter triple, divides the upper 128 spectral bins by the lower
/// 128 to get the high-band energy ratio, and turns that plus the four
/// eight-band vectors into the per-band voicing metric and weights, which land
/// back in the same tracker. `mode` is the word at `cfg + 0x180`.
///
/// **Then the filter-bank history ages and two new taps are taken.** The three
/// 80-word output rows at `+0x72e`, `+0x7ce`, `+0x86e` shift up one place, as do
/// their exponent pairs at `+0x90e`, `+0x912`, `+0x916`; then two consecutive
/// 80-sample windows of the analysis ring — starting at ring word
/// `2·(0x72 − advance)` and 40 words later — are copied into the scratch buffer
/// and run through [`subband_filterbank`] (`0x1031cb10`) with `n = 40` at
/// exponent 15, writing 40 sub-band amplitudes each into the newest row and
/// their block exponents into `+0x90e` and `+0x910`.
///
/// So the two taps overlap by half a window and land 40 samples apart: a 10 ms
/// hop at 8 kHz, one per half-frame.
///
/// The original passes a sixth argument to `0x1030f470` that the callee never
/// reads; it is not modelled.
///
/// Domain: `advance` is the number of new samples, `<= 0x72` so the ring index
/// stays non-negative, and the ring must hold `2·(0x72 − advance) + 0x78` words.
/// `work` is the caller's 258-word scratch, of which the first 80 are used.
#[allow(clippy::too_many_arguments)]
pub fn analyse_frame_spectrum(
    sp: &mut [i16],
    active: i16,
    ring: &[i16],
    work: &mut [i16],
    advance: i16,
    cfg: &[i16],
    cos_tbl: &[i16],
    fb_win: &[i16],
    fb_tbl: &[i16],
) {
    // Both exponents are formed in 16 bits and zero-extended into the call.
    let power_exp = sp[sp::EXP] as u16 as i32;
    let meas_exp = (sp[sp::EXP] >> 1) as u16 as i32;

    {
        let mut amp = [0i16; 8];
        let mut lvl = [0i16; 8];
        amp.copy_from_slice(&sp[sp::BANDS..sp::BANDS + 8]);
        lvl.copy_from_slice(&sp[sp::LVL..sp::LVL + 8]);
        let lvl_exp = sp[sp::LVL_EXP];
        let mut trk = [0i16; sp::TRK_N];
        trk.copy_from_slice(&sp[sp::TRK..sp::TRK + sp::TRK_N]);
        spectral_envelope_frame(
            &mut trk, work, meas_exp, sp, power_exp, &amp, &lvl, lvl_exp, cos_tbl,
        );
        sp[sp::TRK..sp::TRK + sp::TRK_N].copy_from_slice(&trk);
    }

    if active > 0 {
        let mut bands = [0i16; sp::BANDS_N];
        let mut mid = [0i16; 8];
        let mut out = [0i16; sp::TRK_V];
        bands.copy_from_slice(&sp[sp::BANDS..sp::BANDS + sp::BANDS_N]);
        mid.copy_from_slice(&sp[sp::MID..sp::MID + 8]);
        out.copy_from_slice(&sp[sp::TRK..sp::TRK + sp::TRK_V]);
        frame_band_voicing(&mut bands, &mut mid, sp, &mut out, cfg[0x180 / 2]);
        sp[sp::BANDS..sp::BANDS + sp::BANDS_N].copy_from_slice(&bands);
        sp[sp::MID..sp::MID + 8].copy_from_slice(&mid);
        sp[sp::TRK..sp::TRK + sp::TRK_V].copy_from_slice(&out);
    }

    // Age the three filter-bank rows and their exponent pairs, oldest first.
    let mut row = sp::FB + 2 * 0x50;
    let mut ew = sp::FB_EXP + 4;
    for _ in 0..2 {
        sp.copy_within(row - 0x50..row, row);
        sp[ew] = sp[ew - 2];
        sp[ew + 1] = sp[ew - 1];
        row -= 0x50;
        ew -= 2;
    }

    // Two 80-sample taps, 40 samples apart.
    let mut at = (0x72i16.wrapping_sub(advance)).wrapping_mul(2);
    let mut out_w = sp::FB;
    let mut exp_w = sp::FB_EXP;
    for _ in 0..2 {
        copy16_batch3(work, &ring[at as i32 as usize..], 0x50);
        let e = subband_filterbank(&mut sp[out_w..], work, 0xf, fb_win, 0x28, fb_tbl);
        sp[exp_w] = e as i16;
        at = at.wrapping_add(0x28);
        out_w += 0x28;
        exp_w += 1;
    }
}

// ======================= from bL7b1 =======================

// ---------------------------------------------------------------------------
// 0x1030b660 — the frame packer
// ---------------------------------------------------------------------------

/// The analysis context `0x1030b660` is handed (`0x1030fe40`'s output), split
/// into the views its callees want.
///
/// The original is one 0x4c-byte pointer block; the two halves overlap in the
/// word at `+0x30`, which `0x1030c3f0` reads as a slot-activity mask and
/// `0x1030c480` forwards to the sink as its first argument. They are the same
/// word and must be set consistently.
pub struct FrameCtx<'a> {
    /// `+0x00 … +0x30` — what the per-slot analysis reads.
    pub slot: SlotCtx<'a>,
    /// `+0x14 … +0x48` — what the amplitude stage and the sink read.
    pub amp: SlotContext,
    /// `+0x24` — the spectral-envelope tracker at `S+0x6f8`, read as
    /// consecutive `i16`: entry `cfg[1]` goes to half-frame 0's word 7 and
    /// entry `cfg[1] + 1` to half-frame 1's.
    pub env: &'a [i16],
    /// `+0x14` — the per-slot magnitude-squared spectrum matrix, 129 `i32` per
    /// slot, borrowed as the raw state words it is stored in.
    /// (`amp.spec_off` is the same pointer as the original sees it.)
    pub spec: &'a [i16],
    /// `+0x18` — the matching per-slot block exponents.
    pub exps: &'a [i16],
}

/// The constant tables `0x1030b660`'s closure reaches as globals. They are
/// arguments so that a differential test can prove the port and the reference share
/// them rather than assuming it.
pub struct PackTables<'a> {
    /// `0x1037f978` — the 512-entry cosine table and the blob around it, which
    /// several callees index far past 512 words.
    pub blob: &'a [i16],
    /// `0x10381920` — the two-group partition table.
    pub partitions: &'a [i16],
    /// `0x10381720` — the voicing codebook, packed dibits, as `i32`.
    pub voicing_book: &'a [u32],
    /// `0x10381428` — packed per-harmonic bit allocation nibbles.
    pub nibbles: &'a [i16],
    /// `0x10381370` — the 64-entry Q11 block-gain codebook.
    pub gain64: &'a [i16],
    /// `0x10382160` — the 32-entry Q11 frame-gain codebook.
    pub gain32: &'a [i16],
    /// `0x103813fc` / `0x1038140e` — higher-order-coefficient weight and
    /// loading vectors.
    pub hoc_weight: &'a [i16],
    pub hoc_loading: &'a [i16],
    /// `0x103800d0` — per-block lengths for the amplitude codec.
    pub block_lens: &'a [i16],
    /// `0x10380130` / `0x10380d30` — the two PRBA vector-quantiser codebooks.
    pub prba9: &'a [i16],
    pub prba7: &'a [i16],
    /// `0x10381130` — the higher-order-coefficient codebooks.
    pub coef_books: &'a [i16],
}

/// `0x1030b660` — turn one analysed 20 ms frame into a channel frame.
///
/// This is the encoder's packing hub, and the whole encode-side parameter path
/// is visible in it. It runs once per 20 ms frame, from `0x1030ff60`, on the
/// **second** half-frame call only, after the front end has already filled the
/// state block for both halves.
///
/// It owns two parameter blocks, one per 10 ms half-frame: `blk0` is the block
/// that will be transmitted and `blk1` the second half's. Both are 0x44 words.
///
/// 1. **Envelope hand-off.** `blk1[7]` and `blk0[7]` are loaded from the
///    spectral-envelope tracker (`ctx.env`) at entries `cfg[1] + 1` and
///    `cfg[1]` — two *consecutive* words, one per half-frame. Word 7 is the
///    harmonic frequency, and it is the input to the joint pitch/voicing code
///    (`0x103132b0`), so this is where the envelope tracker reaches the pitch
///    field.
///
/// 2. **Slot analysis — only for `class == 1`.** [`slot_analysis`]
///    (`0x1030c3f0`) runs twice, half 1 into `blk1` and then half 0 into
///    `blk0`, over slot base `cfg[1]`. Each call produces three eight-word
///    vectors — spectral weights, band levels and floored magnitudes — and
///    writes the chosen pitch into word 6 of its block. The original stacks
///    them so that **the second call's** three vectors land at byte offsets 0,
///    0x20 and 0x40 of one 96-byte area, which is exactly the layout the
///    voicing quantiser wants; the first call's land in the gaps and are
///    dead. So the transmitted voicing decision is made from half-frame 0's
///    weights, levels and magnitudes, and half-frame 1's slot analysis exists
///    only to set `blk1[6]`.
///
///    For any other class the 96-byte area is **never written**: the quantiser
///    below is handed uninitialised stack. On the AMBE side that is harmless —
///    `0x1030b920` only searches the voicing codebook when the class is 1. On
///    the IMBE side it is not: `0x1030bda0`'s codebook path *does* read the
///    three vectors, and it is taken by every class but 1 except a class-3
///    frame whose pitch configuration was accepted. Those combinations read
///    stack residue and are not reproducible by any implementation; this port
///    passes a zeroed area, and the differential tests exclude them rather than
///    pretend to match.
///
/// 3. **Parameter quantisation**, selected by the allocation vector's mode
///    word:
///    * `cfg[0] == 0` (IMBE) → [`encode_frame_params`] (`0x1030bda0`), which
///      returns the voicing field and leaves the cursor past it;
///    * otherwise (AMBE) → [`pack_pitch_and_voicing`] (`0x1030b920`), which
///      emits both fields itself and yields no code.
///
///    Both rebuild `cfg` in place, clear the payload and reset the cursor, so
///    the frame layout is established here and nowhere earlier.
///
/// 4. **Amplitudes**, skipped entirely for `class == 3` (the signalling-tone
///    frame, whose payload is already complete). First the *second* half-frame
///    is dispatched to the registered sink — slot 1, absolute slot
///    `cfg[1] + 1`, gated on `blk1[0] != 3` — and then
///    [`slot_amplitudes_and_dispatch`] (`0x1030c480`) computes half-frame 0's
///    harmonic amplitude vector and dispatches that as slot 0. So the sink sees
///    the two half-frames in the order 1, 0, and only half-frame 0's
///    amplitudes are actually derived from the spectrum: `blk1` is dispatched
///    as analysed, never amplitude-coded.
///
/// 5. **Spectral envelope and packing**, again by mode:
///    * IMBE → [`encode_frame_imbe_spectrum`] (`0x1030c360`) codes the log
///      magnitudes, interleaves the 88-bit frame and promotes `blk0` into
///      `pblk`. It returns immediately afterwards — the IMBE frame is done.
///    * AMBE → for any class but 3, [`code_gain_and_amplitudes`]
///      (`0x10311470`) writes the gain and harmonic fields and
///      [`frame_bit_permute`] (`0x10312cb0`) applies the two-pass AMBE bit
///      permutation of SPEC §3.1. Then, for **every** class,
///      [`copy_params_batch2`] (`0x10311a20`) promotes `blk0` into `pblk` for the next
///      frame's prediction.
///
/// The inline dispatch's `blk1[0] != 3` guard is **dead**: it sits inside the
/// `class != 3` arm and the quantiser has already written `class` into
/// `blk1[0]`, so the two tests are the same one. It is reproduced because it is
/// there, and a mutation to `blk0[0]` is (correctly) undetectable.
///
/// Returns the two sink dispatches in the order the original makes them —
/// half-frame 1 first, then half-frame 0 — as [`SinkCall`] argument vectors
/// rather than performing the indirect call. Argument 2 of each is the address
/// of the parameter block, which the port cannot express, and is left zero.
///
/// The original's fourth and sixth arguments (both `0x44`, the parameter-block
/// size) are pushed and never read; they are not modelled. Its cursor lives in
/// its own argument slot for `cfg`, which the caller cannot observe (SPEC
/// §4.2), so it is internal here.
///
/// `unit` is the channel-unit descriptor the caller took from the signalling
/// tracker: the IMBE hub treats it as a unit *length* (`== 0x20` is the
/// signalling-tone length, `> 0x7a` disables the all-ones voicing mask) and the
/// AMBE hub as the *tone index* written into `blk0[1]`. It is one value with
/// two readings, not two arguments.
#[allow(clippy::too_many_arguments)]
pub fn pack_frame(
    bits: &mut [i16],
    rate: i16,
    blk0: &mut [i16],
    blk1: &mut [i16],
    pblk: &mut [i16],
    unit: i16,
    peaks: &[i16],
    tone_flag: i16,
    template: &[i16],
    class: i16,
    cfg: &mut [i16],
    ctx: &FrameCtx<'_>,
    allow_high: i16,
    tbl: &PackTables<'_>,
) -> [Option<SinkCall>; 2] {
    let base = cfg[1];
    let bi = base as isize;
    blk1[7] = ctx.env[(bi + 1) as usize];
    blk0[7] = ctx.env[bi as usize];

    // The original's six 16-byte stack slots, in ascending address order:
    // local_60, local_50, local_40, local_30, local_20, local_10.
    let mut locals = [0i16; 48];
    if class == 1 {
        let (l60, rest) = locals.split_at_mut(8);
        let (l50, rest) = rest.split_at_mut(8);
        let (l40, rest) = rest.split_at_mut(8);
        let (l30, rest) = rest.split_at_mut(8);
        let (l20, l10) = rest.split_at_mut(8);
        slot_analysis(blk1, l50, l10, l30, 1, base as i32, &ctx.slot, tbl.blob);
        slot_analysis(blk0, l60, l20, l40, 0, base as i32, &ctx.slot, tbl.blob);
    }

    let mut cursor = 0usize;
    let code: u16;
    if cfg[0] == 0 {
        code = encode_frame_params(
            bits,
            &mut cursor,
            blk0,
            blk1,
            pblk,
            unit,
            peaks,
            tone_flag,
            class,
            cfg,
            &locals,
            tbl.blob,
            tbl.partitions,
        );
    } else {
        let mut c13 = [0i16; 13];
        c13.copy_from_slice(&cfg[..13]);
        pack_pitch_and_voicing(
            bits,
            &mut cursor,
            rate,
            blk0,
            blk1,
            0,
            unit,
            peaks,
            class,
            &mut c13,
            &locals,
            tbl.voicing_book,
            allow_high,
        );
        cfg[..13].copy_from_slice(&c13);
        code = 0;
    }

    let mut sinks = [None, None];
    if class != 3 {
        if ctx.amp.sink_enable != 0 && blk1[0] != 3 {
            sinks[0] = Some(SinkCall {
                args: [
                    ctx.amp.sink_arg0 as u32,
                    ctx.amp.sink_d,
                    0, // the address of `blk1`
                    ctx.amp.sink_e,
                    ctx.amp.sink_c,
                    ctx.amp.sink_a,
                    ctx.amp.sink_b,
                    1,
                    (base as u16).wrapping_add(1) as u32,
                    ctx.amp.sink_arg9 as u32,
                ],
            });
        }
        sinks[1] = slot_amplitudes_and_dispatch(
            blk0, pblk, 0, base, &ctx.amp, template, ctx.spec, ctx.exps,
        );
    }

    if cfg[0] == 0 {
        let mut pos = cursor as i32;
        encode_frame_imbe_spectrum(
            bits,
            &mut pos,
            blk0,
            pblk,
            code as i16,
            cfg,
            tbl.nibbles,
            tbl.gain64,
            tbl.hoc_weight,
            tbl.hoc_loading,
            tbl.blob,
            tbl.blob,
        );
        return sinks;
    }

    if class != 3 {
        let mut pos = cursor as i32;
        code_gain_and_amplitudes(
            pblk,
            blk0,
            cfg,
            bits,
            &mut pos,
            1,
            tbl.gain32,
            tbl.block_lens,
            tbl.prba9,
            tbl.prba7,
            tbl.coef_books,
            tbl.blob,
        );
        frame_bit_permute(bits, cfg, 0);
    }
    copy_params_batch2(pblk, blk0, 0);
    sinks
}

// ---------------------------------------------------------------------------
// 0x1030e020 — the frame analyser
// ---------------------------------------------------------------------------

/// `0x1030e020` — advance the sub-band front end, and once per frame run the
/// spectral analysis and the pitch / voicing decision.
///
/// Called from the front end (`0x1030a190`) for **every** half-frame, after the
/// DC blocker has pushed the new audio into the 383-word analysis ring and
/// after the primary 199-tap Hamming analysis has produced this frame's
/// magnitude-squared spectrum. It splits cleanly in two:
///
/// 1. [`subband_analysis_step`] (`0x1030e5a0`) always runs. It converts
///    `advance` new ring samples into eight-sample analysis hops, ages the
///    band hold counters and peak trackers, folds the sixteen sub-band
///    energies into the long-term level/gain trackers in `trk`, and — when
///    `half > 0` — builds the eight-band × two-half-frame spectrum and its
///    per-band exponents.
///
/// 2. Everything else runs **only when `half > 0`**, i.e. once per 20 ms
///    frame, on the second half-frame call. That is the whole reason the
///    encoder has a one-frame delay (SPEC §2.1): the pitch and voicing decision
///    cannot be made until both halves of the frame have been analysed.
///
///    * [`analysis_fft`] (`0x1030d890`) takes the **secondary** analysis: 255
///      ring samples through the 128-tap wide ramp window at `0x1037f780`, a
///      256-point real FFT at exponent −7, and the magnitude-squared spectrum
///      of its 128 bins. This is a *different* window from the front end's
///      199-tap Hamming — the encoder runs two analyses of the same audio, a
///      narrow one for the amplitude envelope and this wide one for pitch.
///    * [`pitch_voicing_frame`] (`0x1030eaa0`) then refines the fundamental
///      against that power spectrum with the sub-band `total` vector as its
///      correlation input, takes the band voicing decision from the eight-band
///      spectrum, applies the octave clamp, and conditions the per-band voicing
///      gains on the noise picture in `trk`.
///
/// **Three stack buffers are aliased in the original**, and all three aliases
/// turn out to be unobservable — which is worth stating explicitly, because two
/// of them look load-bearing.
///
/// * The 516-byte spectral buffer is passed to `0x1030d890` as *both* the
///   transform buffer and the magnitude output. The magnitudes overwrite the
///   128 complex bins in place, which is exact because each `i32` result lands
///   on precisely the pair it was computed from. The transform buffer is
///   `N + 2 = 258` words and the magnitudes cover only 256, so the 129th `i32`
///   is the Nyquist bin's two words reinterpreted rather than a magnitude.
///   This port reconstructs that word anyway; a mutation test confirms the
///   pitch stage reads only the 128 real magnitudes and never the 129th, while
///   perturbing the 128th does break the comparison.
/// * The 64-byte slot serves as both the sub-band energy accumulator and the
///   eight-band total. The accumulator is consumed before the total is written,
///   and the total is written in full (32 words), so separating them is exact.
/// * The 512-byte band spectrum abuts that slot, and band 7's `i32` scratch
///   runs 64 bytes past its end into it; the total overwrites those bytes
///   afterwards, so nothing survives the call that the caller reads.
///
/// The original's seventh and ninth arguments (the caller's `flags & 0x80` and
/// `S+0x16b8`), and the word it fetches from `trk[0xc0]` to pass on, are pushed
/// and read by nothing.
#[allow(clippy::too_many_arguments)]
pub fn analyse_frame(
    sub: &mut [i16],
    st: &mut [i16],
    half: i16,
    ring: &[i16],
    advance: i16,
    trk: &mut [i16],
    flag: &mut i16,
    taper: &[i16],
    interp: &[i16],
    subband_win: &[i16],
    wide_win: &[i16],
) {
    // The original's stack frame, in ascending address order: local_454 (the
    // eight band exponents), local_444 (the band spectrum), local_244 (the
    // sub-band accumulator, then the eight-band total), local_204 (the
    // transform buffer, then the power spectrum).
    let mut band_exp = [0i16; 8];
    let mut spec = [0i16; 0x120];
    let mut total = [0i16; 32];
    let mut acc = [0i32; 16];

    subband_analysis_step(
        &mut total,
        &mut spec,
        &mut band_exp,
        &mut acc,
        sub,
        half,
        ring,
        advance,
        trk,
        taper,
        interp,
        subband_win,
    );

    if half >= 1 {
        let mut fft_buf = [0i16; 258];
        let mut energy = [0i32; 129];
        let exp = analysis_fft(
            Some(&mut energy),
            ring,
            wide_win,
            -7,
            0xff,
            0,
            &mut fft_buf,
            8,
            0,
        );
        // The 129th entry is the transform buffer's Nyquist pair, which the
        // in-place magnitude pass never reaches.
        energy[128] = ((fft_buf[257] as u16 as u32) << 16 | fft_buf[256] as u16 as u32) as i32;

        pitch_voicing_frame(
            st,
            &total,
            &spec[..0x100],
            &band_exp,
            trk,
            &energy,
            exp,
            flag,
        );
    }
}

// ======================= from bL8b1 =======================

// ---------------------------------------------------------------------------
// Encoder state map — byte offsets, as the two functions hand them out.
// ---------------------------------------------------------------------------

/// `S+0x018` — the magnitude-squared spectrum block, through to the newest
/// filter-bank exponent pair at `S+0x92e`. This is what `0x1030e020` and
/// `0x1030f710` are both given.
const SPEC: usize = 0x18 / 2;

const SPEC_W: usize = 0x490;

/// `S+0x21c`, `S+0x420` — the two older spectrum rows.
const SPEC_ROW1: usize = 0x21c / 2;

const SPEC_ROW2: usize = 0x420 / 2;

/// `0x102` words — 129 `i32` bins — per row.
const ROW_W: usize = 0x102;

/// `S+0x624` — the newest row's block exponent; `+0x626`, `+0x628` the others.
const EXP0: usize = 0x624 / 2;

/// `S+0x938` — the level and noise trackers.
const TRK: usize = 0x938 / 2;

const TRK_W: usize = 0x10c;

/// `S+0xab8` — the tone classifier's enable word (tracker byte `+0x180`).
const TONE_ENABLE: usize = 0xab8 / 2;

/// `S+0xabc` — the previous frame's parameter block.
const PBLK: usize = 0xabc / 2;

const BLK_W: usize = 0x44;

/// `S+0xb50` — the sub-band analysis sub-state.
const SUB: usize = 0xb50 / 2;

const SUB_W: usize = 0x370;

/// `S+0x1370` — the signalling / tone tracker.
const TONE_TRK: usize = 0x1370 / 2;

const TONE_TRK_W: usize = 0x20;

/// `S+0x138a` — the bit-allocation vector.
const CFG: usize = 0x138a / 2;

const CFG_W: usize = 0x10;

/// `S+0x13a4` — the 383-word analysis history ring.
const RING: usize = 0x13a4 / 2;

const RING_W: usize = 0x17f;

/// `S+0x13dc` — where the 199-tap Hamming analysis starts inside the ring,
/// i.e. 28 words in.
const RING_ANALYSIS: usize = 0x1c;

/// `S+0x16ba` — the 17-symbol signalling sequence tracker.
const SEQ: usize = 0x16ba / 2;

const SEQ_W: usize = 0x12;

/// The number of words the encoder state block occupies.
pub const ENC_STATE_WORDS: usize = 0xb80;

#[inline]
fn rd32(s: &[i16], w: usize) -> i32 {
    ((s[w] as u16 as u32) | ((s[w + 1] as u16 as u32) << 16)) as i32
}

#[inline]
fn wr32_bL8b1(s: &mut [i16], w: usize, v: i32) {
    s[w] = v as i16;
    s[w + 1] = (v >> 16) as i16;
}

// ---------------------------------------------------------------------------
// 0x1030a190 — the analysis front end
// ---------------------------------------------------------------------------

/// Every constant table `0x1030a190`'s closure reaches as a global. They are
/// arguments so a differential test can prove the port and the reference share them
/// rather than assuming it.
pub struct FrontEndTables<'a> {
    /// `0x1037f880` — 100 Q15 taps, half of the symmetric 199-tap Hamming
    /// **primary** analysis window (SPEC §5).
    pub hamming: &'a [i16],
    /// `0x1037f780` — 128 Q15 taps, half of the 255-tap wide ramp used by the
    /// **secondary** (pitch) analysis.
    pub wide_win: &'a [i16],
    /// `0x1037fed4` — half of the 32-wide sub-band taper.
    pub taper: &'a [i16],
    /// `0x1037fef4` — the 7 × 16 interpolation filter; four taps are read.
    pub interp: &'a [i16],
    /// `0x1037ff04` — the 64-tap sub-band analysis window.
    pub subband_win: &'a [i16],
    /// `0x1037f978` — the 512-entry Q15 cosine table.
    pub cos: &'a [i16],
    /// `0x10380080` — 80 Q15 taps for the 40-band filter bank.
    pub fb_win: &'a [i16],
    /// `0x1037ff40` — the 160-entry DCT-IV kernel of that filter bank.
    pub fb_tbl: &'a [i16],
    /// `0x10381ba8` — the signalling-tone class and frequency tables.
    pub tone: ToneTables<'a>,
}

/// `0x1030a190` — take 10 ms of new audio and run the whole analysis front end
/// over it.
///
/// This is the encoder's per-half-frame sequencer, and it establishes the order
/// the whole analysis happens in. Five stages:
///
/// 1. **The analysis ring advances.** The 383-word history at `S+0x13a4`
///    shifts down by `nsamp` words and the new audio lands at the top. Unless
///    bit `0x100` of `flags` is set the audio is filtered on the way in by the
///    third-order DC blocker of SPEC §7.2 ([`dc_blocker`], `0x1030fa40`) whose
///    six-word `i32` memory is the very base of the state block; with that bit
///    set the samples are copied straight through and the filter memory is not
///    touched. So the ring holds *filtered* audio and everything downstream
///    sees a signal with a triple zero at DC.
///
/// 2. **The spectrum shift register ages.** The three `0x102`-word rows at
///    `S+0x18`, `S+0x21c` and `S+0x420` move down one place, as do their three
///    block exponents at `S+0x624`…`S+0x628`. The encoder therefore keeps the
///    three most recent power spectra, which is what lets the packer address
///    them by slot number.
///
/// 3. **The primary analysis.** [`analysis_fft`] (`0x1030d890`) windows 199
///    ring samples — starting 28 words in, so the window's centre sits a fixed
///    distance behind the newest sample — with the symmetric Hamming window at
///    `0x1037f880`, takes a 256-point real FFT at exponent −7, and writes the
///    129 magnitude-squared bins into the newest spectrum row. Its returned
///    block exponent becomes `S+0x624`. This is the *narrow* analysis, and it
///    is the one the amplitude envelope is measured from.
///
/// 4. **The sub-band front end and the pitch / voicing decision.**
///    [`analyse_frame`] (`0x1030e020`) always advances the sixteen-band filter
///    bank and the noise trackers, and on the odd half-frame additionally runs
///    the *wide* 255-tap secondary analysis and the pitch refinement. Its
///    out-parameter is dead: it is forwarded to stage 5, which never reads it.
///
/// 5. **Spectral envelope, band voicing, filter-bank taps**
///    ([`analyse_frame_spectrum`], `0x1030f710`) and then **signalling-tone
///    classification** ([`tone_class_half_frame`], `0x1030f920`) over the
///    freshly written spectrum, at the exponent stage 3 returned, gated by the
///    enable word at `S+0xab8`.
///
/// The 516-byte scratch is shared between stages 3 and 5 — it is the FFT
/// buffer and then the filter bank's window scratch — which is exact because
/// nothing reads it across the boundary.
///
/// Domain: `0 < nsamp <= 0x72`, the half-frame length `enc_frame` clamps to
/// `[0x4c, 0x54]`; `pcm` at least `nsamp` words; `st` the whole
/// [`ENC_STATE_WORDS`]-word encoder state, brought up by `enc_init`.
pub fn front_end(
    st: &mut [i16],
    pcm: &[i16],
    nsamp: i16,
    flags: u32,
    half: i16,
    t: &FrontEndTables<'_>,
) {
    // 1a. age the ring. `0x17f - nsamp` is formed in 16 bits and the copy
    // count is signed, so a `nsamp` past the ring length copies nothing.
    let keep = (0x17fu16.wrapping_sub(nsamp as u16)) as i16;
    if keep > 0 {
        let src = RING + (nsamp as usize);
        st.copy_within(src..src + keep as usize, RING);
    }

    // 1b. the new audio, filtered unless bit 0x100 says otherwise.
    let at = RING + keep as usize;
    if flags & 0x100 != 0 {
        if nsamp > 0 {
            st[at..at + nsamp as usize].copy_from_slice(&pcm[..nsamp as usize]);
        }
    } else {
        let mut mem = [0i32; 6];
        for (k, m) in mem.iter_mut().enumerate() {
            *m = rd32(st, 2 * k);
        }
        let n = nsamp.max(0) as usize;
        // The original filters straight into the ring; `pcm` is the caller's
        // buffer and cannot alias the state, so no scratch is needed.
        dc_blocker(&mut mem, &mut st[at..at + n], pcm, nsamp as u16);
        for (k, m) in mem.iter().enumerate() {
            wr32_bL8b1(st, 2 * k, *m);
        }
    }

    // 2. age the spectrum shift register, oldest row first.
    st.copy_within(SPEC_ROW1..SPEC_ROW1 + ROW_W, SPEC_ROW2);
    st[EXP0 + 2] = st[EXP0 + 1];
    st.copy_within(SPEC..SPEC + ROW_W, SPEC_ROW1);
    st[EXP0 + 1] = st[EXP0];

    // 3. the primary 199-tap Hamming analysis into the newest row.
    let mut work = [0i16; 258];
    {
        let mut mag = [0i32; 0x81];
        let e = analysis_fft(
            Some(&mut mag),
            &st[RING + RING_ANALYSIS..RING + RING_ANALYSIS + 199],
            t.hamming,
            -7,
            199,
            0,
            &mut work,
            8,
            1,
        );
        for (i, v) in mag.iter().enumerate() {
            wr32_bL8b1(st, SPEC + 2 * i, *v);
        }
        st[EXP0] = e;
    }

    // Stages 4 and 5a work directly on the state block, exactly as the original
    // does: the spectrum, level-tracker and sub-band sub-states are three
    // *adjacent* windows (`0x18 + 0x490 == 0x938`, `0x938 + 0x10c == 0xb50`),
    // so one `split_at_mut` chain hands out all three with their original
    // lengths, and the analysis ring sits above all of them.
    const _: () = assert!(SPEC + SPEC_W == TRK && TRK + TRK_W == SUB && SUB + SUB_W <= RING);
    {
        let (lo, hi) = st.split_at_mut(RING);
        let ring = &hi[..RING_W];
        let (_, rest) = lo.split_at_mut(SPEC);
        let (sp, rest) = rest.split_at_mut(SPEC_W);
        let (trk, rest) = rest.split_at_mut(TRK_W);
        let sub = &mut rest[..SUB_W];

        // 4. sub-band front end, and once per frame the pitch / voicing
        //    decision.
        // The original's `local_208`, initialised to zero and never read again.
        let mut dead = 0i16;
        analyse_frame(
            sub,
            sp,
            half,
            ring,
            nsamp,
            trk,
            &mut dead,
            t.taper,
            t.interp,
            t.subband_win,
            t.wide_win,
        );

        // 5a. spectral envelope, band voicing and the two filter-bank taps.
        analyse_frame_spectrum(
            sp, half, ring, &mut work, nsamp, trk, t.cos, t.fb_win, t.fb_tbl,
        );
    }

    // 5b. signalling-tone classification over the new spectrum.
    {
        let mut energy = [0i32; 0x81];
        for (i, d) in energy.iter_mut().enumerate() {
            *d = rd32(st, SPEC + 2 * i);
        }
        let exp = st[EXP0];
        let enable = st[TONE_ENABLE];
        let mut tone = [0i16; TONE_TRK_W];
        tone.copy_from_slice(&st[TONE_TRK..TONE_TRK + TONE_TRK_W]);
        tone_class_half_frame(
            &mut tone,
            &energy,
            exp,
            flags,
            half,
            enable,
            &t.tone,
            &TONE_BANDS,
        );
        st[TONE_TRK..TONE_TRK + TONE_TRK_W].copy_from_slice(&tone);
    }
}

// ---------------------------------------------------------------------------
// 0x1030ff60 — the frame tail
// ---------------------------------------------------------------------------

/// The two runtime addresses `0x1030fe40` bakes into the packer's context view.
///
/// They reach no arithmetic: the state base and the sink's own address are
/// forwarded verbatim to the registered sink, so a port needs them only to
/// reproduce that call's argument vector. Pass zeros when the sink is disabled
/// — which is every path the P25 wrapper takes — and the addresses are not
/// wanted.
#[derive(Clone, Copy, Default)]
pub struct Bases {
    /// The runtime address of the encoder state block.
    pub state: u32,
    /// `0x1031f530` — the sink the view carries at `+0x3c`.
    pub sink: u32,
}

/// The frame disposition [`frame_tail`] returns, as named bits.
pub mod disposition {
    /// A signalling-tone frame was taken from the pending-entry queue.
    pub const TONE: u16 = 0x8000;
    /// The activity timer at tracker `+0x40` was re-armed this frame.
    pub const REARMED: u16 = 0x0400;
    /// Speech, or a background-level frame whose floor moved.
    pub const ACTIVE: u16 = 0x0002;
}

/// `0x1030ff60` — classify the finished 20 ms frame and hand the analysis to
/// the packer.
///
/// `enc_frame` calls this on the **odd** half-frame only, after [`front_end`]
/// has filled the state block for both halves, and it is where the encoder
/// decides what kind of frame to transmit. Its `gain` is the AGC follower's
/// per-frame value, `0x1030db40` scaled by `0x1542c / 2^16`, and zero unless
/// bit `0x2000` of `flags` is set. Four steps:
///
/// 1. **A gate and a timer.** Bit `0x800` of `flags` — the caller's request for
///    a background-noise frame — is cleared outright when the allocation
///    vector's mode word at `S+0x138a` is zero, i.e. **IMBE never emits one**;
///    an IMBE frame is always class 1 or class 3.
///
///    Then [`rearm_timer`] (`0x1031d680`) reloads the activity countdown at
///    tracker `+0x40` with 1024000 if either bit of the pair at bit
///    `cfg[1] + 2` is set in the tracker's flag word at `+0x44` — the same word
///    the packer's per-slot activity mask comes from. The decompiler types that
///    helper `void`, but it leaves `0x400` (fired) or `0` in `eax` and the
///    caller uses it: it is the first bit of the returned disposition *and* the
///    gate on step 2's second arm.
///
///    That countdown is the very word [`band_level_floor`] latches its running
///    minimum in, so a re-arm resets the noise floor high — and the two can
///    never collide, because the class-2 arm is reachable only when the timer
///    did *not* fire.
///
/// 2. **The class**, one of three, in the order the original tests them:
///    * **3, signalling tone.** [`take_signalling_entry`] (`0x103100e0`) found
///      a pending entry, either forced by bit `0x4000` or waiting in one of the
///      tracker's two slots. Its two-word data pair is biased by `gain` and
///      becomes the packer's `peaks`; the disposition gains `0x8002`.
///    * **2, background level.** Only when the timer did *not* re-arm and bit
///      `0x800` survived step 1. [`band_level_floor`] (`0x1031d6c0`) writes the
///      16 log-domain band levels, biased by `gain`, into the packer's
///      `template`, and **replaces** the disposition with its own result — `2`
///      if the minimum-statistics noise floor moved down this frame, `0` if
///      not.
///    * **1, speech**, otherwise. `gain` goes into the context's log-magnitude
///      bias instead, so the AGC reaches the harmonic amplitudes rather than a
///      parameter vector; the disposition gains `2`.
///
///    So the AGC gain lands in exactly one of three places depending on what
///    the frame turned out to be, and never in more than one.
///
/// 3. **The signalling sequence tracker** ([`tone_seq_track`], `0x1030d2c0`,
///    always mode 1) sees the entry's tag whatever the class was, and while its
///    countdown runs it *substitutes* a scripted symbol for it. The packer
///    receives the substituted tag as its channel-unit word, not the original.
///
/// 4. **The packer** ([`pack_frame`], `0x1030b660`) turns all of that into the
///    channel frame, with `flags & 0x200` as the tone flag and `flags & 0x80`
///    as the high-band permission. Its 0x4c-byte context is `0x1030fe40`'s
///    block of nineteen interior pointers into the state — the sub-band history
///    and magnitudes, the five pitch candidates and their errors, the previous
///    frame's pitch, the three-slot spectrum matrix and its exponents, the
///    spectral-envelope tracker, the per-slot band energies and exponents, the
///    activity mask at `S+0x97c` and the level trackers — re-expressed here as
///    the views the port's packer wants. Bit `0x40` of `flags` becomes the
///    sink's enable and `p4` its last argument.
///
/// Two things this establishes that are worth writing down. `flags & 0x80`,
/// the high-pitch permission, is **unreachable from here**: it only changes the
/// answer when the reconstructed fundamental clears `0x6bca`, and the AMBE
/// pitch codebook's own reconstruction is bounded below that, so no input to
/// `0x1030ff60` can exercise it. And `S+0xac2` — word 3 of the retained
/// parameter block — is written from one of the two stack blocks that no path
/// ever fills, so its value is stack residue; nothing reads it back.
///
/// Returns the disposition word and the packer's two sink dispatches (see
/// [`pack_frame`]); the original performs the calls, which bit `0x40` of
/// `flags` enables and which are disabled on every path the P25 wrapper takes.
///
/// **The two `0x44`-word parameter blocks are uninitialised stack in the
/// original.** For a speech frame the slot analysis and the quantiser fill
/// every word anything reads; for the other classes a handful of words are
/// simply promoted into the state block unwritten. They are arguments here so
/// the caller decides what the packer sees in them, and so a differential test
/// can plant the same contents on both sides instead of passing by luck.
///
/// Domain: `st` the whole [`ENC_STATE_WORDS`]-word encoder state after a
/// [`front_end`] pair; `bits` at least the payload; `blk0`/`blk1` at least
/// `0x44` words.
#[allow(clippy::too_many_arguments)]
pub fn frame_tail(
    st: &mut [i16],
    bits: &mut [i16],
    rate: i16,
    flags: u32,
    p4: i16,
    gain: i16,
    blk0: &mut [i16],
    blk1: &mut [i16],
    bases: Bases,
    tbl: &PackTables<'_>,
) -> (u16, [Option<SinkCall>; 2]) {
    // 1a. IMBE has no background-noise frame.
    let flags = if st[CFG] == 0 {
        flags & 0xffff_f7ff
    } else {
        flags
    };

    // 1b. the activity timer. Its `eax` is both a bit of the disposition and
    // the gate below, so the condition is formed here as well as inside the
    // call.
    let sel = get_word1(&st[CFG..]);
    let trk_flags = st[TRK + 0x44 / 2];
    let hit =
        3i32.wrapping_shl(((sel as u16 as u32).wrapping_add(2)) & 0x1f) & (trk_flags as i32) != 0;
    {
        let mut timer = rd32(st, TRK + 0x40 / 2);
        rearm_timer(trk_flags, sel as i8, &mut timer);
        wr32_bL8b1(st, TRK + 0x40 / 2, timer);
    }
    let mut code: u16 = if hit { disposition::REARMED } else { 0 };

    // 2. the class.
    let mut unit = 0i16;
    let mut peaks = [0i16; 2];
    let mut template = [0i16; 0x10];
    let class: i16;
    let mut level_bias = 0i16;

    let tone = take_signalling_entry(
        &mut unit,
        &mut peaks,
        flags,
        &mut st[TONE_TRK..TONE_TRK + TONE_TRK_W],
    );

    if tone {
        class = 3;
        code |= 0x8002;
        let src = peaks;
        vec_add_scalar(&mut peaks, &src, gain, 2);
    } else if code == 0 && (flags & 0x800) != 0 {
        class = 2;
        code = band_level_floor(&mut template, &mut st[TRK..TRK + TRK_W]) as u16;
        let src = template;
        vec_add_scalar(&mut template, &src, gain, 0x10);
    } else {
        class = 1;
        code |= disposition::ACTIVE;
        // The original bumps the context view's own log-magnitude bias word,
        // which `0x1030fe40` had just set to zero.
        level_bias = level_bias.wrapping_add(gain);
    }

    // 3. the signalling sequence tracker may rewrite the tag.
    {
        tone_seq_track(&mut unit, &mut st[SEQ..SEQ + SEQ_W], 1);
    }

    // 4. the packer. The context's pointer slots become value views over a
    // snapshot of the state; the two regions the packer writes — the retained
    // parameter block and the allocation vector — are put back afterwards, and
    // neither overlaps anything the context points at.
    //
    // No snapshot is taken: the two written regions are a strict suffix of the
    // state and every view the context hands out lives below them, so a pair of
    // `split_at_mut`s reproduces the original's aliasing exactly and for free.
    // `pblk` (`S+0xabc`) and `cfg` (`S+0x138a`) overlapped in the snapshot form
    // — `cfg` was copied back last and therefore won — and here they are simply
    // adjacent, which is the same thing as long as the packer never reaches
    // past `S+0x138a` through `pblk`.
    let (snap, tail) = st[..ENC_STATE_WORDS].split_at_mut(PBLK);
    let (pblk, cfg) = tail.split_at_mut(CFG - PBLK);
    let ctx = FrameCtx {
        slot: SlotCtx {
            bands: &snap[0x64e / 2..],
            mags: &snap[0x69e / 2..],
            pitch_cand: &snap[0x63a / 2..0x644 / 2],
            pitch_err: &snap[0x644 / 2..0x64e / 2],
            pitch_target: snap[0x630 / 2],
            energies: &snap[0x710 / 2..],
            energy_exp: &snap[0x740 / 2..],
            active_mask: snap[0x97c / 2] as i32,
        },
        amp: SlotContext {
            spec_off: bases.state.wrapping_add(0x18),
            exps_off: bases.state.wrapping_add(0x624),
            sink_a: bases.state.wrapping_add(0x746),
            sink_b: bases.state.wrapping_add(0x926),
            sink_arg0: snap[0x97c / 2] as u16,
            sink_enable: ((flags >> 6) & 1) as i16,
            sink_c: bases.state.wrapping_add(0xa2e),
            sink_d: bases.state.wrapping_add(0x122e),
            sink_e: bases.state.wrapping_add(0x938),
            sink_fn: bases.sink,
            level_bias,
            sink_arg9: p4 as u16,
        },
        env: &snap[0x6f8 / 2..],
        spec: &snap[SPEC..],
        exps: &snap[EXP0..],
    };

    let sinks = pack_frame(
        bits,
        rate,
        blk0,
        blk1,
        pblk,
        unit,
        &peaks,
        (flags & 0x200) as i16,
        &template,
        class,
        cfg,
        &ctx,
        (flags & 0x80) as i16,
        tbl,
    );
    (code, sinks)
}

// ======================= from bL9b1 =======================

/// `S+0x938` — the level / noise tracker block the AGC follower measures.
const LEVEL: usize = 0x938 / 2;

/// `S+0x63a` — the packer's per-slot analysis vectors; the follower reads one
/// word out of them (`ctrl[5]`, i.e. `S+0x644`) as a confidence gate.
const FOLLOW_CTRL: usize = 0x63a / 2;

/// `S+0x16a2` — the AGC follower's own 12-word parameter / state vector.
const AGC: usize = 0x16a2 / 2;

/// The half-frame length window `enc_frame` forces its `nsamp` argument into.
pub const NSAMP_MIN: i16 = 0x4c;

/// See [`NSAMP_MIN`].
pub const NSAMP_MAX: i16 = 0x54;

/// `flags` bit that enables the AGC level follower. Clear in the shipped path.
pub const F_AGC_BIAS: u32 = 0x2000;

/// The Q16 factor the AGC follower's Q15-ish gain is scaled by before it
/// becomes the packer's log-magnitude bias: `0x1542c` = 87084 ≈ 1.3287.
///
/// It is a full 32-bit `imul` of the sign-extended `i16` follower output
/// followed by an arithmetic `>> 16`; the product can exceed `i32` for a
/// follower output near `0x7fff`, and the original wraps rather than
/// saturating, so the multiply is `wrapping_mul`.
pub const GAIN_SCALE: i32 = 0x1542c;

/// `0x1030a0f0` — encode one 10 ms half-frame.
///
/// Runs the analysis front end unconditionally, then — on odd half-frames only
/// — the AGC level follower and the frame tail, which is what emits the
/// payload. Returns the frame disposition (`0` on even half-frames) together
/// with the packer's two deferred sink calls, exactly as
/// [`frame_tail`] reports them.
///
/// `blk0`/`blk1` are the frame tail's two 136-byte parameter blocks. In the
/// original they are *raw stack* inside `0x1030ff60`; for a speech frame every
/// word anything reads is written before it is read, but the background-level
/// and signalling-tone classes promote a handful of never-written words
/// straight into the retained parameter block. They are arguments here for the
/// same reason they are arguments on [`frame_tail`]: so a caller
/// (and a differential test) decides what those words hold rather than
/// inheriting an unreproducible stack residue.
///
/// Domain: `st` at least [`ENC_STATE_WORDS`] words, brought up by
/// [`enc_init`]; `pcm` at least `clamp(nsamp)` words; `bits` at
/// least the payload width of the configured mode; `blk0`/`blk1` at least
/// `0x44` words.
#[allow(clippy::too_many_arguments)]
pub fn enc_frame(
    bits: &mut [i16],
    rate: i16,
    pcm: &[i16],
    nsamp: i16,
    flags: u32,
    half: i16,
    unit: i16,
    st: &mut [i16],
    blk0: &mut [i16],
    blk1: &mut [i16],
    bases: Bases,
    fe: &FrontEndTables<'_>,
    tbl: &PackTables<'_>,
) -> (u16, [Option<SinkCall>; 2]) {
    debug_assert!(st.len() >= ENC_STATE_WORDS);

    // 1. the half-frame length window. `cmp ax,0x54 / jle` then
    //    `cmp ax,cx / cmovl` — both compares are 16-bit *signed*, so a
    //    negative request lands on the floor rather than the ceiling.
    //
    //    The original works in `eax` and only the in-range path preserves the
    //    caller's high half; the callee reads 16 bits, so an `i16` is exact.
    // Spelled out rather than `clamp`: the original tests the ceiling first
    // and the floor second, and only the second is a `cmov`.
    #[allow(clippy::manual_clamp)]
    let nsamp = if nsamp > NSAMP_MAX {
        NSAMP_MAX
    } else if nsamp < NSAMP_MIN {
        NSAMP_MIN
    } else {
        nsamp
    };

    // 2. analysis, on every half-frame.
    front_end(st, pcm, nsamp, flags, half, fe);

    // 3. `test BYTE PTR [half],1` — bit 0 of the low byte, nothing else.
    if half & 1 == 0 {
        return (0, [None, None]);
    }

    // 4. the AGC log-domain bias, gated by `flags & 0x2000`.
    let gain: i16 = if flags & F_AGC_BIAS == 0 {
        0
    } else {
        // Three disjoint windows of the one state block: the follower reads
        // the trackers at S+0x938 and one gate word out of S+0x63a, and
        // updates its own vector at S+0x16a2.
        let (head, tail) = st.split_at_mut(AGC);
        let g = level_follow_gain(&head[LEVEL..], &head[FOLLOW_CTRL..], tail) as i32;
        // `cwde ; imul eax,eax,0x1542c ; sar eax,0x10 ; movzx eax,ax`
        (g.wrapping_mul(GAIN_SCALE) >> 16) as i16
    };

    // 5. classify and pack.
    frame_tail(st, bits, rate, flags, unit, gain, blk0, blk1, bases, tbl)
}

// ======= pulled in from batch1 (encoder-only) =======

/// `0x10310b10` — eight-band `min(max(prev, next), cur)`, promoted to Q31.
///
/// `flags` is a bitmap of which neighbouring bands are usable: bit `a+b+2`
/// gates `next`, bit `b-a+2` gates `prev`, and a clear bit substitutes `cur`
/// for that neighbour (making the corresponding term a no-op). The eight
/// results are stored as `value << 16` in a 32-bit array. `a` and `b` are
/// signed byte band indices in the original.
pub fn band_min_of_max(
    out: &mut [i32],
    cur: &[i16],
    prev: &[i16],
    next: &[i16],
    flags: i16,
    a: i8,
    b: i8,
) {
    let mut prev = prev;
    let mut next = next;
    if (flags as i32) >> ((a as i32 + b as i32 + 2) & 0x1f) & 1 == 0 {
        next = cur;
    }
    if (flags as i32) >> ((b as i32 - a as i32 + 2) & 0x1f) & 1 == 0 {
        prev = cur;
    }
    for i in 0..8 {
        let (x, y, c) = (prev[i], next[i], cur[i]);
        let m = if y < x {
            if c <= x {
                c
            } else {
                x
            }
        } else if c <= y {
            c
        } else {
            y
        };
        out[i] = (m as i32) << 16;
    }
}

/// `0x1030a440` — compare two mantissa/exponent numbers: is `m1@e1 > m2@e2`?
///
/// A zero mantissa borrows the other operand's exponent so that
/// `0@anything == 0@anything`, then the smaller-exponent operand is shifted
/// down to meet the larger. Past a 30-bit difference the shifted operand
/// collapses to its sign (the original spells this as `m / 0x8001 + (m >> 31)`,
/// which is 0 for a positive mantissa and -1 for a negative one).
///
/// The `d > -1` test inside the first arm is dead in the original too: `d` has
/// just been negated from a strictly positive value. It is kept so the two arms
/// stay visibly symmetric.
pub fn mant_exp_gt(m1: i16, e1: i16, m2: i16, e2: i16) -> bool {
    let a = (m1 as i32) << 16;
    let b = (m2 as i32) << 16;
    let ea = if a != 0 { e1 } else { e2 };
    let eb = if b != 0 { e2 } else { ea };
    let mut d = eb.wrapping_sub(ea);
    if d > 0 {
        d = -d;
        if d > -0x21 {
            if d > -1 {
                return b < a.wrapping_shl(d as u32 & 0x1f);
            }
            if d > -0x1f {
                return b < a >> ((d as i32).wrapping_neg() as u32 & 0x1f);
            }
        }
        return b < (m1 as i32) / 0x8001 + ((m1 as i32) >> 31);
    }
    if d > -0x21 {
        if d > -1 {
            return b.wrapping_shl(d as u32 & 0x1f) < a;
        }
        if d > -0x1f {
            return b >> ((d as i32).wrapping_neg() as u32 & 0x1f) < a;
        }
    }
    (m2 as i32) / 0x8001 + ((m2 as i32) >> 31) < a
}

/// `0x1031b110` — accumulate one 32-bit vector into another after aligning
/// their block exponents.
///
/// `acc[i] += src[i] << (e_src - e_acc)`, with a negative difference becoming
/// an arithmetic right shift and anything past -30 a full sign fill. The add
/// wraps; there is no saturation here.
pub fn acc_add_shifted(src: &[i32], e_src: i16, acc: &mut [i32], e_acc: i16, n: u16) {
    if (n as i16) > 0 {
        let sh = e_src.wrapping_sub(e_acc);
        for i in 0..n as usize {
            acc[i] = acc[i].wrapping_add(shift_i32_batch1(src[i], sh));
        }
    }
}

/// `0x1031ffa0` — 32-point real DFT, in place.
///
/// The input is folded into its even (sum) and odd (difference) halves with a
/// 1/2 scale, then each of bins 1..15 is formed by a 15-tap dot product against
/// the Q15 twiddle table at `0x103821a0`, accumulated in 64 bits and scaled
/// down by 2^19. Bin 0 is the mean (sum of the even half, /8) with a zero
/// imaginary part; the output overwrites the input as interleaved
/// `(real, imag)` pairs.
///
/// `buf` must be 32 words.
pub fn dft32_real(buf: &mut [i16]) {
    let mut t = [0i16; 32];
    t[0] = buf[0] >> 1;
    for k in 0..15 {
        let hi = buf[31 - k] as i32;
        let lo = buf[1 + k] as i32;
        t[1 + k] = ((hi + lo) >> 1) as i16;
        t[17 + k] = ((lo - hi) >> 1) as i16;
    }
    let mid = buf[16];
    t[16] = mid >> 1;

    let mut sum = 0i32;
    for &x in &t[..17] {
        sum = sum.wrapping_add(x as i32);
    }
    buf[0] = (sum >> 3) as i16;
    buf[1] = 0;

    for m in 1i16..0x10 {
        let mut acc_c: i64 = ((t[0] as i32).wrapping_shl(16)) as i64;
        let mut acc_s: i64 = 0;
        let mut ti = m;
        for j in 0..15usize {
            let k = (ti as usize & 0x1f) * 2;
            acc_c = acc_c.wrapping_add(
                (tables_batch1::DFT32_TW[k] as i32)
                    .wrapping_mul(t[1 + j] as i32)
                    .wrapping_mul(2) as i64,
            );
            acc_s = acc_s.wrapping_add(
                (tables_batch1::DFT32_TW[k + 1] as i32)
                    .wrapping_mul(t[17 + j] as i32)
                    .wrapping_mul(2) as i64,
            );
            ti = ti.wrapping_add(m) & 0x1f;
        }
        acc_c = acc_c.wrapping_add(
            (tables_batch1::DFT32_TW[(ti as usize & 0x1f) * 2] as i32)
                .wrapping_mul(t[16] as i32)
                .wrapping_mul(2) as i64,
        );
        buf[2 * m as usize] = (acc_c >> 0x13) as i16;
        buf[2 * m as usize + 1] = (acc_s >> 0x13) as i16;
    }
}

/// `0x10315b00` — decide whether a weighted spectrum is concentrated enough.
///
/// Forms two moments of `v[1..n]`, `S1 = sum 2*i*v[i]` and
/// `S2 = sum 2*i^2*v[i]`, normalises each to a 16-bit mantissa with its own
/// block exponent, and tests `S2*thr - S1^2` (both brought to a common
/// exponent) against `0.6 * thr^2` — a second-moment-versus-mean-squared test,
/// i.e. a spread/peakiness decision. `exp0` and `exp1` are the incoming block
/// exponents of the accumulator and of `v`.
pub fn spread_test(thr: i16, exp0: i32, v: &[i16], exp1: i32, n: i16) -> bool {
    let mut s1: u32 = 0;
    if n > 1 {
        let mut w = 1i32;
        for i in 1..n as u16 as usize {
            let term = (v[i] as i32).wrapping_mul(w);
            w += 1;
            s1 = s1.wrapping_add(term.wrapping_mul(2) as u32);
        }
    }
    let sh1: u32 = if s1 == 0 {
        0
    } else {
        norm_shift_batch1(s1) as u32 & 0xffff
    };
    let m1 = ((s1 << (sh1 & 0x1f)) >> 16) as i16 as i32;
    let sq = m1.wrapping_mul(m1).wrapping_mul(2);

    let mut s2: u32 = 0;
    if n > 1 {
        let mut w: i16 = 1;
        loop {
            let i = w as usize;
            let ww = w.wrapping_mul(w);
            w = w.wrapping_add(1);
            s2 = s2.wrapping_add((ww as i32).wrapping_mul(v[i] as i32).wrapping_mul(2) as u32);
            if w >= n {
                break;
            }
        }
    }
    let sh2: u32 = if s2 == 0 {
        0
    } else {
        norm_shift_batch1(s2) as u32 & 0xffff
    };
    let m2 = ((s2 << (sh2 & 0x1f)) >> 16) as i16 as i32;

    let e2 = (exp1 as u32).wrapping_sub(sh2).wrapping_add(0xf) & 0xffff;
    let align = (((exp1 as u32).wrapping_sub(sh1).wrapping_add(0xf) & 0xffff).wrapping_mul(2))
        .wrapping_sub(e2)
        .wrapping_sub(exp0 as u32) as i32;
    let t = thr as i32;
    let lhs_sq = shift_i32_batch1(sq, align as i16);
    let rhs_exp = (exp0 as u32).wrapping_sub(e2) as i32;
    let rhs = shift_i32_batch1(
        (((t.wrapping_mul(0x999a)) as u32 >> 16) as i16 as i32)
            .wrapping_mul(t)
            .wrapping_mul(2),
        rhs_exp as i16,
    );
    m2.wrapping_mul(t).wrapping_mul(2).wrapping_sub(lhs_sq) < rhs
}

/// `0x10316510` — pick the candidate lag whose windowed energy sum is largest.
///
/// For each of `ncand` candidates it sums `win` consecutive 32-bit values
/// starting at `table[offs[k]]` in a 64-bit accumulator, scales by 1/8, and
/// keeps the largest. The comparison is `<=`, so ties resolve to the **last**
/// candidate. Writes the winning value and its index.
pub fn argmax_block_sum(
    best: &mut u32,
    best_idx: &mut i16,
    ncand: i16,
    table: &[i32],
    offs: &[i16],
    win: u16,
) {
    let mut idx: i16 = 0;
    let mut top: u32 = 0;
    let mut out_idx: i16 = 0;
    let mut ret: u32 = 0;
    let mut k: i16 = 0;
    while k < ncand {
        let mut acc: i64 = 0;
        if (win as i16) > 0 {
            let base = offs[k as usize] as isize;
            for j in 0..win as isize {
                acc = acc.wrapping_add(table[(base + j) as usize] as i64);
                idx = out_idx;
            }
        }
        let v = ((acc as u64) >> 3) as u32;
        if (top as i32) <= (v as i32) {
            top = v;
            idx = k;
            out_idx = k;
        }
        k += 1;
        ret = top;
    }
    *best = ret;
    *best_idx = idx;
}

/// `0x1031c0c0` — update the seven-word adaptive background-level tracker.
///
/// State layout: `[0]` mantissa and `[1]` exponent of the tracked level, `[2]`
/// a slow 0.3/0.7 smoothing of the input, `[3]` a 0.02/0.98 leak, `[4]` the
/// published level, `[5]` a decayed copy of it, `[6]` a hangover counter of
/// consecutive frames in which the input stayed within 0.6 of `ref_level`.
///
/// When the frame looks quiet and stationary (`a` and `c` both below 0.2, and
/// the input within 0.6 of `ref_level`) the tracker latches onto the current
/// input; otherwise it decays the stored level by 0.8. The published value is
/// clamped to `[0x666, 0x3e00]`, i.e. roughly 0.05 to 0.49 of full scale.
///
/// The multiplier constants (`0x999a`, `0xcccc`, `0xe666`, `0xb334`, `0xa3d8`,
/// `0xfae2`) are *unsigned* Q16 fractions — 0.6, 0.8, 0.9, 0.7, 0.64, 0.98 —
/// not negative Q15 values.
pub fn level_track_update(st: &mut [i16], level: i16, a: i16, ref_level: i16, c: i16) {
    let mut smooth = st[2];
    let mut leak = st[3];
    let mut pub_level = st[4];

    if a < 0xccd {
        smooth = ((((level as i32).wrapping_mul(0x4ccc))
            .wrapping_add((smooth as i32).wrapping_mul(0xb334)) as u32)
            >> 16) as i16;
    }

    let d = (level as i32)
        .wrapping_shl(16)
        .wrapping_sub((ref_level as i32).wrapping_shl(16));
    let mag = if d < 0 {
        if d as u32 == 0x8000_0000 {
            0x7fff_ffff
        } else {
            d.wrapping_neg()
        }
    } else {
        d
    };
    let thr = (level as i32).wrapping_mul(0x999a) >> 2;
    let hang = if mag < thr { st[6].wrapping_add(1) } else { 0 };

    let mant;
    let exp;
    let decayed;
    if a < 0x199a && c < 0x199a && mag < thr {
        let sh: i16 = if d == 0 {
            0
        } else {
            norm_shift_batch1(d as u32) as i16
        };
        mant = (((d as u32) << (sh as u32 & 0x1f)) >> 16) as i16;
        exp = -sh;
        decayed = 0x7fff;
        leak = (((((level as i32).wrapping_mul(0xa3d8) >> 5)
            .wrapping_add((leak as i32).wrapping_mul(0xfae2)))
        .wrapping_add(0x8000) as u32)
            >> 16) as i16;
        pub_level = level;
    } else {
        let u = (st[0] as i32).wrapping_mul(0xcccc);
        let sh: i16 = if u == 0 {
            0
        } else {
            norm_shift_batch1(u as u32) as i16
        };
        exp = st[1].wrapping_sub(sh);
        mant = (((u as u32) << (sh as u32 & 0x1f)) >> 16) as i16;
        decayed = (((st[5] as i32).wrapping_mul(0xe666) as u32) >> 16) as i16;
    }

    let m = mant as i32;
    let scaled = if exp < 0 {
        if exp < -0x1e {
            m / 0x8e3a + (m >> 31)
        } else {
            m.wrapping_mul(0xe666) >> ((exp as i32).wrapping_neg() as u32 & 0x1f)
        }
    } else {
        m.wrapping_mul(0xe666).wrapping_shl(exp as u32 & 0x1f)
    };

    let acc = sat_add_batch1((pub_level as i32).wrapping_mul(0xe666), scaled);
    let acc = sat_add_batch1((leak as i32).wrapping_mul(0xcccc) >> 3, acc);

    let mut out = (acc >> 16) as i16;
    if out < 0x3e01 {
        if out < 0x666 {
            out = 0x666;
        }
    } else {
        out = 0x3e00;
    }

    st[4] = out;
    st[5] = decayed;
    st[1] = exp;
    st[2] = smooth;
    st[0] = mant;
    st[3] = leak;
    st[6] = hang;
}

/// `0x103174e0` — maximum of `n` signed 16-bit words.
///
/// The first element is read unconditionally, so `n < 1` still returns `v[0]`
/// rather than a sentinel.
pub fn vec_max(v: &[i16], n: i16) -> i16 {
    let mut m = v[0];
    if n > 1 {
        for &x in &v[1..(n as u16) as usize] {
            if m < x {
                m = x;
            }
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Ported functions.
// ---------------------------------------------------------------------------

/// `0x1030b8d0` — read the second 16-bit field of a structure.
///
/// Ghidra types this `undefined2 f(int)`; it is `mov ax, [arg+2]`, so only the
/// low half of the returned register is meaningful. Used as an accessor for a
/// packed pair whose first word is the one the caller already has.
pub fn get_word1(p: &[i16]) -> i16 {
    p[1]
}

/// `0x1030ff10` — `dst[i] = src[i] + val`, wrapping at 16 bits.
///
/// A zero `val` is short-circuited to nothing at all — the original does not
/// even copy — so `dst` keeps its previous contents in that case.
pub fn vec_add_scalar(dst: &mut [i16], src: &[i16], val: i16, n: u16) {
    if val != 0 && (n as i16) > 0 {
        for i in 0..n as usize {
            dst[i] = src[i].wrapping_add(val);
        }
    }
}

// ---------------------------------------------------------------------------
// Constant tables_batch1 lifted from the reference's .rdata.
// ---------------------------------------------------------------------------

#[allow(clippy::all)]
pub(crate) mod tables_batch1 {
    /// `0x10381920` — dibit allocation for `voicing_pattern`: six rows of eight,
    /// indexed by `nbits - 3`. Each row says how many of the eight output dibits
    /// each of the `nbits` input bits owns; every row sums to 8.
    pub(crate) static BIT_ALLOC: [i16; 48] = [
        3, 2, 3, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 1, 2, 1, 2, 0, 0, 0, 1, 1, 2, 1, 1, 2,
        0, 0, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    ];

    /// `0x10381980` — the 56 pre-baked voicing patterns used for index < 0x38.
    pub(crate) static VOICING_LUT: [u32; 56] = [
        0x00020000, 0x00080000, 0x000a0000, 0x00200000, 0x002a0000, 0x00aa0000, 0x02000000,
        0x02800000, 0x00008000, 0x00000280, 0x00000200, 0x00000aa8, 0x00000800, 0x00000a00,
        0x00000a80, 0x000002a0, 0x00000002, 0x0000002a, 0x00000008, 0x00000020, 0x0000000a,
        0x000000aa, 0x00000080, 0x000000a0, 0x80000000, 0x0a800000, 0xa8000000, 0x02a00000,
        0xa0000000, 0xaa000000, 0x02aa0000, 0x0a000000, 0xaaaa0000, 0x00800000, 0x0aaa0000,
        0xaa800000, 0x08000000, 0xaaa00000, 0x2aaa0000, 0xaaa80000, 0x0000aaaa, 0x0000aa80,
        0x0000a800, 0x00002aaa, 0x0000a000, 0x000002aa, 0x00000aaa, 0x0000aaa0, 0xaaaaaaaa,
        0xaaaa0aaa, 0xa800a800, 0x02aa02aa, 0x0aaa0aaa, 0xaaaaa800, 0xa000a000, 0x08000800,
    ];

    /// `0x1037f978` — `round(32767*cos(2*pi*k/512))`, k = 0..511. The reference keeps a
    /// second symbol at `0x1037fa78`, which is just this table offset by 128 words,
    /// i.e. `-sin`.
    pub(crate) static COS512: [i16; 512] = [
        32767, 32766, 32758, 32746, 32729, 32706, 32679, 32647, 32610, 32568, 32522, 32470, 32413,
        32352, 32286, 32214, 32138, 32058, 31972, 31881, 31786, 31686, 31581, 31471, 31357, 31238,
        31114, 30986, 30853, 30715, 30572, 30425, 30274, 30118, 29957, 29792, 29622, 29448, 29269,
        29086, 28899, 28707, 28511, 28311, 28106, 27897, 27684, 27467, 27246, 27020, 26791, 26557,
        26320, 26078, 25833, 25583, 25330, 25073, 24812, 24548, 24279, 24008, 23732, 23453, 23170,
        22884, 22595, 22302, 22006, 21706, 21403, 21097, 20788, 20475, 20160, 19841, 19520, 19195,
        18868, 18538, 18205, 17869, 17531, 17190, 16846, 16500, 16151, 15800, 15447, 15091, 14733,
        14373, 14010, 13646, 13279, 12910, 12540, 12167, 11793, 11417, 11039, 10660, 10279, 9896,
        9512, 9127, 8740, 8351, 7962, 7571, 7180, 6787, 6393, 5998, 5602, 5205, 4808, 4410, 4011,
        3612, 3212, 2811, 2411, 2009, 1608, 1206, 804, 402, 0, -402, -804, -1206, -1608, -2009,
        -2411, -2811, -3212, -3612, -4011, -4410, -4808, -5205, -5602, -5998, -6393, -6787, -7180,
        -7571, -7962, -8351, -8740, -9127, -9512, -9896, -10279, -10660, -11039, -11417, -11793,
        -12167, -12540, -12910, -13279, -13646, -14010, -14373, -14733, -15091, -15447, -15800,
        -16151, -16500, -16846, -17190, -17531, -17869, -18205, -18538, -18868, -19195, -19520,
        -19841, -20160, -20475, -20788, -21097, -21403, -21706, -22006, -22302, -22595, -22884,
        -23170, -23453, -23732, -24008, -24279, -24548, -24812, -25073, -25330, -25583, -25833,
        -26078, -26320, -26557, -26791, -27020, -27246, -27467, -27684, -27897, -28106, -28311,
        -28511, -28707, -28899, -29086, -29269, -29448, -29622, -29792, -29957, -30118, -30274,
        -30425, -30572, -30715, -30853, -30986, -31114, -31238, -31357, -31471, -31581, -31686,
        -31786, -31881, -31972, -32058, -32138, -32214, -32286, -32352, -32413, -32470, -32522,
        -32568, -32610, -32647, -32679, -32706, -32729, -32746, -32758, -32766, -32767, -32766,
        -32758, -32746, -32729, -32706, -32679, -32647, -32610, -32568, -32522, -32470, -32413,
        -32352, -32286, -32214, -32138, -32058, -31972, -31881, -31786, -31686, -31581, -31471,
        -31357, -31238, -31114, -30986, -30853, -30715, -30572, -30425, -30274, -30118, -29957,
        -29792, -29622, -29448, -29269, -29086, -28899, -28707, -28511, -28311, -28106, -27897,
        -27684, -27467, -27246, -27020, -26791, -26557, -26320, -26078, -25833, -25583, -25330,
        -25073, -24812, -24548, -24279, -24008, -23732, -23453, -23170, -22884, -22595, -22302,
        -22006, -21706, -21403, -21097, -20788, -20475, -20160, -19841, -19520, -19195, -18868,
        -18538, -18205, -17869, -17531, -17190, -16846, -16500, -16151, -15800, -15447, -15091,
        -14733, -14373, -14010, -13646, -13279, -12910, -12540, -12167, -11793, -11417, -11039,
        -10660, -10279, -9896, -9512, -9127, -8740, -8351, -7962, -7571, -7180, -6787, -6393,
        -5998, -5602, -5205, -4808, -4410, -4011, -3612, -3212, -2811, -2411, -2009, -1608, -1206,
        -804, -402, 0, 402, 804, 1206, 1608, 2009, 2411, 2811, 3212, 3612, 4011, 4410, 4808, 5205,
        5602, 5998, 6393, 6787, 7180, 7571, 7962, 8351, 8740, 9127, 9512, 9896, 10279, 10660,
        11039, 11417, 11793, 12167, 12540, 12910, 13279, 13646, 14010, 14373, 14733, 15091, 15447,
        15800, 16151, 16500, 16846, 17190, 17531, 17869, 18205, 18538, 18868, 19195, 19520, 19841,
        20160, 20475, 20788, 21097, 21403, 21706, 22006, 22302, 22595, 22884, 23170, 23453, 23732,
        24008, 24279, 24548, 24812, 25073, 25330, 25583, 25833, 26078, 26320, 26557, 26791, 27020,
        27246, 27467, 27684, 27897, 28106, 28311, 28511, 28707, 28899, 29086, 29269, 29448, 29622,
        29792, 29957, 30118, 30274, 30425, 30572, 30715, 30853, 30986, 31114, 31238, 31357, 31471,
        31581, 31686, 31786, 31881, 31972, 32058, 32138, 32214, 32286, 32352, 32413, 32470, 32522,
        32568, 32610, 32647, 32679, 32706, 32729, 32746, 32758, 32766,
    ];

    /// `0x103821a0` — 32 interleaved `(cos, -sin)` twiddles at `2*pi*k/32`, Q15.
    pub(crate) static DFT32_TW: [i16; 64] = [
        32767, 0, 32138, -6393, 30274, -12540, 27246, -18205, 23170, -23170, 18205, -27246, 12540,
        -30274, 6393, -32138, 0, -32767, -6393, -32138, -12540, -30274, -18205, -27246, -23170,
        -23170, -27246, -18205, -30274, -12540, -32138, -6393, -32767, 0, -32138, 6393, -30274,
        12540, -27246, 18205, -23170, 23170, -18205, 27246, -12540, 30274, -6393, 32138, 0, 32767,
        6393, 32138, 12540, 30274, 18205, 27246, 23170, 23170, 27246, 18205, 30274, 12540, 32138,
        6393,
    ];

    /// `0x10381cd8` — incremental swap list for the 8-point bit reversal.
    pub(crate) static BITREV8: [i16; 5] = [2, 2, 8, 3, 3];

    /// `0x10381ce4` — incremental swap list for the 16-point bit reversal.
    pub(crate) static BITREV16: [i16; 13] = [6, 2, 16, 1, -9, 1, 15, 3, -5, 3, 7, 7, -3];

    /// `0x10381d00` — incremental swap list for the 32-point bit reversal.
    pub(crate) static BITREV32: [i16; 25] = [
        12, 2, 32, 1, -17, 1, 31, 3, -9, 1, -17, 1, 31, 3, -21, 3, 15, 3, -9, 3, 15, 7, -11, 7, 7,
    ];

    /// `0x10381d38` — incremental swap list for the 64-point bit reversal.
    pub(crate) static BITREV64: [i16; 57] = [
        28, 2, 64, 1, -33, 1, 63, 1, -81, 1, 63, 1, -33, 1, 63, 3, -41, 1, -33, 1, 63, 3, -17, 1,
        -33, 1, 63, 3, -53, 3, 31, 3, -17, 1, -33, 1, 63, 3, -41, 3, 31, 3, -17, 3, 31, 7, -27, 3,
        -17, 3, 31, 7, -9, 7, 15, 15, -5,
    ];

    /// `0x10381db0` — incremental swap list for the 128-point bit reversal.
    pub(crate) static BITREV128: [i16; 113] = [
        56, 2, 128, 1, -65, 1, 127, 1, -161, 1, 127, 1, -65, 1, 127, 3, -81, 1, -65, 1, 127, 1,
        -161, 1, 127, 1, -65, 1, 127, 3, -105, 1, -65, 1, 127, 3, -33, 1, -65, 1, 127, 3, -81, 1,
        -65, 1, 127, 3, -33, 1, -65, 1, 127, 3, -117, 3, 63, 3, -33, 1, -65, 1, 127, 3, -81, 3, 63,
        3, -33, 1, -65, 1, 127, 3, -105, 3, 63, 3, -33, 3, 63, 3, -81, 3, 63, 3, -33, 3, 63, 7,
        -59, 3, -33, 3, 63, 7, -17, 3, -33, 3, 63, 7, -41, 7, 31, 7, -17, 7, 31, 15, -21, 15, 15,
    ];
}

/// Saturating 32-bit add, same shape as [`sat_sub_batch1`].
fn sat_add_batch1(a: i32, b: i32) -> i32 {
    let s = a.wrapping_add(b);
    if ((s ^ a) & (s ^ b)) < 0 {
        if a < 0 {
            i32::MIN
        } else {
            i32::MAX
        }
    } else {
        s
    }
}

// ======= pulled in from batch2 (encoder-only) =======

/// `0x10317590` — index of the maximum element in `a[0..n]`.
///
/// Returns the position of the first strict maximum. `n <= 1` yields 0.
pub fn argmax(a: &[i16], n: i16) -> i16 {
    let mut idx: i16 = 1;
    let mut max = a[0];
    let mut prev: i16 = 0;
    let mut best: i16 = 0;
    if n > 1 {
        let mut p = 0usize;
        loop {
            p += 1;
            best = prev;
            if max < a[p] {
                max = a[p];
                best = idx;
            }
            idx += 1;
            prev = best;
            if !(idx < n) {
                break;
            }
        }
    }
    best
}

/// `0x1030a4f0` — compare two block-floating values `p1·2^p2` and `p3·2^p4`.
///
/// Returns whether `p1·2^p2 < p3·2^p4`, aligning the two by their exponent
/// difference (with the usual TI shift clamps and a `/0x8001` fallback for
/// large gaps). Zero mantissas fold the exponent handling accordingly.
pub fn compare_scaled(p1: i16, p2: i16, p3: i16, p4: i16) -> bool {
    let iv1 = p1 as i32;
    let iv2 = iv1.wrapping_mul(0x10000);
    let mut s5 = p4;
    if iv2 != 0 {
        s5 = p2;
    }
    let iv3 = p3 as i32;
    let iv4 = iv3.wrapping_mul(0x10000);
    let mut d = p4;
    if iv4 == 0 {
        d = s5;
    }
    d = d.wrapping_sub(s5);
    if d > 0 {
        d = d.wrapping_neg();
        if d > -0x21 {
            if d > -1 {
                return iv2.wrapping_shl(d as u32 & 0x1f) < iv4;
            }
            if d > -0x1f {
                return (iv2 >> ((-(d as i32)) as u32 & 0x1f)) < iv4;
            }
        }
        return (iv1 / 0x8001 + (iv1 >> 0x1f)) < iv4;
    }
    if d > -0x21 {
        if d > -1 {
            return iv2 < iv4.wrapping_shl(d as u32 & 0x1f);
        }
        if d > -0x1f {
            return iv2 < (iv4 >> ((-(d as i32)) as u32 & 0x1f));
        }
    }
    iv2 < (iv3 / 0x8001 + (iv3 >> 0x1f))
}

/// `0x10314380` — generate interpolated cosine pairs from a source spectrum.
///
/// For each of `p4` input words, block-shifts it (masking to a Q-window via
/// `p3`), then produces two cosine-table samples 90° apart (`+0x1800000` in
/// phase) written as consecutive output pairs. Uses the 512-entry table at
/// `0x1037f978`. `src`/`out` must hold `2*p4` words each.
pub fn cos_pairs(out: &mut [i16], src: &[i16], p3: i16, p4: u16, table: &[i16]) {
    let (s6, mask): (i16, u32) = if p3 < 0 {
        (0, 0x7fff_ffff)
    } else if p3 != 0 {
        (
            p3,
            (1u32 << ((0x1f - (p3 as i8 as i32)) as u32 & 0x1f)).wrapping_sub(1),
        )
    } else {
        (0, 0x7fff_ffff)
    };

    if (p4 as i16) > 0 {
        let sh = p3 - s6;
        let mut oi = 0usize;
        let mut si = 0usize;
        for _ in 0..p4 {
            let iv1 = (src[si] as i32) << 16;
            let uv2 = shr_or_shl(iv1, sh) & mask;

            let s3: i16 = if s6 < 0 {
                if s6 < -0x1e {
                    ((uv2 as i32) >> 0x1f) as i16
                } else {
                    (((uv2 as i32) >> ((-(s6 as i32)) as u32 & 0x1f)) as u32 >> 16) as i16
                }
            } else {
                (uv2.wrapping_shl(s6 as u32 & 0x1f) >> 16) as i16
            };

            let mut u2 = ((s3 as i32) << 16) >> 6;
            let uv5 = if u2 < 0 {
                if (u2 as u32) == 0x8000_0000 {
                    0x7fff_ffff
                } else {
                    (u2 as u32).wrapping_neg()
                }
            } else {
                u2 as u32
            };
            let idx0 = (uv5 >> 16) as i16 as usize;
            let idx1 = ((((uv5 as i32) >> 16) as u32 & 0xffff).wrapping_add(1) & 0x1ff) as usize;
            out[oi] = ((table[idx0] as i32 * (0x10000 - (uv5 & 0xffff) as i32)
                + table[idx1] as i32 * (uv5 & 0xffff) as i32)
                >> 16) as i16;

            u2 = u2.wrapping_add(0x1800000);
            let uv2b = if u2 < 0 {
                if (u2 as u32) == 0x8000_0000 {
                    0x7fff_ffff
                } else {
                    (u2 as u32).wrapping_neg()
                }
            } else {
                u2 as u32
            };
            let idx2 = (((uv2b as i32) >> 16) as u32 & 0x1ff) as usize;
            out[oi + 1] = ((table[idx2] as i32 * (0x10000 - (uv2b & 0xffff) as i32)
                + table[(idx2 + 1) & 0x1ff] as i32 * (uv2b & 0xffff) as i32)
                >> 16) as i16;

            oi += 2;
            si += 2;
        }
    }
}

/// `0x10315cc0` — pick the band index maximising a (difference of) energy sum.
///
/// For each of `p2` candidate bands described by `pairs` (a start index and a
/// span), sums 64-bit energies from `energy`, optionally subtracting a second
/// span when the band exceeds `p5`, and records the argmax. Writes the winning
/// band index to `out`. `energy` is 32-bit; indices come from `pairs` and
/// `p5`, so callers keep them in range.
pub fn pick_best_band(out: &mut i16, p2: i16, energy: &[i32], pairs: &[i16], p5: i16) {
    let mut uv6: u32 = 0xffff_ffff;
    let mut local_20: u32 = 0xffff_ffff;
    let mut s9: i16 = 0;
    let mut s4: i16 = 0;
    let mut s2: i16 = 0;
    if p2 > 0 {
        let iv10 = p5 as i32;
        let mut pi = 0usize;
        loop {
            let iv5 = pairs[pi] as i32;
            let mut local_8: u32 = 0;
            s2 = (((pairs[pi + 1] as i32)
                .wrapping_mul(0x10000)
                .wrapping_add(iv5.wrapping_mul(-0x10000)) as u32)
                >> 16) as i16;
            let mut iv8 = iv10 + s2 as i32;
            let mut cnt = ((iv8.wrapping_mul(0x10000) >> 16) as u32) & 0xffff;
            let mut istk: i32 = 0;
            if (iv8 as i16) > 0 {
                let mut k = iv5 as usize;
                loop {
                    let val = energy[k] as u32;
                    k += 1;
                    let (s, carry) = local_8.overflowing_add(val);
                    local_8 = s;
                    istk = istk
                        .wrapping_add(val as i32 >> 0x1f)
                        .wrapping_add(carry as i32);
                    cnt -= 1;
                    uv6 = local_20;
                    if cnt == 0 {
                        break;
                    }
                }
            }
            let mut acc = (local_8 >> 3) | ((istk as u32) << 0x1d);
            if p5 < s2 {
                iv8 = (s2 as i32)
                    .wrapping_mul(0x10000)
                    .wrapping_add(iv10.wrapping_mul(-0x10000));
                let mut cnt2 = ((iv8 >> 16) as u32) & 0xffff;
                local_8 = 0;
                istk = 0;
                if (((iv8 as u32) >> 16) as i16) > 0 {
                    let mut k = (iv10 + iv5) as usize;
                    loop {
                        let val = energy[k] as u32;
                        k += 1;
                        let (s, carry) = local_8.overflowing_add(val);
                        local_8 = s;
                        istk = istk
                            .wrapping_add(val as i32 >> 0x1f)
                            .wrapping_add(carry as i32);
                        cnt2 -= 1;
                        if cnt2 == 0 {
                            break;
                        }
                    }
                }
                acc = acc.wrapping_sub((local_8 >> 3) | ((istk as u32) << 0x1d));
                uv6 = local_20;
            }
            if (uv6 as i32) <= (acc as i32) {
                uv6 = acc;
                local_20 = acc;
                s9 = s4;
            }
            s4 += 1;
            pi += 2;
            s2 = s9;
            if !(s4 < p2) {
                break;
            }
        }
    }
    *out = s2;
}

/// `0x103107b0` — spectral-envelope peak smear with leaky decay.
///
/// Spreads a scaled version of `p3` into up to three low bins of `arr`
/// (`arr[0..4]`), taking the running max, then updates the scalar `state` with
/// a leaky-integrator blend of `p3` and its previous value. `p3 >= 0x3334`
/// skips the spread. `arr` must be ≥4 words.
pub fn smooth_peak(arr: &mut [i16], state: &mut i16, p3: i16) {
    let s2 = *state;
    if p3 < 0x3334 {
        let mut s7: i16 = 0;
        if p3 < 0x199b {
            s7 = (p3 < 0x1112) as i16 + 1;
        }
        let mut iv: i32 = ((s7 as i32 + 1)
            .wrapping_mul(p3 as i32)
            .wrapping_mul(0x10000))
        .wrapping_add((s2 as i32).wrapping_mul(-0xcccc));
        loop {
            let s4 = ((iv as u32) >> 16) as i16;
            if s4 < 1 {
                break;
            }
            let nxt = (arr[s7 as usize + 1] as i32).wrapping_mul(0x10000) as u32;
            let prod = (p3 as i32).wrapping_mul(-0x3c00) as u32;
            let uv5 = prod.wrapping_add(nxt);
            let sat = if (((uv5 ^ prod) & (uv5 ^ nxt)) as i32) < 0 {
                (((nxt as i32) < 0) as u32).wrapping_add(0x7fff_ffff)
            } else {
                uv5
            };
            let s3 = (sat >> 16) as i16;
            if arr[s7 as usize] < s3 {
                arr[s7 as usize] = s3;
            }
            s7 -= 1;
            iv = (s4 as i32)
                .wrapping_mul(0x10000)
                .wrapping_add((p3 as i32).wrapping_mul(-0x10000));
            if !(s7 >= 0) {
                break;
            }
        }
    }
    *state = ((((p3 as i32).wrapping_mul(0xcccc) >> 3)
        .wrapping_add((s2 as i32).wrapping_mul(0xe666)) as u32)
        >> 16) as i16;
}

/// `0x10315970` — weighted-slope peak test over `a[1..p3]`.
///
/// Forms `2 * sum(k * a[k])`, block-shifts it by `p2 - p5`, adds linear terms
/// in `p4`, and compares the magnitude against a `p8`-scaled threshold.
/// Returns the boolean comparison (false when `p4 <= 0`).
pub fn peak_test(a: &[i16], p2: i16, p3: i16, p4: i16, p5: i16, p6: i16, p7: i16, p8: i16) -> bool {
    let mut result = false;
    if p4 > 0 {
        let mut acc: i32 = 0;
        if p3 > 1 {
            let mut mult: i32 = 1;
            let mut idx = 0usize;
            let mut cnt = (p3 - 1) as u16;
            loop {
                idx += 1;
                acc = acc.wrapping_add((a[idx] as i32).wrapping_mul(mult).wrapping_mul(2));
                mult += 1;
                cnt = cnt.wrapping_sub(1);
                if cnt == 0 {
                    break;
                }
            }
        }
        let shift = p2 - p5;
        if shift < 0 {
            if shift < -0x1e {
                acc >>= 0x1f;
            } else {
                acc >>= (-(shift as i32)) as u32 & 0x1f;
            }
        } else {
            acc = acc.wrapping_shl(shift as u32 & 0x1f);
        }
        let iv2 = (p6 as i32).wrapping_mul(p4 as i32).wrapping_mul(2);
        let uv5 = iv2 >> 9;
        acc = acc
            .wrapping_add((p7 as i32).wrapping_mul(p4 as i32).wrapping_mul(2))
            .wrapping_sub(uv5);
        if acc < 0 {
            acc = if acc == i32::MIN { 0x7fff_ffff } else { -acc };
        }
        let t1 = (((uv5 as u32 & 0xffff) as i32).wrapping_mul(p8 as i32)) >> 0xf;
        let t2 = ((iv2 >> 0x19) as i16 as i32)
            .wrapping_mul(p8 as i32)
            .wrapping_mul(2);
        result = acc < (t1.wrapping_add(t2) >> 5);
    }
    result
}

/// `0x1030f820` — block-exponent aligned saturating add of two 32-bit vectors.
///
/// `dst[i] = sat(a[i] << (p3-max) + b[i] << (p5-max))`, where `max = max(p3,p5)`
/// (so both shifts are ≤ 0), with 32-bit signed-saturating add. Returns `max`,
/// the common exponent. `n <= 0` returns `max` without touching `dst`.
pub fn block_add_sat(dst: &mut [i32], a: &[i32], b: &[i32], p3: i16, p5: i16, n: u16) -> i16 {
    let s1 = if p3 <= p5 { p5 } else { p3 };
    if (n as i16) > 0 {
        let sh_a = p3 - s1;
        let sh_b = p5 - s1;
        for i in 0..n as usize {
            let ua = shr_or_shl(a[i], sh_a);
            let ub = shr_or_shl(b[i], sh_b);
            let sum = ua.wrapping_add(ub);
            let res = if (((sum ^ ub) & (sum ^ ua)) as i32) < 0 {
                (((ua as i32) < 0) as u32).wrapping_add(0x7fff_ffff)
            } else {
                sum
            };
            dst[i] = res as i32;
        }
    }
    s1
}

/// `0x10316ee0` — select the first "active" entry in a two-slot table.
///
/// The state block holds two tag words at byte offsets 10 and 12; the first
/// whose tag != 0xff is chosen, its tag stored to `out_tag` and its two data
/// words (at `+0xe/+0x10` for slot 0, `+0x12/+0x14` for slot 1) copied to
/// `out_pair`. Returns whether an active entry was found (`tag != 0xff`).
/// `out_pair` is left untouched when nothing is active.
pub fn find_active_entry(state: &[u8], out_tag: &mut i16, out_pair: &mut [i16; 2]) -> bool {
    let rd = |off: usize| i16::from_le_bytes([state[off], state[off + 1]]);
    *out_tag = 0xff;
    for i in 0..2usize {
        let t = rd(10 + i * 2);
        if t != 0xff {
            *out_tag = t;
            out_pair[0] = rd(0xe + i * 4);
            out_pair[1] = rd(0x10 + i * 4);
            return *out_tag != 0xff;
        }
    }
    *out_tag != 0xff
}

/// `0x1030af60` — base-2 log-ish transform with a block exponent.
///
/// Normalises `x`, evaluates a 4th-order polynomial (a `log2`/gain-mapping
/// curve), renormalises the result, and folds the two exponents into `*exp`
/// (scaled by 1/4). `tbl` must be the 4 `i16` at `0x1037f970` (indices 1..3
/// are used). Returns the mantissa; `x == 0` returns 0 with `*exp = 0`.
pub fn log2_transform(param_1: u32, exp: &mut i16, tbl: &[i16]) -> u32 {
    if param_1 == 0 {
        *exp = 0;
        return 0;
    }
    let uv = if (param_1 as i32) < 0 {
        !param_1
    } else {
        param_1
    };
    let mut i1: i32 = 0x1f;
    if uv != 0 {
        while uv >> i1 == 0 {
            i1 -= 1;
        }
    }
    let iv3 = ((param_1 << ((0x1e - i1) as u32 & 0x1f)) >> 0x10) as i16 as i32;

    let m1 = (iv3.wrapping_mul(-0x1f1a) as u32).wrapping_add(0x8000) & 0xffff_0000;
    let h1 = ((m1.wrapping_add(0x3f26_0000) >> 16) as u16 as i16) as i32;
    let m2 = (h1.wrapping_mul(iv3).wrapping_mul(2) as u32).wrapping_add(0x8000) & 0xffff_0000;
    let h2 = ((m2.wrapping_add(0x9292_0000) >> 16) as u16 as i16) as i32;
    let m3 = (h2.wrapping_mul(iv3).wrapping_mul(2) as u32).wrapping_add(0x8000) & 0xffff_0000;
    let h3 = ((m3.wrapping_add(0x7b83_0000) >> 16) as u16 as i16) as i32;
    let t = h3.wrapping_mul(iv3).wrapping_mul(2).wrapping_add(0x8000);
    let mut uv2 = (((t >> 1) as u32) & 0xffff_8000).wrapping_add(0x2128_8000);

    let uv4_norm: u32 = if uv2 == 0 {
        0
    } else {
        let u = if (uv2 as i32) < 0 { !uv2 } else { uv2 };
        let mut i3: i32 = 0x1f;
        if u != 0 {
            while u >> i3 == 0 {
                i3 -= 1;
            }
        }
        (0x1eu32.wrapping_sub(i3 as u32)) & 0xffff
    };
    uv2 = uv2.wrapping_shl(uv4_norm & 0x1f);

    let mut uv4 = (((*exp).wrapping_sub((0x1e - i1) as i16)) as u16 as u32)
        .wrapping_add(1u32.wrapping_sub(uv4_norm));
    let mut s5 = uv4 as i16;
    uv4 &= 3;
    if uv4 != 0 {
        uv2 = ((tbl[uv4 as usize] as i32)
            .wrapping_mul((uv2 >> 16) as i16 as i32)
            .wrapping_mul(2)
            .wrapping_add(0x8000) as u32)
            & 0xffff_0000;
        s5 += 1;
    }
    *exp = (s5 + 3) >> 2;
    uv2
}

/// `0x1031d640` — population count of the low `n` bits of a 16-bit value.
///
/// Counts set bits while repeatedly arithmetic-shifting the value as a signed
/// 16-bit word (the sign bit is replicated, but only `n` low bits are ever
/// examined). `n <= 0` returns 0.
pub fn popcount_bits(mut x: u32, n: u16) -> i32 {
    let mut count: i32 = 0;
    if (n as i16) > 0 {
        let mut u = x as u16;
        for _ in 0..n {
            u = ((u as i16) >> 1) as u16;
            count += (x & 1) as i32;
            x = u as u32;
        }
    }
    count
}

/// `0x103102d0` — store a 32-bit word through a pointer (`*dst = val`).
pub fn store32(dst: &mut u32, val: u32) {
    *dst = val;
}

/// `0x10310680` — a five-argument threshold test returning 0/1.
///
/// True when `p3` is small (< 0x199a) and below half of both `p1` and `p2`
/// (in Q16), and `|p4 - p5/2|` is under `p5/8` (all in Q16). A voicing/pitch
/// consistency gate.
pub fn thresh5(p1: i16, p2: i16, p3: i16, p4: i16, p5: i16) -> u32 {
    if p3 < 0x199a
        && (p3 as i32).wrapping_mul(0x10000) < (((p2 as i32) << 16) >> 1)
        && (p3 as i32).wrapping_mul(0x10000) < (((p1 as i32) << 16) >> 1)
    {
        let mut v = (p4 as i32)
            .wrapping_mul(0x10000)
            .wrapping_sub(((p5 as i32) << 16) >> 1);
        if v < 0 {
            v = if v == i32::MIN {
                0x7fff_ffff
            } else {
                v.wrapping_neg()
            };
        }
        if v < (((p5 as i32) << 16) >> 3) {
            return 1;
        }
    }
    0
}

/// `0x1030fe40` — populate a "synthesis view" struct with pointers into a
/// state block.
///
/// Writes 19 pointer slots (`base + fixed offset`), three 16-bit fields copied
/// from the state (`+0x630`, `+0x97c`, `+0x980`, `+0xa72`), a bit extracted
/// from `flags` (`(flags>>6)&1`), a caller word `p4`, a zero, and one absolute
/// code pointer (`0x1031f530`). Because several slots are written 16 bits at a
/// time, `out` must be pre-initialised and compared whole.
///
/// `out` is the ≥0x4c-byte view struct; `base` is the runtime address of the
/// state block; `state` supplies the copied words; `lab` is the runtime
/// address of `LAB_1031f530`.
pub fn init_synth_view(out: &mut [u8], base: u32, state: &[u8], flags: u32, p4: u16, lab: u32) {
    let put32 = |out: &mut [u8], off: usize, v: u32| {
        out[off..off + 4].copy_from_slice(&v.to_le_bytes());
    };
    let put16 = |out: &mut [u8], off: usize, v: u16| {
        out[off..off + 2].copy_from_slice(&v.to_le_bytes());
    };
    let rd16 = |off: usize| u16::from_le_bytes([state[off], state[off + 1]]);

    put32(out, 0x00, base.wrapping_add(0x64e));
    put32(out, 0x04, base.wrapping_add(0x69e));
    put32(out, 0x0c, base.wrapping_add(0x644)); // slot 3
    put32(out, 0x08, base.wrapping_add(0x63a)); // slot 2
    put16(out, 0x10, rd16(0x630)); // slot 4, low half
    put32(out, 0x14, base.wrapping_add(0x18));
    put32(out, 0x18, base.wrapping_add(0x624));
    put32(out, 0x1c, base.wrapping_add(0x746));
    put32(out, 0x20, base.wrapping_add(0x926));
    put32(out, 0x28, base.wrapping_add(0x710)); // slot 10
    put32(out, 0x2c, base.wrapping_add(0x740)); // slot 0xb
    put32(out, 0x34, base.wrapping_add(0xa2e)); // slot 0xd
    put32(out, 0x38, base.wrapping_add(0x122e)); // slot 0xe
    put16(out, 0x30, rd16(0x97c)); // slot 0xc, low half
    put16(out, 0x32, ((flags >> 6) & 1) as u16); // high half of slot 0xc
    put16(out, 0x42, p4); // high half of slot 0x10
    put16(out, 0x40, 0); // slot 0x10, low half
    put32(out, 0x24, base.wrapping_add(0x6f8)); // slot 9
    put32(out, 0x3c, lab); // slot 0xf
    put16(out, 0x44, rd16(0x980)); // slot 0x11, low half
    put16(out, 0x46, rd16(0xa72)); // high half of slot 0x11
    put32(out, 0x48, base.wrapping_add(0x938)); // slot 0x12
}

// ======= pulled in from batch3 (encoder-only) =======

/// `0x1031b410` — turn a normalised fundamental into the four band edges of a
/// harmonic group, reporting whether the group is usable.
///
/// `w0` (plus a half-LSB bias) is normalised, scaled by `scale` and requantised
/// to exponent `exp`; `out[1]` is the centre. If that centre is below
/// `floor`, the routine also emits `out[0]` (centre less one full step, +1),
/// `out[2]` (less half a step, +1) and `out[3]` (the upper edge), all derived
/// by nudging the *normalised* value by ±0x40000000 / 0x20000000 — that is, by
/// ±1 and ±1/2 in units of the current block exponent — and rescaling.
///
/// Returns 1 if the band is to be used, 0 if the derived edges collapsed
/// (`out[3] >= out[2]`, i.e. the interval is empty). Ghidra types the return
/// `undefined4`; it is a boolean.
pub fn band_edges(out: &mut [i16], scale: i16, exp: i32, w0: i16, floor: i16) -> u32 {
    let v = (w0 as i32).wrapping_mul(0x10000).wrapping_add(0x8000);
    let e = if v == 0 {
        0u32
    } else {
        0x1eu32.wrapping_sub(top_bit_batch3(if v < 0 { !(v as u32) } else { v as u32 }) as u32)
            & 0xffff
    };
    let mut norm = ((v as u32) << (e & 0x1f)) as i32;
    let q = 0xfu32.wrapping_sub(e) & 0xffff;
    let sh = (exp.wrapping_sub(0xf).wrapping_add(q as i32)) as i16;
    let k = scale as i32;

    let centre = shift_to_i16(
        (high16_batch3(norm) as i32).wrapping_mul(k).wrapping_mul(2),
        sh,
    );
    out[1] = centre;
    if centre >= floor {
        return 1;
    }

    let step = (q as u32).wrapping_neg() as i16;

    norm = norm.wrapping_add(shift32_al2b2(0x4000_0000, step).wrapping_mul(-2));
    out[0] = shift_to_i16(
        (high16_batch3(norm) as i32).wrapping_mul(k).wrapping_mul(2),
        sh,
    )
    .wrapping_add(1);

    norm = norm.wrapping_add(shift32_al2b2(0x2000_0000, step));
    out[2] = shift_to_i16(
        (high16_batch3(norm) as i32).wrapping_mul(k).wrapping_mul(2),
        sh,
    )
    .wrapping_add(1);

    let top = shift32_al2b2(0x4000_0000, step).wrapping_add(norm);
    let hi = shift_to_i16(
        (high16_batch3(top) as i32).wrapping_mul(k).wrapping_mul(2),
        sh,
    );
    out[3] = hi;
    if out[2] <= hi {
        return 0;
    }
    1
}

/// `0x1030efd0` — pick the mid-frame value of a parameter, rejecting
/// transitions.
///
/// `w` is a normalised confidence/energy triple `[prev, mid, next]` in Q15 and
/// `v` the parameter triple to interpolate. If one end dominates the other
/// (above 0.4 while the other is below 0.4, or above 0.2 while the other is
/// below half of it) the mid value is copied wholesale from the dominant end
/// instead of being interpolated — an onset or offset inside the frame. Failing
/// that both `w[1]` and `v[1]` become the midpoint average, and when `w[0]` is
/// very small (< 0.1) the interpolation point is nudged to 1/4 or 1 depending
/// on which end `v` is nearer. Both triples are updated in place.
pub fn interp_mid(w: &mut [i16], v: &mut [i16]) {
    let w0 = w[0];
    let w2 = w[2];

    if w0 > 0x3333 && w2 < 0x3333 {
        v[1] = v[2];
        w[1] = w[2];
        return;
    }
    if w0 > 0x1999 && w2 < (w0 >> 1) {
        v[1] = v[2];
        w[1] = w[2];
        return;
    }
    if (w2 > 0x3333 && w0 < 0x3333) || (w2 > 0x1999 && w0 < (w2 >> 1)) {
        v[1] = v[0];
        w[1] = w[0];
        return;
    }

    w[1] = high16_batch3(
        (((w2 as i32).wrapping_mul(0x10000)) >> 1)
            .wrapping_add(((w0 as i32).wrapping_mul(0x10000)) >> 1),
    );
    let next = (v[2] as i32).wrapping_mul(0x10000);
    let half_prev = ((v[0] as i32).wrapping_mul(0x10000)) >> 1;
    let half_next = next >> 1;
    v[1] = high16_batch3(half_next.wrapping_add(half_prev));

    if w[0] < 0xccc {
        let d = abs_sat32(((v[0] as i32).wrapping_mul(0x10000)).wrapping_sub(half_next));
        if d.wrapping_add((v[0] as i32).wrapping_mul(-0x3332)) < 0 {
            v[1] = high16_batch3((next >> 2).wrapping_add(half_prev));
            return;
        }
        let d2 = abs_sat32(next.wrapping_sub(half_prev));
        if d2.wrapping_add((v[0] as i32).wrapping_mul(-0x1998)) < 0 {
            v[1] = high16_batch3(next.wrapping_add(half_prev));
        }
    }
}

/// `0x103106f0` — add a fixed 170 to each word with saturation, then clamp to
/// 0x77ff.
///
/// Works in the 32-bit domain (`x << 16` plus `170 << 16`) purely so the
/// saturating add is the same primitive used elsewhere; the effect is a
/// saturating `+170` on each 16-bit word followed by an upper clamp. This is a
/// noise-floor bias applied to a log-magnitude spectrum.
pub fn bias_clamp(v: &mut [i16], n: i16) {
    if n > 0 {
        for x in &mut v[..n as usize] {
            let a = (*x as i32).wrapping_mul(0x10000);
            let s = sat_add32(a, 0x00aa_0000);
            let mut r = high16_batch3(s);
            if r > 0x77ff {
                r = 0x77ff;
            }
            *x = r;
        }
    }
}

/// Fixed-point exponent of the harmonic log-magnitude words the amplitude
/// stage produces and the spectral quantiser consumes: they are base-2
/// logarithms at Q11, so the linear magnitude is `2^(word / 2048)`.
///
/// Two independent places in the codec pin this:
///
/// * `expand_log_mags` (`0x103176b0`) turns the vector back
///   into linear magnitudes with `pow2_scaled(word, 4, _)`, which evaluates
///   `2^(word · 2^(4−15))` = `2^(word / 2048)`, and derives the shared block
///   exponent as `word >> 11`;
/// * the 6-bit gain codebook [`crate::engine::tables::GAIN64`], whose entries are added
///   straight onto this vector, is the published Annex E gain table times 2048.
pub const LOG_MAG_Q_SHIFT: i32 = 11;

/// The newest analysis spectrum's harmonic log magnitudes, biased exactly as
/// the packer's own amplitude stage biases them.
///
/// This is [`slot_harmonic_levels`] restricted to the newest spectrum row
/// (`slot = 0`) and to the two steps that are unconditional there: the band
/// integration of [`harmonic_log_mags`] over the 129-bin magnitude-squared
/// spectrum at `S+0x18` with its block exponent at `S+0x624`, then
/// [`bias_clamp`]'s `+170`. The optional peak smear is *not* applied — the
/// packer enables it only when the fundamental's own voicing bit is set, and it
/// carries a leaky tracker that only a full encode should advance.
///
/// `step` is the harmonic spacing as a Q11 bin increment over that 129-bin
/// spectrum: at a 256-point transform of 8 kHz audio one bin is 31.25 Hz, so a
/// fundamental of `ω₀` radians/sample is `ω₀ · 128/π` bins and
/// `step = round(ω₀ · 128/π · 2048)`. `offset` is the packer's log-magnitude
/// bias, zero on every path the P25 wrapper takes.
///
/// Domain: `out` at least `n` words, `n` in `1..=56`, and `step` positive and
/// small enough that `n` bands stay inside the spectrum.
pub fn spectrum_harmonic_log_mags(out: &mut [i16], st: &[i16], step: i16, n: i16, offset: i16) {
    harmonic_log_mags(out, step, n as u16, &st[SPEC..], st[EXP0], offset);
    bias_clamp(out, n);
}

/// `0x1031d680` — if either bit of the pair selected by `sel` is set in the
/// flag word at state `+0x44`, re-arm the timer at state `+0x40` to 0xfa000.
///
/// The decompiled `-(ushort)(cond) & 0x400` is just the compiler materialising
/// a boolean; the mask value is irrelevant to the branch. The pair mask is
/// `3 << (sel + 2)`, so `sel` selects a two-bit field starting at bit 2.
/// 0xfa000 = 1024000, a countdown reload.
pub fn rearm_timer(flags: i16, sel: i8, timer: &mut i32) {
    let mask = 3i32.wrapping_shl((sel as i32 + 2) as u32 & 0x1f);
    if mask & (flags as i32) != 0 {
        *timer = 0xfa000;
    }
}

/// `0x1030a730` — as [`div32_half`] but clamped to ±0x7fff.
///
/// The clamp is expressed as a range test on the operands: if `|num| >> 16` is
/// already at or above `|den|` the quotient cannot fit in 15 bits, so 0x7fff is
/// returned without dividing. This is the routine used where the result feeds a
/// 16-bit word directly.
pub fn div32_sat15(num: i32, den: i16) -> i32 {
    let sign = (den as i32) ^ num;
    let n = if num < 0 {
        if num == i32::MIN {
            i32::MAX
        } else {
            -num
        }
    } else {
        num
    };
    let d = if den < 0 {
        if den == i16::MIN {
            i16::MAX
        } else {
            -den
        }
    } else {
        den
    } as i32;
    let q = if n < d {
        0
    } else if (n >> 16) < d {
        (n / d) >> 1
    } else {
        0x7fff
    };
    if sign < 0 {
        -q
    } else {
        q
    }
}

/// 32-bit absolute value with −2^31 clamped to 2^31−1.
fn abs_sat32(x: i32) -> i32 {
    if x < 0 {
        if x == i32::MIN {
            i32::MAX
        } else {
            -x
        }
    } else {
        x
    }
}

/// The [`shift32_al2b2`] variant whose result is truncated to 16 bits *after* the
/// shift, so the "shift right by more than 30" case yields the sign of the
/// operand (0 or −1) rather than its high half.
fn shift_to_i16(v: i32, s: i16) -> i16 {
    if s < 0 {
        if s < -0x1e {
            (v >> 31) as i16
        } else {
            high16_batch3(v >> ((-s) as u32 & 0x1f))
        }
    } else {
        high16_batch3(((v as u32) << (s as u32 & 0x1f)) as i32)
    }
}

// ======= pulled in from batch4 (encoder-only) =======

// ---------------------------------------------------------------------------
// array reductions
// ---------------------------------------------------------------------------

/// `0x103172c0` — sum of `n` 16-bit words into a 32-bit accumulator.
///
/// `n` is signed: a non-positive count yields 0. No saturation.
pub fn sum_i16(x: &[i16], n: i16) -> i32 {
    let mut acc = 0i32;
    if n > 0 {
        for &v in &x[..n as usize] {
            acc = acc.wrapping_add(v as i32);
        }
    }
    acc
}

/// `0x10317650` — block exponent of `n` 32-bit words: the negated
/// normalisation shift of the largest magnitude, or 0 for an all-zero block.
///
/// `abs(0x80000000)` saturates to `0x7fffffff`. The sign is flipped relative
/// to [`cplx_block_exp`] because callers add this to an exponent rather than
/// using it as a left shift.
pub fn i32_block_exp(x: &[i32], n: i16) -> i32 {
    let mut max = 0i32;
    if n > 0 {
        for i in 0..n as usize {
            let mut v = x[i];
            if v < 0 {
                v = if v == i32::MIN {
                    i32::MAX
                } else {
                    v.wrapping_neg()
                };
            }
            if max < v {
                max = v;
            }
        }
    }
    if max == 0 {
        return 0;
    }
    -((norm_shift_batch4(max) as u16) as i32)
}

/// `0x10310ba0` — scale eight spectral amplitudes according to two voicing
/// bits, producing Q31 accumulators.
///
/// For each of the 8 words: if bit `(a + 2 + b)` and bit `(b + 2)` of `vmask`
/// are both set, the corresponding `aux` value selects a gain — below `0x1127`
/// a fixed 0xab86 (an unsigned 43910, confirmed in the disassembly, not a
/// negative constant), between `0x1127` and `0x1999` the value `aux*5`
/// truncated to 16 bits, and above that no scaling. Otherwise the input passes
/// through as `src << 16`. Shift counts are taken modulo 32 on a 16-bit
/// register, so out-of-range bit numbers select the sign bit.
pub fn voiced_scale8(dst: &mut [i32], src: &[i16], aux: &[i16], vmask: i16, a: i16, b: i16) {
    let sh1 = (a.wrapping_add(2).wrapping_add(b) as u16 & 0x1f) as u32;
    let sh2 = (b.wrapping_add(2) as u16 & 0x1f) as u32;
    let m = vmask as i32;
    let gated = ((m >> sh1) & 1) != 0 && ((m >> sh2) & 1) != 0;

    for i in 0..8usize {
        let s = src[i] as i32;
        let v = if !gated {
            s << 16
        } else {
            let t = aux[i];
            if t < 0x1127 {
                s.wrapping_mul(0xab86)
            } else if t > 0x1999 {
                s << 16
            } else {
                (t.wrapping_mul(5) as i32).wrapping_mul(s).wrapping_mul(2)
            }
        };
        dst[i] = v;
    }
}

/// `0x10317610` — squared magnitude of `n` interleaved complex words,
/// doubled: `dst[i] = 2 * (re^2 + im^2)`.
///
/// The doubling renormalises the Q15xQ15 products to Q31. Wraps rather than
/// saturates.
pub fn cplx_mag_sq_x2(dst: &mut [i32], src: &[i16], n: i16) {
    if n <= 0 {
        return;
    }
    for i in 0..n as usize {
        let re = src[2 * i] as i32;
        let im = src[2 * i + 1] as i32;
        dst[i] = re
            .wrapping_mul(re)
            .wrapping_add(im.wrapping_mul(im))
            .wrapping_mul(2);
    }
}

/// `0x1031adf0` — the 16-band, 7-tap interpolation filter used to expand a
/// coarse spectral grid into the synthesis grid.
///
/// Sixteen output rows of `n` columns each (row stride 11 words in the output,
/// starting at word `off`). Every output is a symmetric 7-tap FIR
/// `[c0,c1,c2,c3,c2,c1,c0]` over `src` sampled at 16-word spacing, phase-offset
/// by the row index and hopped 32 words per column. Products are doubled Q31
/// words that may individually wrap, but they are summed in 48 bits — that is
/// what Ghidra's `CARRY4` chain encodes — then rounded with `+ 0x10000` and
/// shifted right by 17 to a 16-bit result.
pub fn interp_filter(out: &mut [i16], coef: &[i16], src: &[i16], n: i16, off: i16) {
    let c = [
        coef[0] as i32,
        coef[1] as i32,
        coef[2] as i32,
        coef[3] as i32,
    ];
    let taps: [(isize, usize); 7] = [
        (-64, 0),
        (-48, 1),
        (-32, 2),
        (-16, 3),
        (32, 0),
        (16, 1),
        (0, 2),
    ];
    let mut p = off as isize;
    for row in 0..16isize {
        if n > 0 {
            for col in 0..n as isize {
                let b = 64 + row + 32 * col;
                let mut sum = 0i64;
                for &(d, ci) in taps.iter() {
                    let prod = (src[(b + d) as usize] as i32)
                        .wrapping_mul(c[ci])
                        .wrapping_mul(2);
                    sum += prod as i64;
                }
                out[p as usize] = ((sum + 0x10000) >> 17) as i16;
                p += 1;
            }
        }
        p += (0xbi16).wrapping_sub(n) as isize;
    }
}

/// `0x10317360` — index of the largest of `n` 32-bit words.
///
/// Strictly-greater comparison, so ties keep the earliest index. Fewer than
/// two elements gives 0. The original returns the index in `ax` with leftover
/// data in the high half of `eax`; only the low 16 bits are meaningful.
pub fn argmax_i32(x: &[i32], n: i16) -> i16 {
    let mut best = 0i16;
    if n > 1 {
        let mut cur = x[0];
        for i in 1..n {
            let v = x[i as usize];
            if cur < v {
                cur = v;
                best = i;
            }
        }
    }
    best
}

/// Twiddle table at `0x1037f978`: `round(32768 * cos(pi * k / 256))` clamped to
/// `i16`, 512 entries — a full period. The FFT stage reads cosines from index 0
/// and sines from index 128, so the "sine" table is really `-sin`.
pub const TWIDDLE_batch4: [i16; 512] = [
    32767, 32766, 32758, 32746, 32729, 32706, 32679, 32647, 32610, 32568, 32522, 32470, 32413,
    32352, 32286, 32214, 32138, 32058, 31972, 31881, 31786, 31686, 31581, 31471, 31357, 31238,
    31114, 30986, 30853, 30715, 30572, 30425, 30274, 30118, 29957, 29792, 29622, 29448, 29269,
    29086, 28899, 28707, 28511, 28311, 28106, 27897, 27684, 27467, 27246, 27020, 26791, 26557,
    26320, 26078, 25833, 25583, 25330, 25073, 24812, 24548, 24279, 24008, 23732, 23453, 23170,
    22884, 22595, 22302, 22006, 21706, 21403, 21097, 20788, 20475, 20160, 19841, 19520, 19195,
    18868, 18538, 18205, 17869, 17531, 17190, 16846, 16500, 16151, 15800, 15447, 15091, 14733,
    14373, 14010, 13646, 13279, 12910, 12540, 12167, 11793, 11417, 11039, 10660, 10279, 9896, 9512,
    9127, 8740, 8351, 7962, 7571, 7180, 6787, 6393, 5998, 5602, 5205, 4808, 4410, 4011, 3612, 3212,
    2811, 2411, 2009, 1608, 1206, 804, 402, 0, -402, -804, -1206, -1608, -2009, -2411, -2811,
    -3212, -3612, -4011, -4410, -4808, -5205, -5602, -5998, -6393, -6787, -7180, -7571, -7962,
    -8351, -8740, -9127, -9512, -9896, -10279, -10660, -11039, -11417, -11793, -12167, -12540,
    -12910, -13279, -13646, -14010, -14373, -14733, -15091, -15447, -15800, -16151, -16500, -16846,
    -17190, -17531, -17869, -18205, -18538, -18868, -19195, -19520, -19841, -20160, -20475, -20788,
    -21097, -21403, -21706, -22006, -22302, -22595, -22884, -23170, -23453, -23732, -24008, -24279,
    -24548, -24812, -25073, -25330, -25583, -25833, -26078, -26320, -26557, -26791, -27020, -27246,
    -27467, -27684, -27897, -28106, -28311, -28511, -28707, -28899, -29086, -29269, -29448, -29622,
    -29792, -29957, -30118, -30274, -30425, -30572, -30715, -30853, -30986, -31114, -31238, -31357,
    -31471, -31581, -31686, -31786, -31881, -31972, -32058, -32138, -32214, -32286, -32352, -32413,
    -32470, -32522, -32568, -32610, -32647, -32679, -32706, -32729, -32746, -32758, -32766, -32767,
    -32766, -32758, -32746, -32729, -32706, -32679, -32647, -32610, -32568, -32522, -32470, -32413,
    -32352, -32286, -32214, -32138, -32058, -31972, -31881, -31786, -31686, -31581, -31471, -31357,
    -31238, -31114, -30986, -30853, -30715, -30572, -30425, -30274, -30118, -29957, -29792, -29622,
    -29448, -29269, -29086, -28899, -28707, -28511, -28311, -28106, -27897, -27684, -27467, -27246,
    -27020, -26791, -26557, -26320, -26078, -25833, -25583, -25330, -25073, -24812, -24548, -24279,
    -24008, -23732, -23453, -23170, -22884, -22595, -22302, -22006, -21706, -21403, -21097, -20788,
    -20475, -20160, -19841, -19520, -19195, -18868, -18538, -18205, -17869, -17531, -17190, -16846,
    -16500, -16151, -15800, -15447, -15091, -14733, -14373, -14010, -13646, -13279, -12910, -12540,
    -12167, -11793, -11417, -11039, -10660, -10279, -9896, -9512, -9127, -8740, -8351, -7962,
    -7571, -7180, -6787, -6393, -5998, -5602, -5205, -4808, -4410, -4011, -3612, -3212, -2811,
    -2411, -2009, -1608, -1206, -804, -402, 0, 402, 804, 1206, 1608, 2009, 2411, 2811, 3212, 3612,
    4011, 4410, 4808, 5205, 5602, 5998, 6393, 6787, 7180, 7571, 7962, 8351, 8740, 9127, 9512, 9896,
    10279, 10660, 11039, 11417, 11793, 12167, 12540, 12910, 13279, 13646, 14010, 14373, 14733,
    15091, 15447, 15800, 16151, 16500, 16846, 17190, 17531, 17869, 18205, 18538, 18868, 19195,
    19520, 19841, 20160, 20475, 20788, 21097, 21403, 21706, 22006, 22302, 22595, 22884, 23170,
    23453, 23732, 24008, 24279, 24548, 24812, 25073, 25330, 25583, 25833, 26078, 26320, 26557,
    26791, 27020, 27246, 27467, 27684, 27897, 28106, 28311, 28511, 28707, 28899, 29086, 29269,
    29448, 29622, 29792, 29957, 30118, 30274, 30425, 30572, 30715, 30853, 30986, 31114, 31238,
    31357, 31471, 31581, 31686, 31786, 31881, 31972, 32058, 32138, 32214, 32286, 32352, 32413,
    32470, 32522, 32568, 32610, 32647, 32679, 32706, 32729, 32746, 32758, 32766,
];

// ======= pulled in from l1_batch1 (encoder-only) =======

/// `0x1030f510` — raise eight band values to a noise floor derived from their
/// block exponents.
///
/// The floor is `0.8 · 2^(-3-e)` in Q15 — the constant `0x66660000` is 0.8 in
/// Q31 — where `e` is one of three exponents in `levels`, chosen by band index
/// and mode:
///
/// * `mode != 0` and band ≥ 5: `alt` is floored at `levels[2]`'s floor and
///   `cur` at `levels[1]`'s, unconditionally.
/// * otherwise: `cur` is floored at `levels[1]`'s floor, but only if `src` for
///   that band exceeds `levels[0]`'s floor **and** the state word at
///   `state[7]` (byte `0xe`) is below `0xccc`.
///
/// The gate word is read once, before the loop.
pub fn band_floor(
    state: &[i16],
    src: &[i16],
    cur: &mut [i16],
    alt: &mut [i16],
    levels: &[i16],
    mode: i16,
) {
    const Q: i32 = 0x6666_0000u32 as i32;
    let gate = state[7] < 0xccc;
    for i in 0..8usize {
        if mode != 0 && i >= 5 {
            let v = (shl_sat(Q, (-3i16).wrapping_sub(levels[2])) >> 16) as i16;
            if alt[i] < v {
                alt[i] = v;
            }
            let v = (shl_sat(Q, (-3i16).wrapping_sub(levels[1])) >> 16) as i16;
            if cur[i] < v {
                cur[i] = v;
            }
        } else {
            let v0 = (shl_sat(Q, (-3i16).wrapping_sub(levels[0])) >> 16) as i16;
            let ok = gate && src[i] > v0;
            let v1 = (shl_sat(Q, (-3i16).wrapping_sub(levels[1])) >> 16) as i16;
            if cur[i] < v1 && ok {
                cur[i] = v1;
            }
        }
    }
}

/// `0x1031dbc0` — fold a new observation into a block-floating-point running
/// mean and reset the confidence pair that goes with it.
///
/// `*val`/`*exp` is the accumulator. The new term `(*val · k)` is normalised
/// and [`add_norm`]-ed with the caller-supplied `(add, add_exp)`; the sum is
/// then divided by `k + 1` so the accumulator holds a mean over `k+1` terms.
/// `*conf` is slammed to `0x7fff` and `*age` to 5 — the "fully confident, five
/// frames to live" reset that accompanies a new estimate.
///
/// The return value is `passthru` in `ax`; the top half of `eax` is a leftover
/// pointer, so only the low 16 bits are meaningful.
pub fn update_pitch_track(
    val: &mut i16,
    exp: &mut u16,
    conf: &mut i16,
    age: &mut i16,
    add: i16,
    add_exp: i16,
    passthru: i16,
    k: i16,
) -> i16 {
    let m = (*val as i32).wrapping_mul(k as i32).wrapping_mul(2);
    let n = if m == 0 {
        0u16
    } else {
        norm_exp_l1_batch1(m) as u16
    };
    let e = (*exp as i32).wrapping_add(0xfi32.wrapping_sub(n as i32));

    let mut oe: i16 = 0;
    let r = add_norm(
        (((m as u32) << (n as u32 & 0x1f)) as i32 >> 16) as i16,
        e as i16,
        add,
        add_exp,
        &mut oe,
    );

    let m2 = ((k as i32) + 1) << 16;
    let n2 = if m2 == 0 {
        0u16
    } else {
        norm_exp_l1_batch1(m2) as u16
    };
    let mut oe2 = oe;
    let q = div_norm(
        (r as i32) << 16,
        oe,
        (((m2 as u32) << (n2 as u32 & 0x1f)) as i32 >> 16) as i16,
        (0xfi32.wrapping_sub(n2 as i32)) as i16,
        &mut oe2,
    );

    *val = q;
    *exp = oe2 as u16;
    *conf = 0x7fff;
    *age = 5;
    passthru
}

/// `0x1031b670` — one step of a running ratio of two block-floating-point
/// accumulators.
///
/// `st` is a ten-word state: `[0]`/`[1]` the ratio and its exponent, `[2]`/`[3]`
/// and `[4]`/`[5]` two input gains, `[6]`/`[7]` the numerator accumulator and
/// `[8]`/`[9]` the denominator accumulator.
///
/// Each call folds a new sample `x` in: `x·st[2]` is normalised, then scaled by
/// `st[4]` and accumulated into the numerator, while `x·st[2]·x` — the squared
/// term — is accumulated into the denominator. The ratio is only re-divided
/// when the denominator is strictly positive, which is the guard against
/// [`div_norm`]'s zero-divisor fault.
pub fn update_ratio_accum(x: i16, st: &mut [i16]) {
    let p = x as i32;

    let m1 = (st[2] as i32).wrapping_mul(p).wrapping_mul(2);
    let n1 = if m1 == 0 {
        0u16
    } else {
        norm_exp_l1_batch1(m1) as u16
    };
    let ex = st[3].wrapping_sub(n1 as i16).wrapping_add(0xf);
    let v = (((m1 as u32) << (n1 as u32 & 0x1f)) as i32 >> 16) as i16;

    let m2 = (st[4] as i32).wrapping_mul(v as i32).wrapping_mul(2);
    let n2 = if m2 == 0 {
        0u16
    } else {
        norm_exp_l1_batch1(m2) as u16
    };
    let e2 = st[5].wrapping_sub(n2 as i16).wrapping_add(ex);
    let mant2 = (((m2 as u32) << (n2 as u32 & 0x1f)) as i32 >> 16) as i16;
    let mut e7 = st[7];
    let r = add_norm(st[6], st[7], mant2, e2, &mut e7);
    st[7] = e7;
    st[6] = r;

    let m3 = (v as i32).wrapping_mul(p).wrapping_mul(2);
    let n3 = if m3 == 0 {
        0u16
    } else {
        norm_exp_l1_batch1(m3) as u16
    };
    let e3 = (ex as u16 as i32).wrapping_sub(n3 as i32).wrapping_add(0xf) as i16;
    let mant3 = (((m3 as u32) << (n3 as u32 & 0x1f)) as i32 >> 16) as i16;
    let mut e9 = st[9];
    let den = add_norm(st[8], st[9], mant3, e3, &mut e9);
    st[9] = e9;
    st[8] = den;

    if den > 0 {
        let mut e1 = st[1];
        let q = div_norm((st[6] as i32) << 16, st[7], den, st[9], &mut e1);
        st[1] = e1;
        st[0] = q;
    }
}

/// `0x1030d960` — multiply a signal segment by a symmetric window stored as its
/// right half, writing the result outwards from the centre.
///
/// `win[h]` is the window's centre (`h = count / 2`) and `win[h-1-j]` walks out
/// towards the edge; the same coefficient scales `input[h+odd+j]` and
/// `input[h-1-j]`, which is what makes the window symmetric with only half of
/// it stored. An odd `count` emits the centre tap first.
///
/// The two halves are written to opposite ends of `out`: forward from
/// `out_off` (or from 0 when `out_off == 0`) and backward from `out_off - 1`
/// (or `len - 1`). The gap left in the middle, `len - count` words, is zeroed —
/// this is the zero-padding that precedes the FFT.
///
/// `out_off` must be 0 or at least `h`, otherwise the backward half runs off
/// the front of the buffer.
pub fn apply_sym_window(
    out: &mut [i16],
    input: &[i16],
    len: i16,
    out_off: i16,
    win: &[i16],
    count: i16,
) {
    let h = ((count >> 1) as u16) as i16 as isize;
    let (mut fwd, mut bwd): (isize, isize) = if out_off == 0 {
        (0, len as isize - 1)
    } else {
        (out_off as isize, out_off as isize - 1)
    };

    let mut in_fwd = h;
    if count & 1 != 0 {
        out[fwd as usize] = mul_round(input[h as usize], win[h as usize]);
        in_fwd = h + 1;
        fwd += 1;
    }
    if h > 0 {
        for j in 0..h {
            let c = win[(h - 1 - j) as usize];
            out[fwd as usize] = mul_round(input[(in_fwd + j) as usize], c);
            fwd += 1;
            out[bwd as usize] = mul_round(input[(h - 1 - j) as usize], c);
            bwd -= 1;
        }
    }
    fill16(&mut out[fwd as usize..], 0, len.wrapping_sub(count));
}

// ---------------------------------------------------------------------------
// pitch selection
// ---------------------------------------------------------------------------

/// `0x10310540` — pick one of five pitch candidates and return it doubled.
///
/// Three selection rules are tried in order:
///
/// 1. If `idx >= 1` and [`thresh5`] accepts the triple around `idx`, take
///    `cand[idx]` (doubled once here, and again by the common tail).
/// 2. Otherwise, if the error `err[idx]` is above `0x3333` the candidate is not
///    trusted: fall back to `alt`, or — if that is no better — to the index of
///    the smallest error among the five.
/// 3. If even that is above `0x3333`, ignore the errors entirely and take the
///    candidate closest in value to `target`.
///
/// The tail is `sat_shl(x << 16, 1) >> 16`, i.e. a saturating doubling.
///
/// Ghidra reports `param_4 = extraout_DX` after the [`thresh5`] call, implying
/// the index is clobbered. It is not: `thresh5` never writes `edx`, so the
/// index survives. Transcribing the artefact literally gives a garbage index.
pub fn pitch_select(cand: &[i16], err: &[i16], target: i16, idx: i16, alt: i16) -> i32 {
    let chosen: i16;
    'done: {
        if idx >= 1 {
            let i = idx as usize;
            if thresh5(err[i + 1], err[i], err[i - 1], cand[i], cand[i - 1]) != 0 {
                chosen = (((cand[i] as i32) << 17) >> 16) as i16;
                break 'done;
            }
        }
        let i = idx as usize;
        if err[i] > 0x3333 {
            let mut j = alt;
            if err[alt as usize] >= 0x3333 {
                j = 0;
                for k in 1..5i16 {
                    if err[k as usize] < err[j as usize] {
                        j = k;
                    }
                }
            }
            if err[j as usize] >= 0x3333 {
                let mut best = 0i16;
                for k in 1..5i16 {
                    let dk = (cand[k as usize] as i32).wrapping_sub(target as i32).abs();
                    let db = (cand[best as usize] as i32)
                        .wrapping_sub(target as i32)
                        .abs();
                    if dk < db {
                        best = k;
                    }
                }
                chosen = cand[best as usize];
            } else {
                chosen = cand[j as usize];
            }
        } else {
            chosen = cand[i];
        }
    }
    shl_sat((chosen as i32) << 16, 1) >> 16
}

/// `0x1031ea20` — attack/release smoother for the level tracker at state words
/// `0x24` (smoothed, byte `0x48`) and `0x25` (instantaneous, byte `0x4a`).
///
/// Three coefficient pairs are selected by how loud the frame is and whether
/// the level is rising:
///
/// | condition | on new | on old | exponent |
/// |---|---|---|---|
/// | rising, `level > 0x4000`, new > −0x2800 | `0xcccc` (0.8) | `0xe666` (0.9) | 4 |
/// | rising, or `level > 0x2000` | `0xa3d8` (0.64) | `0xfd70` (0.99) | 1 |
/// | otherwise | `0x8312` (0.51) | `0xffbe` (~1.0) | −2 |
///
/// The multipliers are **unsigned Q16 fractions**; the immediates are 32-bit
/// `imul` constants, not sign-extended words. A falling level that is already
/// below −0x2800 is left exactly where it is (a floor, not a decay).
///
/// Each pair is a one-pole smoother `y = α·new + (1−α)·old` in disguise: the
/// exponent gap between the two terms is 3, 6 and 9 bits, so the new-sample
/// weights are `0.8/8 = 0.1`, `0.64/64 = 0.01` and `0.512/512 = 0.001` against
/// old-sample weights of `0.9`, `0.99` and `0.999`. The three paths are a fast
/// attack, a normal track and a slow decay, and each pair sums to exactly 1 —
/// which is why `new == old` is a fixed point of all three.
///
/// The two terms are combined by [`add_norm`] and the result is rescaled back
/// to Q15 by the exponent it reports.
pub fn smooth_level(state: &mut [i16], level: i16) {
    let cur = state[0x25];
    let prev = state[0x24];

    let (m1, ebase, m2): (i32, i32, i32) = if cur > prev {
        if level > 0x4000 && cur > -0x2800 {
            (
                (cur as i32).wrapping_mul(0xcccc),
                4,
                (prev as i32).wrapping_mul(0xe666),
            )
        } else {
            (
                (cur as i32).wrapping_mul(0xa3d8),
                1,
                (prev as i32).wrapping_mul(0xfd70),
            )
        }
    } else if level > 0x2000 {
        (
            (cur as i32).wrapping_mul(0xa3d8),
            1,
            (prev as i32).wrapping_mul(0xfd70),
        )
    } else if prev <= -0x2800 {
        state[0x24] = prev;
        return;
    } else {
        (
            (cur as i32).wrapping_mul(0x8312),
            -2,
            (prev as i32).wrapping_mul(0xffbe),
        )
    };

    let n1 = if m1 == 0 {
        0u16
    } else {
        norm_exp_l1_batch1(m1) as u16
    };
    let e1 = ebase.wrapping_sub(n1 as i32) as i16;
    let a = (((m1 as u32) << (n1 as u32 & 0x1f)) as i32 >> 16) as i16;

    let n2 = if m2 == 0 {
        0u16
    } else {
        norm_exp_l1_batch1(m2) as u16
    };
    let b = (((m2 as u32) << (n2 as u32 & 0x1f)) as i32 >> 16) as i16;

    let mut e: i16 = 0;
    let r = add_norm(a, e1, b, (7i32.wrapping_sub(n2 as i32)) as i16, &mut e);
    state[0x24] = (shift32_al3b2((r as i32) << 16, e.wrapping_sub(7)) >> 16) as i16;
}

/// `0x1031c8f0` — align an array of 32-bit values to a common block exponent.
///
/// Finds the largest of `exps` with [`vec_max`], then shifts each value by its
/// own exponent minus that maximum, in place over `dst` (reading from `src`).
/// Returns the maximum in the low half of `eax`; the high half is whatever the
/// last shifted value's top word was, which is why the return type is `u32`.
///
/// `n` must be at least 1: `vec_max` reads element 0 unconditionally.
pub fn align_to_max_exp(dst: &mut [i32], src: &[i32], exps: &[i16], n: u16) -> u32 {
    let mx = vec_max(exps, n as i16);
    let mut last = 0i32;
    if (n as i16) > 0 {
        for i in 0..n as usize {
            let v = shift32_al3b2(src[i], exps[i].wrapping_sub(mx));
            dst[i] = v;
            last = v;
        }
    }
    ((last as u32) & 0xffff_0000) | (mx as u16 as u32)
}

/// `0x103201d0` — integrate a magnitude spectrum into per-harmonic energies.
///
/// `step` is the fundamental in units of `1/1024` of a bin (`inc = step · 64`
/// as a Q16 bin position). Successive harmonic band edges are the running
/// positions `a`, `a+inc`, `a+2·inc`, …; for each band the whole bins strictly
/// inside it are summed in 64 bits and the two partial bins at the edges are
/// added with a Q15 fractional weight — the low edge's contribution is
/// *subtracted* from its bin (`v − v·frac`) and the high edge's is added.
///
/// Each band sum is normalised by [`norm64`]; the low word of the normalised
/// value goes to `mags[k]` and the *negated* shift to `exps[k]`. Bands are
/// produced until the upper edge reaches bin 128, and the count is returned.
///
/// The decompilation's `((short)(u >> 16) << 16) | (u & 0xffff)` on each edge
/// is an identity — the compiler re-materialising a value it had split into
/// halves — so the edges are simply `a += inc`.
///
/// `step` must be strictly positive, otherwise the edge never reaches bin 128
/// and the loop does not terminate.
pub fn band_energy(mags: &mut [i32], exps: &mut [i16], step: i16, spec: &[i32]) -> i16 {
    let inc = ((step as i32) << 9) >> 3;
    let mut a = (((step as i32) << 9) >> 4).wrapping_add(0x8000);
    let mut b = a.wrapping_add(inc);
    let mut k: i16 = 0;

    while ((b >> 16) as i16) < 0x80 {
        let i0 = ((a as u32) >> 16) as u16 as i16;
        let i1 = ((b as u32) >> 16) as u16;
        let cnt = i1.wrapping_sub(i0 as u16) as i16;

        // Low edge: remove the fraction of the bin that lies below the band.
        let f0 = ((a as u32 & 0xffff) >> 1) as i32;
        let v = spec[i0 as usize];
        let p = (((v as u32 & 0xffff) as i32).wrapping_mul(f0) >> 15)
            .wrapping_add(((v >> 16) as i16 as i32).wrapping_mul(f0).wrapping_mul(2));
        let p = if p == i32::MIN {
            i32::MAX
        } else {
            p.wrapping_neg()
        };
        let mut acc = v.wrapping_add(p) as i64;

        // High edge: add the fraction of the last bin that lies inside it.
        let f1 = (((b as u32) >> 1) & 0x7fff) as i32;
        let v2 = spec[(i0 as usize).wrapping_add(cnt as u16 as usize)];
        let q = (((v2 as u32 & 0xffff) as i32).wrapping_mul(f1) >> 15)
            .wrapping_add(((v2 >> 16) as i16 as i32).wrapping_mul(f1).wrapping_mul(2));
        acc = acc.wrapping_add(q as i64);

        if cnt > 1 {
            for j in 1..cnt as usize {
                acc = acc.wrapping_add(spec[i0 as usize + j] as i64);
            }
        }

        let s = norm64(acc as u32, (acc >> 32) as u32) as i16;
        let sh = shift64_l1_batch1(acc, s);
        exps[k as usize] = s.wrapping_neg();
        mags[k as usize] = sh as i32;
        k = k.wrapping_add(1);

        a = b;
        b = b.wrapping_add(inc);
    }
    k
}

/// `0x10314660` — windowed complex correlation of two interleaved spectra.
///
/// `a` and `b` hold `n` interleaved `(re, im)` pairs and `w` a real window.
/// The window is applied to `a` first — `(a·w) << 2 >> 16`, truncated back to
/// 16 bits — and the product with `b` is then accumulated in **two 64-bit**
/// accumulators, one for `re·re + im·im` and one for `re·im − im·re`. Each
/// individual product is formed as a *wrapping 32-bit* `x·y·2` and only then
/// sign-extended to 64 bits, so a product that overflows `i32` wraps rather
/// than widening.
///
/// Each accumulator is normalised by [`norm64`], its exponent set to
/// `exp_a − shift + exp_b`, and both are then brought to the larger of the two
/// exponents. `out[0]` and `out[1]` receive the high words.
///
/// The return value's low half is the shared exponent; the high half is the
/// sign fill of `out[1]`, which is why it is returned as `u32`.
#[allow(clippy::too_many_arguments)]
pub fn complex_correlate(
    out: &mut [i16],
    w: &[i16],
    a: &[i16],
    exp_a: i16,
    b: &[i16],
    exp_b: i16,
    n: u16,
) -> u32 {
    let mut acc1: i64 = 0;
    let mut acc2: i64 = 0;

    if (n as i16) > 0 {
        for i in 0..n as usize {
            let (ar, ai) = (a[2 * i], a[2 * i + 1]);
            let (br, bi) = (b[2 * i], b[2 * i + 1]);
            let wi = w[i] as i32;

            let x = ((((ar as i32).wrapping_mul(wi)) << 2) >> 16) as i16 as i32;
            let y = ((((ai as i32).wrapping_mul(wi)) << 2) >> 16) as i16 as i32;

            let t1 = (bi as i32).wrapping_mul(y).wrapping_mul(2);
            let t2 = (br as i32).wrapping_mul(x).wrapping_mul(2);
            acc1 = acc1.wrapping_add((t1 as i64).wrapping_add(t2 as i64));

            let u1 = (br as i32).wrapping_mul(y).wrapping_mul(2);
            let u2 = (bi as i32).wrapping_mul(x).wrapping_mul(2);
            acc2 = acc2.wrapping_add((u1 as i64).wrapping_sub(u2 as i64));
        }
    }

    let s1 = norm64(acc1 as u32, (acc1 >> 32) as u32) as i16;
    let sh1 = shift64_l1_batch1(acc1, s1);
    let e1 = (exp_a as i32)
        .wrapping_sub(s1 as u16 as i32)
        .wrapping_add(exp_b as i32) as i16;

    let s2 = norm64(acc2 as u32, (acc2 >> 32) as u32) as i16;
    let sh2 = shift64_l1_batch1(acc2, s2);
    let e2 = (exp_a as i32)
        .wrapping_sub(s2 as u16 as i32)
        .wrapping_add(exp_b as i32) as i16;

    let mx = if e1 > e2 { e1 } else { e2 };
    let r1 = shift64_l1_batch1(sh1, e1.wrapping_sub(mx));
    let r2 = shift64_l1_batch1(sh2, e2.wrapping_sub(mx));

    out[0] = ((r1 as i32) >> 16) as i16;
    out[1] = ((r2 as i32) >> 16) as i16;
    ((((r2 as i32) >> 16) as u32) & 0xffff_0000) | (mx as u16 as u32)
}

/// `0x1030e530` — advance sixteen 49-word band histories by `n` samples.
///
/// The buffer is a 16 × 49 `i16` matrix. Each row is shifted left by `n` words
/// and the vacated tail is filled from `src`, which is read with a stride of 11
/// words per row. This is the analysis delay line's per-frame update.
///
/// `n` must lie in `0..=0x31`: the original computes `0x31 - n` as a word count
/// and indexes the row at that offset.
pub fn shift_band_history(buf: &mut [i16], src: &[i16], n: i16) {
    let keep = 0x31i16.wrapping_sub(n);
    let mut row = 0usize;
    let mut s = 0usize;
    for _ in 0..16 {
        if keep > 0 {
            let k = keep as usize;
            let off = n as usize;
            buf.copy_within(row + off..row + off + k, row);
        }
        copy16_batch3(&mut buf[row + (0x31 - n as usize)..], &src[s..], n);
        row += 0x31;
        s += 0xb;
    }
}

// ---------------------------------------------------------------------------
// windowing and filtering
// ---------------------------------------------------------------------------

#[inline]
fn mul_round(a: i16, b: i16) -> i16 {
    (((a as i32).wrapping_mul(b as i32).wrapping_mul(2)).wrapping_add(0x8000) >> 16) as i16
}

/// 64-bit variable shift as the original spells it: a non-negative count goes
/// to `__allshl`, a negative one is negated and goes to `__allshr`. Both take
/// the count in `cl`, so only its low byte is significant.
#[inline]
fn shift64_l1_batch1(v: i64, c: i16) -> i64 {
    if c < 0 {
        allshr(v, c.wrapping_neg() as u8)
    } else {
        allshl_l1_batch1(v, c as u8)
    }
}

/// MSVC's `__allshl` (`0x10305930`) — the mirror of [`allshr`]. Not in the
/// ported map because no layer-0 batch reached it; it is three instructions and
/// is reproduced here rather than depending on another batch's file.
fn allshl_l1_batch1(v: i64, count: u8) -> i64 {
    let hi = (v >> 32) as u32;
    let lo = v as u32;
    if count >= 0x40 {
        return 0;
    }
    if count >= 0x20 {
        let n = count & 0x1f;
        return (((lo << n) as u64) << 32) as i64;
    }
    let n = count & 0x1f;
    let (new_hi, new_lo) = if n == 0 {
        (hi, lo)
    } else {
        ((hi << n) | (lo >> (32 - n)), lo << n)
    };
    (((new_hi as u64) << 32) | new_lo as u64) as i64
}

// ======= pulled in from l1_batch2 (encoder-only) =======

/// `0x1031bed0` — sum a 32-bit spectrum over successive harmonic bands.
///
/// The band width is `(den << 9) · 2^sh`; the first band centre is placed at
/// `1.0 - width/4` plus `q = num/den` whole widths, and `*start_out` records
/// that Q16 position. Bands then advance by half a width at a time, each
/// summing the bins between its two integer edges, until the position passes
/// bin `0x80`; whatever remains up to bin `0x80` is added in one final block.
/// The 64-bit total is normalised and returned as a mantissa with
/// `*exp_out = -shift`.
///
/// Domain: the walk must start at a non-negative bin, so `den`, `sh` and `num`
/// have to keep the Q16 position positive; the original indexes unguarded.
pub fn band_energy_walk(
    exp_out: &mut i16,
    table: &[i32],
    num: i32,
    den: i16,
    sh: i16,
    start_out: &mut i32,
) -> i16 {
    let q = div32(num, den) as u16;
    let width = shift_i32_l1_batch2((den as i32) << 9, sh);
    let half = width >> 1;
    let quarter = width >> 2;
    let mut acc32 = if quarter == i32::MIN {
        i32::MAX
    } else {
        quarter.wrapping_neg()
    }
    .wrapping_add(0x10000);
    if (q as i16) > 0 {
        acc32 = acc32.wrapping_add((q as i32).wrapping_mul(width));
    }

    let mut pos = high16_al2b1(acc32) as u16;
    *start_out = acc32;
    acc32 = acc32.wrapping_add(half);
    let mut next = high16_al2b1(acc32) as u16;
    let mut span = next.wrapping_sub(pos);

    let (mut lo, mut hi) = (0u32, 0i32);
    while (next as i16) <= 0x80 {
        if (span as i16) > 0 {
            let base = (pos as i16) as i32 as usize;
            for k in 0..span as usize {
                acc64(&mut lo, &mut hi, table[base + k] as u32);
            }
        }
        let mid = acc32.wrapping_add(half);
        acc32 = mid.wrapping_add(half);
        pos = high16_al2b1(mid) as u16;
        next = high16_al2b1(acc32) as u16;
        span = next.wrapping_sub(pos);
    }
    if (pos as i16) < 0x80 {
        let cnt = 0x80u16.wrapping_sub(pos);
        if (cnt as i16) > 0 {
            let base = (pos as i16) as i32 as usize;
            for k in 0..cnt as usize {
                acc64(&mut lo, &mut hi, table[base + k] as u32);
            }
        }
    }

    let (s, low) = norm64_align(lo, hi as u32);
    *exp_out = -s;
    high16_al2b1(low)
}

/// `0x10312700` — codebook search over voicing patterns.
///
/// Three per-harmonic weight tables are precomputed from the two input vectors
/// (16 words each, read *backwards*, so table slot `k` corresponds to source
/// index `15 - k`):
///
/// ```text
/// t_unv[k] = (1 - (1-A)^2)·B^2 / 2 + (1-A)^2      "unvoiced"
/// t_mid[k] = (1 - A^2)·B^2 / 2 + A^2              "mixed"
/// t_voi[k] = (1-B)^2·(1 - A^2) / 2 + A^2          "voiced"
/// ```
///
/// where `A = w_a[j]`, `B = w_b[j]` and every product is the Q15 `x·y·2 >> 16`
/// idiom. Each codebook entry is a 32-bit word of 16 two-bit symbols; symbol 1
/// selects `t_unv`, 2 selects `t_voi` and 0/3 select `t_mid`. The selected
/// weights are dotted against `w_c` in 64 bits, normalised, and the entry with
/// the smallest (exponent, mantissa) wins — an exact zero returns immediately.
///
/// `mode == 0` scans indices `0 ..= 0x38` and then jumps to
/// `0x38 + 2^(nbits-1)`, covering `0x38 + 2^nbits` candidates at stride 1;
/// otherwise it scans `2^nbits` candidates at stride `2^(7-nbits)`, i.e. a
/// 128-entry book sub-sampled. `nbits <= 7`.
pub fn voicing_codebook_search(
    w_a: &[i16],
    w_b: &[i16],
    w_c: &[i16],
    book: &[u32],
    nbits: u8,
    mode: i16,
) -> u32 {
    let mut t_unv = [0i16; 16];
    let mut t_mid = [0i16; 16];
    let mut t_voi = [0i16; 16];

    for k in 0..16usize {
        let j = 15 - k;
        let a = w_a[j] as i32;
        let b = w_b[j] as i32;

        let a2 = high16_al2b1(a.wrapping_mul(a).wrapping_mul(2)) as i32;
        let na = ((0x7fff - a) as i16) as i32;
        let na2 = high16_al2b1(na.wrapping_mul(na).wrapping_mul(2)) as i32;
        let b2 = high16_al2b1(b.wrapping_mul(b).wrapping_mul(2)) as i32;
        let ia2 = ((0x7fff - a2) as i16) as i32;
        let nb = ((0x7fff - b) as i16) as i32;
        let nb2 = high16_al2b1(nb.wrapping_mul(nb).wrapping_mul(2)) as i32;
        let ina2 = ((0x7fff - na2) as i16) as i32;

        t_unv[k] = high16_al2b1(
            (ina2.wrapping_mul(b2).wrapping_mul(2) >> 1).wrapping_add(na2.wrapping_shl(16)),
        );
        t_mid[k] = high16_al2b1(
            (ia2.wrapping_mul(b2).wrapping_mul(2) >> 1).wrapping_add(a2.wrapping_shl(16)),
        );
        t_voi[k] = high16_al2b1(
            (nb2.wrapping_mul(ia2).wrapping_mul(2) >> 1).wrapping_add(a2.wrapping_shl(16)),
        );
    }

    let (limit, stride, wrap): (i16, i16, u16);
    if mode == 0 {
        limit = ((1u16 << (nbits & 0x1f)) as i16).wrapping_add(0x38);
        stride = 1;
        wrap = ((((limit as i32) - 0x38) >> 1) + 0x38) as u16;
    } else {
        let nb = nbits as i8 as i16;
        stride = (1u16 << ((7i16.wrapping_sub(nb)) as u32 & 0x1f)) as i16;
        limit = (1u16 << (nb as u32 & 0x1f)) as i16;
        wrap = 0; // never read: the wrap only fires when `mode == 0`
    }

    let mut best_exp: i16 = 0x7fff;
    let mut best_mant: i16 = 0x7fff;
    let mut best_idx: u32 = 0;

    if limit > 0 {
        let mut idx: u16 = 0;
        loop {
            if mode == 0 && (idx as i16) == 0x39 {
                idx = wrap;
            }
            let mut code = book[((idx as i16) as i32 * stride as i32) as usize] as i32;
            let (mut lo, mut hi) = (0u32, 0i32);
            for k in 0..16usize {
                let s = match (code as u16) & 3 {
                    1 => t_unv[k],
                    2 => t_voi[k],
                    _ => t_mid[k],
                };
                let prod = (w_c[15 - k] as i32).wrapping_mul(s as i32).wrapping_mul(2);
                acc64(&mut lo, &mut hi, prod as u32);
                code >>= 2;
            }
            let (s, low) = norm64_align(lo, hi as u32);
            let e = -s;
            let m = high16_al2b1(low);
            if m == 0 {
                return idx as u32;
            }
            if e <= best_exp && (e != best_exp || m < best_mant) {
                best_idx = idx as u32;
                best_mant = m;
                best_exp = e;
            }
            idx = idx.wrapping_add(1);
            if (idx as i16) >= limit {
                break;
            }
        }
    }
    best_idx
}

/// `0x1031dcb0` — leaky blend of a level towards a new observation.
///
/// Normally `new = 0.8·x·2^(x_exp-3) + 0.9·old`, i.e. the observation is
/// attenuated by a further factor of 8 relative to the retained level. If the
/// observation both exceeds the retained level *and* exceeds `-0x7000 @ 2^5`,
/// the weights switch to 0.5/0.5 at the observation's own exponent, so the
/// tracker snaps upwards quickly and decays slowly.
///
/// `0x6666`, `0x7333` and `0x4000` are Q15 0.8, 0.9 and 0.5. The two scratch
/// outputs are unconditionally set to `0x7fff` and `5`. The return value is
/// argument 7 passed straight through (only `ax` is meaningful).
pub fn blend_toward(
    val: &mut i16,
    exp: &mut u16,
    out_a: &mut i16,
    out_b: &mut i16,
    x: i16,
    x_exp: u16,
    ret: u16,
) -> u16 {
    let mut ca: i16 = 0x6666;
    let mut cb: i16 = 0x7333;
    let mut ea = x_exp.wrapping_sub(3);
    let v = *val;
    let ev = *exp;

    if compare_scaled(x, x_exp as i16, v, ev as i16) && compare_scaled(x, x_exp as i16, -0x7000, 5)
    {
        ca = 0x4000;
        cb = 0x4000;
        ea = x_exp;
    }

    let pa = (ca as i32).wrapping_mul(x as i32).wrapping_mul(2);
    let na = norm_exp_l1_batch2(pa as u32);
    let pb = (v as i32).wrapping_mul(cb as i32).wrapping_mul(2);
    let nb = norm_exp_l1_batch2(pb as u32);

    let mut e: i16 = 0;
    let m = add_norm(
        high16_al2b1(((pa as u32) << (na & 0x1f)) as i32),
        ea.wrapping_sub(na as u16) as i16,
        high16_al2b1(((pb as u32) << (nb & 0x1f)) as i32),
        ev.wrapping_sub(nb as u16) as i16,
        &mut e,
    );
    *val = m;
    *exp = e as u16;
    *out_a = 0x7fff;
    *out_b = 5;
    ret
}

/// `0x1031d3f0` — `log2` of a scaled product, in block floating point.
///
/// Forms `a·b` (Q15×Q15, doubled), normalises it into `(*mant, *exp)` with
/// `*exp = a_exp - norm + tail`, adds the term `(0x20 - n)·2^25 @ 2^-30` — a
/// small bias that grows as `n` falls below 32 — and takes the base-2 logarithm
/// of the sum. The logarithm is renormalised and returned as a 16-bit mantissa
/// with its exponent in `*exp2`.
///
/// A sum that cancels to zero returns `0xffff8000` (i.e. −1.0 in Q15, sign
/// extended) with `*exp2 = 15`.
pub fn log_scaled_product(
    mant: &mut i16,
    exp: &mut i16,
    exp2: &mut i16,
    n: i16,
    a: i16,
    a_exp: i16,
    b: i16,
    tail: i16,
) -> u32 {
    let p = (a as i32).wrapping_mul(b as i32).wrapping_mul(2);
    let s = if p == 0 {
        0i16
    } else {
        norm_shift_l1_batch2(p as u32) as i16
    };
    *exp = a_exp.wrapping_sub(s).wrapping_add(tail);
    *mant = high16_al2b1(((p as u32) << (s as u32 & 0x1f)) as i32);

    let q = (0x20i32 - n as i32).wrapping_shl(0x19);
    let nq = norm_exp_l1_batch2(q as u32);
    let m = add_norm(
        *mant,
        *exp,
        high16_al2b1(((q as u32) << (nq & 0x1f)) as i32),
        (-0x1ei32 - nq as i32) as i16,
        exp2,
    );
    if m == 0 {
        *exp2 = 0xf;
        return 0xffff_8000;
    }
    let l = log2_q(m, *exp2);
    let sl = if l == 0 {
        0i16
    } else {
        norm_shift_l1_batch2(l as u32) as i16
    };
    *exp2 = 0xfi16.wrapping_sub(sl);
    high16_al2b1(((l as u32) << (sl as u32 & 0x1f)) as i32) as u16 as u32
}

/// `0x103139a0` — summed squared difference of two block-floating vectors.
///
/// The operand with the larger exponent is halved and the other is aligned to
/// one bit below it (`e_lo - e_hi - 1`), so the difference cannot overflow; the
/// squares are accumulated in 64 bits, normalised, and returned as a mantissa
/// with `*exp_out = 2·e_hi + 2 - shift`. The doubled exponent is because the
/// value is a *square*.
///
/// `n` counts complex points — the loop runs `2n` times. The operands are
/// swapped internally so the caller's order does not matter.
pub fn sq_diff_norm(
    exp_out: &mut i16,
    x: &[i16],
    x_exp: i16,
    y: &[i16],
    y_exp: i16,
    n: i16,
) -> i16 {
    let (a, b, e_lo, e_hi) = if x_exp < y_exp {
        (y, x, x_exp, y_exp)
    } else {
        (x, y, y_exp, x_exp)
    };

    let (mut lo, mut hi) = (0u32, 0i32);
    let cnt = (n as i32).wrapping_mul(2);
    if cnt > 0 {
        let sh = e_lo.wrapping_sub(e_hi).wrapping_sub(1);
        for i in 0..cnt as usize {
            let bv = shift_i32_l1_batch2((b[i] as i32) << 16, sh);
            let d = high16_al2b1((((a[i] as i32) << 16) >> 1).wrapping_sub(bv)) as i32;
            acc64(&mut lo, &mut hi, d.wrapping_mul(d).wrapping_mul(2) as u32);
        }
    }
    let (s, low) = norm64_align(lo, hi as u32);
    *exp_out = ((e_hi as i32) * 2 + 2 - s as i32) as i16;
    high16_al2b1(low)
}

/// `0x10310dd0` — turn eight block-floating magnitudes into eight Q15 weights
/// on a log scale.
///
/// For each band: shift the 32-bit magnitude by its per-band exponent with
/// [`shl_sat`], take the high word, and if that is zero emit `0x7fff`.
/// Otherwise `w = -log2(m)/2`, offset by `0xa934` and rescaled by `2^15`,
/// clamped to `[0, 0x7fff]`. So a large magnitude gives a small weight — this
/// is an inverse-energy weighting, and the `0xa934` offset is the log of the
/// magnitude at which the weight reaches zero.
///
/// The original's third argument is never read; it is kept in the signature so
/// the call site still lines up.
pub fn log_weight8(out: &mut [i16], src: &[i32], _unused: u32, shifts: &[i16]) {
    for i in 0..8 {
        let hi = high16_al2b1(shl_sat(src[i], shifts[i]));
        out[i] = if hi == 0 {
            0x7fff
        } else {
            let h = log2_q(hi, 0) >> 1;
            let neg = if h == i32::MIN {
                i32::MAX
            } else {
                h.wrapping_neg()
            };
            let d = neg.wrapping_sub(0xa934);
            if d < 0 {
                0
            } else if d <= 0xfffe {
                high16_al2b1(d.wrapping_add(1).wrapping_shl(0xf))
            } else {
                0x7fff
            }
        };
    }
}

/// `0x1030da60` — install the default adaptive-gain parameter set at
/// `state + 0x16a2`.
///
/// A thin wrapper over `0x1030daa0`, which stores eleven 16-bit
/// parameters: the first is duplicated into two slots, five of them are scaled
/// by 256 on the way in, and four are stored raw. The constants supplied here
/// are `-40, 0x4000, 0xccc, 0x333, 0xa3, 10, 0x19`.
///
/// `st` is the state block viewed as `i16`; the parameters land at word indices
/// `0xb51 ..= 0xb5b`.
pub fn set_agc_defaults(st: &mut [i16], a: i16, b: i16, c: i16) {
    set_agc_params(st, a, b, c, -0x28, 0x4000, 0xccc, 0x333, 0xa3, 10, 0x19);
}

/// `0x1031ebf0` — one step of the two-pole level smoother held at `state+0x50`.
///
/// `new = 0.8 · state[0x4a] · 2^4 + 0.9 · state[0x50] · 2^7`, computed in block
/// floating point (`0xcccc` and `0xe666` are *unsigned* Q16 0.8 and 0.9) and
/// written back to `state+0x50` requantised to exponent 7.
///
/// `st` is the state block viewed as `i16`, so the two words are `st[0x25]` and
/// `st[0x28]`.
pub fn update_smoothed_level(st: &mut [i16]) {
    let a = (st[0x25] as i32).wrapping_mul(0xcccc);
    let na = norm_exp_l1_batch2(a as u32);
    let b = (st[0x28] as i32).wrapping_mul(0xe666);
    let nb = norm_exp_l1_batch2(b as u32);

    let mut e: i16 = 0;
    let m = add_norm(
        high16_al2b1(((a as u32) << (na & 0x1f)) as i32),
        (4u32.wrapping_sub(na) & 0xffff) as i16,
        high16_al2b1(((b as u32) << (nb & 0x1f)) as i32),
        (7u32.wrapping_sub(nb)) as i16,
        &mut e,
    );
    let sh = e.wrapping_sub(7);
    st[0x28] = if sh >= 0 {
        high16_al2b1(((m as i32) << 16).wrapping_shl(sh as u32 & 0x1f))
    } else if sh < -0x1e {
        m >> 0xf
    } else {
        high16_al2b1(((m as i32) << 16) >> ((sh.wrapping_neg() as u8) as u32 & 0x1f))
    };
}

/// `0x10320390` — per-harmonic band energies with fractional band edges.
///
/// `step` is a Q7 harmonic spacing; each output band runs from the previous
/// fractional edge to the next, and the two partial bins at the ends are
/// weighted by their fractional coverage using the 32×16 multiply
/// [`mul32x16`]. The whole bins between them are added at full weight into the
/// same 64-bit accumulator. Each band's total is normalised into
/// `out[k]` (Q31 mantissa) and `exps[k] = -shift`. The upper edge is clamped to
/// bin `0x7f`, so the last band never runs off the 129-bin spectrum.
///
/// Domain: `step > 0` — a negative spacing walks the index backwards out of the
/// table.
pub fn band_energy_sum(out: &mut [i32], exps: &mut [i16], step: i16, n: u16, table: &[i32]) {
    if (n as i16) <= 0 {
        return;
    }
    let inc = ((step as i32) << 9) >> 4;
    let mut cur = ((((step as i32) << 9) >> 5).wrapping_add(0x8000)) as u32;

    for k in 0..n as usize {
        cur = cur.wrapping_add(inc as u32);
        let i0 = ((cur >> 16) as i16) as i32;
        let nxt = (((i0 as u32) << 16) | (cur & 0xffff)).wrapping_add(inc as u32);
        let mut top = ((nxt as i32) >> 16) as u32 & 0xffff;
        if high16_al2b1(nxt as i32) > 0x7f {
            top = 0x7f;
        }
        let span = (top.wrapping_sub(((cur as i32) >> 16) as u32 & 0xffff)) as u16;

        // Leading partial bin: `v · (1 - frac)`.
        let v0 = table[i0 as usize] as u32;
        let f0 = (((cur & 0xffff) as i32) >> 1) as i16 as i32;
        let t = mul32x16(v0, f0);
        let t = if t == i32::MIN {
            i32::MAX
        } else {
            t.wrapping_neg()
        };
        let head = t.wrapping_add(v0 as i32) as u32;

        // Trailing partial bin: `v · frac`.
        let f1 = ((((nxt as i32) >> 1) as u16) & 0x7fff) as i16 as i32;
        // `span` is signed here: once the upper edge saturates at bin 0x7f it
        // can fall *below* the lower edge, and the original indexes backwards.
        let edge = (i0 as isize).wrapping_add((span as i16) as isize);
        let tail = mul32x16(table[edge as usize] as u32, f1) as u32;

        let mut lo = head.wrapping_add(tail);
        let mut hi = ((head as i32) >> 31)
            .wrapping_add((tail as i32) >> 31)
            .wrapping_add((((head as u64) + (tail as u64)) >> 32) as i32);

        if (span as i16) > 1 {
            for j in 1..span as usize {
                acc64(&mut lo, &mut hi, table[i0 as usize + j] as u32);
            }
        }

        let (s, low) = norm64_align(lo, hi as u32);
        exps[k] = -s;
        out[k] = low;
        cur = (((top as i16) as u32) << 16) | (nxt & 0xffff);
    }
}

/// `0x1030e660` — fold a 125-bin energy vector into 16 sub-bands, update the
/// per-band level trackers, and re-run the speech/noise decision.
///
/// Bin group 0 is 5 bins wide and groups 1..15 are 8 bins each. Each group's
/// 64-bit total is scaled by `0x4800/2^15 = 0.5625` after a `>> 2`
/// (the guard bits) to give one 32-bit sub-band energy.
///
/// The 16 sub-band energies are summed and normalised into the *wideband* level;
/// then, if `scale_flag == 0`, the upper eight are attenuated by
/// `5·st_a[0x63e + i]` whenever that word is below `0x199a` — a per-band noise
/// weighting that only bites when the stored coefficient is small — and the
/// upper eight alone are summed into a *high-band* level.
///
/// Each of the eight upper bands then updates its own tracker at
/// `st_b + 0x150` (mantissa) / `st_b + 0x170` (exponent) with
/// `new = 0.995·old + 0.0050·energy` (`0xfeb8` is unsigned Q16 0.995 and
/// `0xa3 = 163` is Q15 0.00497), and the eight trackers are summed into a
/// running total. The wideband tracker at `st_b + 0x13c/0x13e` gets the same
/// leak from the wideband level.
///
/// Decision: `st_b + 0x182` is a shift register of per-frame "high band is
/// well above the tracked total" votes, and `st_b + 0x180` is the resulting
/// speech flag. With no vote in the last four frames the flag clears; otherwise
/// the total is tested against three absolute thresholds (`0x7a12 @ 2^-10`,
/// `0x9c4 @ 2^-13`, `0xc35 @ 2^-10`) and, in the two lower cases, additionally
/// against 0.8 and 0.55 of the wideband tracker. This is a voice-activity
/// detector.
///
/// `energy` is 125 words; `st_a` must reach word `0x322` and `st_b` word `0xc1`.
pub fn subband_energy_update(
    energy: &[i32],
    exp_in: i16,
    st_a: &[i16],
    st_b: &mut [i16],
    scale_flag: i16,
) {
    const MB: usize = 0x150 / 2;
    const EB: usize = MB + 0x10;
    const WM: usize = 0x13c / 2;
    const WE: usize = 0x13e / 2;
    const FLAG: usize = 0x180 / 2;
    const VOTES: usize = 0x182 / 2;

    let mut groups = [0i32; 16];
    let mut p = 0usize;
    let mut cnt: i16 = 5;
    for g in groups.iter_mut() {
        let mut v: i64 = 0;
        if cnt != 0 {
            for _ in 0..cnt {
                v = v.wrapping_add(energy[p] as i64);
                p += 1;
            }
        }
        cnt = 8;
        *g = mul32x16((v >> 2) as u32, 0x4800);
    }

    let (s1, lo1) = {
        let (mut lo, mut hi) = (0u32, 0i32);
        for g in groups.iter() {
            acc64(&mut lo, &mut hi, *g as u32);
        }
        norm64_align(lo, hi as u32)
    };
    let e1 = ((exp_in as i32) - (s1 as i32) + 3) as i16;

    if scale_flag == 0 {
        for i in 0..4usize {
            let sv = st_a[0x63e / 2 + i];
            if sv < 0x199a {
                let m = sv.wrapping_mul(5) as i32;
                groups[8 + 2 * i] = mul32x16(groups[8 + 2 * i] as u32, m);
                groups[8 + 2 * i + 1] = mul32x16(groups[8 + 2 * i + 1] as u32, m);
            }
        }
    }

    let (s2, lo2) = {
        let (mut lo, mut hi) = (0u32, 0i32);
        for g in groups[8..].iter() {
            acc64(&mut lo, &mut hi, *g as u32);
        }
        norm64_align(lo, hi as u32)
    };
    let e2 = ((exp_in as i32) - (s2 as i32) + 3) as i16;
    let m2 = high16_al2b1(lo2);

    for i in 0..8usize {
        let v = groups[8 + i] as u32;
        let n = norm_exp_l1_batch2(v);
        let vs = v << (n & 0x1f);
        let a = high16_al2b1((st_b[MB + i] as i32).wrapping_mul(0xfeb8));
        let b = high16_al2b1(mul32x16(vs, 0xa3));
        let mut e = st_b[EB + i];
        let r = add_norm(a, e, b, ((exp_in as i32) - (n as i32) + 3) as i16, &mut e);
        st_b[EB + i] = e;
        st_b[MB + i] = r;
    }

    let mut acc_m: i16 = 0;
    let mut acc_e: i16 = 0;
    for i in 0..8usize {
        acc_m = add_norm(acc_m, acc_e, st_b[MB + i], st_b[EB + i], &mut acc_e);
    }

    let a = high16_al2b1((st_b[WM] as i32).wrapping_mul(0xfeb8));
    let b = high16_al2b1(mul32x16(lo1 as u32, 0xa3));
    let mut we = st_b[WE];
    let nw = add_norm(a, we, b, e1, &mut we);
    st_b[WE] = we;
    st_b[VOTES] = st_b[VOTES].wrapping_mul(2);
    st_b[WM] = nw;

    if mant_exp_gt(
        m2,
        e2,
        high16_al2b1((acc_m as i32).wrapping_mul(0x7ae)),
        acc_e,
    ) && mant_exp_gt(m2, e2, 2000, -0x10)
    {
        st_b[VOTES] = (st_b[VOTES] as u16 | 1) as i16;
    }
    if (st_b[VOTES] as u16 as u8) & 0xf == 0 {
        st_b[FLAG] = 0;
        return;
    }
    if mant_exp_gt(acc_m, acc_e, 0x7a12, -10) {
        st_b[FLAG] = 1;
        return;
    }
    if mant_exp_gt(acc_m, acc_e, 0x9c4, -13) {
        if mant_exp_gt(
            acc_m,
            acc_e,
            high16_al2b1((st_b[WM] as i32).wrapping_mul(0xcccc)),
            st_b[WE],
        ) {
            st_b[FLAG] = 1;
        }
        return;
    }
    if mant_exp_gt(acc_m, acc_e, 0xc35, -10) {
        if mant_exp_gt(
            acc_m,
            acc_e,
            high16_al2b1((st_b[WM] as i32).wrapping_mul(0x8ccc)),
            st_b[WE],
        ) {
            st_b[FLAG] = 1;
        }
        return;
    }
    st_b[FLAG] = 0;
}

/// `0x1030fa40` — the decoder's third-order DC blocker and output stage.
///
/// Input words are promoted to Q31 and scaled by 1/4, then run through
///
/// ```text
/// y1[n] = C1·y1[n-1] + x[n] - x[n-1]
/// Y[n]  = y1[n] + (y1[n-2] - 2·y1[n-1]) + C2·Y[n-1] - C3·Y[n-2]
/// ```
///
/// with `C1 = 0x78ab/2^15 = 0.9424`, `C2 = 2·0x7c20/2^15 = 1.9398` and
/// `C3 = 0x78af/2^15 = 0.9428` — a triple zero at DC. Both recursions are
/// accumulated in 64 bits and clamped to `i32` on store, but note that **the
/// unclamped `y1` feeds the second stage**: the saturated value is only what
/// gets remembered as `y1[n-1]`.
///
/// The output stage multiplies by `0x78aa`, saturating-shifts left by 2 and
/// rounds with `+0x8000` before taking the high word.
///
/// `st` is the six-word `i32` state: `[x[n-1], y1[n-1], y1[n-1], Y[n-1],
/// y1[n-2], Y[n-2]]` — slots 1 and 2 are genuinely duplicated by the original.
/// `n <= 84` (the scratch buffer is 84 words) and a non-positive `n` is a
/// complete no-op, state included.
pub fn dc_blocker(st: &mut [i32], out: &mut [i16], src: &[i16], n: u16) {
    let mut tmp = [0i32; 84];
    if (n as i16) > 0 {
        for i in 0..n as usize {
            tmp[i] = ((src[i] as i32) << 16) >> 2;
        }
    }

    if (n as i16) > 0 {
        let mut y1p = st[1];
        let mut y1p2 = st[4];
        let mut y_p = st[3];
        let mut y1b = st[2];
        let mut y_p2 = st[5];

        for slot in tmp.iter_mut().take(n as usize) {
            let a = y1b;
            let b = y_p;
            let x = *slot;

            let prod = mul32x16(y1p as u32, 0x78ab);
            let xprev = st[0];
            let v = (prod as i64) - (xprev as i64) + (x as i64);
            let y1 = sat64(v);
            st[0] = x;

            let v2 = v + (y1p2 as i64) - 2 * (a as i64) + 2 * (mul32x16(b as u32, 0x7c20) as i64)
                - (mul32x16(y_p2 as u32, 0x78af) as i64);
            y1p2 = a;
            let y = sat64(v2);
            *slot = y;

            y1b = y1;
            y1p = y1;
            y_p = y;
            y_p2 = b;
        }
        st[2] = y1p;
        st[3] = y_p;
        st[1] = y1p;
        st[4] = y1p2;
        st[5] = y_p2;
    }

    if (n as i16) > 0 {
        for i in 0..n as usize {
            let v = shl_sat(mul32x16(tmp[i] as u32, 0x78aa), 2);
            let r = v.wrapping_add(0x8000);
            let r = if ((r ^ v) & r) < 0 {
                (0x7fff_ffffu32).wrapping_add((v < 0) as u32) as i32
            } else {
                r
            };
            out[i] = high16_al2b1(r);
        }
    }
}

/// Exponent-align shift: left for a non-negative count, arithmetic right for a
/// negative one, full sign fill past −30. Counts are masked to 5 bits, as x86
/// does.
fn shift_i32_l1_batch2(v: i32, sh: i16) -> i32 {
    if sh < 0 {
        if sh < -0x1e {
            v >> 31
        } else {
            v >> ((sh as i32).wrapping_neg() as u32 & 0x1f)
        }
    } else {
        v.wrapping_shl(sh as u32 & 0x1f)
    }
}

/// `0x1030daa0` — the eleven-parameter store behind [`set_agc_defaults`].
///
/// Reproduced privately here rather than shared: it is a leaf of the AGC
/// initialization path and has no other caller in scope.
fn set_agc_params(
    st: &mut [i16],
    p2: i16,
    p3: i16,
    p4: i16,
    p5: i16,
    p6: i16,
    p7: i16,
    p8: i16,
    p9: i16,
    p10: i16,
    p11: i16,
) {
    const W: usize = 0x16a2 / 2;
    let v = p2.wrapping_add(4).wrapping_mul(0x100);
    st[W] = v;
    st[W + 1] = v;
    st[W + 2] = p3.wrapping_mul(0x100);
    st[W + 3] = p4.wrapping_mul(0x100);
    st[W + 4] = p5.wrapping_mul(0x100);
    st[W + 5] = p6;
    st[W + 6] = p7;
    st[W + 7] = p8;
    st[W + 8] = p9;
    st[W + 9] = p10.wrapping_mul(0x100);
    st[W + 10] = p11.wrapping_mul(0x100);
}

/// The library's standard "normalise a 64-bit accumulator" sequence:
/// [`norm64`] gives a shift, the pair is shifted by it (left or right), and the
/// caller reads the mantissa out of the **low** word — `norm64`'s −9 bias is
/// chosen so the msb lands at bit 30 of the low word.
///
/// Returns `(shift, low_word)`.
fn norm64_align(lo: u32, hi: u32) -> (i16, i32) {
    let s = norm64(lo, hi) as i16;
    let v = ((hi as u64) << 32 | lo as u64) as i64;
    let r = if s < 0 {
        allshr(v, s.wrapping_neg() as u8)
    } else {
        allshl(v, s as u8)
    };
    (s, r as i32)
}

/// 64-bit accumulate of a signed 32-bit term, in the `add`/`adc` form the
/// decompiler renders as an explicit carry.
fn acc64(lo: &mut u32, hi: &mut i32, x: u32) {
    let carry = ((*lo as u64) + (x as u64)) >> 32;
    *lo = lo.wrapping_add(x);
    *hi = hi.wrapping_add((x as i32) >> 31).wrapping_add(carry as i32);
}

/// 32×16 fractional multiply, `(lo·g) >> 15 + hi·g·2` — the library's way of
/// multiplying a Q31 accumulator by a Q15 word without a 64-bit product.
fn mul32x16(a: u32, g: i32) -> i32 {
    (((a & 0xffff) as i32).wrapping_mul(g) >> 15)
        .wrapping_add(((a >> 16) as i16 as i32).wrapping_mul(g).wrapping_mul(2))
}

/// Clamp a 64-bit accumulator into `i32`, the operation the decompiler renders
/// as a thicket of `SCARRY4`/`SBORROW4` tests. See SPEC §4.2.
fn sat64(v: i64) -> i32 {
    if v > i32::MAX as i64 {
        i32::MAX
    } else if v < i32::MIN as i64 {
        i32::MIN
    } else {
        v as i32
    }
}

/// MSVC's `__allshl` (`0x10305930`), the companion of [`allshr`]. The count is
/// only ever read from `cl`, so it is a byte.
fn allshl(v: i64, count: u8) -> i64 {
    let hi = (v >> 32) as u32;
    let lo = v as u32;
    if count >= 0x40 {
        return 0;
    }
    let (nh, nl) = if count >= 0x20 {
        (lo << (count & 0x1f), 0u32)
    } else if count == 0 {
        (hi, lo)
    } else {
        ((hi << count) | (lo >> (32 - count)), lo << count)
    };
    ((nh as u64) << 32 | nl as u64) as i64
}

// ======= pulled in from l1_batch3 (encoder-only) =======

// ---------------------------------------------------------------------------

/// `0x1030a900` — arctangent of a block-floating-point value.
///
/// Returns the Q15 mantissa of `atan(x * 2^-exp)` and writes its exponent to
/// `out_exp`. The argument is first folded into the first octant: above
/// `0x4d41@2` (≈ tan 67.5°) the reciprocal is taken and `pi/4` (`0x6488`)
/// is added at full weight; above `0x6a0a@-1` (≈ tan 22.5°) the identity
/// `atan(x) = pi/8 + atan((x-t)/(1+xt))` is used, adding `pi/4` at half
/// weight; below that the series is evaluated directly.
///
/// The core is a degree-4 odd polynomial in `z = x^2` evaluated by Horner in
/// block floating point — every stage renormalises and carries its own
/// exponent, which is why the shift bookkeeping dominates the code. The
/// coefficients appear as the unsigned Q16 immediates `0xe38e`, `0xb6db`,
/// `0x6666` and `0xaaab`.
///
/// The sign is stripped up front (with `-0x8000` clamped to `0x7fff`) and
/// re-applied to the mantissa at the end.
pub fn atan_bfp(x: i16, exp: i16, out_exp: &mut i16) -> i16 {
    let neg = x < 0;
    let mag: i16 = if x < 0 {
        if x == i16::MIN {
            0x7fff
        } else {
            x.wrapping_neg()
        }
    } else {
        x
    };

    let mut e = exp;
    let m: i16;
    let quad: i16;
    let oct: i16;

    if mant_exp_gt(mag, exp, 0x4d41, 2) {
        let mut ex = 0i16;
        let q = div_norm(0x4000_0000, 1, mag, exp, &mut ex);
        m = if q == i16::MIN { 0x7fff } else { -q };
        quad = 0x6488;
        oct = 1;
        e = ex;
    } else if !mant_exp_gt(mag, exp, 0x6a0a, -1) {
        m = mag;
        quad = 0;
        oct = 0;
    } else {
        let mut ea = 0i16;
        let mut es = 0i16;
        let den = add_norm(0x4000, 1, mag, exp, &mut ea);
        let d = sub_norm(0x4000, 1, mag, exp, &mut es);
        let n32 = (d as i32).wrapping_mul(0x10000);
        let mut ex = 0i16;
        if n32 < 0 {
            let p = if n32 == i32::MIN { i32::MAX } else { -n32 };
            let q = div_norm(p, es, den, ea, &mut ex);
            m = if q == i16::MIN { 0x7fff } else { -q };
        } else {
            m = div_norm(n32, es, den, ea, &mut ex);
        }
        quad = 0x6488;
        oct = 0;
        e = ex;
    }

    // z = m^2 in block floating point.
    let xm = m as i32;
    let zw = xm.wrapping_mul(xm).wrapping_mul(2);
    let n0 = norm_u16(zw);
    let ez = ((e as i32).wrapping_mul(2).wrapping_sub(n0 as i32)) as u32 & 0xffff;
    let z = high16(((zw as u32) << (n0 & 0x1f)) as i32) as i32;

    // Horner: (((c0*z + c1)*z + c2)*z + c3)*z*x + x
    let t = shift32(z.wrapping_mul(0xe38e), (ez.wrapping_sub(1) as u16) as i16);
    let a1 = t.wrapping_add(0xb6db_0000u32 as i32);
    let n1 = norm_u16(a1);
    let s1 = ez
        .wrapping_add(2)
        .wrapping_add(0u32.wrapping_sub(n1).wrapping_sub(2) & 0xffff) as i32;
    let t = shift32(
        (high16(((a1 as u32) << (n1 & 0x1f)) as i32) as i32)
            .wrapping_mul(z)
            .wrapping_mul(2),
        s1 as i16,
    );

    let a2 = t.wrapping_add(0x6666_0000);
    let n2 = norm_u16(a2);
    let s2 = ez
        .wrapping_add(1)
        .wrapping_add(0u32.wrapping_sub(n2).wrapping_sub(2) & 0xffff) as i32;
    let t = shift32(
        (high16(((a2 as u32) << (n2 & 0x1f)) as i32) as i32)
            .wrapping_mul(z)
            .wrapping_mul(2),
        s2 as i16,
    );

    let a3 = t.wrapping_add(0xaaab_0000u32 as i32);
    let n3 = norm_u16(a3);
    let p = (high16(((a3 as u32) << (n3 & 0x1f)) as i32) as i32)
        .wrapping_mul(z)
        .wrapping_mul(2);
    let n4 = norm_u16(p);
    let s3 =
        (0u32.wrapping_sub(n3).wrapping_sub(1) & 0xffff).wrapping_add(ez.wrapping_sub(n4)) as i32;
    let r = shift32(
        (high16(((p as u32) << (n4 & 0x1f)) as i32) as i32)
            .wrapping_mul(xm)
            .wrapping_mul(2),
        s3 as i16,
    );
    let r = r.wrapping_add(xm.wrapping_mul(0x10000));
    let r = shift32(r, (e as i32).wrapping_sub(1) as i16);

    // Fold in the octant offset.
    let os = oct.wrapping_sub(1);
    let qterm = if os < 0 {
        if os < -0x1e {
            0
        } else {
            ((quad as i32) << 16) >> ((os as u8).wrapping_neg() & 0x1f) as u32
        }
    } else {
        ((((quad as i32) << 16) as u32) << ((os as u8) & 0x1f) as u32) as i32
    };

    let sum = r.wrapping_add(qterm);
    let sh = if sum == 0 { 0i16 } else { norm_e(sum) as i16 };
    let mant = high16(((sum as u32) << (sh as u32 & 0x1f)) as i32) as u16;
    if mant == 0 {
        *out_exp = 0;
        return 0;
    }
    *out_exp = 1i16.wrapping_sub(sh);
    if !neg {
        mant as i16
    } else if mant == 0x8000 {
        0x7fff
    } else {
        (mant as i16).wrapping_neg()
    }
}

/// `0x10312a90` — build the voicing-pattern lookup for a sub-mode.
///
/// Fills `out[i]` with [`voicing_pattern`] for every reachable codebook index:
/// the 56 pre-baked patterns plus `2^nbits` raw-bit patterns. Index 0x39 is a
/// jump point — on reaching it the walk skips forward to `0x38 + 2^(nbits-1)`,
/// so the entries in between are left untouched. `out` must hold
/// `0x38 + 2^nbits` dwords.
///
/// Domain: `nbits >= 3` (see [`voicing_pattern`]; `nbits <= 2` makes the skip
/// target 0x39 itself and the walk never terminates).
pub fn build_voicing_table(out: &mut [u32], prev: u32, nbits: i32) {
    let end = ((1i32 << (nbits as u8 & 0x1f)) as i16).wrapping_add(0x38);
    let mut i: u32 = 0;
    if end <= 0 {
        return;
    }
    loop {
        if (i as i16) == 0x39 {
            i = ((end - 0x38) >> 1).wrapping_add(0x38) as u32 & 0xffff;
        }
        let v = voicing_pattern(i as i16, prev, nbits as i16);
        let at = i as i16;
        i = i.wrapping_add(1);
        out[at as usize] = v;
        if (i as i16) >= end {
            break;
        }
    }
}

/// `0x1031d4f0` — derive the 16-band spectral gain envelope.
///
/// Two passes over the 16 `(mantissa, exponent)` band energies. The first
/// accumulates them in 64 bits at a common Q, keeping bits 20.. of the sum as
/// a mean level (floored at `-0x5a00`); the second forms, per band, a ramp
/// starting at `mean + 0x2800` and falling by `0x266.6` per band, minus the
/// band's own level, clamped into `[0x400, 0x2800]`.
///
/// The band term is *negated* before shifting — the decompiler drops the
/// `neg`, so the shape is `ramp - level`, not `ramp + level`. Finally the
/// whole envelope is normalised so the 16 values sum to a constant, by scaling
/// with `0x3ffffff / (sum >> 4) / 2`.
pub fn band_gain_envelope(mant: &[i16], exp: &[i16], out: &mut [i16]) {
    let mut acc: i64 = 0;
    for i in 0..16usize {
        let v = shift32((mant[i] as i32) << 16, exp[i].wrapping_sub(5));
        acc = acc.wrapping_add(v as i64);
    }
    let t = ((((acc as u64) >> 4) as u32) as i32) >> 16;
    let mean = if (t as i16) < -0x5a00 {
        -0x5a00i32
    } else {
        (t as i16) as i32
    };

    let mut ramp = (mean.wrapping_add(0x2800) as u32).wrapping_shl(16) as i32;
    let mut total = 0i32;
    for i in 0..16usize {
        let x = ((mant[i].wrapping_neg() as i32) << 16) as i64;
        let s = exp[i].wrapping_sub(5) as i32;
        let sv = if s < 0 {
            allshr(x, (-s) as u32 as u8)
        } else {
            allshl_l1_batch3(x, s as u32 as u8)
        } as i32;
        let mut v = high16(sv.wrapping_add(ramp));
        if v < 0x400 {
            v = 0x400;
        } else if v > 0x2800 {
            v = 0x2800;
        }
        ramp = ramp.wrapping_sub(0x266_6000);
        out[i] = v;
        total = total.wrapping_add(v as i32);
    }
    let g = div32_half(0x3ff_ffff, (total >> 4) as i16) as i16;
    for i in 0..16usize {
        out[i] = high16((out[i] as i32).wrapping_mul(g as i32).wrapping_mul(2));
    }
}

/// `0x1031dde0` — update the running / peak level pair of the noise tracker.
///
/// With `d_m == 0` the call is the frame boundary: the running level in
/// `(cur_m, cur_e)` is promoted to the peak in `(max_m, max_e)` if it is
/// larger, and the running level is reset. Otherwise one measurement is
/// folded in:
///
/// * a smoothing weight is built from `sel` (`1 - sel/16` in Q30, times `g1`)
///   and `(c_m, c_e) / (that + (c_m, c_e))` gives a normalised step, scaled by
///   `g2`;
/// * that step accumulates into `(b_m, b_e)`, and the step times the magnitude
///   of `(d) - (e)` accumulates into `(a_m, a_e)`;
/// * the running level becomes that magnitude, optionally halved (`sel == 0`)
///   or scaled by 0.75 (`sel == 1`) when the frame is quiet or far below the
///   reference (`lvl - ref_lvl > 0x800`, or `lvl < -0x1900`);
/// * the peak is raised to the new running level if it exceeds it.
///
/// Domain: the smoothing denominator must be non-zero, i.e.
/// `add_norm(c_m, c_e, …) != 0`.
#[allow(clippy::too_many_arguments)]
pub fn noise_track_update(
    a_m: &mut i16,
    a_e: &mut i16,
    b_m: &mut i16,
    b_e: &mut i16,
    c_m: i16,
    c_e: i16,
    d_m: i16,
    d_e: i16,
    e_m: i16,
    e_e: i16,
    cur_m: &mut i16,
    cur_e: &mut i16,
    max_m: &mut i16,
    max_e: &mut i16,
    sel: i16,
    g1: i16,
    g2: i16,
    lvl: i16,
    ref_lvl: i16,
) {
    if d_m == 0 {
        let mut ex = 0i16;
        let s = sub_norm(*max_m, *max_e, *cur_m, *cur_e, &mut ex);
        if s > 0 {
            *max_m = *cur_m;
            *max_e = *cur_e;
        }
        *cur_m = 0;
        *cur_e = 0;
        return;
    }

    let t = ((sel as i32) << 16) >> 5;
    let t = if t == i32::MIN { i32::MAX } else { -t };
    let w = (high16(t.wrapping_add(0x10000).wrapping_mul(0x4000)) as i32)
        .wrapping_mul(g1 as i32)
        .wrapping_mul(2);
    let nw = norm_u16(w);
    let mut e8 = 0i16;
    let den = add_norm(
        c_m,
        c_e,
        high16(((w as u32) << (nw & 0x1f)) as i32),
        ((-0x13i32) - nw as i32) as i16,
        &mut e8,
    );
    let mut p7 = 0i16;
    let q = div_norm((c_m as i32) << 16, c_e, den, e8, &mut p7);
    p7 = p7.wrapping_add(5);
    let step = high16((q as i32).wrapping_mul(g2 as i32).wrapping_mul(2));

    let bm = *b_m;
    let be = *b_e;
    let r = add_norm(bm, be, step, p7, b_e);
    *b_m = r;

    let mut lc = 0i16;
    let sn = sub_norm(d_m, d_e, e_m, e_e, &mut lc);
    let mag: i16 = if sn < 0 {
        if sn == i16::MIN {
            0x7fff
        } else {
            sn.wrapping_neg()
        }
    } else {
        sn
    };
    let iv = mag as i32;

    let prod = (step as i32).wrapping_mul(iv).wrapping_mul(2);
    let sh = if prod == 0 { 0i16 } else { norm_e(prod) as i16 };
    p7 = p7.wrapping_add(lc.wrapping_sub(sh));
    let am = *a_m;
    let ae = *a_e;
    let r = add_norm(
        am,
        ae,
        high16(((prod as u32) << (sh as u32 & 0x1f)) as i32),
        p7,
        a_e,
    );
    *a_m = r;

    let saved_e = lc;
    let mut out_m = mag;
    if ((lvl as i32) - (ref_lvl as i32) > 0x800) || (lvl < -0x1900) {
        if sel == 0 {
            out_m = ((iv << 15) >> 16) as i16;
        } else if sel == 1 {
            out_m = high16(iv.wrapping_mul(0xc000));
        }
    }

    let mut e8b = 0i16;
    let nm = add_norm(*cur_m, *cur_e, out_m, lc, &mut e8b);
    let mut lc2 = 0i16;
    let s = sub_norm(*max_m, *max_e, nm, e8b, &mut lc2);
    if s > 0 {
        *max_m = nm;
        *max_e = e8b;
    }
    *cur_m = out_m;
    *cur_e = saved_e;
}

/// `0x1031c040` — refine a spectral peak to sub-bin resolution by fitting a
/// parabola to the peak bin and its two neighbours.
///
/// Returns the interpolated position as a Q15 fraction of a bin
/// (`(pos << 15) * 2`, i.e. Q16 in a 32-bit word). The peak is nudged inward
/// when it sits on either end of the array. If the fitted parabola is not
/// concave (the second difference is non-negative) the integer bin is
/// returned unchanged; otherwise the offset is `(y[-1] - y[+1]) / (2 * d2)`,
/// clamped to ±0x7fff by [`div32_sat15`] — which also covers the case of a
/// second difference too small to be represented, since it returns the clamp
/// without dividing.
pub fn parabolic_peak(peak: i16, y: &[i16], n: i16) -> i32 {
    let orig = peak as i32;
    let mut p = peak;
    if peak == 0 {
        p = 1;
    } else if orig == (n as i32) - 1 {
        p = peak - 1;
    }
    let i = p as i32;
    let ym = (y[(i - 1) as usize] as i32) << 16;
    let yp = (y[(i + 1) as usize] as i32) << 16;
    let d2 = (ym >> 1)
        .wrapping_add((y[i as usize] as i32).wrapping_mul(-0x10000))
        .wrapping_add(yp >> 1);
    if d2 < 0 {
        let q = div32_sat15((ym >> 2).wrapping_sub(yp >> 2), (d2 >> 16) as i16) as i16;
        return ((q as i32).wrapping_add(i.wrapping_mul(0x8000))).wrapping_mul(2);
    }
    orig << 16
}

/// `0x10313c10` — element-wise reciprocal of a vector at a chosen Q point.
///
/// `dst[i] = (0x8000 << shift) / src[i] / 2`, with a zero entry mapped to
/// `0x7fff` instead of dividing. Returns the exponent of the result block,
/// `(0x10 - shift) - exp_in`, so the caller can carry the block scaling.
pub fn reciprocal_vec(dst: &mut [i16], src: &[i16], exp_in: i32, shift: i16, n: u16) -> i32 {
    let num = if shift < 0 {
        if shift < -0x1e {
            0
        } else {
            0x8000i32 >> ((shift as u8).wrapping_neg() & 0x1f) as u32
        }
    } else {
        ((0x8000u32) << ((shift as u8) & 0x1f) as u32) as i32
    };
    if (n as i16) > 0 {
        for i in 0..n as usize {
            let s = src[i];
            dst[i] = if s == 0 {
                0x7fff
            } else {
                div32_half(num, s) as i16
            };
        }
    }
    ((0x10i16.wrapping_sub(shift)) as u16) as i32 - exp_in
}

/// `0x1030ed30` — adaptive voicing threshold for one harmonic band.
///
/// Builds a threshold that falls with harmonic index (the two ramps
/// `0x10000000 - k*0x2222` and `-0x30000000 - k*0x4444`, whichever is lower,
/// biased by `base`), floors it with the per-band history at state words
/// `0x29+k` / `0x39+k`, and — for bands above 8 when `use_hist` is set — with
/// the log-magnitude of the band held at words `0xa0+k` / `0xb0+k`.
///
/// The margin between the two is then exponentiated: a negative margin
/// (threshold below the floor) gives `2^margin` as a normalised
/// mantissa/exponent pair via [`pow2_frac`]; a non-negative margin saturates
/// the result at `0x4000@1` = 1.0. Returns the exponent, writes the mantissa
/// to `out_mant`.
pub fn voicing_threshold(
    out_mant: &mut i16,
    st: &[i16],
    base: i16,
    exp: i32,
    band: i16,
    use_hist: i32,
) -> i16 {
    let k = band.wrapping_shl(11);
    let s = (5i32 - exp) as i16;

    let mut a = (k as i32).wrapping_mul(-0x2222).wrapping_add(0x10000000);
    a = shift32(a, s);
    a = a.wrapping_add((base as i32).wrapping_mul(0x10000));
    let mut b = (k as i32).wrapping_mul(-0x4444).wrapping_add(-0x30000000);
    b = shift32(b, s);
    if a < b {
        b = a;
    }

    let idx = band as usize;
    let mut floor = (st[0x29 + idx] as i32) << 16;
    floor = shift32(floor, st[0x39 + idx].wrapping_sub(exp as i16));

    if (use_hist as i16) != 0 && band > 8 {
        let l = log2_q(st[0xa0 + idx], st[0xb0 + idx]);
        let ln = norm_u16(l);
        let v = high16(shl_sat(
            ((l as u32) << (ln & 0x1f)) as i32,
            (0x0fu32.wrapping_sub(ln) as i32).wrapping_sub(exp) as i16,
        ));
        if high16(floor) < v {
            floor = (v as i32) << 16;
        }
    }

    let margin = b.wrapping_sub(floor);
    if margin < 0 {
        let arg = ((high16(margin) as i32)
            .wrapping_mul(0x8000)
            .wrapping_add(((((margin as u32) & 0xffff) << 0xe) as i32) >> 0xf))
            >> 9;
        let mut e = 0i16;
        let m = pow2_frac(arg, &mut e);
        let sh = if m == 0 { 0i16 } else { norm_e(m) as i16 };
        *out_mant = high16(((m as u32) << (sh as u32 & 0x1f)) as i32);
        return e.wrapping_sub(sh);
    }
    *out_mant = 0x4000;
    1
}

/// `0x103100e0` — fetch the active signalling entry, or synthesise one from
/// the caller's flags.
///
/// With bit 0x4000 set in `flags` the entry is taken straight from the state
/// block (`state[2]` as the tag, `state[1]` duplicated into both data words)
/// and `state[0]` is armed with 1; otherwise [`find_active_entry`] scans the
/// two slots and `state[0]` is cleared. Returns whether an entry was produced.
pub fn take_signalling_entry(
    out_tag: &mut i16,
    out_pair: &mut [i16; 2],
    flags: u32,
    state: &mut [i16],
) -> bool {
    if (flags & 0x4000) != 0 {
        *out_tag = state[2];
        out_pair[0] = state[1];
        out_pair[1] = state[1];
        state[0] = 1;
        return true;
    }
    let mut bytes = [0u8; 0x16];
    for (k, w) in state[..0x0b].iter().enumerate() {
        bytes[2 * k..2 * k + 2].copy_from_slice(&w.to_le_bytes());
    }
    let found = find_active_entry(&bytes, out_tag, out_pair);
    state[0] = 0;
    found
}

/// `0x1030db40` — one step of the adaptive noise/level follower, returning the
/// suppression gain for the frame.
///
/// `p` is an 11-word tuning block: `p[0]` the tracked level, `p[4]` its floor,
/// `p[5..9]` four attack/release coefficients, `p[9]` the per-frame step
/// limit, `p[1..4]` the gain curve (offset, ceiling, floor) and `p[10]` the
/// knee width. `st[0x25]` is the frame's measured level and `st[0x28]` the
/// reference; `st[0x7b]` and `ctrl[5]` gate which coefficient is used.
///
/// The level moves toward the measurement by `coef * delta` (a Q15 one-pole),
/// with `delta` limited to ±`p[9]`; below `0x2001` of confidence the update is
/// skipped entirely. The gain is `p[1] - level` clamped into `[p[3], p[2]]`,
/// then faded to zero linearly over the knee when the level is within `p[10]`
/// of the reference.
///
/// Domain: `p[10] != 0` — the knee divide faults on a zero width.
pub fn level_follow_gain(st: &[i16], ctrl: &[i16], p: &mut [i16]) -> i16 {
    let lim = p[9];
    let meas = (st[0x25] as i32) << 16;
    let mut d = high16(meas.wrapping_add((p[0] as i32).wrapping_mul(-0x10000)));
    if (d as i32) < -(lim as i32) {
        d = (-(lim as i32)) as i16;
    }

    let mut coef = 0i32;
    let mut update = true;
    if lim < d {
        coef = p[5] as i32;
    } else if d < 1 {
        if st[0x7b] < 0x4001
            || ctrl[5] > 0xccb
            || high16(meas.wrapping_add((st[0x28] as i32).wrapping_mul(-0x10000))) <= p[10]
        {
            if st[0x7b] < 0x2001 {
                update = false;
            } else {
                coef = p[8] as i32;
            }
        } else {
            coef = p[7] as i32;
        }
    } else {
        coef = p[6] as i32;
    }
    if update {
        p[0] = high16(
            coef.wrapping_mul(d as i32)
                .wrapping_add((p[0] as i32).wrapping_mul(0x8000))
                .wrapping_mul(2),
        );
    }
    if p[0] < p[4] {
        p[0] = p[4];
    }

    let mut g = high16(
        (p[1] as i32)
            .wrapping_mul(0x10000)
            .wrapping_add((p[0] as i32).wrapping_mul(-0x10000)),
    );
    if p[2] < g {
        g = p[2];
    } else if g < p[3] {
        g = p[3];
    }

    let above = high16(
        (p[0] as i32)
            .wrapping_mul(0x10000)
            .wrapping_add((st[0x28] as i32).wrapping_mul(-0x10000)),
    );
    if above < p[10] {
        let q = div32_half(((above as i32) << 16) >> 4, p[10]);
        let f = (((q as u32) << 0x14) >> 0x10) as i16;
        return high16((f as i32).wrapping_mul(g as i32).wrapping_mul(2));
    }
    g
}

// ---------------------------------------------------------------------------
// Local copies of the arithmetic idioms these functions are built from. They
// are the same shapes already verified in `batch1`..`batch4`, repeated here so
// this module does not depend on another batch's private helpers.
// ---------------------------------------------------------------------------

/// High half of a 32-bit accumulator, i.e. the Q15 value of a Q31 accumulator.
#[inline]
fn high16(x: i32) -> i16 {
    (x >> 16) as i16
}

/// x86 variable shift as the reference spells it: the count is a signed 16-bit
/// value, the shift amount is its low byte masked to 5 bits, and a count below
/// -30 is special-cased to a full sign fill.
#[inline]
fn shift32(x: i32, s: i16) -> i32 {
    if s < 0 {
        if s < -0x1e {
            x >> 31
        } else {
            x >> ((s as u8).wrapping_neg() & 0x1f) as u32
        }
    } else {
        ((x as u32) << ((s as u8) & 0x1f) as u32) as i32
    }
}

// ======= pulled in from l1_batch4 (encoder-only) =======

/// `0x1031c310` — as [`accumulate_exp`], but returning the normalised sum as a
/// Q15 mantissa.
///
/// Same 64-bit accumulate / normalise / `*exp -= shift`, then the mantissa is
/// the high word of the shifted low half. This is the block-floating-point
/// "sum a vector of energies" primitive: `(mantissa, *exp)` is the sum.
pub fn sum_mant_exp(x: &[i32], n: i16, exp: &mut i16) -> i16 {
    ((sum_norm64(x, n, exp) as i32) >> 16) as i16
}

/// `0x1031d970` — accumulate one band's weighted voicing error.
///
/// `|a - b|` is formed as a block-floating value by running `sub_norm`
/// (`0x1030b490`) both ways round and keeping whichever difference came out
/// non-negative; that difference is reported through `out_m`/`out_e`
/// (`out_*` always carry the `a - b` direction, not the chosen one). For the
/// first six bands (`idx < 6`) the magnitude is multiplied by the band weight
/// `w`, renormalised, and added into the running error accumulator
/// `(acc_m, acc_e)` while `w` itself is added into `(wsum_m, wsum_e)` — so the
/// caller ends up with a weighted mean.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_weighted_diff(
    acc_m: &mut i16,
    acc_e: &mut i16,
    wsum_m: &mut i16,
    wsum_e: &mut i16,
    out_m: &mut i16,
    out_e: &mut i16,
    a_m: i16,
    a_e: i16,
    b_m: i16,
    b_e: i16,
    w_m: i16,
    w_e: i16,
    idx: i16,
) {
    let mut e1: i16 = 0;
    let r1 = sub_norm(a_m, a_e, b_m, b_e, &mut e1);
    *out_m = r1;
    *out_e = e1;
    let mut e2: i16 = 0;
    let r2 = sub_norm(b_m, b_e, a_m, a_e, &mut e2);
    let (mant, exp) = if r1 < 0 { (r2, e2) } else { (r1, e1) };

    if idx < 6 {
        let p = (mant as i32).wrapping_mul(w_m as i32).wrapping_mul(2);
        let nrm = if p == 0 {
            0u16
        } else {
            (norm_shift(p as u32) as u16) & 0xffff
        };
        let ne = (exp as u16).wrapping_sub(nrm).wrapping_add(w_e as u16) as i16;
        let m = ((p.wrapping_shl(nrm as u32 & 0x1f)) >> 16) as i16;

        let (am, ae) = (*acc_m, *acc_e);
        *acc_m = add_norm(am, ae, m, ne, acc_e);
        let (sm, se) = (*wsum_m, *wsum_e);
        *wsum_m = add_norm(sm, se, w_m, w_e, wsum_e);
    }
}

/// `0x10314530` — bring a block-floating vector onto a single shared exponent.
///
/// The largest of the `n` per-element exponents is found with `vec_max`
/// (`0x103174e0`); every mantissa is then shifted by `exps[i] - max` and
/// written to `dst`, so all `n` values end up expressed at the common exponent
/// `max`. Returns that exponent. This is the "maximise the per-element
/// exponents" step the codec's block floating point is built on.
pub fn align_block_exp(dst: &mut [i16], src: &[i16], exps: &[i16], n: i16) -> i16 {
    let m = vec_max(exps, n);
    if n > 0 {
        for i in 0..n as u16 as usize {
            let v = (src[i] as i32) << 16;
            let sh = exps[i].wrapping_sub(m);
            dst[i] = if sh < 0 {
                if sh < -0x1e {
                    src[i] >> 15
                } else {
                    high16_al2b3(v >> ((-sh) as u32 & 0x1f))
                }
            } else {
                high16_al2b3(v.wrapping_shl(sh as u32 & 0x1f))
            };
        }
    }
    m
}

/// `0x10317140` — accumulate an energy vector and renormalise the caller's
/// block exponent.
///
/// Sums `n` 32-bit words into a 64-bit accumulator, normalises it, and folds
/// the normalisation shift into `*exp` (`*exp -= shift`). The shifted
/// accumulator itself is written to a dead stack slot and dropped — the
/// original is a `void` function whose only visible effect is the exponent
/// update. The low half is still left in `eax`, so it is returned here for the
/// test to pin down.
pub fn accumulate_exp(x: &[i32], n: i16, exp: &mut i16) -> u32 {
    sum_norm64(x, n, exp)
}

// ---------------------------------------------------------------------------
// Ported functions.
// ---------------------------------------------------------------------------

/// `0x1030b0b0` — log2 of a block-floating value, re-aligned to a caller-chosen
/// exponent and rounded to a 16-bit result.
///
/// Runs `log2_transform` (`0x1030af60`) on `x` with the incoming exponent
/// `exp`, then shifts the mantissa by `new_exp - target_exp` and rounds with
/// the usual `+ 0x8000, >> 16`. Only the low 16 bits of the exponent
/// difference matter — the original tests `cx` and shifts by `cl`.
pub fn log2_at_exp(x: u32, exp: i16, target_exp: i16) -> i32 {
    let mut e = exp;
    let m = log2_transform(x, &mut e, &LOG_FIXUP) as i32;
    shift_round16_l1_batch4(m, e.wrapping_sub(target_exp))
}

/// `0x10312b90` — read a bit field out of `word` on a logarithmically spaced
/// index schedule.
///
/// `scale` is normalised and inverted (`0x3b39c0ec / scale`, halved); for each
/// of the `n` output bits the index is `7 - round(k · quotient)` with
/// `k = 0x20, 0x50, 0x80, …` (step 0x30), floored at 0 and capped at the number
/// of bits still to go. The selected bit of `word` is shifted into the result
/// MSB first. This is how the per-harmonic quantiser allocation is turned into
/// a bit-plane selection.
pub fn extract_bit_pattern(word: i16, n: u16, scale: i16) -> u16 {
    let v = (scale as i32) << 16;
    let nrm = if v == 0 {
        0u16
    } else {
        (norm_shift(v as u32) as u16) & 0xffff
    };
    let mant = ((v.wrapping_shl(nrm as u32 & 0x1f)) >> 16) as i16;
    let q = div32_half(0x3b39_c0ec, mant) as u16 as i16 as i32;
    let sh = nrm.wrapping_sub(0xf) as i16;

    if (n as i16) <= 0 {
        return 0;
    }
    let mut acc: u16 = 0;
    let mut k: i16 = 0x20;
    let mut left = (n as i16 as i32).wrapping_sub(1);
    let mut floor_idx = n.wrapping_sub(1);
    for _ in 0..n {
        let e = shift_i32_l1_batch4((k as i32).wrapping_mul(q).wrapping_mul(2), sh);
        let mut idx = (7i32.wrapping_sub(e.wrapping_add(0x4000) >> 16)) as u16;
        if (idx as i16) < 0 {
            idx = 0;
        }
        if (idx as i16 as i32) > left {
            idx = floor_idx;
        }
        let c = idx & 0x1f;
        let bit = if c >= 16 {
            ((word >> 15) & 1) as u16
        } else {
            ((word >> c) & 1) as u16
        };
        acc = acc.wrapping_mul(2) | bit;
        k = k.wrapping_add(0x30);
        floor_idx = floor_idx.wrapping_sub(1);
        left -= 1;
    }
    acc
}

/// `0x1031e080` — apply the frame-classification penalty to a level word.
///
/// Reads the tracker's recent-peak history — `state[0x24]`/`state[0x25]` (a
/// long- and a short-term level), the flag word `state[0x22]`, the hangover
/// count `state[0x26]` and the frame counter `state[0x9c]` — and subtracts a
/// penalty from `*level` in the log domain (Q16 of the level word). The
/// penalty is a five-segment piecewise-linear function of the level
/// *difference* `(state[0x25] - state[0x24]) << 16`, breaking at −0x14000000,
/// −0x1e000000, −0x28000000 and −0x32000000, with slopes `0xccca`, `-0xcccc/2`
/// and `0x33328`; the deepest segment additionally requires at least two set
/// bits in the low 18 of the flag word (`popcount_bits`, `0x1031d640`).
///
/// Nothing happens at all unless the hangover count (as `state[0x26] << 8`)
/// exceeds −0x5a00 and is at least `state[0x25]`. Returns 1 if the resulting
/// level is strictly positive, 0 otherwise.
pub fn level_penalty(state: &[i16], level: &mut i16) -> i32 {
    let s48 = state[0x24] as i32;
    let s4a = state[0x25];
    let flags = state[0x22] as u16;
    let d = ((s4a as i32) << 16).wrapping_sub(s48 << 16);
    let t = (((state[0x26] as i32) << 24) >> 16) as i16;
    let mut acc = (*level as i32) << 16;

    if t > -0x5a00 && s4a <= t {
        if state[0x9c] < 8 {
            acc = acc.wrapping_sub(0x3800_0000);
        } else if d > -0x13ff_fb00
            && ((flags & 1) != 0 || s48.wrapping_sub(state[0x28] as i32) < 0x800)
        {
            acc = acc.wrapping_sub(0x0999_8000);
        } else if d > -0x1400_0000 {
            acc = acc.wrapping_sub(0x1000_0000);
        } else if d > -0x1e00_0000 {
            let s = ((d.wrapping_add(0x1e00_0000) >> 16) as i16 as i32).wrapping_mul(0xccca);
            acc = acc.wrapping_add(0xe800_0000u32 as i32).wrapping_add(s);
        } else if d > -0x2800_0000 {
            let s = ((d.wrapping_add(0x2800_0000) >> 16) as i16 as i32)
                .wrapping_mul(0xffff_3334u32 as i32);
            acc = acc.wrapping_add((0xe400_0000u32 as i32).wrapping_sub(s >> 1));
        } else if d > -0x3200_0000 && popcount_bits(flags as u32, 0x12) >= 2 {
            let s = ((d.wrapping_add(0x3200_0000) >> 16) as i16 as i32).wrapping_mul(0x33328);
            acc = acc.wrapping_add(0xc400_0000u32 as i32).wrapping_add(s);
        } else {
            acc = acc.wrapping_sub(0x1c00_0000);
        }
    }
    let r = high16_al2b3(acc);
    *level = r;
    if r > 0 {
        1
    } else {
        0
    }
}

/// `0x1031b180` — one minus a normalised ratio, in Q15.
///
/// The tail of `tail` starting at `off = (hint · 2^(hint_exp-16)) + 1` and
/// running to index 32 is summed (`sum_i16`, `0x103172c0`); a non-positive sum
/// short-circuits to `0x7fff`. Otherwise `num@num_exp` is divided by that sum
/// with `div_norm` (`0x1030a660`) and the Q15 result is subtracted from
/// `0x7fff`. A quotient whose exponent is positive — i.e. a ratio of 1 or more
/// — returns 0. Used to turn "how much of the band energy is accounted for"
/// into a voicing measure.
///
/// The offset must land inside `tail`; the original computes it as a raw
/// pointer and would read out of bounds for a wild `hint`.
pub fn unvoiced_ratio(num: i16, num_exp: i16, tail: &[i16], hint: i16, hint_exp: i16) -> u16 {
    let v = shift_i32_l1_batch4((hint as i32) << 16, hint_exp.wrapping_sub(0x10));
    let off = (v.wrapping_add(0x10000) >> 16) as i16;
    let cnt = (0x20i32 - (off as u16 as i32)) as i16;
    let sum = sum_i16(&tail[off as usize..], cnt);
    if sum <= 0 {
        return 0x7fff;
    }
    let nrm = (0x1e - (31 - sum.leading_zeros() as i32)) as u16;
    let den = ((sum.wrapping_shl(nrm as u32 & 0x1f)) >> 16) as i16;
    let mut e: i16 = 0;
    let q = div_norm(
        (num as i32) << 16,
        num_exp,
        den,
        (0u16.wrapping_sub(nrm)) as i16,
        &mut e,
    );
    if e >= 1 {
        return 0;
    }
    let val = shift_i32_l1_batch4((q as i32) << 16, e);
    if val == i32::MIN {
        return 0xfffe;
    }
    ((val.wrapping_neg().wrapping_add(0x7fff_0000) >> 16) as u32 & 0xffff) as u16
}

/// `0x1030eed0` — convert an accumulated log-energy into a block-floating gain.
///
/// Combines the two 32-bit log terms at `state[0x9a]`/`state[0x9b]` (halved so
/// their sum cannot overflow); on the second half-frame the sum is instead
/// re-referenced against `state[0x89] + state[0x91]`. Subtracting `0x12000000`
/// puts it in the log domain the exponential expects: a non-negative remainder
/// means the gain has railed, and `0x4000 · 2^1` is returned. Otherwise the
/// value is scaled by the split 32×16 constant `0xaac0.5560` (≈ 0.6669),
/// shifted down by 9 and fed to `pow2_frac` (`0x1030a7a0`); the mantissa is
/// renormalised and clamped from below at `0x51eb · 2^-6` (≈ 0.32) using
/// `compare_scaled` (`0x1030a4f0`).
///
/// Returns the exponent; the mantissa is written to `out`. The original leaves
/// stale bits in the high half of `eax` on the non-clamped path, so only the
/// low 16 bits are meaningful.
pub fn log_energy_to_gain(out: &mut i16, state: &[i16], half: i16) -> i16 {
    let mut acc =
        (((state[0x9b] as i32) << 16) >> 1).wrapping_add(((state[0x9a] as i32) << 16) >> 1);
    if half > 0 {
        let c = (state[0x91] as i32).wrapping_add(state[0x89] as i32);
        acc = (c.wrapping_mul(0x10000).wrapping_sub(acc >> 1)).wrapping_mul(2);
    }
    let d = acc.wrapping_sub(0x1200_0000);
    if d >= 0 {
        *out = 0x4000;
        return 1;
    }
    let hi = ((d >> 16) as i16 as i32).wrapping_mul(0xaac0);
    let lo = ((d as u32 & 0xffff) as i32).wrapping_mul(0x5560) >> 15;
    let mut exp: i16 = 0;
    let m = pow2_frac(hi.wrapping_add(lo) >> 9, &mut exp);
    let nrm = if m == 0 {
        0u16
    } else {
        (norm_shift(m as u32) as u16) & 0xffff
    };
    let e = exp.wrapping_sub(nrm as i16);
    let mant = ((m.wrapping_shl(nrm as u32 & 0x1f)) >> 16) as i16;
    if compare_scaled(mant, e, 0x51eb, -6) {
        *out = 0x51eb;
        return -6;
    }
    *out = mant;
    e
}

/// `0x1030dd30` — the speech-activity gate.
///
/// Three tests over the tracker state decide whether the frame carries speech:
/// the largest of the sixteen band levels at `state[0x29..0x39]` (mantissas)
/// with exponents at `state[0x39..0x49]`, the total of the eight band energies
/// at `state[0xa8..0xb0]`/`state[0xb8..0xc0]` summed with `add_norm`, and the
/// long/short-term level pair at `state[0x24]`/`state[0x28]`. The total above
/// `0x7a12·2^-10` is speech outright; above `0xc35·2^-10` it is speech if it
/// also exceeds `0.55 ·` the stored threshold at `state[0x9e..0x9f]`. Below
/// that, a loud enough peak (`> -0x5000·2^5`) or a level that has risen by more
/// than `0xf00` also counts.
///
/// `*flag` is preloaded from `state[0xc0]` and set to 1 when the gate fires.
/// On a positive decision the mean of the first eight band levels, aligned to
/// exponent 5 and floored at `-0x7000`, is written to `*level` with `*exp = 5`,
/// and 1 is returned; a rejected frame returns 0 and writes nothing.
pub fn speech_activity_gate(state: &[i16], level: &mut i16, exp: &mut i16, flag: &mut i16) -> i32 {
    let mut best_m = state[0x29];
    let mut best_e = state[0x39];
    for i in 0..15 {
        if mant_exp_gt(state[0x2a + i], state[0x3a + i], best_m, best_e) {
            best_m = state[0x2a + i];
            best_e = state[0x3a + i];
        }
    }

    let mut tot_e: i16 = 0;
    let mut tot_m: i16 = 0;
    for j in 0..8 {
        tot_m = add_norm(tot_m, tot_e, state[0xa8 + j], state[0xb8 + j], &mut tot_e);
    }
    *flag = state[0xc0];

    let mut active = mant_exp_gt(tot_m, tot_e, 0x7a12, -10);
    if active {
        *flag = 1;
    } else {
        if mant_exp_gt(tot_m, tot_e, 0xc35, -10) {
            let t = (((state[0x9e] as i32).wrapping_mul(0x8ccc) >> 16) as u32 & 0xffff) as u16;
            if mant_exp_gt(tot_m, tot_e, t as i16, state[0x9f]) {
                *flag = 1;
                active = true;
            }
        }
        if !active
            && (state[0x28] as i32).wrapping_add(0x1e00) <= 0
            && !mant_exp_gt(best_m, best_e, 0xb000u16 as i16, 5)
            && (state[0x24] as i32)
                .wrapping_sub(state[0x28] as i32)
                .wrapping_sub(0xf00)
                >= 0
        {
            return 0;
        }
    }

    let mut acc: i64 = 0;
    for k in 0..8 {
        let v = shift_i32_l1_batch4(
            (state[0x29 + k] as i32) << 16,
            state[0x39 + k].wrapping_sub(5),
        );
        acc = acc.wrapping_add(v as i64);
    }
    let mut r = (((acc >> 3) as u32 as i32) >> 16) as i16;
    if r < -0x7000 {
        r = -0x7000;
    }
    *level = r;
    *exp = 5;
    1
}

/// Exponent-align shift with x86 semantics: left for a non-negative count,
/// arithmetic right for a negative one, full sign fill past -30. The count is
/// masked to five bits exactly as `shl`/`sar` do.
fn shift_i32_l1_batch4(v: i32, sh: i16) -> i32 {
    if sh < 0 {
        if sh < -0x1e {
            v >> 31
        } else {
            v >> ((sh as i32).wrapping_neg() as u32 & 0x1f)
        }
    } else {
        v.wrapping_shl(sh as u32 & 0x1f)
    }
}

// ---------------------------------------------------------------------------
// Shared idioms (same shapes as the layer-0 batches; repeated here so this
// module does not depend on another batch's private helpers).
// ---------------------------------------------------------------------------

/// `0x1e - bsr(|x|)`, the library's block-exponent search, with its two
/// degenerate inputs intact: `x == 0` and `x == -1` fall through the
/// `if (mag != 0)` guard with the search variable still at 31 and yield -1.
/// Call sites that guard `x == 0` themselves are written to match.
fn norm_shift(x: u32) -> i32 {
    let mag = if (x as i32) < 0 { !x } else { x };
    let mut i = 31i32;
    if mag != 0 {
        while mag >> i == 0 {
            i -= 1;
        }
    }
    0x1e - i
}

/// The 64-bit accumulate-then-normalise idiom shared by `0x10317140` and
/// `0x1031c310`: sign-extend and sum `n` 32-bit words into a 64-bit
/// accumulator, ask `norm64` (`0x10310150`) how far it must move to normalise,
/// shift it there with `__allshl`/`__allshr`, and *subtract* the shift from the
/// caller's running exponent. Returns the low half of the shifted accumulator,
/// which is where both callers take their result from.
fn sum_norm64(x: &[i32], n: i16, exp: &mut i16) -> u32 {
    let mut acc: i64 = 0;
    if n > 0 {
        for i in 0..n as u16 as usize {
            acc = acc.wrapping_add(x[i] as i64);
        }
    }
    let s = norm64(acc as u32, (acc >> 32) as u32) as i16;
    let r = if s < 0 {
        allshr(acc, ((s as i32).wrapping_neg() as u32 & 0xff) as u8)
    } else {
        allshl_l1_batch4(acc, (s as u32 & 0xff) as u8)
    };
    *exp = exp.wrapping_sub(s);
    r as u32
}

/// The four `i16` at `0x1037f970` — the 2^(k/4) fixup factors
/// `log2_transform` uses to fold a quarter-exponent back into the mantissa.
const LOG_FIXUP: [i16; 4] = [16977, 19484, 23170, 27554];

// ======= pulled in from l1_batch5 (encoder-only) =======

/// `0x1031c390` — `1 - num/den` in Q15, floored at zero.
///
/// Divides with [`div_norm`] and, provided the quotient's exponent is
/// non-positive (i.e. the ratio is below 1), brings it to Q15 and subtracts it
/// from `0x7fff`. A non-positive denominator short-circuits to `0x7fff` (ratio
/// zero) and a ratio at or above 1 to `0`. Only the low 16 bits of the return
/// are meaningful.
pub fn one_minus_ratio(num: i16, num_exp: i16, den: i16, den_exp: i16) -> u16 {
    if den < 1 {
        return 0x7fff;
    }
    let mut e: i16 = 0;
    let q = div_norm(up16_l1_batch5(num), num_exp, den, den_exp, &mut e);
    if e > 0 {
        return 0;
    }
    let v = shift_i32_l1_batch5(up16_l1_batch5(q), e);
    if v == i32::MIN {
        return 0xfffe;
    }
    ((0x7fff_0000u32 as i32).wrapping_sub(v) >> 16) as u16
}

/// `0x1030f120` — per-band voicing metric: a floor over the low three bands and
/// a normalised ratio over the upper five.
///
/// Bands 0..2 take `min(a[i], c[i])` — the plain smaller of the two energies.
/// Bands 3..7 form two block-floating sums,
/// `S1 = d[k]@exp_d + f[k]@exp_f` and `S2 = a[k]·d[k]@exp_d + c[k]·f[k]@exp_f`,
/// each aligned to one above the larger exponent (the guard bit against the
/// add overflowing), and store `S2 / S1` brought back to Q15 with a saturating
/// shift. A zero `S1` mantissa yields `0x7fff` rather than dividing.
///
/// In both halves `w[i] = high16_l1_batch5(2·d[i]·coef)` is written; if all eight come out
/// zero they are forced to 1, so a downstream divide by the weight cannot fault.
#[allow(clippy::too_many_arguments)]
pub fn band_ratio_weights(
    a: &[i16],
    out: &mut [i16],
    c: &[i16],
    d: &[i16],
    w: &mut [i16],
    f: &[i16],
    exp_d: i32,
    exp_f: i32,
    coef: i16,
) {
    for i in 0..3usize {
        out[i] = if a[i] < c[i] { a[i] } else { c[i] };
        w[i] = high16_l1_batch5((d[i] as i32).wrapping_mul(coef as i32).wrapping_mul(2));
    }

    // Align two block-floating values to one above the larger exponent and add.
    let combine = |x: i32, ex: i32, y: i32, ey: i32| -> (i32, u32) {
        let nx = nrm(x as u32);
        let ny = nrm(y as u32);
        let ax = x.wrapping_shl(nx as u32 & 0x1f);
        let ay = y.wrapping_shl(ny as u32 & 0x1f);
        let e1 = ex.wrapping_sub(nx as i32);
        let e2 = ey.wrapping_sub(ny as i32);
        let top = if (e1 as i16) <= (e2 as i16) { e2 } else { e1 };
        let common = (top as i16 as i32 + 1) as u32 & 0xffff;
        let sx = shift_i32_l1_batch5(ax, ((e1 & 0xffff) - common as i32) as i16);
        let sy = shift_i32_l1_batch5(ay, ((e2 & 0xffff) - common as i32) as i16);
        (sx.wrapping_add(sy), common)
    };

    for k in 3..8usize {
        let dk = d[k] as i32;
        let (s1, e_s1) = combine(dk.wrapping_shl(16), exp_d, up16_l1_batch5(f[k]), exp_f);
        let n1 = nrm(s1 as u32);
        let num = s1.wrapping_shl(n1 as u32 & 0x1f);

        let p = (a[k] as i32).wrapping_mul(dk).wrapping_mul(2);
        let q = (c[k] as i32).wrapping_mul(f[k] as i32).wrapping_mul(2);
        let (s2, e_s2) = combine(p, exp_d, q, exp_f);
        let n2 = nrm(s2 as u32);

        out[k] = if high16_l1_batch5(num) == 0 {
            0x7fff
        } else {
            let mut e: i16 = 0;
            let r = div_norm(
                s2.wrapping_shl(n2 as u32 & 0x1f),
                e_s2.wrapping_sub(n2 as u32) as i16,
                high16_l1_batch5(num),
                e_s1.wrapping_sub(n1 as u32) as i16,
                &mut e,
            );
            high16_l1_batch5(shl_sat(up16_l1_batch5(r), e))
        };
        w[k] = high16_l1_batch5((d[k] as i32).wrapping_mul(coef as i32).wrapping_mul(2));
    }

    let mut any: u16 = 0;
    for i in 0..8usize {
        any |= w[i] as u16;
    }
    if any == 0 {
        for i in 0..8usize {
            w[i] = 1;
        }
    }
}

/// `0x1030df20` — ratio of the sums of two 128-element 32-bit blocks.
///
/// Each block is summed in 64 bits, normalised, and reduced to a
/// mantissa/exponent pair; the quotient is then formed with [`div_norm`] and its
/// exponent written to `exp`. A denominator whose normalised mantissa is zero is
/// replaced by the fixed floor `0x68db · 2^-13` (≈ 0.82 · 2^-13), which keeps
/// the divide alive on an all-zero block instead of faulting.
///
/// Returns the quotient mantissa (the original leaves it in `ax` and Ghidra
/// types the function `void`).
pub fn ratio_of_block_sums(
    num: &[i16],
    num_exp: i32,
    den: &[i16],
    den_exp: i16,
    exp: &mut i16,
) -> i16 {
    let dsum: i64 = (0..0x80).map(|i| rd32(den, 2 * i) as i64).sum();
    let (dlo, dsh) = norm_sum64(dsum);
    let mut dm = ((dlo >> 16) & 0xffff) as i16;
    let mut de = den_exp.wrapping_sub(dsh);
    if dm == 0 {
        dm = 0x68db;
        de = -13;
    }

    let nsum: i64 = (0..0x80).map(|i| rd32(num, 2 * i) as i64).sum();
    let (nlo, nsh) = norm_sum64(nsum);
    div_norm(
        nlo as i32,
        num_exp.wrapping_sub(nsh as u16 as i32) as i16,
        dm,
        de,
        exp,
    )
}

/// `0x1031b5c0` — `a / (b - a)` as a percentage, capped at 100.
///
/// Both operands are normalised, divided with [`div_norm`], and the quotient is
/// accepted only if its exponent is below 8 — i.e. the ratio is under 256. At
/// exponent exactly 7 the mantissa is capped at `0x6400`, which with that
/// exponent is 100.0; every rejected case returns that same cap with exponent 7.
/// A negative (normalised) numerator yields 0 rather than a negative percentage.
/// The exponent is written to `exp`.
///
/// `a` is treated as a full 32-bit value, `b - a` likewise; `b <= a` takes the
/// capped path.
pub fn ratio_percent(a: u32, b: i32, exp: &mut i16) -> i16 {
    let d = (b as u32).wrapping_sub(a);
    let mut shifted = a;
    if (d as i32) > 0 {
        let na = nrm(a);
        // The difference is known positive here, so the original's magnitude
        // fold is skipped and the top-bit search runs on `d` directly.
        let mut i = 31i32;
        if d != 0 {
            while d >> i == 0 {
                i -= 1;
            }
        }
        let nd = (0x1eu32.wrapping_sub(i as u32) & 0xffff) as u16;
        shifted = a.wrapping_shl(na as u32 & 0x1f);
        let mut e: i16 = 0;
        let mut q = div_norm(
            shifted as i32,
            (na as i16).wrapping_neg(),
            high16_l1_batch5(d.wrapping_shl(nd as u32 & 0x1f) as i32),
            (nd as i16).wrapping_neg(),
            &mut e,
        );
        if e < 8 {
            if e == 7 && (q as u16) > 0x63ff {
                q = 0x6400;
            }
            if (shifted as i32) < 0 {
                q = 0;
            }
            *exp = e;
            return q;
        }
    }
    *exp = 7;
    if (shifted as i32) < 0 {
        0
    } else {
        0x6400
    }
}

/// `0x1031da80` — attack/decay tracker for a block-floating-point level.
///
/// `(*mant, *exp)` is the tracked level and `new@new_exp` the frame's value.
/// Three regimes:
///
/// - `new` above `level - 1.0` (the [`sub_norm`] of `0x4000@2`): the tracker
///   jumps straight to `new` — instant attack.
/// - `new` above `level + thr@7`: the level moves to `0.9·level + 0.1·new`
///   (`0xcccc` is 0.8 and the extra `-3` on its exponent makes it 0.1).
/// - otherwise: the level is left alone.
///
/// `0xe666` and `0xcccc` are *unsigned* Q16 fractions (0.9, 0.8), not negative
/// Q15 values. Returns `hold - 1`, the caller's hangover counter.
pub fn track_level_ar(
    mant: &mut i16,
    exp: &mut i16,
    new: i16,
    new_exp: i16,
    hold: i32,
    thr: i16,
) -> i32 {
    let m = *mant;
    let mut e = *exp;

    let mut e_lo: i16 = 0;
    let lo = sub_norm(0x4000, 2, m, e, &mut e_lo);
    let mut e_hi: i16 = 0;
    let hi = add_norm(thr, 7, m, e, &mut e_hi);

    let (out_m, out_e) = if !compare_scaled(new, new_exp, lo, e_lo) {
        let mut r = m;
        if compare_scaled(new, new_exp, hi, e_hi) {
            let a = (m as i32).wrapping_mul(0xe666u32 as i32);
            let na = nrm(a as u32);
            let ea = e.wrapping_sub(na as i16);
            let ma = high16_l1_batch5(a.wrapping_shl(na as u32 & 0x1f));

            let b = (new as i32).wrapping_mul(0xccccu32 as i32);
            let nb = nrm(b as u32);
            let eb = new_exp.wrapping_sub(nb as i16).wrapping_sub(3);

            r = add_norm(
                ma,
                ea,
                high16_l1_batch5(b.wrapping_shl(nb as u32 & 0x1f)),
                eb,
                &mut e,
            );
        }
        (r, e)
    } else {
        (new, new_exp)
    };

    *mant = out_m;
    *exp = out_e;
    hold.wrapping_sub(1)
}

/// `0x10320130` — sum a vector in contiguous bands of width `step`.
///
/// `step` arrives as a mantissa/exponent pair and is brought to Q16. The first
/// band starts at `0.75·step + 1` and each subsequent one begins half a step
/// further, so the bands tile the range `[0.75·step, count/2)` without gaps; the
/// last is truncated at `count/2`. Each band's words are summed with
/// [`sum_i16`] and the band sums accumulated.
///
/// Domain: `step` must be at least 1 after scaling or the walk never advances;
/// `count/2` must not exceed the vector length.
pub fn sum_bands(v: &[i16], step: i16, step_exp: i16, count: i16) -> i32 {
    let half = count >> 1;
    let s = shift_i32_l1_batch5(up16_l1_batch5(step), step_exp.wrapping_sub(0xf));
    let mut acc = 0i32;
    let mut pos = s.wrapping_sub(s >> 2).wrapping_add(0x10000);
    while high16_l1_batch5(pos) < half {
        let mut nxt = pos.wrapping_add(s >> 1);
        let hi = if half < high16_l1_batch5(nxt) {
            half as u16 as u32
        } else {
            ((nxt as u32) >> 16) & 0xffff
        };
        nxt = nxt.wrapping_add(s >> 1);
        let start = high16_l1_batch5(pos);
        let cnt = hi.wrapping_sub((pos as u32 >> 16) & 0xffff) as i16;
        acc = acc.wrapping_add(sum_i16(&v[start as usize..], cnt));
        pos = nxt;
    }
    acc
}

/// `0x10311170` — map a pitch lag to a 0..207 quantiser index.
///
/// `0x8000000 / lag` (halved by [`div32_half`], then truncated to 16 bits) is
/// the fundamental frequency; scaling it by 2^11, subtracting `0x4e0000` and
/// shifting down 17 gives an index that is clamped to `[0, 0xcf]`. The clamp
/// bounds are the pitch codebook's size.
///
/// Domain: `lag != 0` — the division faults on a zero divisor.
pub fn lag_to_pitch_index(lag: i16) -> i16 {
    let q = div32_half(0x800_0000, lag) as i16;
    let v = ((((q as i32) << 16) >> 5).wrapping_sub(0x4e_0000) >> 0x11) as i16;
    if v > 0xcf {
        return 0xcf;
    }
    if v < 0 {
        return 0;
    }
    v
}

/// `0x1031e8c0` — log of a 16-block mean energy, as a biased Q0 value.
///
/// The block is summed in 64 bits with `-v[0]/2` pre-loaded (so `v[0]` ends up
/// weighted 0.5 while the other fifteen count fully), normalised, divided by
/// `den`, and passed through [`log2_q`]. The logarithm is then renormalised and
/// scaled by `0xc0a8` — an *unsigned* Q16 fraction, 0.7526 — with the
/// normalisation exponent as the shift, and biased by `0x4ba7000`. A zero
/// quotient or an exponent below -31 returns the floor `-0x59ff`.
///
/// The `iVar8 / 0xaa16 + (iVar8 >> 31)` in the decompilation is the decompiler's
/// rendering of `(iVar8 * 0xc0a8) >> 31`; that branch is only reached when the
/// shift count is exactly 31.
///
/// Domain: `den` must be neither `0` nor `-1` — both normalise to a zero
/// divisor mantissa and fault.
pub fn log_mean_energy(v: &[i32], exp: i32, den: i16) -> i32 {
    let x0 = v[0];
    let neg = if x0 as u32 == 0x8000_0000 {
        0x7fff_ffff
    } else {
        x0.wrapping_neg()
    };
    let mut acc: i64 = (neg >> 1) as i64;
    for &x in &v[..0x10] {
        acc = acc.wrapping_add(x as i64);
    }
    let (lo, sh) = norm_sum64(acc);

    let mut e = (exp.wrapping_sub(sh as u16 as i32).wrapping_add(1) & 0xffff) as u16 as i16;
    let dv = up16_l1_batch5(den);
    let nd = nrm(dv as u32);
    let q = div_norm(
        lo as i32,
        e,
        high16_l1_batch5(dv.wrapping_shl(nd as u32 & 0x1f)),
        (0xf - nd as i32) as i16,
        &mut e,
    );
    if q == 0 || e < -0x1f {
        return -0x59ff;
    }

    let l = log2_q(q, e);
    let n = nrm(l as u32);
    let m = high16_l1_batch5(l.wrapping_shl(n as u32 & 0x1f)) as i32;
    let s = ((0xfu32.wrapping_sub(n as u32) & 0xffff) as i32).wrapping_sub(5);
    let scaled = m.wrapping_mul(0xc0a8u32 as i32);
    let t = if (s as i16) >= 0 {
        scaled.wrapping_shl(s as u32 & 0x1f)
    } else if (s as i16) < -0x1e {
        scaled >> 31
    } else {
        scaled >> ((s as u8).wrapping_neg() as u32 & 0x1f)
    };
    t.wrapping_add(0x4ba_7000) >> 16
}

/// `0x103145a0` — expand eight band voicing decisions into a 128-bin mask.
///
/// Band `k` is voiced (`0x4000`) when its amplitude `amp[k]` reaches `0x1999`
/// (0.2) *and* its level `lvl[k]` is representable at exponent `exp` — the
/// `exp - norm < -0x2a` test rejects a band more than 42 binary points below
/// scale, and a `lvl[k]` of 0 or -1 rejects outright (for -1 the normalisation
/// shift is the degenerate 31, which shifts the value to zero).
///
/// Bin boundaries are 0, 13, 29, 45, 61, 77, 93, 109, 128: the first band is 13
/// bins wide, the middle six are 16, and the last runs to the end of the
/// 128-bin half-spectrum.
pub fn voiced_band_mask(dst: &mut [i16], amp: &[i16], lvl: &[i16], exp: i16) {
    let mut lo: u32 = 0;
    let mut hi: u32 = 0xd;
    for k in 0..8usize {
        if k == 7 {
            hi = 0x80;
        }
        let v = up16_l1_batch5(lvl[k]);
        let n = if v == 0 {
            0
        } else {
            (0x1e - top_bit_l1_batch5(v as u32)) as i16
        };
        let word = if v.wrapping_shl(n as u32 & 0x1f) == 0 {
            0
        } else if exp.wrapping_sub(n) < -0x2a {
            0
        } else if amp[k] < 0x1999 {
            0
        } else {
            0x4000
        };
        fill16(
            &mut dst[(lo as i16) as usize..],
            word,
            hi.wrapping_sub(lo) as i16,
        );
        lo = hi & 0xffff;
        hi = hi.wrapping_add(0x10);
    }
}

/// Exponent-align shift: left for a positive count, arithmetic right for a
/// negative one, full sign fill past -30. The count is masked to five bits as
/// x86 does.
fn shift_i32_l1_batch5(v: i32, sh: i16) -> i32 {
    if sh < 0 {
        if sh < -0x1e {
            v >> 31
        } else {
            v >> ((sh as i32).wrapping_neg() as u32 & 0x1f)
        }
    } else {
        v.wrapping_shl(sh as u32 & 0x1f)
    }
}

/// Sum 128 32-bit words into 64 bits, normalise with [`norm64`], and return the
/// low 32 bits of the shifted sum together with the shift. This is the shared
/// prologue of `0x1030df20` and `0x1031e8c0`; the original inlines a call to
/// `__allshl` / `__allshr` and then reads only `eax`.
fn norm_sum64(sum: i64) -> (u32, i16) {
    let sh = norm64(sum as u32, (sum >> 32) as u32) as i16;
    let shifted = if sh < 0 {
        sum >> ((-sh) as u32 & 0x3f)
    } else {
        ((sum as u64) << (sh as u32 & 0x3f)) as i64
    };
    (shifted as u32, sh)
}

/// Saturating 32-bit add in the original's overflow-detect form, saturating
/// towards the sign of the operands.
fn sat_add(a: i32, b: i32) -> i32 {
    let s = a.wrapping_add(b);
    if ((s ^ a) & (s ^ b)) < 0 {
        if a < 0 {
            i32::MIN
        } else {
            i32::MAX
        }
    } else {
        s
    }
}
