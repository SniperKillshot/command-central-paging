//! Decoder layers: payload unpacking, parameter reconstruction, synthesis.
#![allow(dead_code, clippy::all)]

use crate::engine::dsp::*;

// ======================= from aL2b1 =======================

/// `0x103813f0` — per-coefficient scale for the five scalar-quantised
/// higher-order cepstral coefficients. Monotonically decreasing: the later
/// coefficients carry less energy and get a proportionally smaller step.
pub const HOC_SCALE: [i16; 5] = [20316, 13173, 11010, 9503, 8651];

/// `0x1038140e` — step-size normaliser indexed by the *number of bits* a field
/// gets, i.e. the standard-deviation multiplier of a uniform quantiser with
/// `2^b` levels. Index 0 is unused (a zero-bit field is skipped). Entries
/// saturate at 20972 from 7 bits up; the table runs out at index 12, where the
/// packed-nibble allocation table at `0x10381428` begins.
pub const HOC_STEP: [i16; 13] = [
    0, 4915, 6963, 10650, 13107, 18350, 19661, 20972, 20972, 20972, 20972, 20972, 20972,
];

/// x86 variable arithmetic shift of an `i32`: a positive count shifts left, a
/// negative count shifts right, both with the count masked to five bits, and a
/// count below `-30` is a full sign fill (SPEC §4).
#[inline]
pub fn shift_i32_al2b1(v: i32, sh: i16) -> i32 {
    if sh >= 0 {
        ((v as u32) << (sh as u32 & 0x1f)) as i32
    } else if sh < -0x1e {
        v >> 31
    } else {
        v >> ((sh.wrapping_neg() as u32) & 0x1f)
    }
}

/// `0x1030b8e0` — install the frame's bit-allocation descriptor and the
/// default AGC configuration.
///
/// Called from both `dec_init` (`0x10309cd0`) and `enc_init` (`0x10309ea0`).
/// `mode == 0` selects IMBE (four fields, 88 payload bits,
/// [`imbe_bit_alloc`]); any other value selects AMBE half-rate
/// with sub-mode 0 (nine fields, 49 payload bits,
/// [`ambe_bit_alloc`]) — note the AMBE sub-mode is hard-wired to
/// 0 here, so the reduced 48/47/46/45-bit variants are not reachable through
/// `init`. The AGC block is written unconditionally, before the branch.
///
/// Returns 0 always (`xor eax,eax`).
pub fn init_bit_alloc_and_agc(mode: i16, alloc: &mut [i16; 13], agc: &mut [i16]) -> i32 {
    init_agc_config(agc);
    if mode == 0 {
        imbe_bit_alloc(alloc);
    } else {
        ambe_bit_alloc(alloc, 0);
    }
    0
}

/// `0x1031a2f0` — forward real-input FFT of `2^log2n` samples.
///
/// A real transform of length `N = 2^log2n` is computed as a complex transform
/// of length `N/2` ([`fft`], `0x1031a370`) followed by one conjugate-symmetric
/// merge stage ([`fft_merge_stage`], `0x1031a1b0`, variant 0). `buf` holds the
/// `N` real samples on entry and the `N/2 + 1` complex bins on exit, so it must
/// be at least `N + 2` words long.
///
/// `stage_flag` selects how the DC/Nyquist pair is handled: with `stage_flag`
/// zero the merge writes the Nyquist bin out to `buf[N]`/`buf[N+1]` and the
/// caller then also clears `buf[1]` (the imaginary part of DC, which is
/// identically zero for real input).
///
/// Both call results are **zero-extended from 16 bits** (`movzx eax,ax`) before
/// use, so the running block-exponent credit that [`fft`] returns is truncated
/// to `u16` on the way in — see SPEC §4.3, this counter is not a small signed
/// exponent.
///
/// Domain: `4 <= log2n <= 8` — the inner complex FFT wants `log2n - 1` in
/// `[2, 7]`.
pub fn rfft_forward(buf: &mut [i16], exp_in: i32, log2n: i16, stage_flag: i16) -> i32 {
    let c = fft(buf, exp_in, log2n.wrapping_sub(1)) as u16 as i32;
    let r = fft_merge_stage(buf, c, log2n, 0, stage_flag) as u16 as i32;
    if stage_flag == 0 {
        buf[1] = 0;
    }
    r
}

/// `0x103176b0` — turn a vector of log2-domain harmonic magnitudes into linear
/// magnitudes sharing one block exponent, and zero-pad it to 56 harmonics.
///
/// Reached from the synthesiser (`0x1030c520`). Three passes over `v[0..n]`:
///
/// 1. Clamp every log value to `±0x77ff` — a *symmetric* clamp one notch inside
///    full scale, not the usual `i16` saturation (SPEC §4, per-operation).
/// 2. Take the maximum, promoted to Q31 (`x << 16`), starting from `i32::MIN`.
/// 3. The shared output exponent is `high16_al2b1((max >> 11) + 0x10000)`, i.e.
///    `ceil` of the largest log value in the coarse units the exponential
///    wants, plus one guard bit. Every element then becomes
///    `pow2_scaled(v[i], 4, out_exp)` (`0x1030a890`), truncated to 16 bits.
///
/// `*exp` receives `out_exp - 15`, the block exponent of the resulting Q15
/// mantissas. The tail `v[n..56]` is zeroed so the harmonic loops downstream
/// can run to the codec's maximum *L* unconditionally.
///
/// `n` is signed: a non-positive count skips all three passes, leaves the
/// exponent at `high16_al2b1(i32::MIN >> 11 + 0x10000) - 15 = -30`, and zero-fills
/// the whole 56-word vector.
pub fn expand_log_mags(v: &mut [i16], exp: &mut i16, n: u16) {
    let cnt = n as i16;
    if cnt > 0 {
        for x in v[..n as usize].iter_mut() {
            *x = if *x >= 0x7800 {
                0x77ff
            } else if *x < -0x77ff {
                -0x77ff
            } else {
                *x
            };
        }
    }

    let mut max = i32::MIN;
    if cnt > 0 {
        for &x in &v[..n as usize] {
            let p = (x as i32) << 16;
            if max < p {
                max = p;
            }
        }
    }
    let scaled = (max >> 0xb).wrapping_add(0x10000);
    let out_exp = high16_al2b1(scaled);

    if cnt > 0 {
        for x in v[..n as usize].iter_mut() {
            *x = pow2_scaled(*x, 4, out_exp) as i16;
        }
    }
    *exp = out_exp.wrapping_sub(0xf);
    fill16(&mut v[cnt as usize..], 0, 0x38i16.wrapping_sub(cnt));
}

/// `0x103114d0` — scalar quantise (or dequantise) the five higher-order
/// cepstral coefficients through the frame's bit cursor.
///
/// Five coefficients live at `params[1..6]`; `nbits[i]` is the field width
/// allocated to coefficient `i`. The quantiser step for coefficient `i` is
///
/// ```text
/// step = (HOC_STEP[nbits[i]] * HOC_SCALE[i] * 2) >> 17
/// ```
///
/// — the per-coefficient standard deviation times the uniform-quantiser
/// normaliser for that many levels.
///
/// With `encode` set, the index is `(x/2 + step/2) / step` computed by
/// `nbits[i]` [`subc`] steps over a Q16 accumulator: a negative numerator
/// floors to index 0 and a numerator at or past the top of the range clamps to
/// `2^b - 1`, which is the codec's *floor*-toward-negative-infinity quantiser
/// (SPEC §4). The index is then emitted at the cursor with [`emit_bits`].
/// With `encode` clear the index is read back from the cursor with
/// [`read_field_l1_batch4`] instead.
///
/// Either way the coefficient is **overwritten with the reconstruction**
/// `(2*idx + 1 - 2^b) * step`, i.e. the centre of the selected cell, scaled by
/// `2^(14 - nbits[i])` with x86 variable-shift semantics and kept as `high16_al2b1`.
/// A zero-bit field therefore reconstructs as exactly `-step << 14`… and in
/// practice `HOC_STEP[0]` is 0, so a zero-bit coefficient reconstructs to 0.
///
/// `2^b` is formed as a **16-bit** `shl ax,cl`, so a width of 16 or more gives
/// 0, and it is then *sign*-extended (`cwde`) in the reconstruction.
///
/// Domain: `nbits[i]` in `0..=12` — index 13 and up runs off the end of
/// `HOC_STEP` into the packed-nibble allocation table at `0x10381428`.
pub fn code_higher_order_coeffs(
    bits: &mut [i16],
    pos: &mut i32,
    nbits: &[u16; 5],
    params: &mut [i16],
    encode: bool,
) {
    for i in 0..5usize {
        let b = nbits[i];
        let prod = (HOC_STEP[b as usize] as i32)
            .wrapping_mul(HOC_SCALE[i] as i32)
            .wrapping_mul(2);
        // 2^b as a 16-bit shift: a count of 16..31 shifts everything out.
        let levels: u16 = if (b & 0x1f) >= 16 {
            0
        } else {
            1u16 << (b & 0x1f)
        };
        let step = (prod >> 0x11) as i16;

        let idx: u16 = if !encode {
            read_field_i16(bits, pos, b as i16)
        } else {
            let acc = ((params[1 + i] as i32) << 16).wrapping_add(((step as i32) << 16) >> 1);
            let v = if acc < 0 {
                0u16
            } else if high16_al2b1(acc) < step {
                let mut a = acc;
                if (b as i16) > 0 {
                    for _ in 0..b {
                        subc(step, &mut a);
                    }
                }
                a as u16
            } else {
                levels.wrapping_sub(1)
            };
            emit_bits_i16(bits, pos, b as i16, v as i16);
            v
        };

        let centre = high16_al2b1(
            ((idx as i16 as i32) << 17)
                .wrapping_sub((levels as i16 as i32) << 16)
                .wrapping_add(0x10000),
        );
        let recon = (centre as i32).wrapping_mul(step as i32).wrapping_mul(2);
        params[1 + i] = high16_al2b1(shift_i32_al2b1(recon, (0xei32 - b as i32) as i16));
    }
}

/// [`read_field_l1_batch4`] (`0x103102e0`) over an `i16` bit array addressed by a word
/// cursor. Only the `n` words the field occupies are ever touched.
pub fn read_field_i16(bits: &[i16], pos: &mut i32, n: i16) -> u16 {
    let r = pack_bits_i16(&bits[*pos as usize..], n);
    *pos = pos.wrapping_add(n as i32);
    r
}

/// [`emit_bits`] over an `i16` bit array with a word cursor.
pub fn emit_bits_i16(bits: &mut [i16], pos: &mut i32, n: i16, word: i16) {
    let mut c = *pos as usize;
    emit_bits(&mut c, bits, n, word);
    *pos = c as i32;
}

#[inline]
pub fn rd32_al2b1(b: &[i16], w: usize) -> u32 {
    (b[w] as u16 as u32) | ((b[w + 1] as u16 as u32) << 16)
}

#[inline]
pub fn wr32_al2b1(b: &mut [i16], w: usize, v: u32) {
    b[w] = v as u16 as i16;
    b[w + 1] = (v >> 16) as u16 as i16;
}

/// `0x103153a0` — the band voicing decision: decide which three-harmonic bands
/// are voiced, subtract the voicing threshold out of the ones that are, and
/// publish the decision as a dibit pattern into both parameter blocks.
///
/// `cur` and `prev` are 68-byte parameter blocks viewed as `i16`: word 2 is the
/// harmonic count *L*, words 4/5 a 32-bit voicing dibit pattern, word 6 the
/// fundamental `w0`, and words `8..` the per-harmonic magnitudes. `st` is the
/// encoder state (only `+0x08`, a dibit history word, and `+0x88`, a frame
/// counter, are read). `mask` is the running band mask, read and written.
///
/// **Guard.** Nothing happens unless the frame is loud enough
/// (`thresh > 0x28f`) or the talkspurt is at least four frames old
/// (`frames >= 4`); *and* neither parameter block already carries a voiced band
/// (`pattern & 0xaaaaaaaa`, the odd bit of every dibit); *and* one of: the state
/// counter is positive, `cur`'s pattern is non-zero, or `w0` falls outside the
/// narrow band `[0x3d63, 0x40b5]`.
///
/// **Per-band test.** Bands run from the top down: band `triads-1` covers
/// harmonics `3*(triads-1) ..= 55`, and band `k` covers `3k ..= 3k+2`, where
/// `triads = triads_capped(L)` (`0x10312480`). For each band the peak magnitude
/// is taken (initialised to `-0x5000`, ties keeping the *first*, i.e. highest,
/// index) and the frequency-dependent floor `log2_q11(w0) / 4` (`0x10314ee0`)
/// is subtracted. In the top half of the spectrum (`2*hi > L`) a negative
/// spectral tilt from [`weighted_slope`] (`0x10314d80`) adds a further
/// `slope / 4`. If what remains exceeds the level from [`voicing_level`]
/// (`0x10314e60`) and the band is not already flagged, the band is marked
/// voiced and the floor is **saturatingly subtracted out of every harmonic in
/// it**, each result clamped to the symmetric `±0x77ff`.
///
/// **Publication.** The 16-bit mask goes back to `*mask`. With more than eight
/// bands it is folded onto the eight fixed frequency bands by
/// [`map_voicing_bits`] (`0x10312620`); the previous frame's dibits are
/// gathered by [`gather_dibits`] (`0x10317060`), the pair is run through the
/// voicing codebook [`voicing_pattern`] (`0x103124b0`) with the mask offset by
/// `0x38`, and [`expand_dibits`] (`0x10312570`) writes the *combined* half into
/// `prev`'s pattern word and the *fresh* half into `cur`'s.
///
/// Note the `& 0xffff` on the already-flagged test: the band bit is shifted
/// left in a 32-bit register but tested 16-bit, so a seventeenth band could
/// never see itself as flagged. `triads` caps at 12, so this is unreachable.
#[allow(clippy::too_many_arguments)]
pub fn voicing_decision(
    cur: &mut [i16],
    prev: &mut [i16],
    st: &[i16],
    mask_io: &mut u16,
    thresh: i16,
    frames: i16,
    cfg: &[i16],
    mant: i16,
    exp: i16,
) {
    if thresh <= 0x28f && frames < 4 {
        return;
    }
    let cur_pat = rd32_al2b1(cur, 4);
    if cur_pat & 0xaaaa_aaaa != 0 {
        return;
    }
    if rd32_al2b1(prev, 4) & 0xaaaa_aaaa != 0 {
        return;
    }
    let w0 = cur[6];
    if !(st[0x44] > 0 || cur_pat != 0 || w0 > 0x40b5 || w0 < 0x3d63) {
        return;
    }

    let l = cur[2];
    let slope = weighted_slope(&cur[8..], l) as i16;
    let level = voicing_level(mant, exp, thresh, cfg) as i16;
    let floor = ((log2_q11(w0) as i16 as i32) << 16) >> 2;
    let triads = triads_capped(l);

    let mut bit: u32 = 1;
    let mut mask: u32 = *mask_io as u32;

    let mut k = (triads.wrapping_sub(1)) as u16; // band counter, tested as i16
    if (k as i16) >= 0 {
        let mut lo = (k as i16 as i32).wrapping_mul(3); // first harmonic of the band
        let mut top = (k.wrapping_mul(3)) as u32; // one past the band's last harmonic
        let mut hi: u32 = 0x38;
        loop {
            let prev_top = top;
            // Peak of the band, seeded at -0x5000 — a floor, not a sentinel:
            // a band entirely quieter than that scores as if it were at
            // -0x5000. The scan runs downward and the comparison is strict, so
            // ties keep the highest index; only the value is used, so the
            // tie rule is not observable.
            let mut peak: i16 = -0x5000;
            let last = hi.wrapping_sub(1);
            let mut c = last as u16;
            while (c as i16 as i32) >= lo {
                let v = cur[8 + (c as i16 as i32) as usize];
                if peak < v {
                    peak = v;
                }
                c = c.wrapping_sub(1);
            }
            let mut s = (((peak as i32) << 16).wrapping_sub(floor) >> 16) as i16;
            if ((hi as u16 as i16 as i32) * 2) > (l as i32) && slope < 0 {
                s = ((((slope as i32) << 16) >> 2).wrapping_add((s as i32) << 16) >> 16) as i16;
            }

            if s > level && ((bit as u16) & (mask as u16)) == 0 {
                mask |= bit;
                let mut j = last as u16;
                while (j as i16 as i32) >= lo {
                    let idx = 8 + (j as i16 as i32) as usize;
                    let a = (cur[idx] as i32) << 16;
                    let d = a.wrapping_sub(floor);
                    let r = if ((d ^ a) & (!floor ^ d)) < 0 {
                        0x7fff_ffffi32.wrapping_add((a < 0) as i32)
                    } else {
                        d
                    };
                    let v = high16_al2b1(r);
                    cur[idx] = if v > 0x77ff {
                        0x77ff
                    } else if v < -0x77ff {
                        -0x77ff
                    } else {
                        v
                    };
                    j = j.wrapping_sub(1);
                }
            }

            lo = lo.wrapping_sub(3);
            bit = bit.wrapping_shl(1);
            k = k.wrapping_sub(1);
            hi = (prev_top as u16) as u32;
            top = prev_top.wrapping_sub(3);
            if (k as i16) < 0 {
                break;
            }
        }
    }

    *mask_io = mask as u16;

    // At exactly eight bands the two arms coincide: map_voicing_bits with
    // nbands = 8 reduces to `mask & 0xff` for every L that yields eight triads
    // (22..=24), and voicing_pattern only reads eight bits of its index anyway.
    let mut nbands = triads;
    if nbands > 8 {
        mask = map_voicing_bits(mask as i16, nbands as i32, l) as u16 as u32;
        nbands = 8;
    }
    let prev_dibits = gather_dibits(rd32_al2b1(st, 4) as i32) as u16;
    let pat = voicing_pattern(
        (mask as u16).wrapping_add(0x38) as i16,
        prev_dibits as u32,
        nbands,
    );
    let mut o0 = rd32_al2b1(prev, 4);
    let mut o1 = rd32_al2b1(cur, 4);
    expand_dibits(&mut o0, &mut o1, pat);
    wr32_al2b1(prev, 4, o0);
    wr32_al2b1(cur, 4, o1);
}

/// One `0x10312f30` copy step, with word indices instead of pointers: moves
/// `n` words between the channel array and the scratch (into the scratch when
/// `store`), and returns the advanced channel cursor. A non-positive `n` leaves
/// the cursor alone.
pub fn copy_seg(chan: &mut [i16], o: i32, scratch: &mut [i16], s: i32, n: i16, store: bool) -> i32 {
    copy16_batch4(
        &mut chan[o as usize..],
        &mut scratch[s as usize..],
        n,
        store,
    );
    if n > 0 {
        o.wrapping_add(n as i32)
    } else {
        o
    }
}

/// `0x10312fe0` — interleave (or de-interleave) an **IMBE** 88-bit frame
/// between natural info-vector order and channel order.
///
/// Reached through the two one-line wrappers `0x10312fa0` (`store = true`, the
/// encoder's pack) and `0x10312fc0` (`store = false`, the decoder's unpack).
/// `chan` is the channel-order bit array, one payload bit per `i16`;
/// `hbits[k]` is the number of bits harmonic `k` was allocated (the output of
/// [`unpack_nibbles`], `0x10311b00`); `alloc` is the frame's
/// bit-allocation descriptor, of which word 2 (the payload total, 88), word 5
/// (8) and word 6 (6) are used.
///
/// The routine works through a 192-word stack scratch holding the frame in
/// natural order. Unpacking snapshots `chan` into it first; packing writes it
/// back at the end. The channel image is then laid out as
///
/// | channel words | source |
/// |---|---|
/// | `0 ..6` | scratch `0..6` |
/// | `6 ..9` | scratch `81..84` |
/// | `9 ..48` | harmonic bit planes, MSB first |
/// | `48 ..48+t` | scratch `8..8+t` |
/// | `48+t..50+t` | scratch `84..86` |
/// | `50+t..84` | more harmonic bit planes |
/// | `84 ..85` | scratch `86` |
/// | `85 ..87` | scratch `6..8` |
/// | `87 ..88` | scratch `87` |
///
/// where `t = triads_capped(nharm)` (`0x10312480`). The harmonic planes are
/// emitted by [`interleave_walk`] (`0x103131f0`), which sweeps a threshold
/// `off` down from the largest allocation: on each pass it visits harmonics
/// `0 ..nharm-1` and emits the bit at position `hbits[k] - off` of harmonic
/// `k`'s field whenever that is non-negative. The two walks share one cursor
/// set, so the second resumes exactly where the first stopped — which is how
/// the most significant harmonic bits end up before the gain/PRBA words and the
/// rest after them.
///
/// **This is IMBE-only.** The fixed `0x30`-word first-walk boundary and the
/// `total - 4` second only balance for an 88-bit payload; applied to the 49-bit
/// AMBE vector the layout runs past the end of the frame, exactly as SPEC §3.1
/// records for the mirror-image AMBE-only packer.
pub fn interleave_imbe_frame(
    chan: &mut [i16],
    nharm: i16,
    hbits: &[i16],
    alloc: &[i16],
    store: bool,
) {
    let mut scratch = [0i16; 192];
    let total = alloc[2];
    let f5 = alloc[5];
    let f6 = alloc[6];

    if !store {
        copy16_batch3(&mut scratch, chan, total);
    }

    let mut o = copy_seg(chan, 0, &mut scratch, 0, 6, store);
    let tail = (total.wrapping_sub(f6).wrapping_sub(1)) as i32; // scratch word of the b2 field
    o = copy_seg(chan, o, &mut scratch, tail, 3, store);

    // The threshold sweep starts at the largest per-harmonic allocation.
    let limit = nharm.wrapping_sub(1);
    let mut off: i16 = 0;
    if limit > 0 {
        for &h in &hbits[..limit as u16 as usize] {
            if off < h {
                off = h;
            }
        }
    }
    let triads = triads_capped(nharm);
    let mut cnt: i16 = 1;
    let base = (triads as i32).wrapping_add(f5 as i32) as isize;
    let mut dpos = base;
    let mut spos: isize = 0;

    if o < 0x30 {
        let mut op = o as usize;
        off = interleave_walk(
            chan,
            &mut op,
            0x30,
            &mut scratch,
            &mut dpos,
            hbits,
            &mut spos,
            &mut cnt,
            base,
            off,
            limit,
            store,
        );
        o = op as i32;
    }

    o = copy_seg(chan, o, &mut scratch, f5 as i32, triads, store);
    o = copy_seg(chan, o, &mut scratch, tail.wrapping_add(3), 2, store);

    let end2 = (total as i32).wrapping_sub(4);
    if o < end2 {
        let mut op = o as usize;
        interleave_walk(
            chan,
            &mut op,
            end2 as usize,
            &mut scratch,
            &mut dpos,
            hbits,
            &mut spos,
            &mut cnt,
            base,
            off,
            limit,
            store,
        );
        o = op as i32;
    }

    o = copy_seg(chan, o, &mut scratch, tail.wrapping_add(5), 1, store);
    o = copy_seg(chan, o, &mut scratch, 6, 2, store);
    copy_seg(
        chan,
        o,
        &mut scratch,
        (total as i32).wrapping_sub(1),
        1,
        store,
    );

    if store {
        copy16_batch3(chan, &scratch, total);
    }
}

// ======================= from aL2b2 =======================

/// As [`shift32_al2b2`] but the count is compared as a full 32-bit quantity. The
/// voicing-bit extraction in [`detect_tone_frame`] forms its count as
/// `-4 * t` in a 32-bit register and tests `cmp ecx, -31`, so a count outside
/// `i16` range must still reach the sign-fill branch.
#[inline]
pub fn shift32_wide(v: i32, s: i32) -> i32 {
    if s < 0 {
        if s < -0x1e {
            v >> 31
        } else {
            v >> ((-s) as u32 & 0x1f)
        }
    } else {
        ((v as u32) << (s as u32 & 0x1f)) as i32
    }
}

/// x86 `shl ax, cl`: the count is masked to five bits and a 16-bit operand
/// shifted by 16 or more becomes zero. Rust's `wrapping_shl` masks to four bits
/// for `u16` and is not a substitute.
///
/// Counts of exactly 16 are unreachable from the in-domain call sites (they
/// would need `bits >= 16`, which faults the divide inside `harmonic_index`
/// anyway), so only the `>= 17` half of the guard is exercised.
#[inline]
pub fn shl16_al2b2(v: u16, count: u32) -> u16 {
    let c = count & 0x1f;
    if c >= 16 {
        0
    } else {
        v << c
    }
}

/// The inlined hardware bit-scan normalisation (SPEC §4.1). `bsr` leaves its
/// destination unchanged for a zero source, and the destination holds the
/// magnitude, so `x == -1` (magnitude `~x == 0`) yields **30**, not 31. A zero
/// input is short-circuited to 0 by the caller's own test and returns 0.
///
/// The result is truncated to 16 bits, exactly as the `movzx ax` does.
///
/// The `-1` case is unreachable from the one call site: the value normalised
/// there is a sum of two `x << 15` terms, hence always a multiple of `2^15`.
/// The zero short-circuit *is* reachable (both tone legs unvoiced).
#[inline]
pub fn norm_bsr(x: i32) -> i32 {
    if x == 0 {
        return 0;
    }
    let mag = if x < 0 { !(x as u32) } else { x as u32 };
    let b = if mag == 0 {
        0
    } else {
        31 - mag.leading_zeros() as i32
    };
    (0x1e - b) as u16 as i32
}

/// Little-endian 32-bit read of two adjacent `i16` words.
#[inline]
pub fn rd32_al2b2(v: &[i16], word: usize) -> i32 {
    (v[word] as u16 as i32) | ((v[word + 1] as i32) << 16)
}

/// `0x1030c970` — force a decoded parameter block to a pure signalling tone.
///
/// The block's word 1 carries a *tone code*. [`frame_size_class`]
/// (`0x1030cdb0`) classifies it and [`band_count_for_mode`] (`0x1030cd50`) is
/// asked twice, with `b = 0` and `b = 1`, for the pair of harmonic indices that
/// carry the tone — for codes `0x80..=0xa3` those come straight out of the
/// two-word-per-code table at `0x1037fdd0` (see the module notes), for the low
/// codes they are computed arithmetically.
///
/// Every *other* harmonic magnitude in `p[8..]` is attenuated by
/// `0xa3c / 2^16 = 0.04` (−28 dB), leaving only the tone pair. For the
/// dual-tone codes `0x80..=0x9f` the two survivors are then replaced by their
/// common RMS `sqrt(a² + b²)` (via [`sqrt_to_exp`], exponent 8 → 4), so the
/// tone comes out with equal energy in both legs.
///
/// Finally, for the three single-tone signalling codes `0x14..=0x16` the
/// fundamental in word 6 is re-derived as `(code · 2^20) / ((p[3]+1) · 256)`
/// halved — [`div32_half`] (`0x1030a600`).
///
/// `p` is the 34-word parameter block: word 1 = tone code, word 2 = harmonic
/// count *L*, word 3 = frame-size word, word 6 = fundamental, words `8..8+L` =
/// harmonic magnitudes.
///
/// Domain: `(p[3] + 1) * 256` must be non-zero as an `i16` — the divide faults
/// otherwise — and the two harmonic indices must be inside `0..L`.
pub fn tone_frame_fixup(p: &mut [i16], cos_tbl: &[i16]) {
    let code = p[1];
    let class = frame_size_class(code) as i32;
    let ia = band_count_for_mode(class, code, 0, cos_tbl) as u16 as i16;
    let ib = band_count_for_mode(class, code, 1, cos_tbl) as u16 as i16;

    let n = p[2];
    let mut i: i16 = 0;
    while i < n {
        if i != ia && i != ib {
            let x = p[8 + i as usize] as i32;
            p[8 + i as usize] = (x.wrapping_mul(0xa3c) >> 16) as i16;
        }
        i += 1;
    }

    if p[1] >= 0x80 && p[1] < 0xa0 {
        let a = p[8 + ia as usize] as i32;
        let b = p[8 + ib as usize] as i32;
        let m = b.wrapping_mul(b).wrapping_add(a.wrapping_mul(a));
        let v = sqrt_to_exp(m, 8, 4) as i16;
        p[8 + ib as usize] = v;
        p[8 + ia as usize] = v;
    }

    let code = p[1];
    if (0x14..=0x16).contains(&code) {
        let den = p[3].wrapping_add(1).wrapping_mul(0x100);
        p[6] = div32_half((code as i32) << 0x14, den) as i16;
    }
}

/// `0x103165e0` — decide whether the frame just decoded is a signalling tone,
/// and if so which one.
///
/// The log-domain envelope in `p[8..8+L]` is exponentiated to linear magnitudes
/// by [`pow2_vector`] (`0x103163b0`) into a 56-word scratch sharing one block
/// exponent, and [`sum_norm`] (`0x10316350`) gives the normalised total. The
/// loudest harmonic is found with a strict `>` scan, so ties keep the *first*
/// index.
///
/// The decision has two arms.
///
/// **Low-index arm** — taken when the peak harmonic is voiced (its dibit in the
/// voicing word at `p[4..6]`, selected by `4 · round(k·p[6]/2^15)`, equals 1)
/// and lies at index ≤ 10. A peak-to-mean threshold of `0x7fdf`, `0x7f5c` or
/// `0x7eb8` (Q15: 0.99899, 0.99500, 0.98999) is chosen by the peak index, and
/// the tone is accepted only if `thresh · mean < peak`. The codeword search
/// [`nearest_codeword`] (`0x10315c60`) then runs over a window derived from the
/// peak index — `12.8·k + 1 … 12.8·(k+1)` in code units, `0xccc00` being 12.8
/// in Q16 — or over the fixed
/// window `0x14..=0x16` when the last two history words disagree — the
/// single-tone signalling codes. A Hamming distance of ≤ 1, or 2 with the code
/// unchanged from the previous frame, accepts immediately; otherwise a distance
/// above 16 rejects.
///
/// **Dual-tone arm** — otherwise the search runs over the whole dual-tone code
/// range `0x80..=0xa3`, and the winner's two harmonic legs are summed in energy,
/// block-normalised and compared against `min(sel, 0.9·L)` where `sel` is
/// `0x3200`, `0x3600` or `0x7800` by code range. The comparison is done at
/// whichever side has the smaller exponent, which is why it appears twice with
/// the shift on opposite operands.
///
/// On acceptance `p[0]` becomes 3 (tone frame), `p[1]` the code, and `p[3]`
/// either the single harmonic index or the packed pair `hi << 8 | lo`.
/// `p[1]` is pre-set to `0xff` (no tone) on every call.
///
/// `hist` is the caller's three-word tone history: `hist[2]` is the received
/// symbol pattern the codeword search matches, `hist[0..2]` the two previous
/// symbols.
///
/// Domain: `1 <= p[2] <= 56`, and `p[2]` must exceed both harmonic legs the
/// tone table can name (23 is enough), because the scratch beyond `p[2]` is
/// left uninitialised by the original.
///
/// Coverage notes — the whole function is bit-exact against the reference, but four
/// details are transcribed rather than *distinguished* by any input found:
///
/// * `0xe666` (the `0.9·L` cap on the accept floor). The cap does bind — in
///   roughly 420 000 of the swept evaluations — but replacing it with `0xe000`
///   changes no decision anywhere in that sweep: the accept margin in the
///   capped regime is never within 3 % of zero.
/// * the `+0xf` / `+5` exponent bias that picks between the two comparison
///   forms, and branch B's `+10` shift bias. Branch B is evaluated often but
///   never yields an accept — algebraically it is branch A with the shift moved
///   to the other operand, and its operands cannot reach the crossing.
/// * the argmax tie-break (`<`, first index wins). A tie at the maximum forces
///   the residual energy to at least equal the peak, which cannot satisfy a
///   99 %-of-total threshold, so the chosen index never reaches an output.
/// * `dist > 0x10` as the give-up distance. `dist` is a ten-bit popcount or the
///   `0x7fff` no-match seed, so any threshold in `10 ..= 0x7ffe` behaves the
///   same.
#[allow(clippy::too_many_arguments)]
pub fn detect_tone_frame(
    p: &mut [i16],
    prev_code: i16,
    hist: &[i16],
    cos_tbl: &[i16],
    codewords: &[u16],
) {
    p[1] = 0xff;

    let n = p[2] as u16;
    let mut lin = [0i16; 56];
    let mut blk_exp: i16 = 0;
    pow2_vector(&mut lin, &mut blk_exp, &p[8..], n);

    let mut mean: i16 = 0;
    let sum_exp = (sum_norm(&mut mean, &lin, blk_exp as i32, n) as u16) as i32;
    let blk = blk_exp as i32;

    let mut peak = lin[0];
    let mut peak_i: i16 = 0;
    let mut i: i16 = 1;
    while i < n as i16 {
        if peak < lin[i as usize] {
            peak = lin[i as usize];
            peak_i = i;
        }
        i += 1;
    }

    if voiced_dibit(p, peak_i) == 1 && peak_i <= 10 {
        let thresh: i16 = if peak_i <= 0 {
            0x7fdf
        } else {
            let mut t = 0x7eb8;
            if peak_i <= 1 && peak_i == band_count_for_mode(0, prev_code, 0, cos_tbl) as i16 {
                t = 0x7f5c;
            }
            t
        };

        let scaled = shift32_al2b2((peak as i32) << 16, blk.wrapping_sub(sum_exp) as i16);
        let cmp = (thresh as i32)
            .wrapping_mul(mean as i32)
            .wrapping_mul(2)
            .wrapping_sub(scaled);
        if cmp < 0 {
            // A tone: pick the code window and search it.
            let mut forced = 0i16;
            let lo;
            let hi;
            let mut matched = false;
            if peak_i == 1 && (prev_code as u16).wrapping_sub(0x14) < 3 {
                let d = hist[2];
                let mut k: i16 = 0;
                while k < 2 {
                    if hist[k as usize] != d {
                        matched = true;
                        break;
                    }
                    k += 1;
                }
            }
            if matched {
                forced = 1;
                lo = 0x14i32;
                hi = 0x16i32;
            } else {
                let c = (peak_i as i32).wrapping_mul(0xccc00);
                lo = ((c >> 16).wrapping_add(1)) as u16 as i32;
                hi = (c.wrapping_add(0xccc00) >> 16) as u16 as i32;
            }

            let mut dist: i16 = 0x7fff;
            let mut code = lo as i16;
            if (prev_code as u16) < 0x80 {
                nearest_codeword(
                    hist[2] as u16,
                    &mut dist,
                    &mut code,
                    prev_code,
                    prev_code,
                    codewords,
                );
            }
            nearest_codeword(
                hist[2] as u16,
                &mut dist,
                &mut code,
                lo as i16,
                hi as i16,
                codewords,
            );

            let accept = if dist <= 1 || (dist <= 2 && code == prev_code) {
                true
            } else if dist > 0x10 {
                return;
            } else {
                forced != 0
            };
            if accept {
                p[1] = code;
                p[0] = 3;
                p[3] = band_count_for_mode(0, code, 0, cos_tbl) as u16 as i16;
            }
            return;
        }
    }

    // Dual-tone arm.
    let mut dist: i16 = 0x7fff;
    let mut code: i16 = 0x80;
    if (0x80..=0xa3).contains(&prev_code) {
        nearest_codeword(
            hist[2] as u16,
            &mut dist,
            &mut code,
            prev_code,
            prev_code,
            codewords,
        );
    }
    nearest_codeword(hist[2] as u16, &mut dist, &mut code, 0x80, 0xa3, codewords);

    if dist > 1 && (dist > 2 || code != prev_code) {
        return;
    }

    let leg_a = band_count_for_mode(1, code, 0, cos_tbl) as u16 as i16;
    let e_a = if voiced_dibit(p, leg_a) == 1 {
        ((lin[leg_a as usize] as i32) << 16) >> 1
    } else {
        0
    };
    let leg_b = band_count_for_mode(1, code, 1, cos_tbl) as u16 as i16;
    let e_b = if voiced_dibit(p, leg_b) == 1 {
        ((lin[leg_b as usize] as i32) << 16) >> 1
    } else {
        0
    };

    let energy = e_a.wrapping_add(e_b);
    let norm = norm_bsr(energy);
    let e_exp = blk.wrapping_sub(norm).wrapping_add(1) as u16 as i32;
    let mant = (((energy as u32) << (norm as u32 & 0x1f)) as i32 >> 16) as i16;

    let sel: i16 = if code < 0x90 {
        0x3200
    } else if code < 0xa0 {
        0x3600
    } else {
        0x7800
    };
    let mut lim = ((sel as i32) << 16) >> 10;
    let cap = (p[2] as i32).wrapping_mul(0xe666);
    if lim > cap {
        lim = cap;
    }
    let floor = high16_al2b1(lim.wrapping_shl(10));

    let below = if (e_exp as i16 as i32).wrapping_add(0xf) > (sum_exp as i16 as i32).wrapping_add(5)
    {
        let d = (floor as i32).wrapping_mul(mean as i32).wrapping_mul(2);
        let cnt = sum_exp.wrapping_sub(e_exp).wrapping_sub(10) as i16;
        let a = (mant as i32).wrapping_mul(p[2] as i32).wrapping_mul(2);
        a.wrapping_sub(shift32_al2b2(d, cnt)) < 0
    } else {
        let a = (mant as i32).wrapping_mul(p[2] as i32).wrapping_mul(2);
        let cnt = e_exp.wrapping_sub(sum_exp).wrapping_add(10) as i16;
        let d = (floor as i32).wrapping_mul(mean as i32).wrapping_mul(2);
        shift32_al2b2(a, cnt).wrapping_sub(d) < 0
    };

    if !below {
        p[1] = code;
        p[0] = 3;
        p[3] = (((leg_a as i32) << 8) | (leg_b as u16 as i32)) as i16;
    }
}

/// The voicing dibit of harmonic `k`, as [`detect_tone_frame`] extracts it: the
/// 32-bit voicing word at `p[4..6]` is shifted right by `4 · t` where
/// `t = round((k+1) · p[6] / 2^15)`, and the low two bits taken. `1` is the
/// unvoiced/tone marker (the all-unvoiced base pattern is `0x55555555`).
///
/// The shift count is formed and tested in a 32-bit register, so it uses
/// [`shift32_wide`].
pub fn voiced_dibit(p: &[i16], k: i16) -> u8 {
    let word = rd32_al2b2(p, 4);
    let t = ((k as i32).wrapping_add(1) as i16 as i32)
        .wrapping_mul(p[6] as i32)
        .wrapping_mul(4)
        .wrapping_add(0x8000)
        >> 0x11;
    let t = t as i16 as i32;
    (shift32_wide(word, t.wrapping_mul(-4)) as u8) & 3
}

/// `0x103116b0` — the uniform scalar quantiser / dequantiser for the
/// higher-order spectral coefficients, run over `blocks` blocks of
/// `per_block` coefficients.
///
/// Each coefficient gets its own step size
/// `step = (loading[bits] · weight[j] · 2) >> 17`, where `weight` is the
/// nine-entry table at `0x103813fc` (`20120, 15794, 13566, 12452, 11731,
/// 11338, 10813, 11141, 11141` — the per-coefficient standard deviations,
/// restarted at every block) and `loading` is the table at `0x1038140e`
/// indexed by the bit count (`4915, 6963, 10650, 13107, 18350, 19661, 20972,
/// 20972 …` for 1…8 bits — the optimum-loading factors of a uniform quantiser).
/// A zero or negative allocation zeroes the coefficient outright.
///
/// With `encode != 0` the value is quantised: the index is
/// `(x + step/2) / step` computed as `bits` [`subc`] steps, clamped to
/// `[0, 2^bits − 1]`, and written to the payload by [`emit_bits`]
/// (`0x10310310`). With `encode == 0` the index is read back instead
/// ([`read_field_al2b2`], `0x103102e0`).
///
/// Either way the coefficient is then **reconstructed at the cell centre**,
/// `(2·idx + 1 − 2^bits) · step`, and rescaled by `2^(14 − bits)` so every
/// coefficient comes back on the same Q scale regardless of its allocation.
///
/// `vals` is walked with one word skipped at the head of each block (that slot
/// holds the block's gain and is quantised elsewhere); `allocs` runs
/// continuously across all blocks.
///
/// Note: the over-range test is `high16_al2b1(acc) < step`, but `<=` is
/// indistinguishable — at exactly `high16_al2b1(acc) == step` the `subc` division
/// produces `2^bits − 1`, which is what the clamp branch writes anyway. Every
/// (bits, coefficient) pair was checked; the two branches agree on the
/// boundary by construction.
#[allow(clippy::too_many_arguments)]
pub fn quantize_coeff_blocks(
    payload: &mut [i16],
    pos: &mut i32,
    blocks: i16,
    per_block: u16,
    vals: &mut [i16],
    allocs: &[u16],
    encode: i16,
    weight: &[i16],
    loading: &[i16],
) {
    let mut left = blocks.wrapping_sub(1);
    let mut vi = 0usize;
    let mut ai = 0usize;
    while left >= 0 {
        left = left.wrapping_sub(1);
        vi += 1;
        if (per_block as i16) > 0 {
            let mut j = 0usize;
            let mut rem = per_block;
            loop {
                let bits = allocs[ai];
                ai += 1;
                if (bits as i16) < 1 {
                    vals[vi] = 0;
                } else {
                    let prod = (loading[bits as usize] as i32)
                        .wrapping_mul(weight[j] as i32)
                        .wrapping_mul(2);
                    let cells = shl16_al2b2(1, bits as u32);
                    let step = (prod >> 0x11) as i16;

                    let idx: u16 = if encode == 0 {
                        read_field_al2b2(payload, pos, bits as i16)
                    } else {
                        let acc = (((step as i32) << 16) >> 1)
                            .wrapping_add((vals[vi] as i32).wrapping_mul(0x10000));
                        let v = if acc < 0 {
                            0u16
                        } else if high16_al2b1(acc) < step {
                            let mut a = acc;
                            for _ in 0..bits {
                                subc(step, &mut a);
                            }
                            a as u16
                        } else {
                            cells.wrapping_sub(1)
                        };
                        let mut cur = *pos as usize;
                        emit_bits(&mut cur, payload, bits as i16, v as i16);
                        *pos = cur as i32;
                        v
                    };

                    let centre = ((idx as i16 as i32).wrapping_mul(0x20000))
                        .wrapping_add((cells as i16 as i32).wrapping_mul(-0x10000))
                        .wrapping_add(0x10000);
                    let acc = (high16_al2b1(centre) as i32)
                        .wrapping_mul(step as i32)
                        .wrapping_mul(2);
                    vals[vi] = high16_al2b1(shift32_al2b2(acc, 0x0eu16.wrapping_sub(bits) as i16));
                }
                j += 1;
                vi += 1;
                rem -= 1;
                if rem == 0 {
                    break;
                }
            }
        }
    }
}

/// `0x103102e0` in its `i16`-payload form: fold `n` one-bit-per-word values
/// into a word and advance the cursor. The layer-1 port takes a `&[u16]`
/// payload; the quantiser above holds the same buffer as `&mut [i16]`, so this
/// adapts the slice rather than duplicating the logic of [`pack_bits`].
pub fn read_field_al2b2(payload: &[i16], pos: &mut i32, n: i16) -> u16 {
    let at = *pos as usize;
    let v = pack_bits_i16(&payload[at..], n);
    *pos = pos.wrapping_add(n as i32);
    v
}

/// `0x103132b0` — split one joint frequency code between two voiced sub-bands.
///
/// `a` and `b` are two band descriptors, each carrying a 32-bit per-band dibit
/// map at word 4 and receiving a residual frequency error at word 7. The dibit
/// alphabet is the codec's usual one: bit 0 of a dibit marks unvoiced, bit 1
/// voiced (the all-unvoiced base pattern is `0x55555555`).
///
/// The gate rejects outright — zeroing both residuals and returning 0 — if
/// either band contains an unvoiced dibit, if `b[0] != 1` while `enable` is
/// set, or if neither band is voiced at all. Otherwise:
///
/// * only `b` voiced → the whole `2^bits` code space goes to `b`;
/// * only `a` voiced → it goes to `a`;
/// * both voiced → the code space is factored as `n × n` with
///   `n = 2^(bits/2)`, scaled by `sqrt(2) = 0x16a08 / 2^16` when `bits` is odd,
///   and `*code` is divided by `n` ([`div32`], `0x1030b430`) into a quotient and
///   remainder, each clamped to `n − 1`.
///
/// Each part is converted to a harmonic index by [`harmonic_index`]
/// (`0x10313540`), whose residual lands in the corresponding word 7. With
/// `enable` set the two clamped parts are recombined into `*code`, itself
/// clamped to `[0, (0x3c << (bits − 6)) − 1]`.
///
/// Returns 1 when the split happened, 0 when the gate rejected.
///
/// Domain: `1 <= bits <= 15` — [`harmonic_index`] divides by `2^bits` truncated
/// to 16 bits, which is zero for `bits >= 16`, and [`div32`] by `n`.
///
/// Note on the decompilation: `0x103132b0` reuses the argument slots of
/// `param_3` and `param_7` as the scratch for the quotient and remainder
/// (SPEC §4.2). When `enable != 0` those slots are never initialised before
/// [`harmonic_index`] overwrites them, which is why the port can start them at
/// zero.
#[allow(clippy::too_many_arguments)]
pub fn split_voiced_freq_code(
    code: &mut i16,
    a: &mut [i16],
    b: &mut [i16],
    freq_a: i16,
    freq_b: i16,
    bits: u32,
    enable: i32,
) -> i32 {
    let map_a = rd32_al2b2(a, 4);
    let map_b = rd32_al2b2(b, 4);
    let a_unvoiced = (map_a as u32 & 0x5555_5555) != 0;
    let b_unvoiced = (map_b as u32 & 0x5555_5555) != 0;
    let a_voiced = (map_a as u32 & 0xaaaa_aaaa) != 0;
    let b_voiced = (map_b as u32 & 0xaaaa_aaaa) != 0;
    let on = enable as i16 != 0;
    let ok = if on { b[0] == 1 } else { true };

    if a_unvoiced || b_unvoiced {
        b[7] = 0;
        a[7] = 0;
        return 0;
    }
    if !ok || (!a_voiced && !b_voiced) {
        b[7] = 0;
        a[7] = 0;
        return 0;
    }

    let cap: i16 = if on {
        (shl16_al2b2(0x3c, bits.wrapping_sub(6)) as i16).wrapping_sub(1)
    } else {
        0
    };

    if !a_voiced {
        // only b carries voiced energy
        if b_voiced {
            a[7] = 0;
            let n = shl16_al2b2(1, bits) as i16;
            b[7] = harmonic_index(code, n, freq_b, cap, enable as i16) as i16;
        }
        return 1;
    }
    if !b_voiced {
        b[7] = 0;
        let n = shl16_al2b2(1, bits) as i16;
        a[7] = harmonic_index(code, n, freq_a, cap, enable as i16) as i16;
        return 1;
    }

    let mut n = shl16_al2b2(1, (bits as i32 >> 1) as u32);
    if bits & 1 != 0 {
        n = high16_al2b1((n as i16 as i32).wrapping_mul(0x16a08)) as u16;
    }
    let ni = n as i16;

    let mut q: i16 = 0;
    let mut r: i16 = 0;
    if !on {
        let quo = div32(*code as i32, ni) as i16;
        let rem = code.wrapping_sub(quo.wrapping_mul(ni));
        q = quo;
        r = rem;
        if quo as i32 > (ni as i32).wrapping_sub(1) {
            q = n.wrapping_sub(1) as i16;
        }
        // UNVERIFIED: dead in the original. `rem` is the remainder of a
        // truncating divide by `ni`, so `|rem| < ni` always holds and this
        // clamp can never fire; no input distinguishes it. Kept because the
        // original carries it. (The matching clamp on the quotient *is*
        // reachable — `ni == 1` with a large code hits it.)
        if rem as i32 > (ni as i32).wrapping_sub(1) {
            r = n.wrapping_sub(1) as i16;
        }
    }

    a[7] = harmonic_index(&mut r, ni, freq_a, cap, enable as i16) as i16;
    b[7] = harmonic_index(&mut q, ni, freq_b, cap, enable as i16) as i16;

    if on {
        let v = r.wrapping_add(q.wrapping_mul(ni));
        *code = v;
        if cap < v {
            *code = cap;
        } else if v < 1 {
            *code = 0;
        }
    }
    1
}

/// `0x103178a0` — spectral slope: extend the envelope, then differentiate it.
///
/// [`build_spectrum_history`] (`0x103178e0`) lays `src[0..n]` into a 0x68-word
/// scratch with symmetric extensions at both edges, and
/// [`hilbert_diff`] (`0x103179d0`) runs the ten-tap antisymmetric
/// (Hilbert / differentiating) filter of `0x10381c18` over it. The extension
/// exists purely so the filter never reads past the data.
///
/// `out` receives `n` slope values with words `n..0x38` zeroed.
///
/// Domain: `1 <= n <= 0x40`.
pub fn spectral_slope(out: &mut [i16], src: &[i16], n: i16, taps: &[i16]) {
    let mut scratch = [0i16; 0x68];
    build_spectrum_history(&mut scratch, src, n);
    hilbert_diff(out, &scratch, n as u16, taps);
}

/// `0x1031a330` — the inverse FFT of a real, conjugate-symmetric spectrum.
///
/// One un-merge stage [`fft_merge_stage`] (`0x1031a1b0`, variant 1) folds the
/// `2^log2n`-point half-spectrum into a `2^(log2n−1)`-point complex sequence,
/// which [`fft`] (`0x1031a370`) then transforms in place. On the first call of a
/// frame (`first == 0`) the Nyquist word `buf[1]` is cleared.
///
/// The exponent handed from the merge stage to the transform is **truncated to
/// 16 bits and zero-extended** (`movzx ecx, ax`), not sign-extended; the return
/// value is the transform's block exponent re-based by `log2n − 1`.
pub fn inverse_real_fft(buf: &mut [i16], exp: i32, log2n: i32, first: i16) -> i32 {
    if first == 0 {
        buf[1] = 0;
    }
    let e = fft_merge_stage(buf, exp, log2n as i16, 1, first) as u16 as i32;
    fft(buf, e, (log2n - 1) as i16)
        .wrapping_sub(log2n)
        .wrapping_add(1)
}

// ======================= from aL2b3 =======================

/// MSVC's `__allshl` (`0x10305930`). Only the low half is ever used by the
/// callers in this codec (SPEC §4.2), but the helper is written honestly.
pub fn allshl_al2b3(v: i64, count: u8) -> i64 {
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

/// The ETSI `L_sub` saturating 32-bit subtract, open-coded at `0x10316fe2`.
/// Note the limit is chosen from the *minuend*'s sign, not the result's
/// (SPEC §4.1).
#[inline]
pub fn sat_sub32(a: i32, b: i32) -> i32 {
    let r = a.wrapping_sub(b);
    if ((!b ^ r) & (r ^ a)) < 0 {
        0x7fff_ffffi32.wrapping_add((a < 0) as i32)
    } else {
        r
    }
}

/// Read the 32-bit voicing mask a harmonic record carries at byte `+8`, given
/// the record viewed as `i16` words.
#[inline]
pub fn mask32(rec: &[i16]) -> u32 {
    (rec[4] as u16 as u32) | ((rec[5] as u16 as u32) << 16)
}

/// `0x1030ca70` — re-select the tracked harmonic from a type-3 pitch message.
///
/// `state` is the pitch tracker viewed as `i16` words: `[1]` is the pitch
/// numerator (the channel-unit length the tracker is running at), `[3]` the
/// currently selected harmonic index. `msg` is a four-word message: `[0]` must
/// be the tag `3`, `[1]` the sender's channel-unit length, `[3]` the harmonic
/// index the sender selected. Both lengths must classify to frame class 0
/// (`frame_size_class`, `0x1030cdb0`) or the message is ignored entirely.
///
/// If the sender's index is within one of ours *and* our own numerator is below
/// `(idx + 1) · 16`, the sender's index is adopted verbatim. Otherwise the
/// index is rescaled into our own numerator,
/// `idx' = round(( (idx+1)·state[1] + msg[1]/2 ) / msg[1])`, and then walked
/// **up** until `idx' · 16` exceeds `state[1]`; the value one below that is
/// adopted. The rescale is `div32` (`0x1030b430`) and the `msg[1]/2` bias
/// truncates toward zero, as the original's `cdq; sub; sar 1` does.
///
/// `flags` is the one-hot harmonic flag array `select_harmonic` (`0x1030d490`)
/// reaches through the pointer at `state+0x80`; it is passed explicitly here
/// because the Rust port of that callee takes it as an argument.
///
/// Domain: `state[1] >= 0` (otherwise the walk never terminates) and the
/// adopted index must land inside the 56-entry harmonic array.
pub fn retune_from_pitch_msg(state: &mut [i16], flags: &mut [i16], msg: &[i16]) {
    if frame_size_class(state[1]) != 0 {
        return;
    }
    if msg[0] != 3 {
        return;
    }
    if frame_size_class(msg[1]) != 0 {
        return;
    }

    let idx = msg[3];
    let d = state[3].wrapping_sub(idx);
    let d = if d < 0 { d.wrapping_neg() } else { d };
    if d <= 1 && (state[1] as i32) < ((idx as i32) + 1) * 0x10 {
        select_harmonic(state, flags, idx);
        return;
    }

    let num = ((idx as i32) + 1)
        .wrapping_mul(state[1] as i32)
        .wrapping_add((msg[1] as i32) / 2);
    let mut e = div32(num, msg[1]) as u16 as u32;
    while ((e as u16 as i16) as i32) * 0x10 <= state[1] as i32 {
        e = e.wrapping_add(1);
    }
    select_harmonic(state, flags, e.wrapping_sub(1) as u16 as i16);
}

/// `0x10311870` — take the DC coefficient of each of `count` consecutive
/// `n`-point harmonic cosine transforms.
///
/// Walks `src` in blocks of `n` words, runs the in-place harmonic DFT
/// (`0x1031f9e0`, [`harmonic_dft`]) over each block with both its length and
/// its output count set to `n`, and copies the block's first (DC) output word
/// into `dst`. **`src` is rewritten in place** — every block is replaced by its
/// own transform — so a caller that still needs the input must copy it first.
///
/// Returns the word offset one past the last block, i.e. the pointer the
/// original leaves in `eax`. A non-positive `count` does nothing and returns 0.
///
/// Domain: `1 <= n <= 56` (the callee's scratch buffer and reciprocal table).
pub fn block_dft_dc(dst: &mut [i16], src: &mut [i16], count: u16, n: u32, cos_tbl: &[i16]) -> i32 {
    let mut off: i32 = 0;
    if (count as i16) > 0 {
        for k in 0..count as usize {
            harmonic_dft(&mut src[off as usize..], n as u16, n as i16, cos_tbl);
            dst[k] = src[off as usize];
            off = off.wrapping_add(n as u16 as i16 as i32);
        }
    }
    off
}

/// `0x1031fb40` — inverse harmonic cosine transform: expand `count` spectral
/// coefficients back into `n` harmonic amplitudes, in place.
///
/// `count` is first clamped to `n`. The coefficients are copied to a 56-word
/// scratch with every term *except the DC one* doubled (`0x103173e0` with a
/// shift of 1 — the standard DCT-I half-weighting of the DC term), and output
/// `k` is
///
/// ```text
/// buf[k] = round16( sum_i c[i] · cos( (0x40 + i·step_k) · 2π/(512·128) ) · 2 )
/// ```
///
/// with `step_k = RECIP_N[n]/2 + k·RECIP_N[n]` accumulated in **16 bits**, so it
/// wraps — that wrap is the frequency folding of the transform and is
/// load-bearing. `RECIP_N` is `0x103820e8`, `trunc(32768/n)` saturated to
/// `0x7fff` at `n = 1`. The accumulator wraps rather than saturating, and the
/// result is rounded with `+0x8000 >> 16`.
///
/// Domain: `1 <= n <= 56` (the reciprocal table ends at 56 and the scratch
/// holds 56 words); a non-positive `n` is a no-op.
pub fn harmonic_idct(buf: &mut [i16], n: i16, count: i16, cos_tbl: &[i16]) {
    let cnt = if count >= n { n } else { count };

    let mut local = [0i16; 56];
    local[0] = buf[0];
    if cnt > 1 {
        shift_copy16(&mut local[1..], &buf[1..], (cnt - 1) as u16, 1);
    }
    if n <= 0 {
        return;
    }

    let recip = RECIP_N[n as usize];
    let mut step = recip >> 1;
    // Both bounds are proved once here: `ti` is masked to 9 bits, and `cnt`
    // and `n` are loop-invariant. Inside the pair of loops nothing is checked.
    let ct = &cos_tbl[..0x200];
    let src = &local[..if cnt > 0 { cnt as usize } else { 0 }];
    for slot in buf[..n as usize].iter_mut() {
        let mut acc: i32 = 0;
        let mut phase: i32 = 0x40;
        for &c in src {
            let ti = ((phase >> 7) & 0x1ff) as usize;
            phase = phase.wrapping_add(step as i32);
            acc = acc.wrapping_add((ct[ti] as i32).wrapping_mul(c as i32).wrapping_mul(2));
        }
        step = step.wrapping_add(recip);
        *slot = high16_al2b3(acc.wrapping_add(0x8000));
    }
}

/// `0x10314a90` — combine two harmonic records' amplitudes according to their
/// voicing masks.
///
/// A record is `i16` words with a 32-bit voicing mask at byte `+8` (words 4–5)
/// and a Q15 amplitude at byte `+0xc` (word 6). The masks are packed dibits: an
/// odd bit set (`& 0x55555555`) means "voiced somewhere", an even bit set
/// (`& 0xaaaaaaaa`) means "unvoiced somewhere".
///
/// If `b` has voiced content and the free-standing mask `flags` has no unvoiced
/// content, the result is `sqrt(a.amp · b.amp)` — the geometric mean, evaluated
/// by `0x1030b380` ([`sqrt_to_exp`]) on the doubled product at input exponent
/// −8 and output exponent −4 — provided `a` also has voiced content; if `a` is
/// wholly unvoiced the result is `b`'s amplitude unchanged. In every other case
/// `a`'s amplitude passes through.
///
/// The original returns `movzx eax,ax`, so only the low 16 bits are meaningful.
pub fn merge_voiced_amplitude(flags: u32, a: &[i16], b: &[i16]) -> u16 {
    if mask32(b) & 0x5555_5555 != 0 && flags & 0xaaaa_aaaa == 0 {
        if mask32(a) & 0x5555_5555 != 0 {
            let p = (a[6] as i32).wrapping_mul(b[6] as i32).wrapping_mul(2);
            return sqrt_to_exp(p, -8, -4) as u16;
        }
        return b[6] as u16;
    }
    a[6] as u16
}

/// `0x10316f40` — copy a 56-harmonic magnitude vector, attenuating the
/// harmonics that fall in unvoiced bands by a fixed log-domain offset.
///
/// The offset is derived from `level` by `log2_q` (`0x1030b100`) at exponent 4:
/// `off = high16_al2b3( (log2_q(level, 4) << 10) − 0x314fc01 )`, negated when
/// `negate` is non-zero. `0x314fc01` is the bias that puts a unity level at a
/// zero offset.
///
/// `nbits` is turned into a band count by `0x10312480` ([`triads_capped`]) —
/// `(nbits+2)/3` capped at 12 — and the vector is then walked **downwards**
/// from harmonic `0x37` to `0`, three harmonics per band. `mask` supplies one
/// voicing bit per band, LSB first and shifted right (arithmetically) as the
/// walk proceeds, **starting at the top band**. A set bit copies the harmonic
/// verbatim; a clear bit subtracts `off` in Q15 with the ETSI `L_sub`
/// saturation and then clamps the word to `[-0x7800, 0x77ff]` — the codec's
/// symmetric log-magnitude range, deliberately one notch inside full scale.
///
/// The top band is not three harmonics wide: it absorbs everything from
/// `3·(bands−1)` up to `0x37`.
///
/// `out` and `inp` must hold 0x38 words; `out` is only written in the swept
/// range, which reaches 0 whenever `bands >= 1`.
pub fn attenuate_unvoiced_bands(
    out: &mut [i16],
    inp: &[i16],
    mask: i16,
    level: i16,
    nbits: i16,
    negate: i16,
) {
    let l = log2_q(level, 4);
    let mut off = high16_al2b3((l.wrapping_shl(10)).wrapping_sub(0x0314_fc01));
    if negate != 0 {
        off = off.wrapping_neg();
    }

    let mut bands = triads_capped(nbits).wrapping_sub(1);
    if bands < 0 {
        return;
    }

    let mut m = mask;
    let mut lo = (bands as i32) * 3;
    let mut idx: i16 = 0x37;
    let mut cur: i32 = 0x37;
    loop {
        let voiced = (m & 1) != 0;
        m >>= 1;
        while lo <= cur {
            let mut v = inp[cur as usize];
            if !voiced {
                let r = sat_sub32((v as i32) << 16, (off as i32) << 16);
                v = high16_al2b3(r);
                if v > 0x77ff {
                    v = 0x77ff;
                } else if v < -0x7800 {
                    v = -0x7800;
                }
            }
            idx = idx.wrapping_sub(1);
            out[cur as usize] = v;
            cur = idx as i32;
        }
        bands = bands.wrapping_sub(1);
        lo -= 3;
        if bands < 0 {
            return;
        }
    }
}

/// `0x10317ff0` — harmonic-vector energy and its correlation with a cosine of
/// the trial fundamental.
///
/// Two block-floating-point results come out of one pass over the amplitude
/// vector `amps`:
///
/// * **Energy** — `sum a[i]²`, taken by the 64-bit dot product `0x103170a0`
///   ([`dot_norm64`]) of the vector with itself. Its mantissa is `high16_al2b3` of
///   that call's return (the decompiler's bare `extraout_var`, SPEC §4.2) and
///   its exponent is the callee's, biased by `2·exp_adj`.
/// * **Correlation** — `sum (a[i]²>>16) · cos(i·omega)`, accumulated in 64 bits
///   and block-normalised by `norm64` (`0x10310150`) plus the 64-bit shift
///   helpers, exponent `2·exp_adj − shift`. A mantissa of `-0x8000` is folded
///   to `-0x7fff` (the symmetric-Q15 clamp of SPEC §4.1).
///
/// If the energy mantissa is zero the correlation is reported as `(0, 0)` and
/// the cosine buffer is left untouched.
///
/// The cosines are also written out to `cos_out`, one per harmonic; the phase
/// runs `omega·64·(i+1)` in Q16 and the table index is the **signed** high half
/// with no mask, so the original relies on the caller keeping the trial
/// fundamental small.
///
/// Domain: `0 <= omega` and `omega·64·n < 512·65536` so the cosine index stays
/// inside the 512-word table; `cos_out` must not alias `amps` (the original
/// stores the cosine before reading the amplitude at the same index).
#[allow(clippy::too_many_arguments)]
pub fn harmonic_energy_and_corr(
    out_energy: &mut i16,
    out_energy_exp: &mut i16,
    out_corr: &mut i16,
    out_corr_exp: &mut i16,
    cos_out: &mut [i16],
    omega: i16,
    amps: &[i16],
    exp_adj: i16,
    n: u16,
    cos_tbl: &[i16],
) {
    let mut e: i16 = 0;
    let energy = dot_norm64(&mut e, amps, amps, n);
    *out_energy = high16_al2b3(energy);
    let adj = exp_adj.wrapping_mul(2);
    *out_energy_exp = e.wrapping_add(adj);

    if *out_energy == 0 {
        *out_corr = 0;
        *out_corr_exp = 0;
        return;
    }

    let step = ((omega as i32) << 10) >> 4;
    let mut phase = step;
    let mut acc: i64 = 0;
    if (n as i16) > 0 {
        for i in 0..n as usize {
            let ti = high16_al2b3(phase) as i32;
            let c = cos_tbl[ti as usize];
            cos_out[i] = c;
            let sq = high16_al2b3(
                (amps[i] as i32)
                    .wrapping_mul(amps[i] as i32)
                    .wrapping_mul(2),
            );
            acc = acc.wrapping_add((sq as i32).wrapping_mul(c as i32).wrapping_mul(2) as i64);
            phase = phase.wrapping_add(step);
        }
    }

    let sh = norm64(acc as u32, (acc >> 32) as u32) as u16 as i16;
    let shifted = if sh >= 0 {
        allshl_al2b3(acc, (sh as u32 & 0xff) as u8)
    } else {
        allshr(acc, ((sh as i32).wrapping_neg() as u32 & 0xff) as u8)
    };
    *out_corr_exp = adj.wrapping_sub(sh);
    let m = high16_al2b3(shifted as i32);
    *out_corr = if m == -0x8000 { -0x7fff } else { m };
}

// ======================= from aL2b4 =======================

/// Hardware `bsr`: index of the highest set bit. The one call site in this
/// batch guards a zero source explicitly, so the "destination unchanged"
/// behaviour of §4.1 is not reachable here.
#[inline]
pub fn bsr(v: u32) -> i32 {
    31 - v.leading_zeros() as i32
}

#[inline]
pub fn up16_al2b4(x: i16) -> i32 {
    ((x as u32) << 16) as i32
}

/// ETSI `L_add` with the saturation limit taken from `sgn` rather than from the
/// result — see §4.1: *which* operand supplies the sign is per-call-site, so
/// every use below names it explicitly.
#[inline]
pub fn add_sat_sgn_al2b4(a: i32, b: i32, sgn: i32) -> i32 {
    let r = a.wrapping_add(b);
    if ((r ^ a) & (r ^ b)) < 0 {
        if sgn < 0 {
            i32::MIN
        } else {
            i32::MAX
        }
    } else {
        r
    }
}

/// `x + 0xb0000000` (i.e. `x - 0x50000000`) with saturation, the "drop the
/// spreading floor 1.25 full-scale below the peak" step that `pitch_config_init`
/// performs three times. The compiler folded the constant operand out of the
/// overflow test, leaving `(r ^ x) & ~r`; that is reproduced literally.
#[inline]
pub fn minus_1p25(x: i16) -> i16 {
    let a = up16_al2b4(x);
    let r = a.wrapping_add(0xb000_0000u32 as i32);
    let r = if ((r ^ a) & !r) < 0 {
        if a < 0 {
            i32::MIN
        } else {
            i32::MAX
        }
    } else {
        r
    };
    high16_al2b1(r)
}

/// Word offsets, from the base of the constant blob at `0x1037f978`, of the two
/// small tables `pitch_config_init` reads. They sit immediately after the
/// 328-word band-count table at `0x1037fbd0` that `band_count_for_mode`
/// (`0x1030cd50`) indexes, which is itself contiguous with the cosine table —
/// the whole region is one blob, and callers must pass it as one slice.
pub const T_STEP: usize = (0x1037_fd88 - 0x1037_f978) / 2; // 520

pub const T_DEV: usize = (0x1037_fe60 - 0x1037_f978) / 2; //  628

/// `0x1030ce40` — build the per-channel band configuration for a channel-unit
/// length, and seed the band-level array with one or two spectral peaks.
///
/// `rate` is the same channel-unit length that [`frame_size_class`]
/// (`0x1030cdb0`) classifies: class 0 (`5..=0x7a`) is the single-peak
/// arithmetic case, classes 1..3 (`0x80..=0xa3`) are the two-peak table-driven
/// cases, and anything else is rejected with a return of 4.
///
/// The 64-word `cfg` block is laid out as
///
/// | word | meaning |
/// |---|---|
/// | 0 | version tag, always 3 |
/// | 1 | the channel-unit length as given |
/// | 2 | harmonic count, clamped to the codec's `[9, 56]` |
/// | 3 | band index (class 0) or the packed pair `hi<<8 \| lo` |
/// | 4,5 | the dword 1 |
/// | 6 | frequency step: computed by division for class 0, read from the table at `0x1037fd88` otherwise |
/// | 7 | 0 |
/// | 8.. | 56 band levels, pre-filled with the floor `-0x4fff` |
///
/// With `apply != 0` the band levels are additionally shaped: each peak is
/// written at its own index and [`spread_peak`] (`0x1030d1b0`) lays a skirt
/// falling 0x400 per band from 1.25 full-scale below it. For classes 1..3 the
/// two peak levels are first replaced by `mid ± dev`, where `mid` is the
/// (halved, saturating) mean of `lo` and `hi` and `dev` comes from the table at
/// `0x1037fe60`; the clamp window is `[-0x5000, 0x6fff]` there but
/// `[-0x5000, 0x77ff]` for class 0.
///
/// `tbl` is the constant blob based at `0x1037f978`; at least 664 words are
/// touched. Returns 0 on success, 4 for an out-of-range `rate`.
pub fn pitch_config_init(
    cfg: &mut [i16],
    rate: i16,
    lo: i16,
    hi: i16,
    apply: i16,
    tbl: &[i16],
) -> u32 {
    let u = rate as u16;
    let cls: i16 = if u.wrapping_sub(5) < 0x76 {
        0
    } else if u.wrapping_sub(0x80) < 0x10 {
        1
    } else if u.wrapping_sub(0x90) < 0x10 {
        2
    } else if u.wrapping_sub(0xa0) < 4 {
        3
    } else {
        return 4;
    };

    cfg[0] = 3;
    cfg[4] = 1;
    cfg[5] = 0;
    cfg[7] = 0;
    fill16(&mut cfg[8..], -0x4fff, 0x38);

    let clamp_l = |v: i16, top: i16| -> i16 {
        if v > top {
            top
        } else if v < -0x5000 {
            -0x5000
        } else {
            v
        }
    };

    if cls == 0 {
        let band = band_count_for_mode(0, rate, 0, tbl) as u16 as i16;
        cfg[1] = rate;
        cfg[3] = band;

        // step = (rate << 20) / (byte-truncated (band+1) << 8), halved.
        let num = ((rate as i32) as u32).wrapping_shl(20) as i32;
        let den = high16_al2b1(((band as i32 + 1) as u32).wrapping_shl(24) as i32) as i16;
        cfg[6] = div32_half(num, den) as i16;

        let l = band.wrapping_add(1);
        cfg[2] = if l > 0x38 {
            0x38
        } else if l < 9 {
            9
        } else {
            l
        };

        let peak = clamp_l(hi, 0x77ff);
        cfg[8 + band as usize] = peak;
        if apply != 0 {
            spread_peak(&mut cfg[8..], band, minus_1p25(peak), 0x400);
        }
        return 0;
    }

    let b_lo = band_count_for_mode(cls as i32, rate, 0, tbl) as u16 as i16;
    let b_hi = band_count_for_mode(cls as i32, rate, 1, tbl) as u16 as i16;
    cfg[1] = rate;
    let k = rate.wrapping_sub(0x80) as usize;
    cfg[3] = ((b_lo as u16 as u32) << 8).wrapping_add(b_hi as u16 as u32) as i16;
    cfg[6] = tbl[T_STEP + k];

    let l = b_hi.wrapping_add(1);
    cfg[2] = if l > 0x38 {
        0x38
    } else if l < 9 {
        9
    } else {
        l
    };

    let (mut v_lo, mut v_hi) = (lo, hi);
    if apply != 0 {
        // mid = (hi/2 + lo/2), saturating with the limit signed by the *lo*
        // term. Both operands are pre-halved, so the sum cannot leave `i32`
        // and this saturation is unreachable; it is kept for fidelity.
        let a = up16_al2b4(hi) >> 1;
        let b = up16_al2b4(lo) >> 1;
        let mid = high16_al2b1(add_sat_sgn_al2b4(a, b, b));

        let dev = up16_al2b4(tbl[T_DEV + k]);
        let mv = up16_al2b4(mid);

        // mid - dev, saturating; the compiler spells the overflow test with the
        // subtrahend complemented, and the limit is signed by `mid`.
        let d = mv.wrapping_sub(dev);
        let d = if (((!dev) ^ d) & (d ^ mv)) < 0 {
            if mv < 0 {
                i32::MIN
            } else {
                i32::MAX
            }
        } else {
            d
        };
        v_lo = high16_al2b1(d);
        v_hi = high16_al2b1(add_sat_sgn_al2b4(dev, mv, mv));
    }

    let p_lo = clamp_l(v_lo, 0x6fff);
    cfg[8 + b_lo as usize] = p_lo;
    if apply != 0 {
        spread_peak(&mut cfg[8..], b_lo, minus_1p25(p_lo), 0x400);
    }

    let p_hi = clamp_l(v_hi, 0x6fff);
    cfg[8 + b_hi as usize] = p_hi;
    if apply != 0 {
        spread_peak(&mut cfg[8..], b_hi, minus_1p25(p_hi), 0x400);
    }
    0
}

/// `0x10312030` — the midpoint of the two band values a channel-unit length
/// selects.
///
/// Classifies `rate` with [`frame_size_class`], asks
/// [`band_count_for_mode`] for its low and high band indices, and returns the
/// mean of `arr` at those two indices. Both terms are promoted to Q31 and
/// halved before the (non-saturating) add, so the mean is exact and cannot
/// overflow; the result is returned as a full 32-bit `arr[hi]/2 + arr[lo]/2`
/// shifted back down by 16, i.e. a Q15 word sign-extended to `i32`.
///
/// `tbl` is the constant blob based at `0x1037f978` (see
/// [`pitch_config_init`]). For class 4 — an out-of-range `rate` — both indices
/// come back 0, so the answer is `arr[0]`.
pub fn band_pair_mean(arr: &[i16], rate: i16, tbl: &[i16]) -> i32 {
    let cls = frame_size_class(rate) as i16 as i32;
    let i_lo = band_count_for_mode(cls, rate, 0, tbl) as u16 as i16;
    let i_hi = band_count_for_mode(cls, rate, 1, tbl) as u16 as i16;
    let a = up16_al2b4(arr[i_hi as usize]) >> 1;
    let b = up16_al2b4(arr[i_lo as usize]) >> 1;
    a.wrapping_add(b) >> 16
}

/// `0x10314be0` — resample a 56-point spectral envelope onto `n` points at the
/// rate `num/den`, with linear interpolation between neighbours.
///
/// The envelope is first edge-extended to 60 points by [`spread_hold`]
/// (`0x10314870`), which delays it one place and repeats the first and last
/// values. The step is the Q16 ratio `num/den` from [`ratio_q16`]
/// (`0x103148e0`); the accumulator starts at one whole step (so output 0 is
/// already one step in, not the envelope's own first point) and advances by it
/// each output sample. Its integer part indexes the table — clamped to 59 at
/// both taps, so the tail saturates rather than running off — and its
/// fractional part, **halved to a Q15 weight**, mixes the two taps:
/// `out[k] = tmp[i]·(0x7fff − w) + tmp[i+1]·w` on a saturating Q31 accumulator.
///
/// Domain: `den != 0` (a zero divisor faults inside `ratio_q16`) and the step
/// must be non-negative with `n · step` not wrapping into the sign bit — the
/// index is used signed and unguarded below zero. `src` must hold 56 words.
pub fn interp_envelope(out: &mut [i16], num: i16, den: i16, src: &[i16], n: u16) {
    let mut tmp = [0i16; 60];
    spread_hold(&mut tmp, src, 0x3c);
    let step = ratio_q16(num, den);

    if (n as i16) <= 0 {
        return;
    }
    let mut acc = step;
    for slot in out.iter_mut().take(n as usize) {
        let idx = high16_al2b1(acc);
        // fractional part, then >>1: a Q16 fraction becomes a Q15 weight.
        let w = high16_al2b1(((acc.wrapping_sub(up16_al2b4(idx)) as u32) << 15) as i32);

        let i0 = if idx > 0x3b { 0x3b } else { idx };
        let i1 = {
            let t = i0.wrapping_add(1);
            if t > 0x3b {
                0x3b
            } else {
                t
            }
        };

        let lo = (tmp[i0 as usize] as i32)
            .wrapping_mul(0x7fffi16.wrapping_sub(w) as i32)
            .wrapping_mul(2);
        // The (-32768, -32768) guard of §4.1. Unreachable here: `w` is a
        // halved 16-bit fraction and so never leaves [0, 0x7fff], so the
        // second operand of the multiply is never -0x8000.
        let hi = if w == i16::MIN && tmp[i1 as usize] == i16::MIN {
            i32::MAX
        } else {
            (tmp[i1 as usize] as i32)
                .wrapping_mul(w as i32)
                .wrapping_mul(2)
        };

        // The two weights sum to exactly 0x7fff, which bounds
        // |lo + hi| by 2 * 0x8000 * 0x7fff = 0x7fff0000: this saturation is
        // also unreachable, and is kept only for fidelity.
        *slot = high16_al2b1(add_sat_sgn_al2b4(hi, lo, lo));
        acc = acc.wrapping_add(step);
    }
}

/// `0x103172f0` — copy `n` words while rebasing them from block exponent
/// `exp_src` to `exp_dst`.
///
/// A one-line wrapper over `shift_copy16` (`0x103173e0`) that forms the shift
/// as the difference of the two exponents, so the caller never has to compute
/// it. Equal exponents degenerate to a plain copy.
pub fn rebase_block_exp(dst: &mut [i16], src: &[i16], n: u16, exp_src: i32, exp_dst: i32) {
    shift_copy16(dst, src, n, exp_src.wrapping_sub(exp_dst) as i16);
}

/// `0x10318720` — RMS amplitude of a `2n`-word vector, in block floating point.
///
/// Computes `sqrt(sum(x^2) / n)` without ever forming the quotient in fixed Q:
/// the sum of squares comes back from `dot_norm64` (`0x103170a0`) as a
/// normalised mantissa plus an exponent, `n` itself is normalised to Q30, one
/// extra halving guarantees the quotient is a proper fraction, and
/// `div32_half` then produces it as a 16-bit mantissa whose exponent is
/// `(15 − norm(n)) [+1] − exp(sum)`. `sqrt_norm` (`0x1030b2d0`) halves that
/// exponent and takes the root.
///
/// The vector length passed to the dot product is `2n`, so `n` counts *pairs*
/// — the caller is measuring a complex or two-channel buffer and `n` is the
/// number of points to divide by.
///
/// A non-positive or zero-energy sum yields `(0, 0)`. Domain: `n > 0` and
/// `vec` at least `2n` words.
pub fn vector_rms(out_mant: &mut i16, out_exp: &mut i16, vec: &[i16], n: i16) {
    let mut sum_exp: i16 = 0;
    let sum = dot_norm64(&mut sum_exp, vec, vec, n.wrapping_mul(2) as u16);

    let m = high16_al2b1(sum);
    if m <= 0 {
        *out_mant = 0;
        *out_exp = 0;
        return;
    }

    let v = up16_al2b4(n);
    let nrm = if v == 0 {
        0u16
    } else {
        let mag = if v < 0 { !(v as u32) } else { v as u32 };
        (0x1e - bsr(mag)) as u16
    };
    let mut den = ((v as u32) << (nrm as u32 & 0x1f)) as i32;
    let mut exp = (0x0fu32.wrapping_sub(nrm as u32) & 0xffff) as u16;
    if m <= high16_al2b1(den) {
        den >>= 1;
        exp = exp.wrapping_add(1);
    }

    let q = div32_half(den, m) as i16;
    let mut e = (exp as i16).wrapping_sub(sum_exp);
    let r = sqrt_norm(((q as i32) as u32).wrapping_shl(16) as i32, &mut e);
    *out_mant = high16_al2b1(r);
    *out_exp = e;
}

/// `0x1031fd80` — encode or decode one vector-quantised field, and hand back
/// the codebook entry it names.
///
/// With `encode != 0` the `nwords`-long vector already in `out` is matched
/// against the codebook by [`vq_search`] (`0x1031fc50`) and the winning index
/// is written to the bit stream through `emit_bits` (`0x10310310`); with
/// `encode == 0` the index is read back out of the bit stream by `read_field_l1_batch4`
/// (`0x103102e0`). Either way `out` is then overwritten with the codebook
/// entry, so the two directions leave the caller holding the same quantised
/// vector.
///
/// `bits_avail` truncates the field: when it is smaller than `nbits` the field
/// is shortened to `bits_avail` bits and the codebook stride is multiplied by
/// `1 << (nbits − bits_avail)`, i.e. the codebook is decimated instead of the
/// index. Both the shift and the stride multiply are 16-bit and wrap; the
/// shift count is masked to five bits, so a gap of 16 or more collapses the
/// stride to zero (§4.2).
///
/// `bits` is the one-bit-per-word payload array and `pos` the shared word
/// cursor into it. Returns the codebook index (the original leaves junk in the
/// high half of `eax`; only `ax` is meaningful).
#[allow(clippy::too_many_arguments)]
pub fn codebook_field(
    out: &mut [i16],
    nwords: u16,
    bits_avail: i16,
    book: &[i16],
    nbits: i16,
    stride: i16,
    bits: &mut [i16],
    pos: &mut i32,
    encode: i16,
) -> i16 {
    let mut nb = nbits;
    let mut st = stride;
    if bits_avail < nbits {
        let k = ((nbits as u8).wrapping_sub(bits_avail as u8)) & 0x1f;
        nb = bits_avail;
        let f: i16 = if k >= 16 { 0 } else { 1i16 << k };
        st = f.wrapping_mul(stride);
    }

    let idx = if encode != 0 {
        let cand: i16 = {
            let k = (nb as u8) & 0x1f;
            if k >= 16 {
                0
            } else {
                1i16 << k
            }
        };
        let found = vq_search(out, book, nwords as i16, cand, st);
        let mut cur = *pos as usize;
        emit_bits(&mut cur, bits, nb, found);
        *pos = cur as i32;
        found
    } else {
        read_field_i16(bits, pos, nb) as i16
    };

    if (nwords as i16) > 0 {
        let base = (idx as i32).wrapping_mul(st as i32);
        for i in 0..nwords as usize {
            out[i] = book[(base + i as i32) as usize];
        }
    }
    idx
}

// ======================= from aL2b5 =======================

/// Block exponent `0x1e - bsr(|x|)` as a 16-bit count, with the call sites'
/// explicit `x == 0 -> 0` guard folded in.
#[inline]
pub fn norm_shift_al2b5(x: i32) -> u16 {
    if x == 0 {
        return 0;
    }
    let m = if x < 0 { !(x as u32) } else { x as u32 };
    (0x1e - top_bit_al2b5(m)) as u16
}

/// x86 variable shift: non-negative counts shift left (wrapping), counts in
/// `-30..0` shift right arithmetically, anything below is a full sign fill.
#[inline]
pub fn shift32_al2b5(v: i32, s: i16) -> i32 {
    if s >= 0 {
        ((v as u32) << (s as u32 & 0x1f)) as i32
    } else if s < -0x1e {
        v >> 31
    } else {
        v >> ((-(s as i32)) as u32 & 0x1f)
    }
}

/// ETSI `L_add` saturation, but with the clamp's sign taken from `lim` rather
/// than from a fixed operand — SPEC §4.1 records that the choice is
/// per-call-site, and both conventions occur inside `0x10314f30`.
#[inline]
pub fn sat_add_al2b5(a: i32, b: i32, lim: i32) -> i32 {
    let s = a.wrapping_add(b);
    if ((s ^ a) & (s ^ b)) < 0 {
        0x7fff_ffffi32.wrapping_add((lim < 0) as i32)
    } else {
        s
    }
}

/// ETSI `L_sub` saturation with an explicit clamp-sign source, as [`sat_add_al2b5`].
#[inline]
pub fn sat_sub_al2b5(a: i32, b: i32, lim: i32) -> i32 {
    let d = a.wrapping_sub(b);
    if ((a ^ b) & (d ^ a)) < 0 {
        0x7fff_ffffi32.wrapping_add((lim < 0) as i32)
    } else {
        d
    }
}

/// `0x10310e60` — turn a log-domain pitch into the fundamental `w0` and the
/// harmonic count `L`.
///
/// `log_pitch` is a Q12 base-2 logarithm. The value is first range-reduced into
/// `[-1, 0)` by adding whole units of `0x1000`, which also fixes the final
/// shift count; the fractional part is then run through a quartic Horner
/// polynomial with the Q12 coefficients `0x13b, 0x71b, 0x1ec0, 0x58b9` and
/// biased by `-0x8000`, i.e. a `2^x` evaluation. The 16-bit result is shifted
/// right by `units - 4`.
///
/// **The shift is a 16-bit `sar` and its count is masked to 5 bits** (SPEC
/// §4.2): for a `log_pitch` at or above `-0x1000` no reduction happens, the
/// count is `0xfc & 0x1f = 28`, and the result collapses to a full sign fill —
/// so the whole high-pitch region simply lands on the lower clamp.
///
/// `w0` is clamped to `[0x1079, 0x6bca]`, the upper limit only when
/// `allow_high == 0`; those two limits are exactly the ends of the codec's
/// `L = 9 … 56` range. `L` is then [`harmonic_count`] of it.
pub fn fundamental_from_log2(w0_out: &mut i16, l_out: &mut i16, log_pitch: i16, allow_high: i16) {
    let mut cl: u8 = 0xfc;
    let mut x = log_pitch;
    if log_pitch < -0x1000 {
        let units = ((0xefffu16.wrapping_sub(log_pitch as u16)) >> 12).wrapping_add(1);
        x = log_pitch.wrapping_add((units as i16).wrapping_mul(0x1000));
        cl = units.wrapping_sub(4) as u8;
    }
    let d = x as i32;
    let mut a = (d.wrapping_mul(0x13b) >> 12).wrapping_add(0x71b) as i16 as i32;
    a = (a.wrapping_mul(d) >> 12).wrapping_add(0x1ec0) as i16 as i32;
    a = (a.wrapping_mul(d) >> 12).wrapping_add(0x58b9) as i16 as i32;
    let v = (a.wrapping_mul(d) >> 12).wrapping_sub(0x8000) as i16;

    let c = (cl & 0x1f) as u32;
    let mut w0 = if c >= 16 { v >> 15 } else { v >> c };

    if w0 < 0x1079 {
        w0 = 0x1079;
    } else if w0 > 0x6bca && allow_high == 0 {
        w0 = 0x6bca;
    }
    *w0_out = w0;
    *l_out = harmonic_count(w0);
}

/// `0x10312090` — the PRBA (pitch-refinement block average) transform: four
/// band DCTs, a 45° rotation of each output pair, and an 8-point DCT.
///
/// `scratch[i] = sat(a[i] - b[i])` over `n` words (symmetric Q15, clamped to
/// `±0x7fff`) forms the prediction residual of the log-magnitude vector. It is
/// then cut into four consecutive bands whose lengths come from `lens`; each
/// band is run through [`harmonic_dft`] asking for 6 outputs, of which only the
/// first two — the band's DC and first cosine coefficient — are kept, and the
/// band's tail from index 6 on is zeroed.
///
/// The eight kept values are read as four `(dc, c1)` pairs and rotated by 45°:
/// `dc' = dc + c1·√2`, `c1' = dc - c1·√2` in Q15 with `0xb504 = 46340` as the
/// (unsigned Q16) `√2/2`, both saturating symmetrically. A final 8-point
/// [`harmonic_dft`] over the rotated vector produces the PRBA coefficients that
/// the vector quantiser searches.
///
/// Domain: exactly four `lens`, each in `6..=56` (the DFT's reciprocal table
/// stops at 56 — SPEC §5), and `scratch` at least `sum(lens)` words long.
pub fn prba_transform(
    out: &mut [i16],
    scratch: &mut [i16],
    b: &[i16],
    a: &[i16],
    n: u16,
    lens: &[u16],
    cos_tbl: &[i16],
) {
    if (n as i16) > 0 {
        for i in 0..n as usize {
            let d = (a[i] as i32) - (b[i] as i32);
            scratch[i] = if d > 0x7fff {
                0x7fff
            } else if d < -0x7fff {
                -0x7fff
            } else {
                d as i16
            };
        }
    }

    let mut off = 0usize;
    for k in 0..4usize {
        let len = lens[k];
        harmonic_dft(&mut scratch[off..], len, 6, cos_tbl);
        out[2 * k] = scratch[off];
        out[2 * k + 1] = scratch[off + 1];
        fill16(&mut scratch[off + 6..], 0, (len as i16).wrapping_sub(6));
        off = off.wrapping_add((len as i16) as usize);
    }

    for k in 0..4usize {
        let p = (out[2 * k + 1] as i32).wrapping_mul(0xb504);
        let q = ((out[2 * k] as i32) << 16) >> 1;
        out[2 * k] = sat15(p.wrapping_add(q) >> 15);
        out[2 * k + 1] = sat15(q.wrapping_sub(p) >> 15);
    }

    harmonic_dft(out, 8, 8, cos_tbl);
}

/// Symmetric Q15 clamp — `-0x8000` is not representable here (SPEC §4).
#[inline]
pub fn sat15(v: i32) -> i16 {
    if v > 0x7fff {
        0x7fff
    } else if v < -0x7fff {
        -0x7fff
    } else {
        v as i16
    }
}

/// `0x103175d0` — renormalise a 16-bit vector into `dst` and credit the shift
/// to a running block exponent.
///
/// Takes the vector's block exponent with [`norm_exp_absmax`], copies it out
/// through [`shift_copy16`] scaled by the *negated* exponent (so the peak ends
/// up filling the word), and adds the exponent to `*exp`. The addition is a
/// 16-bit `add`, and the shift argument is formed by a full 32-bit `neg` of the
/// zero-extended return, so only the low half of either is significant.
pub fn normalise_to_exp(dst: &mut [i16], src: &[i16], n: u16, exp: &mut i16) {
    let e = norm_exp_absmax(src, n) as u16;
    shift_copy16(dst, src, n, e.wrapping_neg() as i16);
    *exp = exp.wrapping_add(e as i16);
}

/// `0x10314f30` — adaptive suppression of spectral peaks that appear suddenly
/// against the previous frame.
///
/// `cur` and `prev` are two parameter blocks, word-indexed here: `[2]` is the
/// harmonic count `L`, `[6]` the fundamental `w0`, `[8..]` the Q15 log
/// magnitudes, and `prev[0x41]` a slow level tracker.
///
/// Three stages:
///
/// 1. **Level tracking.** The RMS of `cur`'s magnitudes is formed as
///    `sqrt(dot(m,m) / L)` in block floating point ([`dot_norm64`],
///    [`div_norm`], [`sqrt_to_exp`] at exponent 4). If that RMS is above
///    `tracker - 0x800` the tracker moves toward it as `0.98·tracker +
///    0.02·rms` — the unsigned Q16 constants `0xfae0` and `0xa3d6 >> 5` of SPEC
///    §4.1 — otherwise it decays by 20 per frame.
///
/// 2. **Threshold.** Gated on `frame_kind >= 4 || sub_mode >= 2`. The previous
///    frame's magnitudes are spread one place right into a 60-tap curve
///    ([`spread_hold_alt`]), and a floor `tracker + 0x2c00 - 0x400·sub_mode` is
///    computed with saturation.
///
/// 3. **Per-harmonic clamp.** For harmonic `k = i+1` the floor is tilted down
///    by `min(6·k·w0 / 256, 0x1000)` — a fixed spectral tilt. The previous
///    frame's curve is resampled at the warped position `k · w0_cur / w0_prev`
///    with linear interpolation, so harmonics are compared across a pitch
///    change. Where the current magnitude exceeds *both* that resampled value
///    by more than `0x1000` *and* the tilted floor, it is pulled back toward
///    the previous frame by `(interp - cur) · min(cur - floor, 0x400)` scaled
///    up 4 bits, and re-clamped to the log-magnitude envelope `±0x77ff`.
///
/// Note the saturating add at the update takes its clamp sign from the
/// **addend**, not the accumulator (SPEC §4.1).
///
/// Domain: `cur[2] = L` in `1..=56`; `cur[6]` and `prev[6]` non-zero (both
/// divides fault otherwise) and their ratio at most 1 so the warped index stays
/// inside the 60-word curve.
pub fn suppress_spectral_peaks(cur: &mut [i16], prev: &mut [i16], frame_kind: i16, sub_mode: i16) {
    let l = cur[2];

    // -- stage 1: RMS and the level tracker ---------------------------------
    let vn = if l > 0 { l as usize } else { 0 };
    let mut e: i16 = 0;
    let energy = {
        let mags = &cur[8..8 + vn];
        dot_norm64(&mut e, mags, mags, l as u16)
    };
    e = e.wrapping_add(8);

    let ll = (l as i32) << 16;
    let sh = norm_shift_al2b5(ll);
    let den = (((ll as u32) << (sh as u32 & 0x1f)) as i32) >> 16;
    let mean = div_norm(energy, e, den as i16, (0xf - sh as i32) as i16, &mut e);
    let rms = sqrt_to_exp((mean as i32) << 16, e, 4) as u16;

    let t = prev[0x41] as i32;
    let s0 = sat_add_al2b5(t << 16, 0xf800_0000u32 as i32, t << 16);
    let s1 = sat_sub_al2b5(s0, ((rms as i16) as i32) << 16, s0);
    prev[0x41] = if s1 < 0 {
        ((t.wrapping_mul(0xfae0)).wrapping_add(((rms as i16 as i32).wrapping_mul(0xa3d6)) >> 5)
            >> 16) as i16
    } else {
        (((t << 16).wrapping_sub(0x0013_0001)) >> 16) as i16
    };

    if frame_kind < 4 && sub_mode < 2 {
        return;
    }

    // -- stage 2: the previous frame's curve and the floor -------------------
    let mut curve = [0i16; 60];
    spread_hold_alt(&mut curve, 0x3c, &prev[8..]);

    let tr = prev[0x41] as i32;
    let base = 0x2c00_0000i32.wrapping_sub(((sub_mode as i32) << 27) >> 1);
    let floor0 = (sat_add_al2b5(tr << 16, base, base) >> 16) as u16;

    // w0_cur / w0_prev in block floating point
    let w1 = (cur[6] as i32) << 16;
    let n1 = norm_shift_al2b5(w1);
    let num = ((w1 as u32) << (n1 as u32 & 0x1f)) as i32;
    let w2 = (prev[6] as i32) << 16;
    let n2 = norm_shift_al2b5(w2);
    let den2 = (((w2 as u32) << (n2 as u32 & 0x1f)) as i32) >> 16;
    let mut ratio_exp: i16 = 0;
    let ratio = div_norm(
        num,
        (0u32.wrapping_sub(n1 as u32).wrapping_sub(4)) as u16 as i16,
        den2 as i16,
        (0xffff_fffcu32.wrapping_sub(n2 as u32)) as u16 as i16,
        &mut ratio_exp,
    );

    if l <= 0 {
        return;
    }

    // -- stage 3: per-harmonic clamp ----------------------------------------
    let floor_q31 = ((floor0 as i16) as i32) << 16;
    let ratio_i = ratio as i32;
    let mut i: u16 = 0;
    loop {
        let k = i.wrapping_add(1) as i16;
        let tilt6 = (((k as i32).wrapping_mul(0x60000)) >> 16) as i16 as i32;
        let tilt = (tilt6.wrapping_mul(cur[6] as i32) << 8) >> 16;
        let mut tilt = tilt as u16 as i32;
        if (tilt as u16 as i16) > 0x1000 {
            tilt = 0x1000;
        }
        let fl = floor_q31.wrapping_sub(((tilt as u16 as i16) as i32) << 16);
        let fl16 = (fl >> 16) as u16;

        let warped = shift32_al2b5(ratio_i.wrapping_mul(k as i32).wrapping_mul(2), ratio_exp);
        let idx = ((warped >> 16) as u16 as i16) as i32;
        let frac = ((warped.wrapping_sub(idx << 16) << 15) >> 16) as u16 as i16 as i32;
        let co = ((0x7fffi32 - frac) as i16) as i32;
        let interp = (((co.wrapping_mul(curve[idx as usize] as i32))
            .wrapping_add((curve[idx as usize + 1] as i32).wrapping_mul(frac)))
        .wrapping_mul(2)
            >> 16) as u16 as i16 as i32;

        let p = 8usize + (i as i16) as usize;
        let interp_q31 = interp << 16;
        let c = cur[p] as i32;
        let d1 = sat_sub_al2b5(c << 16, interp_q31, c << 16);
        if sat_add_al2b5(d1, -0x1000_0000, d1) > 0 {
            let c = cur[p] as i32;
            let d3 = sat_sub_al2b5(c << 16, ((fl16 as i16) as i32) << 16, c << 16);
            if d3 > 0 {
                let mut g = (d3 >> 16) as u16 as i16;
                if g > 0x400 {
                    g = 0x400;
                }
                let c = cur[p] as i32;
                let delta = (sat_sub_al2b5(interp_q31, c << 16, interp_q31) >> 16) as i16;
                let prod = if delta == -0x8000 && g == -0x8000 {
                    0x7fff_ffff
                } else {
                    (delta as i32).wrapping_mul(g as i32).wrapping_mul(2)
                };
                let step = shl_sat(prod, 4);
                let c = cur[p] as i32;
                let ns = ((sat_add_al2b5(c << 16, step, step)) >> 16) as i16;
                cur[p] = ns;
                cur[p] = if ns > 0x77ff {
                    0x77ff
                } else if ns < -0x77ff {
                    -0x77ff
                } else {
                    ns
                };
            }
        }

        i = i.wrapping_add(1);
        if (i as i16) >= l {
            break;
        }
    }
}

/// `0x10319c80` — lay one harmonic down across the synthesis buffer with a
/// raised-cosine rise, a flat body and a matching fall.
///
/// `(pos, pos_exp)` is the harmonic's block-floating start position and `step`
/// its per-sample phase increment. [`ceil_and_residual`] splits the start into
/// an integer sample index and the sub-sample residual that seeds the
/// oscillator's phase; [`add_recip_split`] adds `1/step` — one period — to get
/// the end index. The end is clamped to sample `0xa6` (166) and the whole span
/// is abandoned if it would end before sample 0.
///
/// The span is then written in three [`synth_harmonic`] calls sharing one phase
/// accumulator and one cursor: a 16-sample rise with taper step `+1`, the body
/// with taper step `0`, and the fall to `end + 16` with taper step `-1`. The
/// taper phases are the two residuals scaled to Q16 (`+0xf0000` on the falling
/// side, so the two windows meet), and every sample is placed at block exponent
/// `exp - 0x10`.
///
/// A negative start index folds into the phase instead of the cursor: the
/// oscillator is advanced by `-start · step · 32` and both taper phases are
/// corrected, so the harmonic is written from sample 0 with the phase it would
/// have had.
///
/// Domain: `step` non-zero (the reciprocal divide faults); the integer start
/// must land inside `dst`.
#[allow(clippy::too_many_arguments)]
pub fn synth_harmonic_span(
    dst: &mut [i32],
    pos: i16,
    pos_exp: i16,
    step: i16,
    table: &[i16],
    exp: i32,
    cos_tbl: &[i16],
) {
    let mut start: i16 = 0;
    let mut start_frac: i16 = 0;
    ceil_and_residual(&mut start, &mut start_frac, pos, pos_exp);

    let mut end: i16 = 0;
    let mut end_frac: i16 = 0;
    add_recip_split(&mut end, &mut end_frac, pos, pos_exp, step);

    let mut e = end;
    if 0xa6 < e {
        e = 0xa6;
    }
    let last: u16;
    if (e as i32) + 0x10 <= 0xa6 {
        last = (e as u16).wrapping_add(0x10);
        if (last as i16) <= 0 {
            return;
        }
    } else {
        last = 0xa6;
    }

    let mut fall_phase = (((end_frac as i32) << 16) >> 15).wrapping_add(0xf_0000);
    let mut rise_phase = ((start_frac as i32) << 16) >> 15;
    let stp = step as i32;
    let mut phase = stp.wrapping_mul(start_frac as i32).wrapping_mul(2) >> 11;

    let mut cur: u16;
    if start < 0 {
        phase = phase.wrapping_add((start.wrapping_neg() as i32).wrapping_mul(stp) << 5);
        if start < -0x10 {
            rise_phase = rise_phase.wrapping_add(0x10_0000);
        } else {
            rise_phase = rise_phase.wrapping_sub((start as i32) << 16);
        }
        if e < 0 {
            fall_phase = fall_phase.wrapping_add((e.wrapping_add(1) as i32) << 16);
        }
        cur = 0;
    } else {
        cur = start as u16;
    }

    let shift = (exp.wrapping_sub(0x10) as u16) as i16;
    let mut cursor = ((cur as i16) as i32) as usize;

    let n_rise = (start as u16).wrapping_sub(cur).wrapping_add(0x10);
    if (n_rise as i16) > 0 {
        cur = cur.wrapping_add(n_rise);
        synth_harmonic(
            &mut phase,
            table,
            step,
            rise_phase,
            &mut cursor,
            dst,
            shift,
            1,
            n_rise,
            cos_tbl,
        );
    }

    let n_body = (e as u16).wrapping_sub(cur).wrapping_add(1);
    if (n_body as i16) > 0 {
        cur = cur.wrapping_add(n_body);
        synth_harmonic(
            &mut phase,
            table,
            step,
            0,
            &mut cursor,
            dst,
            shift,
            0,
            n_body,
            cos_tbl,
        );
    }

    let n_fall = last.wrapping_sub(cur).wrapping_add(1);
    synth_harmonic(
        &mut phase,
        table,
        step,
        fall_phase,
        &mut cursor,
        dst,
        shift,
        -1,
        n_fall,
        cos_tbl,
    );
}

// ======================= from aL3b1 =======================

/// The open-coded variable shift, followed by the `sar 16` every call site in
/// this batch applies: a non-negative count shifts left (wrapping, the count
/// masked to five bits as x86 does), a count in `-30..0` shifts right
/// arithmetically, and anything at or below −31 is a full sign fill.
#[inline]
pub fn shifted_hi(v: i32, sh: i16) -> i16 {
    let t = if sh >= 0 {
        ((v as u32) << (sh as u32 & 0x1f)) as i32
    } else if sh < -0x1e {
        v >> 31
    } else {
        v >> ((-(sh as i32)) as u32 & 0x1f)
    };
    (t >> 16) as i16
}

/// `0x10309cd0` — `dec_init(state, mode)`: the decoder's power-on reset.
///
/// This is the public `dec_init` entry point of SPEC §2. Word indices below are
/// halved byte offsets from the state base; the state block is a little under
/// 2 KB.
///
/// 1. Zero the first `0x3f4` words (`0x7e8` bytes) — everything except the
///    trailing configuration tail.
/// 2. Seed two configuration words: `+0x7c2 = -30` and `+0x7be = 0x1c94`
///    (7316), the initial level/exponent pair the output AGC starts from.
/// 3. Call [`frame_size_class`] (`0x1030cdb0`) on 0x20 and **discard the
///    result** — the compiler kept the call for its (absent) side effects.
/// 4. Store the frame-type code `0x20` at `+0x79c` and derive its starting gain
///    into `+0x79e` with [`gain_for_rate`] at −10 dB.
/// 5. Reset the two adaptive-level trackers at `+0x470` and `+0x4f8`
///    ([`init_level_tracker`]).
/// 6. Prime the 0x54-word unvoiced-excitation ring at `+0x64a` by running the
///    LCG at `+0x648` ([`rand_step`]) 84 times. The seed word itself was just
///    zeroed, so the noise sequence is identical on every `dec_init` — that is
///    why a decode is reproducible.
/// 7. Build the bit-allocation vector at `+0x7a0` and the AGC defaults at
///    `+0x580` for `mode` ([`init_bit_alloc_and_agc`], `0x1030b8e0`).
///
/// `state` must be at least `0x3e2` words long (and rather more for the
/// callees; `0x400` words is the natural size).
pub fn dec_init(state: &mut [i16], mode: i16) {
    fill16(state, 0, 0x3f4);
    state[0x7c2 / 2] = -0x1e;
    state[0x7be / 2] = 0x1c94;
    let _ = frame_size_class(0x20);
    state[0x79c / 2] = 0x20;
    {
        let mut g: i16 = 0;
        let _ = gain_for_rate(&mut g, 0x20, -10);
        state[0x79e / 2] = g;
    }
    init_level_tracker(&mut state[0x470 / 2..]);
    init_level_tracker(&mut state[0x4f8 / 2..]);

    for i in 0..0x54usize {
        let mut seed = state[0x648 / 2];
        rand_step(&mut seed);
        state[0x648 / 2] = seed;
        state[0x64a / 2 + i] = seed;
    }

    let (lo, hi) = state.split_at_mut(0x7a0 / 2);
    let alloc: &mut [i16; 13] = (&mut hi[..13]).try_into().unwrap();
    let _ = init_bit_alloc_and_agc(mode, alloc, &mut lo[0x580 / 2..]);
}

/// `0x10311620` — spread `total` higher-order spectral coefficients over
/// **six** blocks as evenly as the count allows, and quantise each block with
/// [`quantize_coeff_blocks`] (`0x103116b0`).
///
/// With `q = total / 6` ([`div32`]) and `r = 6·(q + 1) − total`, the split is
/// `r` blocks of `q` coefficients followed by `6 − r` blocks of `q + 1`, so the
/// short blocks come first and the total is exactly `total`. Each block's first
/// slot is skipped by the quantiser (it carries the block mean, coded in the
/// PRBA vector instead), and the allocation vector starts five entries in — the
/// first five nibbles belong to the gain and PRBA fields.
///
/// The two calls therefore hand the second pass `vals + q·r` and
/// `allocs + 5 + (q−1)·r`, both computed with 16-bit multiplies.
///
/// Decompiler artefact (SPEC §4.2): the routine parks `q − 1` in its own
/// `total` argument slot and reloads it for the second call, which Ghidra
/// renders as an assignment to the parameter.
///
/// Domain: `total >= 6` so that `q >= 1` and both offsets are non-negative.
#[allow(clippy::too_many_arguments)]
pub fn quantize_coeffs_six_blocks(
    payload: &mut [i16],
    pos: &mut i32,
    vals: &mut [i16],
    allocs: &[u16],
    total: i16,
    encode: i16,
    weight: &[i16],
    loading: &[i16],
) {
    let q = div32(total as i32, 6) as u16;
    let r = (((q as i32).wrapping_add(1).wrapping_mul(6)).wrapping_sub(total as i32)) as u16;

    quantize_coeff_blocks(
        payload,
        pos,
        r as i16,
        q.wrapping_sub(1),
        vals,
        &allocs[5..],
        encode,
        weight,
        loading,
    );

    let v_off = (q as i16).wrapping_mul(r as i16) as usize;
    let a_off = 5 + (q.wrapping_sub(1) as i16).wrapping_mul(r as i16) as usize;
    quantize_coeff_blocks(
        payload,
        pos,
        (6i32 - r as i32) as i16,
        q,
        &mut vals[v_off..],
        &allocs[a_off..],
        encode,
        weight,
        loading,
    );
}

/// `0x103121d0` — rebuild the `L` log2 spectral magnitudes from the decoded
/// PRBA vector, the higher-order coefficients and the frame gain.
///
/// This is the inverse of the encoder's amplitude analysis, and runs in four
/// stages:
///
/// 1. **PRBA inverse transform.** `prba` holds eight coded values; an 8-point
///    inverse harmonic cosine transform ([`harmonic_idct`], `0x1031fb40`)
///    expands them in place.
/// 2. **Radix-2 butterfly.** Each adjacent pair becomes
///    `(a + b)/2, (a − b)/2 · 0x16a08/2^16`, i.e. sum and scaled difference at
///    `1/sqrt(2) = 0xb504/2^16` (an *unsigned* Q16 constant, SPEC §4.1). The
///    four sums are the four blocks' mean log-magnitudes.
/// 3. **Per-block expansion.** Block `i` covers `block_len[i]` harmonics. Its
///    first two words are seeded from the butterfly output; the remaining four
///    slots already hold the decoded higher-order coefficients, and a 6-term
///    inverse harmonic cosine transform over the block's own length turns the
///    six coefficients into `block_len[i]` log magnitudes, written back into
///    `mags`.
/// 4. **Gain restoration.** The frame's mean is removed and the coded gain put
///    back: `offset = (2·L·(gain − log2(L)/2^5) − 2·Σ block_len[i]·prba[2i]) / L`
///    ([`log2_at_q`], [`div32_half`]), and every magnitude becomes
///    `sat(mags[i] + offset)` through [`shl_sat`] on the halved Q31 forms.
///
/// `prba` and `mags` are modified in place; `out` receives the final `L` words.
///
/// Domain: `1 <= L`, `1 <= block_len[i] <= 56` (the transform's reciprocal
/// table stops at 56), `Σ block_len[i] <= mags.len()`.
#[allow(clippy::too_many_arguments)]
pub fn expand_prba_to_log_mags(
    out: &mut [i16],
    l: u16,
    prba: &mut [i16],
    mags: &mut [i16],
    block_len: &[i16],
    gain: i16,
    cos_tbl: &[i16],
) {
    harmonic_idct(prba, 8, 8, cos_tbl);

    for i in 0..4usize {
        let a = (prba[2 * i] as i32).wrapping_mul(0x8000);
        let b = (prba[2 * i + 1] as i32).wrapping_mul(0x8000);
        prba[2 * i] = high16_al2b1(b.wrapping_add(a));
        let d = high16_al2b1(a.wrapping_sub(b)) as i32;
        prba[2 * i + 1] = high16_al2b1(d.wrapping_mul(0xb504));
    }

    let mut dst = 0usize;
    for i in 0..4usize {
        mags[dst] = prba[2 * i];
        mags[dst + 1] = prba[2 * i + 1];
        let n = block_len[i];
        harmonic_idct(&mut mags[dst..], n, 6, cos_tbl);
        dst = dst.wrapping_add(n as usize);
    }

    let lg = log2_at_q(l as i16, 0xf, 5) as i16;
    let mut acc = (gain.wrapping_sub(lg) as i32)
        .wrapping_mul(l as i16 as i32)
        .wrapping_mul(2);
    for i in 0..4usize {
        let t = (block_len[i] as i32)
            .wrapping_mul(prba[2 * i] as i32)
            .wrapping_mul(2);
        acc = acc.wrapping_sub(t);
    }
    let offset = div32_half(acc, l as i16) as i16;

    if (l as i16) > 0 {
        let base = (offset as i32).wrapping_mul(0x8000);
        for i in 0..l as usize {
            let x = (mags[i] as i32).wrapping_mul(0x8000);
            out[i] = (shl_sat(x.wrapping_add(base), 1) >> 16) as i16;
        }
    }
}

/// `0x10317d70` — rescale a block-floating-point vector so its energy equals a
/// target, in place.
///
/// The vector is first renormalised to fill the word ([`normalise_to_exp`],
/// `0x103175d0`), which credits the shift to `*exp`. Its energy is then
/// `dot(v, v)` in a 64-bit accumulator ([`dot_norm64`], `0x103170a0`) with
/// exponent `dot_exp + 2·(*exp)` — twice the vector exponent, because energy is
/// a square.
///
/// If the energy mantissa is zero the vector is left alone. Otherwise the gain
/// is `sqrt(target / energy)` in block floating point ([`div_norm`],
/// `0x1030a660`, then [`sqrt_norm`], `0x1030b2d0`), applied with
/// [`scale_vec`]. The gain's exponent is added to `*exp`; if the result is
/// positive the vector is shifted back down to exponent 0 with
/// [`scale_block_exp`] so it cannot be left over-scaled.
///
/// Decompiler artefact (SPEC §4.2): the `exp` *argument slot* is reused as the
/// scratch exponent handed to `div_norm` and `sqrt_norm`, which is why Ghidra
/// shows `&param_2` being passed after `param_2` was a pointer. `div_norm`
/// always writes it before `sqrt_norm` reads it, so its initial contents never
/// matter — confirmed by mutation: seeding `scratch` non-zero leaves every test
/// green.
pub fn scale_to_target_energy(
    buf: &mut [i16],
    exp: &mut i16,
    n: u16,
    target: i16,
    target_exp: i16,
) {
    // `buf` is both source and destination in the original; every pass is
    // element-wise, so a snapshot in a fixed scratch is exactly equivalent.
    // `n` is a harmonic count, bounded by 56 (SPEC §3).
    let nn = if (n as i16) > 0 { n as usize } else { 0 };
    let mut src = [0i16; 56];
    src[..nn].copy_from_slice(&buf[..nn]);
    normalise_to_exp(buf, &src, n, exp);

    let mut e: i16 = 0;
    let energy = dot_norm64(&mut e, buf, buf, n);
    let e = e.wrapping_add(exp.wrapping_mul(2));

    let den = high16_al2b1(energy);
    if den == 0 {
        return;
    }

    let mut scratch: i16 = 0;
    let m = div_norm(
        (target as i32).wrapping_mul(0x10000),
        target_exp,
        den,
        e,
        &mut scratch,
    );
    let g = sqrt_norm((m as i32).wrapping_mul(0x10000), &mut scratch);

    src[..nn].copy_from_slice(&buf[..nn]);
    scale_vec(buf, &src, n as i16, high16_al2b1(g));

    *exp = exp.wrapping_add(scratch);
    if *exp > 0 {
        src[..nn].copy_from_slice(&buf[..nn]);
        scale_block_exp(buf, &src, n, *exp as u16 as i32, 0);
        *exp = 0;
    }
}

/// `0x10319140` — build the voiced half of the synthesis spectrum and inverse
/// transform it.
///
/// This is the sparse-spectrum construction of SPEC §7.1. Harmonics `k` in
/// `[kmin, kmax)` whose voicing tag equals `tag` are written into `spec` as
/// interleaved `(real, imaginary)` pairs at complex index `k + 1`; everything
/// else stays zero, so the buffer is genuinely sparse and one 256-point inverse
/// real FFT produces the whole voiced contribution.
///
/// Per harmonic:
///
/// * the phase is `((w0·(k+1) + 0x20) + phase_resid[k]) << 10` — the
///   fundamental times the harmonic number plus the cached per-lag residual,
///   both left-shifted by 10 as SPEC §7.1 describes;
/// * that goes through [`frac_at_exp`] (`0x1030b230`, exponent 6) to keep only
///   the fractional turn, then `<< 10` and [`shift_round16_batch2`] (`0x1030b3d0`,
///   exponent 15), and the top nine bits index the 512-entry cosine table;
/// * the imaginary index is the same one **minus 0x80**, formed as
///   `high16_al2b1((i << 16) − 0x7f0001)`, i.e. a quarter period — sine and cosine
///   share one table (SPEC §4.1);
/// * both are multiplied by the harmonic's magnitude (`c·m·2`) and shifted by
///   `−e`, where `e` is the block exponent of the largest tagged magnitude
///   ([`norm_exp_max_tagged`], `0x10319f80`).
///
/// The spectrum exponent handed to [`inverse_real_fft`] is
/// `exp_in + 0x17 + e` with `e` **zero-extended** from 16 bits (`movzx eax,ax`
/// — `e` is negative when anything matched, so this is a 65536 offset, not a
/// no-op). It is not observable: `*exp_out` keeps only the low half and
/// `fft_merge_stage` truncates its exponent to 16 bits too. Reproduced anyway,
/// since the same idiom is load-bearing elsewhere (SPEC §4.3). If no harmonic
/// matched, the transform is skipped entirely and the exponent is set to the
/// −32 "silence" sentinel. Finally `spec[0x100]` is set to `spec[0]` — the
/// wrap-around sample the overlap-add stage reads.
///
/// Returns the number of harmonics written.
///
/// Domain: `spec` is 0x101 words; `0 <= kmin`, `kmax <= 127` so the last pair
/// fits, and `freq`/`tags`/`mags` are at least `kmax` long.
#[allow(clippy::too_many_arguments)]
pub fn build_voiced_spectrum(
    spec: &mut [i16],
    exp_out: &mut i16,
    w0: i16,
    phase_resid: &[i16],
    tags: &[i16],
    mags: &[i16],
    exp_in: i32,
    kmax: i16,
    kmin: i16,
    tag: i16,
    cos_tbl: &[i16],
) -> i16 {
    let e = norm_exp_max_tagged(mags, tags, kmin, kmax, tag) as u16;
    fill16(spec, 0, 0x100);

    let mut cnt: i16 = 0;
    if kmin < kmax {
        let sh = (e as i16).wrapping_neg();
        let mut idx = kmin as u16 as usize;
        let mut kp1 = (kmin as u16).wrapping_add(1);
        let mut left = kmax.wrapping_sub(kmin) as u16;
        loop {
            if tags[idx] == tag {
                let k = kp1 as i16 as i32;
                let phase = (w0 as i32)
                    .wrapping_mul(k)
                    .wrapping_add(0x20)
                    .wrapping_mul(0x400)
                    .wrapping_add(((phase_resid[idx] as i32) << 16) >> 6);
                let f = high16_al2b1(frac_at_exp(phase, 6)) as i32;
                let r = shift_round16_batch2(f.wrapping_mul(0x400), 0xf);
                let ci = (((r as i32) >> 16) & 0x1ff) as usize;

                let m = mags[idx] as i32;
                let p = (cos_tbl[ci] as i32).wrapping_mul(m).wrapping_mul(2);
                spec[k as usize * 2] = shifted_hi(p, sh);

                let ci2 = ((((ci as i32) << 16).wrapping_sub(0x7f_0001)) >> 16) as usize & 0x1ff;
                let p2 = (cos_tbl[ci2] as i32).wrapping_mul(m).wrapping_mul(2);
                cnt = cnt.wrapping_add(1);
                spec[k as usize * 2 + 1] = shifted_hi(p2, sh);
            }
            kp1 = kp1.wrapping_add(1);
            idx += 1;
            left = left.wrapping_sub(1);
            if left == 0 {
                break;
            }
        }
    }

    let ex = exp_in.wrapping_add(0x17).wrapping_add(e as i32);
    *exp_out = ex as i16;
    let r = if cnt < 1 {
        -0x20i32
    } else {
        inverse_real_fft(spec, ex, 8, 0)
    };
    *exp_out = r as i16;
    spec[0x100] = spec[0];
    cnt
}

// ======================= from aL3b2 =======================

/// `0x10310330` — decode a tone (signalling) frame from the payload bits and
/// build the synthetic spectrum it describes.
///
/// The frame carries, in payload order:
///
/// | field | bits | use |
/// |---|---|---|
/// | — | 6 | discarded (read only to advance the cursor) |
/// | amplitude, high part | 6 | |
/// | tone index | 8 × `n` | the same 8-bit index repeated `n` times |
/// | amplitude, low bit | 1 | |
///
/// with `n = (alloc[2] - 13) >> 3` derived from the frame's payload length, so
/// the tone index occupies whatever whole bytes are left after the 13 bits of
/// amplitude and header. The repeats are a majority-style integrity check
/// rather than a code: copy 0 is taken as the value, and the frame is rejected
/// if the **high nibble** of copy 1 differs from copy 0 at all, or if the total
/// Hamming distance from copy 0 over all later copies reaches 6. A rejected
/// frame passes `-1` as the tone index, which
/// [`pitch_config_init`] (`0x1030ce40`) classifies as out of range and refuses.
///
/// The 7-bit amplitude `v` becomes the band level `((2·v - 17) · 0x78f2 · 256)
/// >> 16` — a log-amplitude step of about 121 units per code — and is used for
/// both peaks of a dual-tone entry.
///
/// On success (`pitch_config_init` returning 0) the freshly built 64-word
/// config is also copied into `prev` by [`copy_params_l1_batch3`] (`0x1031f990`), so the
/// interpolator sees the same spectrum on both half-frames. Returns the
/// `pitch_config_init` status: 0 accepted, 4 rejected.
///
/// Domain: `n <= 8` — the repeats land in an eight-word stack buffer — i.e.
/// `alloc[2] <= 84`. `tbl` is the constant blob based at `0x1037f978`.
pub fn decode_tone_frame(
    cfg: &mut [i16],
    prev: &mut [i16],
    bits: &[i16],
    pos: &mut i32,
    alloc: &[i16],
    tbl: &[i16],
) -> i16 {
    read_field_i16(bits, pos, 6);
    let amp_hi = read_field_i16(bits, pos, 6) as i16;
    // `add ax,ax`: the doubled field is kept as a 16-bit quantity.
    let amp2 = amp_hi.wrapping_mul(2);

    let mut rep = [0i16; 8];
    fill16(&mut rep, 0, 8);

    let n = (((alloc[2] as i32) - 0xd) >> 3) as i16;
    if n > 0 {
        for slot in rep.iter_mut().take(n as u16 as usize) {
            *slot = read_field_i16(bits, pos, 8) as i16;
        }
    }

    let mut tone = rep[0];
    let mut errs: i16 = 0;
    if n > 1 {
        for k in 1..(n as u16 as usize) {
            let d = (rep[k] as i32) ^ (rep[0] as i32);
            errs = errs.wrapping_add(popcount_low(d as u32, 8) as i16);
        }
    }
    if ((rep[1] as u8) ^ (rep[0] as u8)) & 0xf0 != 0 || errs >= 6 {
        tone = -1;
    }

    let lsb = read_field_i16(bits, pos, 1) as i16;
    let v = (lsb as i32) | (amp2 as i32);
    let r = ((v << 17).wrapping_sub(0x0010_0001) >> 16) as i16;
    let level = (((r as i32).wrapping_mul(0x0078_f200)) >> 16) as i16;

    let st = pitch_config_init(cfg, tone, level, level, 0, tbl) as u16 as i16;
    if st == 0 {
        copy_params_l1_batch3(prev, cfg);
    }
    st
}

/// `0x103118c0` — walk a `count × n` matrix of harmonic coefficients, seeding
/// each row's DC term from `src` and expanding the row in place.
///
/// For row `i` the routine writes `src[i]` into word 0 of the row and then runs
/// [`harmonic_idct`] (`0x1031fb40`) over the row with `n` points and `n`
/// coefficients. **Only the DC word is written by this function** — the rest of
/// the row is whatever the caller left there, and the transform reads all `n`
/// of them, so the caller owns the higher coefficients.
///
/// Returns the word offset one past the last row, i.e. `count · n`, which is
/// the original's returned end pointer (a `count <= 0` returns `dst`
/// untouched).
///
/// Domain: `1 <= n <= 56` — [`harmonic_idct`]'s reciprocal table ends at 56.
pub fn expand_rows_idct(
    dst: &mut [i16],
    src: &[i16],
    count: i16,
    n: i16,
    cos_tbl: &[i16],
) -> usize {
    if count <= 0 {
        return 0;
    }
    let stride = n as isize;
    let mut off: isize = 0;
    for i in 0..(count as u16 as usize) {
        dst[off as usize] = src[i];
        harmonic_idct(&mut dst[off as usize..], n, n, cos_tbl);
        off += stride;
    }
    off as usize
}

/// `0x10312fa0` — the encoder's IMBE frame packer.
///
/// A one-line wrapper that calls [`interleave_imbe_frame`] (`0x10312fe0`) with
/// its store flag set, i.e. natural info-vector order in, channel order out.
/// The matching de-interleaver is `0x10312fc0`.
pub fn pack_imbe_frame(chan: &mut [i16], nharm: i16, hbits: &[i16], alloc: &[i16]) {
    interleave_imbe_frame(chan, nharm, hbits, alloc, true);
}

/// `0x10317e20` — halve the bottom eighth of a spectrum and shape the rest by
/// how much headroom each bin has under a threshold.
///
/// The whole block's exponent is raised by one, so every bin is implicitly
/// halved. The lowest `n/8` bins are rescaled to that new exponent by
/// [`rebase_block_exp`] (`0x103172f0`) and therefore keep their value exactly.
/// Each remaining bin `i` gets a gain
///
/// ```text
/// margin = (thresh · 2^16) @ exp_thresh  -  (ref[i] · scale · 2) @ exp_scale
/// g      = high16_al2b3( sqrt( sqrt(margin) · spec[i] · 2 ) << (e - 1) )
/// spec[i] = spec[i] · clamp(g, 0x2000, 0x4ccd) · 2 >> 16
/// ```
///
/// where both operands are first brought to the common exponent
/// `e = max(exp_scale, exp_thresh) + 1`, the two [`sqrt_norm`] (`0x1030b2d0`)
/// calls thread one exponent through (with the block exponent added in
/// between), and a result still normalised at exponent 2 or more is taken as
/// "wide open" and given the ceiling directly.
///
/// The clamp window `[0x2000, 0x4ccd]` is Q15 `[0.25, 0.6]`, so against the
/// doubled block exponent the effective per-bin gain is `[0.5, 1.2]`: the low
/// eighth is passed through and the upper bins are attenuated or mildly lifted
/// according to their margin. A zero margin takes the floor.
///
/// `exp` is the block exponent cell, read and post-incremented. A count with
/// `n/8 >= n` — which includes every non-positive `n` — only bumps the exponent
/// and returns.
#[allow(clippy::too_many_arguments)]
pub fn shape_by_headroom(
    spec: &mut [i16],
    exp: &mut i16,
    n: i16,
    scale: i16,
    exp_scale: i16,
    thresh: i16,
    exp_thresh: i16,
    reference: &[i16],
) {
    let lo = n >> 3;

    // `lo = n >> 3` with `n <= 56`, so at most 7 words are snapshotted.
    let keep = if lo > 0 { lo as usize } else { 0 };
    let mut head = [0i16; 7];
    head[..keep].copy_from_slice(&spec[..keep]);
    let e_blk = *exp as u16 as i32;
    rebase_block_exp(spec, &head, lo as u16, e_blk, e_blk + 1);

    // Common exponent for the two operands, formed 16-bit and zero-extended.
    let e = if exp_thresh > exp_scale {
        exp_thresh
    } else {
        exp_scale
    }
    .wrapping_add(1);

    if lo >= n {
        *exp = exp.wrapping_add(1);
        return;
    }

    let sh_scale = exp_scale.wrapping_sub(e);
    let sh_thresh = exp_thresh.wrapping_sub(e);
    let t = (thresh as i32) << 16;
    let g = scale as i32;

    let count = (n as u16).wrapping_sub(lo as u16);
    for k in 0..count as usize {
        let i = (lo as u16 as usize).wrapping_add(k);
        let a = shift32_al3b2(
            (reference[i] as i32).wrapping_mul(g).wrapping_mul(2),
            sh_scale,
        );
        let margin = shift32_al3b2(t, sh_thresh).wrapping_sub(a);

        let gain: i16 = if margin == 0 {
            // exponent 0, mantissa 0 — the shift below yields 0 and the clamp
            // takes the floor.
            clamp_gain(0, -1)
        } else {
            let mut ex = e;
            let s1 = sqrt_norm(margin, &mut ex);
            ex = ex.wrapping_add(*exp);
            let arg = (high16_al2b3(s1) as i32)
                .wrapping_mul(spec[i] as i32)
                .wrapping_mul(2);
            let s2 = sqrt_norm(arg, &mut ex);
            if ex < 2 {
                clamp_gain(s2, ex.wrapping_sub(1))
            } else {
                0x4ccd
            }
        };

        spec[i] = high16_al2b3((spec[i] as i32).wrapping_mul(gain as i32).wrapping_mul(2));
    }

    *exp = exp.wrapping_add(1);
}

/// The gain clamp of [`shape_by_headroom`]: renormalise, take the high word and
/// hold it inside Q15 `[0.25, 0.6]`.
#[inline]
pub fn clamp_gain(v: i32, sh: i16) -> i16 {
    let h = high16_al2b3(shift32_al3b2(v, sh));
    if h > 0x4ccd {
        0x4ccd
    } else if h < 0x2000 {
        0x2000
    } else {
        h
    }
}

/// `0x10319330` — lay a whole bank of voiced harmonics into the synthesis
/// buffer.
///
/// [`solve_quad_ramp`] (`0x10319760`) first solves the harmonic-position
/// quadratic for all `count` harmonics, producing a block-floating
/// (mantissa, exponent) pair per harmonic; each pair is then handed to
/// [`synth_harmonic_span`] (`0x10319c80`), which writes that harmonic's
/// raised-cosine span into `dst` at the shared phase increment `step`, window
/// `table` and block exponent `exp`. The last harmonic's position is returned
/// through `last_pos` / `last_exp` so the caller can continue the ramp into the
/// next subframe.
///
/// Domain: `1 <= count <= 20` — the two stack arrays hold twenty words each,
/// and the trailing read of element `count - 1` happens unconditionally, so a
/// non-positive `count` reads off the front of the frame. `a` and `c` must be
/// non-zero and `step` non-zero (both callees divide unguarded).
#[allow(clippy::too_many_arguments)]
pub fn synth_harmonic_bank(
    dst: &mut [i32],
    last_pos: &mut i16,
    last_exp: &mut i16,
    step: i16,
    table: &[i16],
    exp: i32,
    x0: i32,
    k: i16,
    a: i16,
    c: i16,
    count: u16,
    cos_tbl: &[i16],
) {
    let mut pos = [0i16; 20];
    let mut pexp = [0i16; 20];
    solve_quad_ramp(&mut pos, &mut pexp, x0, a, k, c, count);

    if (count as i16) > 0 {
        for i in 0..(count as usize) {
            synth_harmonic_span(dst, pos[i], pexp[i], step, table, exp, cos_tbl);
        }
    }

    let last = (count as i16 - 1) as usize;
    *last_pos = pos[last];
    *last_exp = pexp[last];
}

// ======================= from aL3b3 =======================

/// ETSI `L_add` with the saturation limit taken from `sgn` rather than from
/// the result — which operand supplies it is per call site (SPEC §4.1).
#[inline]
pub fn add_sat_sgn_al3b3(a: i32, b: i32, sgn: i32) -> i32 {
    let r = a.wrapping_add(b);
    if ((r ^ a) & (r ^ b)) < 0 {
        (0x7fff_ffffu32).wrapping_add((sgn < 0) as u32) as i32
    } else {
        r
    }
}

/// ETSI `L_sub`, saturation limit taken from the minuend.
#[inline]
pub fn sub_sat(a: i32, b: i32) -> i32 {
    let r = a.wrapping_sub(b);
    if ((!b ^ r) & (r ^ a)) < 0 {
        (0x7fff_ffffu32).wrapping_add((a < 0) as u32) as i32
    } else {
        r
    }
}

/// `fill16` (`0x1030b620`) as the original uses it here: it returns `dst + n`
/// **words**, with `n` truncated to `i16` and sign-extended, whether or not it
/// wrote anything. Callers in this batch walk a cursor with that return value,
/// so the pointer arithmetic has to be reproduced even for `n <= 0`.
pub fn fill_at(buf: &mut [i16], base: isize, val: i16, n: i16) -> isize {
    if n > 0 {
        let a = base as usize;
        fill16(&mut buf[a..], val, n);
    }
    base + n as isize
}

/// `0x10310ff0` — pick the frame's fundamental from a coarse pitch-lag index,
/// or fall back to a fixed default outside AMBE mode.
///
/// For `mode != 1` (i.e. IMBE, whose pitch arrives already quantised elsewhere)
/// this just writes the constants `w0 = 0x4027`, `L = 0xf` and returns.
///
/// For AMBE the lag index is turned into a base-2 logarithm and handed to
/// [`fundamental_from_log2`] (`0x10310e60`). The lag is taken at half-sample
/// resolution — `2·lag + 1` — scaled by `0x2a52` and divided by
/// `2·(0x3c << (win − 6))`, the analysis window length for the current
/// sub-sampling factor, by [`div32`] (`0x1030b430`). The quotient is a Q12
/// log-period which is negated and biased by `-0x44fd` to become the Q12
/// log-frequency the fundamental helper expects.
///
/// Note the divisor is formed in **16-bit** arithmetic: `0x3c << k` wraps, so
/// `k >= 13` yields a zero divisor and faults inside `div32`. `k` is
/// `(win − 6) & 0x1f`.
pub fn fundamental_from_lag(
    mode: i16,
    w0: &mut i16,
    l: &mut i16,
    lag: i16,
    win: i16,
    allow_high: i16,
) {
    if mode != 1 {
        *w0 = 0x4027;
        *l = 0xf;
        return;
    }
    let k = ((win as i32).wrapping_sub(6) as u32) & 0x1f;
    let den = (if k >= 16 { 0u16 } else { 0x3cu16 << k }).wrapping_mul(2) as i16;
    let num = (lag.wrapping_mul(2).wrapping_add(1) as i32).wrapping_mul(0x2a52);
    let q = div32(num, den) as i16;
    let log_pitch = (0xbb03u16 as i16).wrapping_sub(q);
    fundamental_from_log2(w0, l, log_pitch, allow_high);
}

/// `0x10311ee0` — build the unvoiced (noise) spectral envelope for one frame
/// and resample it onto `n` synthesis bins.
///
/// Three steps, all in the log-magnitude domain:
///
/// 1. [`attenuate_unvoiced_bands`] (`0x10316f40`) copies the 56 harmonic
///    magnitudes into scratch, pushing every harmonic that falls in a *voiced*
///    band down by a level-dependent offset — what is left is the noise floor.
/// 2. [`resample_lerp`] (`0x10311910`) stretches those 56 points onto `n`
///    output bins, `l` setting the phase increment, and returns the summed
///    output (the envelope's mean energy).
/// 3. [`scale_bias_vec`] (`0x10312400`) mean-removes and gain-scales the
///    result in place.
///
/// The gain ramps with the output width: `0x3333` (0.4) below 16 bins,
/// `0x599a` (0.7) from 25 up, and a straight line between — `0x3d70800·n`
/// minus `0x6660000`, taken from the high word, which is 0.4 at n = 15 and 0.7
/// at n = 25. That is a deliberate widening of the unvoiced contribution as the
/// synthesis band count grows.
///
/// Domain: `l >= 1`, or the attenuation pass writes nothing and the scratch
/// buffer is read uninitialised.
pub fn unvoiced_envelope(out: &mut [i16], n: i16, mags: &[i16], l: i16, level: i16, mask: i16) {
    let mut scratch = [0i16; 0x38];
    attenuate_unvoiced_bands(&mut scratch, mags, mask, level, l, 0);
    let sum = resample_lerp(out, n as u16, &scratch, l);
    let coef: i16 = if n < 0x10 {
        0x3333
    } else if n < 0x19 {
        high16_al2b3((n as i32).wrapping_mul(0x3d7_0800).wrapping_sub(0x666_0000))
    } else {
        0x599a
    };
    scale_bias_vec(out, sum, n, coef);
}

/// `0x10314ae0` — average the two neighbouring frames' spectral envelopes onto
/// the current frame's harmonic grid.
///
/// Both neighbours are resampled by [`interp_envelope`] (`0x10314be0`) from
/// their own fundamental onto this frame's, then averaged elementwise with a
/// floor-halving 48-bit add (`shrd`/`sar`, i.e. `(a + b) >> 1`) and clamped to
/// the codec's symmetric log-magnitude range `±0x77ff`. The result is written
/// into the current frame's own envelope slot, and the tail out to harmonic 55
/// is filled by repeating the last live harmonic so downstream code can index
/// the full 56 words unconditionally.
///
/// Each parameter block is addressed in words: `[2]` is `L`, `[6]` is the
/// fundamental `w0`, and `[8 ..= 0x3f]` the 56-point envelope.
///
/// Domain: `1 <= cur[2] <= 0x38`, and both neighbours' `w0` non-zero (a zero
/// divisor faults inside `ratio_q16`).
pub fn average_neighbour_envelopes(cur: &mut [i16], prev: &[i16], next: &[i16]) {
    let l = cur[2];
    let mut a = [0i16; 0x38];
    let mut b = [0i16; 0x38];
    interp_envelope(&mut a, cur[6], prev[6], &prev[8..], l as u16);
    interp_envelope(&mut b, cur[6], next[6], &next[8..], l as u16);

    let mut i: i16 = 0;
    while i < cur[2] {
        let s = (b[i as usize] as i32).wrapping_add(a[i as usize] as i32) >> 1;
        let mut v = s as i16;
        if v < -0x77ff {
            v = -0x77ff;
        } else if v > 0x77ff {
            v = 0x77ff;
        }
        cur[8 + i as usize] = v;
        i = i.wrapping_add(1);
    }

    let n = cur[2];
    if n < 0x38 {
        let pad = cur[7 + n as usize];
        let mut k = (0x38u16.wrapping_sub(n as u16)) as u32;
        let mut d = 8 + n as usize;
        while k != 0 {
            cur[d] = pad;
            d += 1;
            k -= 1;
        }
    }
}

/// `0x10318430` — shape the pre-filled noise spectrum band by band, so each
/// unvoiced harmonic band carries the amplitude the parameters ask for.
///
/// `spec` arrives holding `n` words of raw (real, imaginary) noise. The band
/// edges are walked with a Q16 accumulator stepping by
/// `(w0 · n · 2) >> 4` — one step per harmonic, half-open, clamped to `n/2` —
/// and for each harmonic `i`:
///
/// * `flags[i] != 0` (a **voiced** band, synthesised elsewhere) zeroes the
///   band's `2·count` words;
/// * otherwise the band is rescaled to `amps[i]` by
///   [`scale_vec_shift`] (`0x10317200`) and the returned count is incremented.
///
/// The per-band coefficient is the harmonic amplitude times a normalisation
/// factor. In mode 2 that factor is a constant for the whole frame — the
/// square root of `L` (`sqrt_norm` of `L · 0x86b6`) further scaled by
/// `0xaff4/2^16` — while in every other mode the band's own measured RMS
/// ([`vector_rms`], `0x10318720`) is divided out first, so each band is
/// normalised individually. `scale`, applied to `L` before the root, is the
/// caller's overall noise gain.
///
/// Afterwards the spectrum's top is cleared: everything from the last band
/// edge up, but never fewer than `n/2 − limit` words where `limit` is `0xf2`
/// for a 512-point transform and `0x79` otherwise; and the first
/// `((zero_head · n · 2) >> 20) · 2` words (the DC region) are cleared too.
///
/// `*exp_out` is read on entry only in mode 2, and always written on exit with
/// the block exponent the spectrum now carries: `st[0x42] + total`.
///
/// `st` is the parameter block in words — `[0]` mode, `[2]` `L`, `[6]` `w0`,
/// `[8..]` amplitudes, `[0x40..0x41]` the pointer to `flags` (passed here
/// separately, since Rust cannot model it), `[0x42]` the exponent base.
///
/// Domain: `w0 > 0` and `L` small enough that `L` band edges fit inside `n/2`
/// — a clamped edge yields a zero-length band, which faults inside the RMS.
#[allow(clippy::too_many_arguments)]
pub fn shape_noise_bands(
    spec: &mut [i16],
    exp_out: &mut i16,
    n: i16,
    st: &[i16],
    flags: &[i16],
    zero_head: i16,
    scale: i16,
) -> i16 {
    let mode2 = st[0] == 2;
    let half = (n >> 1) as u16;
    let limit: i16 = if n == 0x200 { 0xf2 } else { 0x79 };

    // Block exponent of the loudest *unvoiced* harmonic amplitude.
    let peak = peak_exp_unflagged(&st[8..], flags, st[2]) as u16;

    let mut exp: u16 = 0x11;
    let mant: i16;
    if mode2 {
        let mut e = exp as i16;
        let r = sqrt_norm((st[2] as i32).wrapping_mul(0x86b6), &mut e);
        // SPEC §4.2: `extraout_var` is high16_al2b3(eax) of the call, not edx.
        let c = ((r >> 16) as i16 as i32).wrapping_mul(0xaff4);
        exp = (e as u16).wrapping_add((*exp_out as u16).wrapping_add(1));
        mant = high16_al2b3(c);
    } else {
        let mut e = exp as i16;
        let x = st[2].wrapping_mul(scale) as i32;
        let r = sqrt_norm(x.wrapping_mul(0x86b6), &mut e);
        exp = e as u16;
        mant = (r >> 16) as i16;
    }

    let total = exp.wrapping_add(peak).wrapping_add(2);
    let nn = n as i32;
    let step = (st[6] as i32).wrapping_mul(nn).wrapping_mul(2) >> 4;

    let mut acc = step.wrapping_add(0x10000) >> 1;
    let mut idx = (acc >> 16) as i16 as u16;
    let mut ptr = fill_at(spec, 0, 0, idx.wrapping_mul(2) as i16);

    let mut voiced_bands: i16 = 0;
    if 0 < st[2] {
        // The shift is `-(peak)` formed in 16 bits, then sign-extended.
        // `peak_exp_unflagged` only ever returns `-(30 - msb)`, i.e. `-30..=0`,
        // so `sh` is non-negative and the right-shift arms of `shift32_al3b2` here
        // are unreachable; they are kept because the original codes them.
        let sh = (peak as i16).wrapping_neg();
        // Hoisted out of the loop: `scale_vec_shift` reads only `src[..2*cnt]`,
        // and those words are re-copied on every iteration that uses them, so
        // the tail is never read. Declaring it inside the loop costs a
        // 224-byte zero-fill per band for nothing.
        let mut src = [0i16; 112];
        let mut i: i16 = 0;
        loop {
            acc = acc.wrapping_add(step);
            let prev = idx;
            let mut cur = (acc >> 16) as i16;
            if cur > half as i16 {
                cur = half as i16;
            }
            idx = cur as u16;
            let cnt = idx.wrapping_sub(prev) as i16;

            if flags[i as usize] != 0 {
                fill_at(spec, ptr, 0, cnt.wrapping_mul(2));
            } else {
                let (m, e) = if mode2 {
                    (mant, exp as i16)
                } else {
                    let (mut rm, mut re) = (0i16, 0i16);
                    vector_rms(&mut rm, &mut re, &spec[ptr as usize..], cnt);
                    (
                        high16_al2b3((rm as i32).wrapping_mul(mant as i32).wrapping_mul(2)),
                        re.wrapping_add(exp as i16),
                    )
                };
                let p = shift32_al3b2(
                    (st[8 + i as usize] as i32)
                        .wrapping_mul(m as i32)
                        .wrapping_mul(2),
                    sh,
                );
                let e2 = e.wrapping_add(peak as i16);
                let coef = high16_al2b3(p);
                let a = ptr as usize;
                let w = (cnt as usize).wrapping_mul(2);
                if cnt > 0 {
                    src[..w].copy_from_slice(&spec[a..a + w]);
                }
                scale_vec_shift(
                    &mut spec[a..],
                    &src,
                    cnt.wrapping_mul(2),
                    coef,
                    e2,
                    total as i16,
                );
                voiced_bands = voiced_bands.wrapping_add(1);
            }
            ptr += (cnt as isize) * 2;
            i = i.wrapping_add(1);
            if i >= st[2] {
                break;
            }
        }
    }

    // Clear the top of the spectrum, never leaving more than `limit` bins live.
    let mut rem = (half as i16).wrapping_sub(idx as i16) as u16;
    if (half as i16 as i32 - limit as i32) > (rem as i16) as i32 {
        rem = (half as i16).wrapping_sub(limit) as u16;
    }
    fill_at(
        spec,
        (nn - 2 * (rem as i16) as i32) as isize,
        0,
        rem.wrapping_mul(2) as i16,
    );
    // …and the DC region at the bottom.
    fill_at(
        spec,
        0,
        0,
        ((((zero_head as i32).wrapping_mul(nn).wrapping_mul(2)) >> 0x14) * 2) as i16,
    );

    *exp_out = st[0x42].wrapping_add(total as i16);
    voiced_bands
}

/// `0x1031fe40` — quantise (or dequantise) the frame gain, with a one-frame
/// predictor.
///
/// On the **encode** side (`encode != 0`) the gain to be coded is derived from
/// the log-magnitude vector: the mean of `mags` ([`mean_i16`], `0x103171d0`)
/// plus `log2(L)` at Q5 ([`log2_at_q`], `0x1030b1f0`) — i.e. the mean log
/// amplitude corrected for the harmonic count, so the value coded is the frame
/// energy rather than the per-harmonic level — minus half the previous frame's
/// gain, which is the predictor. Every step saturates. Mode 2 then adds
/// `0x1000`.
///
/// [`codebook_field`] (`0x1031fd80`) codes or decodes that scalar against the
/// 32-entry Q11 gain codebook at `0x10382160`, leaving the reconstructed value
/// in `gain[0]`; mode 2 removes its `0x1000` offset again afterwards.
///
/// Finally the predictor is undone — `gain = (prev + 2·gain) / 2` — the result
/// is clamped at `0x6800`, and outside mode 2 it becomes the new `*prev`. Mode
/// 2 therefore quantises without disturbing the predictor state, which is what
/// makes it usable for a trial encode.
///
/// Domain: `nmags >= 1` (a zero count faults in `mean_i16`'s divide), and
/// `nbits` at most 5 — beyond that `vq_search` runs off the end of the
/// 32-entry codebook.
#[allow(clippy::too_many_arguments)]
pub fn quantise_gain(
    gain: &mut [i16],
    mode: i16,
    mags: &[i16],
    nmags: i16,
    prev: &mut i16,
    nbits: i16,
    book: &[i16],
    bits: &mut [i16],
    pos: &mut i32,
    encode: i16,
) {
    if encode != 0 {
        gain[0] = mean_i16(mags, nmags as u16) as i16;
        let lg = log2_at_q(nmags, 0xf, 5) as i16;
        // The original takes the saturation limit from the second operand
        // (`mean << 16`). For a two-operand `L_add` that is indistinguishable
        // from the first — an add can only overflow when both operands share a
        // sign — so the choice is immaterial here; see the module notes.
        let s1 = add_sat_sgn_al3b3(
            (lg as i32) << 16,
            (gain[0] as i32) << 16,
            (gain[0] as i32) << 16,
        );
        let s2 = sub_sat(s1, (*prev as i32) << 15);
        let s3 = shl_sat(s2, 0);
        gain[0] = high16_al2b3(add_sat_sgn_al3b3(s3, 0x8000, s3));
        if mode == 2 {
            let v = (gain[0] as i32) << 16;
            gain[0] = high16_al2b3(add_sat_sgn_al3b3(v, 0x1000_0000, v));
        }
    }

    codebook_field(gain, 1, nbits, book, nbits, 1, bits, pos, encode);

    if mode == 2 {
        gain[0] = high16_al2b3(((gain[0] as i32) << 16).wrapping_sub(0x0fff_0001));
    }
    let mut g = high16_al2b3(
        ((*prev as i32).wrapping_add((gain[0] as i32).wrapping_mul(2)) as u32).wrapping_shl(15)
            as i32,
    );
    if g > 0x67ff {
        g = 0x6800;
    }
    gain[0] = g;
    if mode != 2 {
        *prev = g;
    }
}

// ======================= from aL4b1 =======================

/// The inlined hardware normalisation of SPEC §4.1: `0x1e - bsr(|x|)`, with
/// `bsr` leaving its destination unchanged for a zero source. The caller
/// guards `x == 0` separately, so the only surviving quirk is `x == -1`, where
/// `~x` is zero and the shift comes out **30**, not 31.
///
/// That quirk is coded for fidelity but is **unreachable at all three call
/// sites in this module**: two normalise `n << 16` and the third a sum of two
/// values that are both multiples of 256, so the argument is never `-1`.
/// Mutating the 30 to a 31 leaves the tests green, which is the evidence.
#[inline]
pub fn norm_shift_al4b1(x: i32) -> u16 {
    if x == 0 {
        return 0;
    }
    let mag = if x < 0 { !x } else { x } as u32;
    // `bsr` with a zero source leaves the destination at its previous value,
    // which at this point is the (zeroed) magnitude itself.
    let msb = if mag == 0 {
        0
    } else {
        31 - mag.leading_zeros()
    };
    (0x1eu32.wrapping_sub(msb)) as u16
}

/// x86 `shl` with the count masked to five bits, then `sar 16`.
#[inline]
pub fn shl_hi16(x: i32, sh: u16) -> i16 {
    (((x as u32) << (sh as u32 & 0x1f)) as i32 >> 16) as i16
}

#[inline]
pub fn get32_al4b1(w: &[i16], i: usize) -> u32 {
    (w[i] as u16 as u32) | ((w[i + 1] as u16 as u32) << 16)
}

#[inline]
pub fn put32_al4b1(w: &mut [i16], i: usize, v: u32) {
    w[i] = v as u16 as i16;
    w[i + 1] = (v >> 16) as u16 as i16;
}

/// `0x10314980` — decide which parameters a frame should be synthesised from
/// when the payload said "repeat".
///
/// `cur` and `prev` are two decoded parameter blocks (words: `[0]` frame
/// class, `[1]` a repeat tag, `[2]` `L`, `[4..6]` the packed voicing dibits,
/// `[6]` `w0`, `[7]` a mute/level word, `[8..0x40]` the log-magnitude envelope,
/// `[0x43]` a hangover counter). `dst` receives the block synthesis will use.
///
/// A repeat tag of `0xff` on **both** blocks means the frame carried no new
/// spectrum, and the four cases below decide what to play instead. Any other
/// tag is an ordinary frame: `dst` is just [`copy_params_l1_batch3`] of `cur`.
///
/// * `prev` is a silence frame (class 2): copy `cur` but keep `dst`'s own
///   voicing word and level, and — unless `cur` is silence too — halve the
///   whole envelope, i.e. fade by 6 dB per repeat.
/// * neither is silence: interpolate. The voicing mask already in `dst`
///   selects a merged fundamental ([`merge_voiced_amplitude`]), that sets `L`
///   ([`harmonic_count`]) and the envelope becomes the average of the two
///   neighbours resampled onto the new grid
///   ([`average_neighbour_envelopes`]).
/// * `cur` is silence: play `prev` instead unless `prev` has been repeating
///   for two frames or is itself a class-3 (tone) frame. Then either decay the
///   level by `0x5000` per frame, or — once it has run out — force every band
///   unvoiced (`mask &= 0x55555555`) and mute.
///
/// The fourth argument the original passes to `average_neighbour_envelopes` is
/// not read by it; the prototype simply has one parameter more than the body.
pub fn select_repeat_params(dst: &mut [i16], cur: &[i16], prev: &[i16]) {
    if cur[1] as u16 != 0xff || prev[1] as u16 != 0xff {
        copy_params_l1_batch3(dst, cur);
        return;
    }

    if prev[0] == 2 {
        // The voicing mask and level survive the copy.
        let mask = get32_al4b1(dst, 4);
        let lvl = dst[7];
        copy_params_l1_batch3(dst, cur);
        put32_al4b1(dst, 4, mask);
        dst[7] = lvl;
        if cur[0] != 2 {
            for w in dst.iter_mut().skip(8).take(0x38) {
                *w >>= 1;
            }
        }
        return;
    }

    if cur[0] != 2 {
        let w0 = merge_voiced_amplitude(get32_al4b1(dst, 4), cur, prev);
        dst[6] = w0 as i16;
        dst[2] = harmonic_count(w0 as i16);
        dst[1] = cur[1];
        average_neighbour_envelopes(dst, cur, prev);
        return;
    }

    let src = if prev[0x43] < 2 && prev[0] != 3 {
        prev
    } else {
        cur
    };
    copy_params_l1_batch3(dst, src);
    if dst[7] > 0 {
        // `((x << 16) - 0x4fff0001) >> 16` is `x - 0x5000`: the constant is
        // `0x5000 << 16` less `0xffff`, so the low half never borrows.
        dst[7] = dst[7].wrapping_sub(0x5000);
    } else {
        put32_al4b1(dst, 4, get32_al4b1(dst, 4) & 0x5555_5555);
        dst[7] = 0;
    }
}

/// `0x10317aa0` — one pass of voicing-driven spectral shaping over a harmonic
/// amplitude vector.
///
/// `amps` holds `n` harmonic amplitudes at block exponent `*exp`. The stage
/// runs four of the layer-2/3 routines in sequence:
///
/// 1. [`harmonic_energy_and_corr`] (`0x10317ff0`) measures the vector's energy
///    and its correlation with a cosine at the trial fundamental `omega`, and
///    leaves that cosine in a 56-word scratch buffer.
/// 2. [`track_level`] (`0x10318130`) folds the energy into the caller's leaky
///    background-level tracker `(level, level_exp)`. This happens
///    unconditionally — the tracker keeps running even on frames that are not
///    shaped.
/// 3. When the energy mantissa is positive, [`voicing_gains`] (`0x10317b80`)
///    splits it into voiced and unvoiced gains, [`shape_by_headroom`]
///    (`0x10317e20`) applies them against the cosine reference, and
///    [`scale_to_target_energy`] (`0x10317d70`) restores the *original*
///    energy, so the shaping redistributes energy across the spectrum without
///    changing its total.
///
/// The energy mantissa slot is reused as the unvoiced gain's exponent by
/// `voicing_gains`, so the target handed to step 3 is the value saved before
/// step 2 — reproducing that save is load-bearing.
///
/// Domain: `omega >= 0` and `omega·64·n < 512·65536` (the cosine index is not
/// masked), and the voiced/unvoiced difference must not cancel exactly, since
/// `voicing_gains` divides by it unguarded.
pub fn shape_spectrum_by_voicing(
    amps: &mut [i16],
    exp: &mut i16,
    level: &mut i16,
    level_exp: &mut i16,
    omega: i16,
    n: i16,
    cos_tbl: &[i16],
) {
    let mut cosines = [0i16; 56];
    let (mut e_m, mut e_e, mut c_m, mut c_e) = (0i16, 0i16, 0i16, 0i16);
    harmonic_energy_and_corr(
        &mut e_m,
        &mut e_e,
        &mut c_m,
        &mut c_e,
        &mut cosines,
        omega,
        amps,
        *exp,
        n as u16,
        cos_tbl,
    );

    let energy = e_m;
    let energy_exp = e_e;
    track_level(level, level_exp, e_m, e_e);

    if energy > 0 {
        let (a0, b0, c0, d0) = (e_m, e_e, c_m, c_e);
        let mut gain = 0i16;
        voicing_gains(
            &mut gain, &mut e_m, &mut c_m, &mut c_e, a0, b0, c0, d0, omega,
        );
        shape_by_headroom(amps, exp, n, gain, e_m, c_m, c_e, &cosines);
        scale_to_target_energy(amps, exp, n as u16, energy, energy_exp);
    }
}

/// `0x103181f0` — the unvoiced half of one synthesis subframe.
///
/// White noise is windowed out of the analysis ring, transformed, shaped so
/// that each unvoiced harmonic band carries the amplitude the parameters ask
/// for, transformed back, and overlap-added into the caller's accumulator:
///
/// 1. The transform length is the smallest power of two at least `n`, floored
///    at 256; `log2n` is 8 for every reachable `n` (SPEC §6 caps a half-frame
///    at 84 samples). The `0x50`-point `sqrt(i/80)` table at `0x10381c30` is
///    resampled onto `n + 1` points ([`resample_linear`]) to give the
///    synthesis window.
/// 2. A class-3 (tone) frame has no noise component at all: the ring and the
///    spectrum are simply zeroed and the exponent is 0.
/// 3. Otherwise [`window_overlap`] (`0x10318330`) draws `2n` windowed noise
///    samples from the ring — refilling it from the LCG at `st[0]` — the
///    forward real FFT runs, [`shape_noise_bands`] (`0x10318430`) imposes the
///    band amplitudes, and the inverse real FFT returns to the time domain.
/// 4. [`correlate_and_shift`] (`0x103187f0`) overlap-adds the result into
///    `acc` and rolls the `0x54`-word history at `st[0x55]`.
///
/// `st` is the noise state in words: `[0]` the LCG seed, `[1..0x55]` the
/// excitation ring, `[0x55..0xa9]` the overlap history, `[0xa9]` its block
/// exponent. `params` is the frame parameter block and `flags` the voicing tag
/// vector its word `0x40` points at.
///
/// The spectrum exponent lives in a 32-bit stack slot that only ever receives
/// a zero-extended 16-bit write, so it is carried as `u16` between the two
/// transforms.
///
/// Domain: `1 <= n <= 0x80` — a larger `n` selects a 512-point transform that
/// does not fit the original's 256-word stack buffer.
#[allow(clippy::too_many_arguments)]
pub fn synthesise_noise_subframe(
    acc: &mut [i32],
    n: i16,
    st: &mut [i16],
    params: &[i16],
    flags: &[i16],
    zero_head: i16,
    sqrt_tbl: &[i16],
) {
    let mut log2n: i32 = 8;
    if n > 0x80 {
        let mut k: i32 = 8;
        loop {
            log2n = k + 1;
            let probe = 1i32 << (k & 0x1f);
            k = log2n;
            if probe >= n as i32 {
                break;
            }
        }
    }
    // 16-bit `shl`: a count of 16 or more shifts the one clean out.
    let fftlen = if (log2n & 0x1f) >= 16 {
        0i16
    } else {
        (1u16 << (log2n & 0x1f)) as i16
    };

    let mut win = [0i16; 86];
    resample_linear(&mut win, n, sqrt_tbl, 0x50);

    let mut spec = [0i16; 256];
    let exp_out = if params[0] == 3 {
        fill16(&mut st[1..], 0, 0x54);
        fill16(&mut spec, 0, 0x100);
        0
    } else {
        let (seed, ring) = st.split_at_mut(1);
        window_overlap(&mut spec, &mut seed[0], ring, &win, n as u16, fftlen);
        let mut e = rfft_forward(&mut spec, 0, log2n as i16, 0) as u16 as i16;
        shape_noise_bands(&mut spec, &mut e, fftlen, params, flags, zero_head, n);
        // The original's exponent slot is a 32-bit stack cell that only ever
        // receives a zero-extended 16-bit write, hence the `as u16`. It makes
        // no observable difference — the transform adds one to the value and
        // truncates it to 16 bits immediately — but it is what the code does.
        inverse_real_fft(&mut spec, e as u16 as i32, log2n, 0) as u16 as i16
    };

    let (head, tail) = st.split_at_mut(0xa9);
    correlate_and_shift(
        acc,
        &spec,
        exp_out,
        &mut head[0x55..],
        &mut tail[0],
        &win,
        n,
    );
}

/// Word offsets, inside the 288-word coefficient-codebook blob at
/// `0x10381130`, of the four per-block codebooks: `0x10381130` (32 entries),
/// `0x10381230` (16), `0x103812b0` (16) and `0x10381330` (8), all of stride 4.
pub const COEF_BOOK_OFF: [usize; 4] = [0, 128, 192, 256];

/// Field widths, in bits, of those four codebooks.
pub const COEF_BOOK_BITS: [i16; 4] = [5, 4, 4, 3];

/// `0x10311250` — code (or decode) a frame's spectral amplitudes.
///
/// This is the amplitude half of the payload, and it is symmetric: `encode`
/// selects the direction and every stage below runs either way.
///
/// The log-magnitude vector is coded as a *prediction residual*. The previous
/// frame's envelope in `prev[8..]` is resampled onto this frame's `L`
/// harmonics ([`resample_lerp`], `0x10311910`) and scaled by `0x5333/2^16`
/// (0.65) with a DC correction taken from the resample's own sum
/// ([`scale_bias_vec`], `0x10312400`) — that is the prediction. `L` also
/// selects a **block split**: the word at `0x103800d0 + 2·(L−9)` packs four
/// nibbles, and block `k` is `nibble_k + 2` harmonics long. The four lengths
/// sum to exactly `L` for every `L` in `9..=56`.
///
/// On encode, [`prba_transform`] (`0x10312090`) subtracts the prediction,
/// takes each block's first two DCT coefficients, rotates the pairs by 45° and
/// runs an 8-point DCT over the result; the eight PRBA coefficients that come
/// out are coded as two vector-quantised fields — 3 words at 9 bits from the
/// codebook at `0x10380130` and 4 words at 7 bits from `0x10380d30` — with
/// the leading coefficient forced to zero, since it is the frame gain and is
/// coded elsewhere. Each block's remaining four coefficients are then coded
/// against its own stride-4 codebook, at 5, 4, 4 and 3 bits. A block only two
/// harmonics long has no coefficients of its own, and its bit field is skipped
/// (read and discarded, or emitted as zero) so the bit budget stays aligned.
///
/// [`expand_prba_to_log_mags`] (`0x103121d0`) then rebuilds the `L` log
/// magnitudes from the decoded PRBA vector, block coefficients and `gain`, and
/// [`avg_and_extend`] (`0x103111b0`) averages the prediction back in and
/// edge-extends the tail out to harmonic 55.
///
/// `cfg` is the bit-allocation block: word 7 is the field width available for
/// the 9-bit PRBA vector, word 8 for the 7-bit one, and words 9..13 for the
/// four blocks. A width narrower than the codebook decimates the codebook
/// rather than the index (see [`codebook_field`]).
///
/// `cur[8..]` is used as the transform's own scratch, so it must hold the full
/// `0x38` words even when `L` is small.
#[allow(clippy::too_many_arguments)]
pub fn code_amplitudes(
    prev: &[i16],
    cur: &mut [i16],
    gain: i16,
    cfg: &[i16],
    bits: &mut [i16],
    pos: &mut i32,
    encode: i16,
    block_lens: &[i16],
    prba_book9: &[i16],
    prba_book7: &[i16],
    coef_books: &[i16],
    cos_tbl: &[i16],
) {
    let l = cur[2] as u16;

    // The prediction: last frame's envelope resampled onto this frame's grid.
    let mut pred = [0i16; 56];
    let sum = resample_lerp(&mut pred, l, &prev[8..], prev[2]);
    scale_bias_vec(&mut pred, sum, l as i16, 0x5333);

    // Four block lengths from one packed word; the shift is a 16-bit `sar`.
    // The original's shift is a 16-bit `sar`, but only four bits are kept and
    // it runs at most three times, so a logical shift is indistinguishable —
    // mutating it does not fail. It is spelled arithmetically anyway.
    let mut packed = block_lens[(l as i16).wrapping_sub(9) as usize] as u16;
    let mut lens = [0i16; 4];
    for slot in lens.iter_mut() {
        *slot = ((packed & 0xf) as i16).wrapping_add(2);
        packed = ((packed as i16) >> 4) as u16;
    }

    let mut prba = [0i16; 8];
    if encode != 0 {
        // `scratch` and `a` are the same buffer in the original; the residual
        // is written element-wise so a copy of `a` is exactly equivalent.
        let mut a = [0i16; 0x38];
        a.copy_from_slice(&cur[8..8 + 0x38]);
        let lens_u: [u16; 4] = [
            lens[0] as u16,
            lens[1] as u16,
            lens[2] as u16,
            lens[3] as u16,
        ];
        prba_transform(&mut prba, &mut cur[8..], &pred, &a, l, &lens_u, cos_tbl);
    }
    // Coefficient 0 is the frame gain; it travels in its own field.
    prba[0] = 0;

    codebook_field(
        &mut prba[1..],
        3,
        cfg[7],
        prba_book9,
        9,
        3,
        bits,
        pos,
        encode,
    );
    codebook_field(
        &mut prba[4..],
        4,
        cfg[8],
        prba_book7,
        7,
        4,
        bits,
        pos,
        encode,
    );

    let mut base = 0usize;
    for k in 0..4usize {
        let coeffs = lens[k].wrapping_sub(2);
        let avail = cfg[9 + k];
        if coeffs < 1 {
            if encode == 0 {
                read_field_i16(bits, pos, avail);
            } else {
                let mut cur_pos = *pos as usize;
                emit_bits(&mut cur_pos, bits, avail, 0);
                *pos = cur_pos as i32;
            }
        } else {
            let nwords = if coeffs > 4 { 4 } else { coeffs } as u16;
            codebook_field(
                &mut cur[8 + base + 2..],
                nwords,
                avail,
                &coef_books[COEF_BOOK_OFF[k]..],
                COEF_BOOK_BITS[k],
                4,
                bits,
                pos,
                encode,
            );
        }
        base = base.wrapping_add(lens[k] as usize);
    }

    // `out` and `mags` are the same buffer: the final pass is element-wise, so
    // splitting them and stitching at `L` reproduces the aliased original.
    // (The two halves in fact agree past `L`, because the block lengths sum to
    // exactly `L` and nothing writes beyond it, so the stitch is defensive:
    // dropping it leaves the tests green.)
    let mut out = [0i16; 0x38];
    out.copy_from_slice(&cur[8..8 + 0x38]);
    let mut mags = out;
    expand_prba_to_log_mags(&mut out, l, &mut prba, &mut mags, &lens, gain, cos_tbl);
    for i in 0..0x38usize {
        cur[8 + i] = if i < l as usize { out[i] } else { mags[i] };
    }

    avg_and_extend(&mut cur[8..], &pred, l);
}

/// `0x10311ba0` — code (or decode) a frame's log-magnitude residual and its
/// gain.
///
/// Where [`code_amplitudes`] predicts from the previous frame's *envelope*,
/// this routine predicts from the previous frame's *unvoiced* envelope and
/// codes the difference as six blocks of harmonic-DCT coefficients plus a
/// scalar gain. It is the second, and larger, half of the amplitude payload.
///
/// The prediction is [`unvoiced_envelope`] (`0x10311ee0`): `prev`'s magnitudes
/// attenuated according to its voicing mask, resampled onto `cur`'s `L`
/// harmonics and widened as `L` grows.
///
/// `L` is split into six blocks as evenly as it allows — with `q = L/6` and
/// `r = L mod 6`, the first `6 − r` blocks are `q` long and the last `r` are
/// `q + 1`, so the lengths sum to exactly `L`. On encode the prediction is
/// subtracted ([`vsub_sat`]) and [`block_dft_dc`] (`0x10311870`) transforms
/// each block in place and collects its DC term; a six-point harmonic DFT over
/// those six DC terms yields the block-mean vector. On decode the same
/// geometry runs backwards through [`expand_rows_idct`] (`0x103118c0`).
///
/// The block-mean vector's own DC term is the frame gain. It is coded
/// separately against the 64-entry Q11 codebook at `0x10381370`
/// ([`quantize_codebook64`], six bits), while the other five means go through
/// [`code_higher_order_coeffs`] (`0x103114d0`) and the per-block coefficients
/// through [`quantize_coeffs_six_blocks`] (`0x10311620`). For a class-3 (tone)
/// frame the gain is measured instead as [`band_pair_mean`] of the residual —
/// once before coding to form the value, once after to remove it again.
///
/// Finally the prediction is added back with [`vec_add3_clamp`]; on decode the
/// envelope tail past `L` is filled by repeating the last live harmonic, and
/// the reconstructed gain is remembered in `prev[0x42]`.
///
/// `alloc` is the per-coefficient bit allocation: its first five nibbles
/// belong to the block means and the rest to the block coefficients.
#[allow(clippy::too_many_arguments)]
pub fn code_log_mag_residual(
    bits: &mut [i16],
    pos: &mut i32,
    prev: &mut [i16],
    cur: &mut [i16],
    alloc: &[u16],
    encode: i16,
    gain_book: &[i16],
    weight: &[i16],
    loading: &[i16],
    band_tbl: &[i16],
    cos_tbl: &[i16],
) {
    let l = cur[2];
    let q = div32(l as i32, 6) as u16;
    // `6·(q+1) − L` is `6 − (L mod 6)`: the count of short blocks, and equally
    // the index at which the long ones start.
    let split = (((q as u32).wrapping_mul(3).wrapping_add(3)).wrapping_mul(2))
        .wrapping_sub(l as u16 as u32) as u16;

    let mut pred = [0i16; 56];
    unvoiced_envelope(&mut pred, l, &prev[8..], prev[2], prev[6], prev[0x45]);

    let mut means = [0i16; 6];
    let mut gain: i16 = 0;
    if encode != 0 {
        vsub_sat(&mut cur[8..], &pred, l as u16);
        if cur[0] == 3 {
            gain = band_pair_mean(&cur[8..], cur[1], band_tbl) as i16;
        }
        let off = block_dft_dc(&mut means, &mut cur[8..], split, q as u32, cos_tbl);
        block_dft_dc(
            &mut means[split as usize..],
            &mut cur[8 + off as usize..],
            6u16.wrapping_sub(split),
            (q as u32).wrapping_add(1),
            cos_tbl,
        );
        harmonic_dft(&mut means, 6, 6, cos_tbl);
        if cur[0] != 3 {
            gain = means[0];
        }
    }
    means[0] = 0;

    let nbits: [u16; 5] = [alloc[0], alloc[1], alloc[2], alloc[3], alloc[4]];
    code_higher_order_coeffs(bits, pos, &nbits, &mut means, encode != 0);
    quantize_coeffs_six_blocks(bits, pos, &mut cur[8..], alloc, l, encode, weight, loading);

    harmonic_idct(&mut means, 6, 6, cos_tbl);
    let off = expand_rows_idct(&mut cur[8..], &means, split as i16, q as i16, cos_tbl);
    expand_rows_idct(
        &mut cur[8 + off..],
        &means[split as usize..],
        (6i32 - split as i32) as i16,
        (q as i16).wrapping_add(1),
        cos_tbl,
    );

    if encode == 0 {
        let idx = read_field_i16(bits, pos, 6) as i16;
        gain = gain_book[idx as usize];
    } else {
        if cur[0] == 3 {
            gain = gain.wrapping_sub(band_pair_mean(&cur[8..], cur[1], band_tbl) as i16);
        }
        let idx = quantize_codebook64(&mut gain);
        let mut cursor = *pos as usize;
        emit_bits(&mut cursor, bits, 6, idx);
        *pos = cursor as i32;
    }

    vec_add3_clamp(&mut cur[8..], &pred, l as u16, gain);

    if encode == 0 && l < 0x38 {
        // Overlapping forward copy: the last live harmonic is replicated.
        for i in 0..(0x38u16.wrapping_sub(l as u16)) as usize {
            cur[8 + l as usize + i] = cur[7 + l as usize + i];
        }
    }
    prev[0x42] = gain;
}

/// Read a little-endian `i32` stored as two state words.
#[inline]
pub fn rd32_al4b1(w: &[i16], i: usize) -> i32 {
    get32_al4b1(w, i) as i32
}

#[inline]
pub fn wr32_al4b1(w: &mut [i16], i: usize, v: i32) {
    put32_al4b1(w, i, v as u32);
}

/// `0x10318b00` — synthesise one voiced half-frame and cross-fade it into the
/// output accumulator.
///
/// This is the voiced counterpart of [`synthesise_noise_subframe`] and the
/// busiest function in the batch. Two harmonic banks are laid down — one
/// continuing the *previous* frame's parameters, one starting the *current*
/// frame's — into a 168-slot Q31 scratch buffer that is twice as long as a
/// half-frame, so a harmonic may run past the frame boundary and be picked up
/// by the next call.
///
/// In order:
///
/// 1. `1/n` is formed once in block floating point ([`div_norm`]) and becomes
///    the cross-fade slope for both [`ramp_mix`] and [`mac_ramp`].
/// 2. [`refine_pitch`] (`0x103193d0`) chooses between the two frames'
///    fundamentals — possibly halving or doubling one of them, which it reports
///    through the two flag words — and returns the running phase accumulator
///    advanced by this half-frame. Its high half is the number of harmonic
///    onsets that fall inside the frame.
/// 3. If the previous frame was voiced, that many harmonics are laid down with
///    [`synth_harmonic_bank`] (`0x10319330`) from the *previous* state, and the
///    result is faded **out** into `acc` by [`ramp_mix`].
/// 4. The scratch is cleared and the current frame's spectrum built:
///    [`build_voiced_spectrum`] (`0x10319140`) with tag 2 (and a starting
///    harmonic taken from the Q15 ratio `p6 / w0`) for the leading partial
///    harmonic, then, if the current frame is voiced, again with tags `0..1`
///    for the full bank. The phase state is advanced either by halving it —
///    when `refine_pitch` reported the "pitch halved" flag — or by
///    [`pitch_advance`] (`0x10319570`).
/// 5. [`mac_ramp`] fades that **in** to `acc`, the scratch's second half
///    becomes the new overlap state, and the leftover fractional harmonic
///    position is recomputed by [`sub_norm`] — or zeroed when neither frame was
///    voiced.
///
/// `state` in words: `[0..0xa8]` the 84-sample Q31 overlap buffer, `[0xa8]` the
/// 32-bit phase accumulator, `[0xaa]`/`[0xab]` the cached fractional harmonic
/// position and its exponent, `[0xad..0xe5]` the per-harmonic phase residual,
/// `[0xe5]` the spectrum's block exponent and `[0xe6..]` the 0x101-word
/// spectrum itself. `cur_tags` is the voicing tag vector `cur`'s word `0x40`
/// points at.
///
/// The word pair at `[0xaa]` is *copied into locals* before the first bank and
/// only written back at the very end, so the second `synth_harmonic_span` still
/// sees the value the frame started with — reproducing that is load-bearing.
///
/// Domain: `n` in `1..=84`; `cur[6]` and `prev[6]` non-zero (both
/// [`ratio_q15`] and [`pitch_advance`] divide by them unguarded).
#[allow(clippy::too_many_arguments)]
pub fn synthesise_voiced_subframe(
    acc: &mut [i32],
    n: i16,
    state: &mut [i16],
    cur: &[i16],
    cur_tags: &[i16],
    prev: &[i16],
    p6: i16,
    cos_tbl: &[i16],
) {
    let mut last_pos = state[0xaa];
    let mut last_exp = state[0xab];
    let prev_voiced = ((get32_al4b1(prev, 4) & 0x5555_5555) != 0) as i16;
    let cur_voiced = ((get32_al4b1(cur, 4) & 0x5555_5555) != 0) as i16;

    let nsig = n as i32;
    let nq = nsig << 16;
    let sh = norm_shift_al4b1(nq);
    let mut fade_exp = (0x0fu16.wrapping_sub(sh)) as i16;
    let fade_step =
        div_norm(0x4000_0000, 1, shl_hi16(nq, sh), fade_exp, &mut fade_exp) as u16 as i16;

    let mut pitch = rd32_al4b1(state, 0xa8);
    let mut pitch_a = pitch;
    let (mut ra, mut rb, mut f_half, mut f_dbl) = (0i16, 0i16, 0i16, 0i16);
    let refined = refine_pitch(
        pitch,
        &mut ra,
        &mut rb,
        &mut f_half,
        &mut f_dbl,
        prev[6],
        cur[6],
        prev_voiced,
        cur_voiced,
        n,
    );
    let mut count = (refined >> 16) as i16;
    let saved = refined;
    let mut acc_b = refined;
    let (mut ca, mut cb) = (ra, rb);

    // The scratch is one 168-slot Q31 buffer; the decompilation splits it in
    // two only because the first half is initialised by a copy and the second
    // by a fill.
    let mut buf = [0i32; 168];
    for (i, slot) in buf.iter_mut().take(84).enumerate() {
        *slot = rd32_al4b1(state, 2 * i);
    }

    if prev_voiced != 0 && ((refined >> 16) as i16) > 0 {
        let exp = state[0xe5] as u16 as i32;
        synth_harmonic_bank(
            &mut buf,
            &mut last_pos,
            &mut last_exp,
            prev[6],
            &state[0xe6..],
            exp,
            pitch,
            rb,
            ra,
            n,
            count as u16,
            cos_tbl,
        );
    }
    ramp_mix(acc, &buf, fade_step, fade_exp, n);
    buf = [0i32; 168];

    // `-(w0 << 5)`: the decompiler's `& 0x7ffffff == 0x4000000` test is a
    // saturation guard for a `neg` that an `i16` operand can never trigger.
    let w_prev = ((cur[6] as i32) << 5).wrapping_neg();
    let kmin = ratio_q15(p6, cur[6]) as u16 as i16;
    let voiced_cnt = {
        let (head, tail) = state.split_at_mut(0xe5);
        let (ex, spec) = tail.split_at_mut(1);
        build_voiced_spectrum(
            spec,
            &mut ex[0],
            ((w_prev << 0xb) >> 0x10) as i16,
            &head[0xad..],
            cur_tags,
            &cur[8..],
            cur[0x42] as u16 as i32,
            cur[2],
            kmin,
            2,
            cos_tbl,
        )
    };

    if voiced_cnt != 0 {
        let v = (((cur[7] as i32) << 16) >> 8).wrapping_add((nsig - 0x10) << 16);
        let sh2 = norm_shift_al4b1(v);
        let exp = state[0xe5] as u16 as i32;
        synth_harmonic_span(
            &mut buf,
            shl_hi16(v, sh2),
            (0x0fu16.wrapping_sub(sh2)) as i16,
            cur[6],
            &state[0xe6..],
            exp,
            cos_tbl,
        );
    }

    let mut out_acc = saved;
    if cur_voiced != 0 {
        if f_half == 0 {
            if f_dbl != 0 {
                cb = (((rb as i32) << 16) >> 0x11) as i16;
                let half = rd32_al4b1(state, 0xa8) >> 1;
                ca = (((ra as i32) << 16) >> 0x11) as i16;
                out_acc = (saved.wrapping_sub(rd32_al4b1(state, 0xa8)) >> 1).wrapping_add(half);
                wr32_al4b1(state, 0xa8, half);
                pitch = half;
            }
        } else {
            pitch_advance(&mut pitch_a, &mut acc_b, &mut ra, &mut rb, prev[6], state);
            pitch = pitch_a;
            ca = ra;
            cb = rb;
            out_acc = acc_b;
        }
        count = (out_acc >> 16) as i16;

        let w_cur = ((cur[6] as i32) << 5).wrapping_neg();
        {
            let (head, tail) = state.split_at_mut(0xe5);
            let (ex, spec) = tail.split_at_mut(1);
            build_voiced_spectrum(
                spec,
                &mut ex[0],
                ((w_cur << 0xb) >> 0x10) as i16,
                &head[0xad..],
                cur_tags,
                &cur[8..],
                cur[0x42] as u16 as i32,
                cur[2],
                0,
                1,
                cos_tbl,
            );
        }
        let exp = state[0xe5] as u16 as i32;
        synth_harmonic_span(
            &mut buf,
            state[0xaa],
            state[0xab],
            cur[6],
            &state[0xe6..],
            exp,
            cos_tbl,
        );
        if count > 0 {
            synth_harmonic_bank(
                &mut buf,
                &mut last_pos,
                &mut last_exp,
                cur[6],
                &state[0xe6..],
                exp,
                pitch,
                cb,
                ca,
                n,
                count as u16,
                cos_tbl,
            );
        }
    }

    wr32_al4b1(state, 0xa8, out_acc.wrapping_sub((count as i32) << 16));
    mac_ramp(acc, &buf, fade_step, fade_exp, n);
    for i in 0..84usize {
        wr32_al4b1(state, 2 * i, buf[n as usize + i]);
    }

    if prev_voiced == 0 && cur_voiced == 0 {
        put32_al4b1(state, 0xaa, 0);
        return;
    }
    let sh3 = norm_shift_al4b1(nq);
    let m = sub_norm(
        shl_hi16(nq, sh3),
        (0x0fu16.wrapping_sub(sh3)) as i16,
        last_pos,
        last_exp,
        &mut state[0xab],
    );
    state[0xaa] = m;
}

// ======================= from aL5b1 =======================

/// Six `i32` of DC-blocker / resonator state (byte `0x00`).
pub const S_RESON: usize = 0x00;

/// Voiced-synthesis state: overlap buffer, phase, spectrum (byte `0x18`).
pub const S_VOICED: usize = 0x0c;

/// The voiced state's per-harmonic phase-residual window (byte `0x172`),
/// which is also where the spectral slope is written.
pub const S_SLOPE: usize = 0xb9;

/// Previous-frame parameter block (byte `0x470`), 0x44 words.
pub const S_PREV_AL5B1: usize = 0x238;

/// Saved parameter block for mode-2 (tone) frames (byte `0x4f8`).
pub const S_TONE: usize = 0x27c;

/// Unvoiced-synthesis (noise) state (byte `0x648`), 0xaa words.
pub const S_NOISE: usize = 0x324;

/// "Use the pitch-message path" selector (byte `0x7a0`).
pub const S_PITCH_MSG: usize = 0x3d0;

/// Bypass flag; only bit 0 of the low byte is tested (byte `0x7ba`).
pub const S_BYPASS: usize = 0x3dd;

/// Tracked spectral level, the noise/voiced gain (byte `0x7be`).
pub const S_LEVEL: usize = 0x3df;

/// Voicing-shaper level and its exponent (bytes `0x7c0`, `0x7c2`).
pub const S_SHAPE_LEVEL: usize = 0x3e0;

pub const S_SHAPE_EXP: usize = 0x3e1;

/// The voicing-tag vector a parameter block points at.
pub const TAG_WORDS: usize = 0x38;

#[inline]
pub fn get32_al5b1(w: &[i16], i: usize) -> i32 {
    (w[i] as u16 as u32 | ((w[i + 1] as u16 as u32) << 16)) as i32
}

#[inline]
pub fn put32_al5b1(w: &mut [i16], i: usize, v: i32) {
    w[i] = v as u16 as i16;
    w[i + 1] = (v >> 16) as u16 as i16;
}

/// `0x1030c520` — synthesise one 10 ms half-frame of PCM from a decoded
/// parameter block.
///
/// This is the decoder's inner sequencer: it conditions the parameter block,
/// builds the two harmonic banks, filters and rescales the result. In order:
///
/// 1. **Conditioning** (skipped entirely when the bypass bit `state[0x7ba].0`
///    is set — the "parameters were already conditioned" path used when a
///    frame is repeated):
///    * mode 1 writes the spectral slope of the harmonic vector into the
///      voiced state's phase-residual window ([`spectral_slope`],
///      `0x103178a0`); every other mode zeroes those 0x38 words. The slope
///      shares storage with the residual, so a mode-1 frame deliberately
///      re-seeds it;
///    * [`expand_log_mags`] (`0x103176b0`) turns the coded log magnitudes into
///      linear amplitudes with a common block exponent in word `0x42`;
///    * mode 1 additionally runs [`shape_spectrum_by_voicing`] (`0x10317aa0`),
///      which folds the voicing decision into the amplitudes and updates the
///      shaper's leaky level at `state[0x7c0]`;
///    * mode 2 (a signalling tone) snapshots the block into the tone slot at
///      byte `0x4f8`.
/// 2. **Voicing tags.** Each block's word-`0x40` pointer is aimed at a 0x38-word
///    stack vector and [`voicing_setup`] (`0x1030cc90`) fills it — once for the
///    current block, once for the previous. Those vectors are locals here.
/// 3. **Parameter fixups**, three mutually exclusive cases:
///    * `params[1] == 0xff` is ordinary speech: on `half == 0` the tracked
///      level is advanced by [`smooth_level_if_voiced`] (`0x10317780`), then
///      [`weight_harmonics_by_freq_sq`] (`0x1030cb40`) applies the `f²` tilt
///      and [`refine_mags`] (`0x1030c700`) renormalises the vector;
///    * otherwise word 1 is a signalling code. With `state[0x7a0] == 1` the
///      previous block is treated as a pitch message
///      ([`retune_from_pitch_msg`], `0x1030ca70`); otherwise the block is
///      forced to a pure tone ([`tone_frame_fixup`], `0x1030c970`).
/// 4. **Synthesis** into an `n`-sample Q31 accumulator: the unvoiced bands via
///    [`synthesise_noise_subframe`] (`0x103181f0`), then the voiced harmonics
///    cross-faded in by [`synthesise_voiced_subframe`] (`0x10318b00`). Both are
///    driven by the same tracked level.
/// 5. **Output**: the DC blocker / resonator cascade [`resonator3`]
///    (`0x1030fca0`) runs *in place* over the accumulator, and
///    [`round_block_i32`] (`0x10317480`) rounds Q31 exponent `0x11` down to
///    `i16` exponent `0xf` — the fixed 1/16384 conversion of SPEC §7.2.
/// 6. The current block is promoted to "previous frame"
///    ([`copy_params_reset_phase`], `0x1030c8b0`) after its own tag pointer is
///    cleared.
///
/// **The previous block's voicing tags are dead.** The second `voicing_setup`
/// in step 2 fills a stack vector that nothing in this call — or anywhere
/// downstream of it — ever reads: the voiced synthesiser takes the *current*
/// block's tags, and the vector's storage dies with the frame. Deleting the
/// call is bit-exactly indistinguishable (confirmed by mutation). It is kept
/// here because the original does it.
///
/// **The previous block's tag pointer is left dangling.** Step 2 stores a stack
/// address into `state[0x4f0/0x4f2]` and nothing ever clears it, so after the
/// call those two words hold a pointer into a dead frame. Only the *current*
/// block's pair is zeroed. The port cannot reproduce a stack address and
/// therefore leaves those two words untouched; they are the one part of the
/// state block a differential test must exclude.
///
/// `out` receives `n` PCM samples. `state` must be at least
/// [`DEC_STATE_WORDS`] long; `params` is the 0x44-word block.
///
/// Domain: `1 <= n <= 84` — the original's accumulator is a 336-byte stack
/// buffer, and a longer half-frame overruns it. SPEC §6 clamps the request to
/// `[76, 84]`.
#[allow(clippy::too_many_arguments)]
pub fn dec_synth_subframe(
    params: &mut [i16],
    out: &mut [i16],
    n: i16,
    half: i16,
    state: &mut [i16],
    hilbert: &[i16],
    cos_tbl: &[i16],
    sqrt_tbl: &[i16],
) {
    if state[S_BYPASS] as u16 & 1 == 0 {
        if params[0] == 1 {
            spectral_slope(&mut state[S_SLOPE..], &params[8..], params[2], hilbert);
        } else {
            fill16(&mut state[S_SLOPE..], 0, 0x38);
        }

        {
            let count = params[2] as u16;
            let (head, tail) = params.split_at_mut(0x42);
            expand_log_mags(&mut head[8..], &mut tail[0], count);
        }

        if params[0] == 1 {
            let omega = params[6];
            let count = params[2];
            let (phead, ptail) = params.split_at_mut(0x42);
            let (shead, stail) = state.split_at_mut(S_SHAPE_EXP);
            shape_spectrum_by_voicing(
                &mut phead[8..],
                &mut ptail[0],
                &mut shead[S_SHAPE_LEVEL],
                &mut stail[0],
                omega,
                count,
                cos_tbl,
            );
        }

        if params[0] == 2 {
            copy_params_reset_phase(&mut state[S_TONE..], params);
        }
    }

    // Each block's word 0x40/0x41 is aimed at one of these; the pointer itself
    // is not modelled (see the note above).
    let mut tags = [0i16; TAG_WORDS];
    voicing_setup(params, &mut tags);
    // The original also runs `voicing_setup` over the *previous* block into a
    // second stack vector. Nothing — here or downstream — ever reads it (see
    // the note above), so the call and its 0x38-word buffer are dropped.

    if params[1] == 0xff {
        if half == 0 {
            let flags = get32_al5b1(params, 4) as u32;
            state[S_LEVEL] = smooth_level_if_voiced(state[S_LEVEL] as u16, params[6], flags) as i16;
        }
        weight_harmonics_by_freq_sq(params, state[S_LEVEL]);
        refine_mags(params);
    } else if state[S_PITCH_MSG] == 1 {
        let (msg_lo, msg_hi) = state.split_at(S_PREV_AL5B1);
        let _ = msg_lo;
        retune_from_pitch_msg(params, &mut tags, msg_hi);
    } else {
        tone_frame_fixup(params, cos_tbl);
    }

    // The original clears `2·n` words of a 336-byte stack buffer; a fresh
    // zeroed array is the same thing.
    let mut acc = [0i32; 84];
    let level = state[S_LEVEL];

    synthesise_noise_subframe(
        &mut acc,
        n,
        &mut state[S_NOISE..],
        params,
        &tags,
        level,
        sqrt_tbl,
    );

    {
        let (lo, hi) = state.split_at_mut(S_PREV_AL5B1);
        synthesise_voiced_subframe(
            &mut acc,
            n,
            &mut lo[S_VOICED..],
            params,
            &tags,
            hi,
            level,
            cos_tbl,
        );
    }

    // `resonator3` is called with `dst == src`; it consumes `src[i]` before
    // writing `dst[i]`, so a snapshot is equivalent.
    let mut reson = [0i32; 6];
    for (k, r) in reson.iter_mut().enumerate() {
        *r = get32_al5b1(state, S_RESON + 2 * k);
    }
    let src = acc;
    resonator3(&mut acc, &src, &mut reson, n);
    for (k, r) in reson.iter().enumerate() {
        put32_al5b1(state, S_RESON + 2 * k, *r);
    }

    round_block_i32(out, 0xf, &acc, 0x11, n as u16);

    params[0x40] = 0;
    params[0x41] = 0;
    copy_params_reset_phase(&mut state[S_PREV_AL5B1..], params);
}

/// `0x10311470` — code (or decode) a frame's gain and then its harmonic
/// amplitudes.
///
/// Two calls in sequence:
///
/// 1. [`quantise_gain`] (`0x1031fe40`) codes the frame energy against the
///    32-entry Q11 gain codebook at `0x10382160`, using `cfg[6]` as the field
///    width and `prev[0x42]` as the one-frame predictor.
/// 2. [`code_amplitudes`] (`0x10311250`) codes the amplitude vector *relative
///    to that gain*.
///
/// **The gain travels through the `cfg` argument slot.** The original passes
/// `&param_3` — the address of its own third argument — as the gain
/// out-parameter, having saved the real `cfg` pointer in a register first.
/// `quantise_gain` writes two bytes there, so the slot ends up holding
/// `hi16(cfg_ptr) : gain`, and `code_amplitudes` reads it as an `i16`. This is
/// exactly the aliasing artefact of SPEC §4.2, and it means the *initial*
/// contents of the gain cell are the low half of a pointer: reproducing the
/// call faithfully requires `cfg[6] >= 1`, so that the codebook field always
/// overwrites the cell before it is read.
///
/// `prev` is the previous frame's parameter block — only word `0x42`, the gain
/// predictor, is written. `cur` is the block being coded. `bits`/`pos` are the
/// payload bit array and cursor.
#[allow(clippy::too_many_arguments)]
pub fn code_gain_and_amplitudes(
    prev: &mut [i16],
    cur: &mut [i16],
    cfg: &[i16],
    bits: &mut [i16],
    pos: &mut i32,
    encode: i16,
    gain_book: &[i16],
    block_lens: &[i16],
    prba_book9: &[i16],
    prba_book7: &[i16],
    coef_books: &[i16],
    cos_tbl: &[i16],
) {
    // The stack cell that is both the gain out-parameter and (in the caller's
    // frame) the `cfg` argument slot.
    let mut gain = [0i16; 1];
    {
        let mode = cur[0];
        let nmags = cur[2];
        let (chead, ctail) = cur.split_at(8);
        let _ = chead;
        quantise_gain(
            &mut gain,
            mode,
            ctail,
            nmags,
            &mut prev[0x42],
            cfg[6],
            gain_book,
            bits,
            pos,
            encode,
        );
    }
    code_amplitudes(
        prev, cur, gain[0], cfg, bits, pos, encode, block_lens, prba_book9, prba_book7, coef_books,
        cos_tbl,
    );
}

/// `0x10311fa0` — code (or decode) the log-magnitude residual, bracketing it
/// with the voicing attenuation so the quantiser only ever sees the *voiced*
/// spectrum.
///
/// [`attenuate_unvoiced_bands`] (`0x10316f40`) subtracts a level-dependent
/// log-domain offset from every harmonic in an unvoiced band. It is applied
/// in place to `cur[8..]`:
///
/// * on **encode**, once with `negate = 0` before coding — the analysis
///   spectrum is pushed down in the unvoiced bands so the quantiser spends its
///   bits on the voiced ones;
/// * always once with `negate = 1` afterwards — undoing it on the encode side
///   and *applying* it on the decode side, which is how the decoder gets the
///   unvoiced bands back at the right level.
///
/// Between them, [`code_log_mag_residual`] (`0x10311ba0`) does the actual
/// coding.
///
/// The `encode` guard on the *first* attenuation is a decode-side no-op rather
/// than a correctness requirement: on decode `code_log_mag_residual` rebuilds
/// `cur[8..]` wholesale from the payload, so anything written there beforehand
/// is discarded. Removing the guard is bit-exactly indistinguishable; removing
/// the call is not (confirmed by mutation both ways).
///
/// On encode the routine also emits **one payload bit: the low bit of a frame
/// counter** kept in `prev[0x44]`, which is then incremented. That is the
/// alternating framing bit; it costs one bit per frame and carries no
/// spectral information.
///
/// `mask` is the packed voicing bit vector (one bit per band, LSB = lowest
/// band). `cur[6]` supplies the attenuation level and `cur[2]` the harmonic
/// count from which the band count is derived.
#[allow(clippy::too_many_arguments)]
pub fn code_log_mags_with_voicing(
    prev: &mut [i16],
    cur: &mut [i16],
    mask: i16,
    alloc: &[u16],
    bits: &mut [i16],
    pos: &mut i32,
    encode: i16,
    gain_book: &[i16],
    weight: &[i16],
    loading: &[i16],
    band_tbl: &[i16],
    cos_tbl: &[i16],
) {
    if encode != 0 {
        attenuate_in_place(cur, mask, 0);
    }

    code_log_mag_residual(
        bits, pos, prev, cur, alloc, encode, gain_book, weight, loading, band_tbl, cos_tbl,
    );

    attenuate_in_place(cur, mask, 1);

    if encode != 0 {
        let mut cursor = *pos as usize;
        emit_bits(&mut cursor, bits, 1, prev[0x44] & 1);
        *pos = cursor as i32;
        prev[0x44] = prev[0x44].wrapping_add(1);
    }
}

/// `attenuate_unvoiced_bands` with `out == inp`, as the original calls it. The
/// sweep runs downwards and reads `inp[i]` before writing `out[i]`, so a
/// snapshot of the source is equivalent to the aliased call.
pub fn attenuate_in_place(cur: &mut [i16], mask: i16, negate: i16) {
    let level = cur[6];
    let nbits = cur[2];
    let mut src = [0i16; TAG_WORDS];
    src.copy_from_slice(&cur[8..8 + TAG_WORDS]);
    attenuate_unvoiced_bands(&mut cur[8..], &src, mask, level, nbits, negate);
}

// ======================= from aL6b1 =======================

/// Frame class (1 = speech, 2 = silence, 3 = tone).
pub const P_CLASS: usize = 0;

/// Repeat tag; `0xff` means "this frame carried a new spectrum".
pub const P_TAG: usize = 1;

/// Harmonic count *L*.
pub const P_L: usize = 2;

/// Packed per-band voicing dibits, 32 bits over words 4 and 5.
pub const P_VOICE: usize = 4;

/// Fundamental `w0`.
pub const P_W0: usize = 6;

/// Pitch-refinement / level word.
pub const P_OFFS: usize = 7;

/// `prev`'s frame tracker (byte `0x88`).
pub const S_TRACK: usize = 0x44;

/// `prev`'s three-word tone-symbol history (byte `0x8c`).
pub const S_TONE_HIST: usize = 0x46;

/// `prev`'s status word (byte `0x90`).
pub const S_STATUS_AL6B1: usize = 0x48;

/// Voicing-decision energy threshold (byte `0x20`).
pub const C_THRESH: usize = 0x10;

/// Start of the voicing-decision config vector (byte `0x22`) — the same words
/// SPEC §2 records as the received FEC error counts.
pub const C_CFG: usize = 0x11;

/// Sub-mode / peak-suppression selector (byte `0x24`).
pub const C_SUBMODE: usize = 0x12;

/// Talkspurt frame count, also the peak-suppression frame kind (byte `0x30`).
pub const C_FRAMES: usize = 0x18;

#[inline]
pub fn get32_al6b1(w: &[i16], i: usize) -> u32 {
    w[i] as u16 as u32 | ((w[i + 1] as u16 as u32) << 16)
}

/// `shl ax, cl`: a **16-bit** left shift whose count is masked to five bits by
/// the hardware, so any count of 16 or more clears the word (SPEC §4.2).
#[inline]
pub fn shl16_al6b1(v: u16, count: i32) -> u16 {
    ((v as u32) << (count as u32 & 0x1f)) as u16
}

/// `0x1030ba70` — decode one AMBE half-rate frame's payload into a parameter
/// block.
///
/// The 49-bit payload opens with a **6-bit frame-type code**, read *without*
/// advancing the cursor (`pack_bits`, `0x10310250`), which sorts the frame into
/// four classes:
///
/// | code | class | meaning |
/// |---|---|---|
/// | `0x3f` | 3 | signalling tone |
/// | `0x3e` | 2 | silence / comfort noise |
/// | `< 0x3c` | 1 | ordinary speech |
/// | `0x3c`, `0x3d` | 0 | reserved — rejected, returns 4 |
///
/// **Bit allocation first.** `alloc` is the 13-word allocation descriptor of
/// SPEC §3. A zero word 0 means "not yet built" and gets the IMBE vector
/// ([`imbe_bit_alloc`], `0x1030c300`); otherwise it is rebuilt for the AMBE
/// sub-mode selected by `rate` through the seven-entry table at `0x1037fd78`,
/// as `0x31 - table[rate]`. That table holds `0x31, 0x30, 0x2f, 0x2e, 0x2d,
/// 0x2d, 0x2d`, i.e. sub-modes `0, 1, 2, 3, 4, 4, 4` — the five reduced
/// allocations of SPEC §3.1 with the last repeated. (The entry Ghidra prints as
/// a wide string `L"10/.---"` is this table.)
///
/// **Speech and silence (classes 1 and 2).** The frame is de-interleaved in
/// place by [`frame_bit_permute`] (`0x10312cb0`) — the AMBE-only two-pass
/// permutation — and then parsed in payload order:
///
/// 1. `b1` bits: the pitch lag;
/// 2. `b0` bits: an index into the pre-baked voicing patterns
///    ([`voicing_from_table`], `0x103125d0`), which writes the dibit word of
///    *both* blocks. Class 2 clears them instead;
/// 3. [`split_voiced_freq_code`] (`0x103132b0`) reinterprets the lag as a joint
///    frequency code across the two blocks' voiced bands. If it accepts, the
///    frame is a **dual tone**: both blocks are forced to `w0 = 0x1079`,
///    `L = 0x38`. If it rejects, the lag is an ordinary pitch and
///    [`fundamental_from_lag`] (`0x10310ff0`) turns it into `w0` and `L`;
/// 4. the gain and amplitude codec ([`code_gain_and_amplitudes`],
///    `0x10311470`) consumes the rest of the payload;
/// 5. [`suppress_spectral_peaks`] (`0x10314f30`) post-processes the envelope.
///
/// **Tone (class 3).** [`decode_tone_frame`] (`0x10310330`) does the whole
/// frame. A non-zero status is returned straight to the caller.
///
/// **Tail, common to every accepted frame.** [`select_repeat_params`]
/// (`0x10314980`) decides what `synth` should actually play, [`copy_params_batch2`]
/// (`0x10311a20`) promotes `cur` to `prev`, and the frame tracker
/// `prev[0x44]` is stamped with `-3` — unconditionally, including on the
/// rejected paths. The AMBE side never uses the tracker as a counter; it just
/// keeps it pinned so an IMBE frame arriving on the same stream starts from a
/// known value.
///
/// Returns 0 on success, 4 for a reserved frame-type code, or the tone
/// decoder's own status.
///
/// Domain: `alloc[0] != 0` (an AMBE vector) for classes 1–3 —
/// [`frame_bit_permute`]'s fixed 35-bit first pass and
/// [`decode_tone_frame`]'s eight-word repeat buffer both overrun on the 88-bit
/// IMBE vector (SPEC §3.1). `rate` indexes the sub-mode table and must be in
/// `0..=6`. `bits` must hold at least `alloc[2]` payload bits.
#[allow(clippy::too_many_arguments)]
pub fn decode_frame_ambe(
    cur: &mut [i16],
    synth: &mut [i16],
    prev: &mut [i16],
    bits: &mut [i16],
    rate: i16,
    ctrl: &[i16],
    alloc: &mut [i16; 13],
    allow_high: i16,
    submode_tbl: &[i16],
    voicing_tbl: &[u32],
    gain_book: &[i16],
    block_lens: &[i16],
    prba_book9: &[i16],
    prba_book7: &[i16],
    coef_books: &[i16],
    blob: &[i16],
) -> i16 {
    if alloc[0] == 0 {
        imbe_bit_alloc(alloc);
    } else {
        // The table entry is read *unsigned*; the subtraction is 32-bit and the
        // callee clamps the result into 0..=4.
        let sub = 0x31i32.wrapping_sub(submode_tbl[rate as i32 as usize] as u16 as i32);
        ambe_bit_alloc(alloc, sub as i16);
    }

    let mut pos: i32 = 0;
    // The frame-type code is a peek: `pack_bits` does not move the cursor.
    let code = pack_bits_i16(bits, 6) as i16;

    let class: i16 = if code as u16 == 0x3f {
        3
    } else if code as u16 == 0x3e {
        2
    } else {
        (code < 0x3c) as i16
    };

    if class <= 0 {
        prev[S_TRACK] = -3;
        return 4;
    }

    if class <= 2 {
        synth[P_CLASS] = class;
        cur[P_CLASS] = class;
        synth[P_TAG] = 0xff;
        cur[P_TAG] = 0xff;

        frame_bit_permute(bits, &alloc[..], 1);

        let mut lag = read_field_i16(bits, &mut pos, alloc[5]) as i16;
        let vidx = read_field_i16(bits, &mut pos, alloc[4]) as i16;

        {
            let mut m_synth = get32_al6b1(synth, P_VOICE);
            let mut m_cur = get32_al6b1(cur, P_VOICE);
            voicing_from_table(
                class,
                &mut m_synth,
                &mut m_cur,
                vidx,
                alloc[4] as u16 as i32,
                voicing_tbl,
            );
            put32_al4b1(synth, P_VOICE, m_synth);
            put32_al4b1(cur, P_VOICE, m_cur);
        }

        // The lag cell is also the joint frequency code; `split_voiced_freq_code`
        // may rewrite it, and whatever it leaves is what the pitch conversion
        // below sees.
        let split = split_voiced_freq_code(&mut lag, synth, cur, 0, 0, alloc[5] as u16 as u32, 0);

        if split != 0 {
            // The two writes into `synth` are dead: `select_repeat_params`
            // rewrites `synth[2]` and `synth[6]` on every one of its paths
            // before anything reads them, and nothing between here and there
            // touches those two words. Confirmed by mutation — perturbing
            // either is bit-exactly indistinguishable, while perturbing the
            // matching `cur` write is caught immediately. Kept because the
            // original does it.
            synth[P_W0] = 0x1079;
            cur[P_W0] = 0x1079;
            synth[P_L] = 0x38;
            cur[P_L] = 0x38;
        } else {
            let mode = cur[P_CLASS];
            let (lo, hi) = cur.split_at_mut(P_W0);
            fundamental_from_lag(mode, &mut hi[0], &mut lo[P_L], lag, alloc[5], allow_high);
        }

        code_gain_and_amplitudes(
            prev,
            cur,
            &alloc[..],
            bits,
            &mut pos,
            0,
            gain_book,
            block_lens,
            prba_book9,
            prba_book7,
            coef_books,
            blob,
        );
        suppress_spectral_peaks(cur, prev, ctrl[C_FRAMES], ctrl[C_SUBMODE]);
    } else {
        let status = decode_tone_frame(cur, synth, bits, &mut pos, &alloc[..], blob);
        if status != 0 {
            prev[S_TRACK] = -3;
            return status;
        }
    }

    select_repeat_params(synth, cur, prev);
    copy_params_batch2(prev, cur, 0);
    prev[S_TRACK] = -3;
    0
}

/// `0x1030bfd0` — decode one IMBE frame's payload into a parameter block.
///
/// The 88-bit payload has no frame-type code; instead its leading pitch field
/// doubles as the sanity check. The **top 6 bits** are peeked first
/// (`pack_bits`, `0x10310250`) and a value of `0x34` or more rejects the frame:
/// `prev[0x48]` becomes `0xffff` and the call returns 4 without touching either
/// parameter block.
///
/// Once accepted:
///
/// 1. the coarse 6-bit pitch is scaled back up to the field's full width
///    (`<< (b1 - 6)`, a 16-bit shift) and run through
///    [`set_pitch_params`] (`0x10310f80`) to get a provisional *L*;
/// 2. that *L* drives [`unpack_nibbles`] (`0x10311b00`), which reads the
///    per-harmonic bit allocation out of the packed table at `0x10381428`, and
///    then [`pack_imbe_frame`] (`0x10312fa0`) permutes the payload in place;
/// 3. the pitch is re-read at full width from the permuted payload and
///    `set_pitch_params` is applied again — this is the *L* everything
///    downstream uses;
/// 4. `triads_capped(L)` (`0x10312480`) bits of voicing field follow.
///
/// `prev[0x48]` records the frame as `pitch << 8 | voicing`, the voicing field
/// first right-shifted down to 8 bits when it is wider.
///
/// **The hangover tracker.** An all-zero voicing field means the frame claims
/// to be entirely unvoiced, which is also what a dead or mis-framed stream
/// looks like. `prev[0x44]` counts the evidence, saturating in `[-3, 10]`: a
/// pitch code in the "suspicious" set — low byte `& 0x7c == 0x18`, or
/// `0x80 <= pitch < 0xb8` — counts *down*, anything else counts *up*. While the
/// count is still positive (or the pitch is outside `0x80..0xb8`) the voicing
/// codebook is indexed at `0x38`, the "all bands unvoiced" entry; once it has
/// gone non-positive with a plausible pitch the index becomes `pitch - 0x80`,
/// i.e. the stream is treated as carrying a genuine 56-entry voicing pattern.
/// A non-zero voicing field bypasses the tracker entirely and indexes at
/// `field + 0x38`, folding down onto eight fixed frequency bands
/// ([`map_voicing_bits`], `0x10312620`) when there are more than eight.
///
/// The index is expanded by [`voicing_pattern`] (`0x103124b0`) against the
/// previous frame's dibits ([`gather_dibits`], `0x10317060`) and written into
/// both blocks by [`expand_dibits`] (`0x10312570`); two chained
/// [`voiced_offset`] calls (`0x10313e60`) then derive each block's pitch
/// refinement from the other's.
///
/// **`voicing_pattern`'s third argument is not lost.** Ghidra shows the call
/// with two arguments; the compiler left the third — the triad count — on the
/// stack from the `gather_dibits` call two instructions earlier and popped only
/// one word (`add esp,4`). The callee genuinely reads it, so a two-argument
/// transcription silently substitutes garbage.
///
/// The rest is the spectral codec ([`code_log_mags_with_voicing`],
/// `0x10311fa0`), the band voicing decision ([`voicing_decision`],
/// `0x103153a0`, which updates the band mask in place), peak suppression, the
/// signalling-tone detector ([`detect_tone_frame`], `0x103165e0`) over a
/// three-word symbol history that is shifted down one place at the very top of
/// the call, and the same `select_repeat_params` / `copy_params_batch2` tail as the
/// AMBE decoder. `cur[0]` is forced back to 1 after the tone detector, so a
/// detected tone changes `cur[1]`/`cur[3]` but never the frame class.
///
/// Returns 0, or 4 when the pitch field was out of range.
///
/// Domain: `alloc[0] == 0` (an IMBE vector) — [`pack_imbe_frame`] is IMBE-only.
/// `L` must be at least 24 for [`detect_tone_frame`], whose scratch beyond `L`
/// is left uninitialised by the original while the dual-tone table can name
/// harmonic 23.
#[allow(clippy::too_many_arguments)]
pub fn decode_frame_imbe(
    cur: &mut [i16],
    synth: &mut [i16],
    prev: &mut [i16],
    bits: &mut [i16],
    voffs: i16,
    ctrl: &[i16],
    alloc: &mut [i16; 13],
    mant: i16,
    exp: i16,
    nib_tbl: &[i16],
    gain_book: &[i16],
    weight: &[i16],
    loading: &[i16],
    codewords: &[u16],
    blob: &[i16],
) -> i16 {
    if alloc[0] == 0 {
        imbe_bit_alloc(alloc);
    } else {
        ambe_bit_alloc(alloc, 0);
    }

    let mut pos: i32 = 0;
    let coarse = pack_bits_i16(bits, 6);

    // `copy16_batch3(prev+0x8c, prev+0x8e, 2)` — a two-word overlapping forward copy
    // that ages the tone-symbol history by one frame.
    {
        let src = [prev[S_TONE_HIST + 1], prev[S_TONE_HIST + 2]];
        copy16_batch3(&mut prev[S_TONE_HIST..], &src, 2);
    }

    if (coarse as i16) >= 0x34 {
        prev[S_STATUS_AL6B1] = -1;
        return 4;
    }

    synth[P_CLASS] = 1;
    cur[P_CLASS] = 1;
    synth[P_TAG] = 0xff;
    cur[P_TAG] = 0xff;

    let widen = (alloc[5] as i32).wrapping_sub(6);
    set_pitch_params(cur, shl16_al6b1(coarse, widen) as i16);

    // The original's 112-byte stack local: the per-harmonic bit allocation.
    let mut hbits = [0u16; 56];
    unpack_nibbles(&mut hbits, cur[P_L], nib_tbl);
    let mut hbits_i = [0i16; 56];
    for (d, &v) in hbits_i.iter_mut().zip(hbits.iter()) {
        *d = v as i16;
    }
    pack_imbe_frame(bits, cur[P_L], &hbits_i, &alloc[..]);

    let pitch = { read_field_i16(bits, &mut pos, alloc[5]) as u32 };
    set_pitch_params(cur, pitch as i16);

    let triads = triads_capped(cur[P_L]);
    let vfield = { read_field_i16(bits, &mut pos, triads) };

    // The status word packs the pitch against the voicing field narrowed to a
    // byte. The narrowing is an arithmetic shift of the sign-extended field.
    let mut vnarrow = vfield as i16 as i32;
    if triads > 8 {
        vnarrow >>= (triads as i32 - 8) as u32 & 0x1f;
    }
    prev[S_STATUS_AL6B1] = ((pitch << 8) as i32).wrapping_add(vnarrow) as u16 as i16;

    let mut mask: u16 = vfield;

    // `idx` is the voicing-codebook index and `nb` the triad count that reaches
    // `voicing_pattern` as its third argument.
    let idx: i32;
    let mut nb = triads;
    if vfield == 0 {
        let track = prev[S_TRACK];
        let below = (pitch as u16).wrapping_sub(0x80);
        idx = if track > 0 || below > 0x37 {
            0x38
        } else {
            below as i32
        };

        let suspicious = ((pitch as u8) & 0x7c) == 0x18
            || ((pitch as u16 as i16) >= 0x80 && (pitch as u16 as i16) < 0xb8);
        if suspicious {
            if track > -3 {
                prev[S_TRACK] = track.wrapping_sub(1);
            }
        } else if track < 10 {
            prev[S_TRACK] = track.wrapping_add(1);
        }
    } else if triads > 8 {
        let folded = map_voicing_bits(vfield as i16, triads as i32, cur[P_L]);
        idx = (folded as u16 as i32).wrapping_add(0x38);
        nb = 8;
    } else {
        idx = (vfield as i32).wrapping_add(0x38);
    }

    let dibits = gather_dibits(get32_al6b1(prev, P_VOICE) as i32) as u16 as u32;
    let pattern = voicing_pattern(idx as i16, dibits, nb);

    {
        let mut m_synth = get32_al6b1(synth, P_VOICE);
        let mut m_cur = get32_al6b1(cur, P_VOICE);
        expand_dibits(&mut m_synth, &mut m_cur, pattern);
        put32_al4b1(synth, P_VOICE, m_synth);
        put32_al4b1(cur, P_VOICE, m_cur);
    }

    let synth_mask = get32_al6b1(synth, P_VOICE);
    voiced_offset(
        &mut synth[P_OFFS],
        synth_mask,
        prev[P_OFFS],
        get32_al6b1(prev, P_VOICE),
        voffs,
    );
    let synth_offs = synth[P_OFFS];
    let cur_mask = get32_al6b1(cur, P_VOICE);
    voiced_offset(&mut cur[P_OFFS], cur_mask, synth_offs, synth_mask, voffs);

    code_log_mags_with_voicing(
        prev,
        cur,
        mask as i16,
        &hbits,
        bits,
        &mut pos,
        0,
        gain_book,
        weight,
        loading,
        blob,
        blob,
    );

    voicing_decision(
        cur,
        synth,
        prev,
        &mut mask,
        ctrl[C_THRESH],
        ctrl[C_FRAMES],
        &ctrl[C_CFG..],
        mant,
        exp,
    );

    suppress_spectral_peaks(cur, prev, ctrl[C_FRAMES], ctrl[C_SUBMODE]);

    {
        let prev_code = prev[P_TAG];
        let mut hist = [0i16; 3];
        hist.copy_from_slice(&prev[S_TONE_HIST..S_TONE_HIST + 3]);
        detect_tone_frame(cur, prev_code, &hist, blob, codewords);
    }

    // UNVERIFIED: this write and the exact three-word window handed to
    // `detect_tone_frame` are both indistinguishable in the differential
    // sweep, because no payload found makes the detector accept — it needs a
    // decoded envelope whose loudest harmonic carries 99 % of the frame's
    // energy, which cannot be steered through the bit stream. `0x103165e0`
    // itself is covered directly by `diff_aL2b2`; what is untested here is
    // only that this decoder resets the class the detector may have set to 3.
    cur[P_CLASS] = 1;
    select_repeat_params(synth, cur, prev);
    copy_params_batch2(prev, cur, mask as i16);
    0
}

// ======================= from aL7b1 =======================

/// Every constant table the two frame decoders reach as a global, gathered so
/// the selector can forward whichever set the chosen codec needs.
///
/// The AMBE decoder uses `submode`, `voicing_patterns`, `gain32`, `block_lens`,
/// `prba9`, `prba7`, `coef_books` and `blob`; the IMBE decoder uses `nibbles`,
/// `gain64`, `hoc_weight`, `hoc_loading`, `codewords` and `blob`. Only `blob`
/// (the `0x1037f978` region: the cosine table and the tone/config data above
/// it) is shared.
pub struct FrameTables<'a> {
    /// `0x1037fd78` — 7 × AMBE sub-mode selector.
    pub submode: &'a [i16],
    /// `0x10381720` — pre-baked per-band voicing dibit patterns.
    pub voicing_patterns: &'a [u32],
    /// `0x10382160` — 32 × Q11 frame-gain codebook.
    pub gain32: &'a [i16],
    /// `0x103800d0` — per-block vector lengths.
    pub block_lens: &'a [i16],
    /// `0x10380130` — 9-bit PRBA codebook.
    pub prba9: &'a [i16],
    /// `0x10380d30` — 7-bit PRBA codebook.
    pub prba7: &'a [i16],
    /// `0x10381130` — higher-order-coefficient codebooks.
    pub coef_books: &'a [i16],
    /// `0x10381428` — packed per-harmonic quantiser bit allocation.
    pub nibbles: &'a [i16],
    /// `0x10381370` — 64 × Q11 block-gain codebook.
    pub gain64: &'a [i16],
    /// `0x103813fc` — higher-order-coefficient weights.
    pub hoc_weight: &'a [i16],
    /// `0x1038140e` — higher-order-coefficient loading factors.
    pub hoc_loading: &'a [i16],
    /// `0x10381a60` — voicing / tone codewords.
    pub codewords: &'a [u16],
    /// `0x1037f978` — cosine table and the tone/config blob above it.
    pub blob: &'a [i16],
}

/// `0x1030b870` — decode one frame's payload with whichever codec this stream
/// is configured for.
///
/// The whole body is one branch on **`alloc[0]`**, word 0 of the 13-word
/// bit-allocation descriptor of SPEC §3:
///
/// | `alloc[0]` | codec | decoder |
/// |---|---|---|
/// | 0 | IMBE, 88 payload bits | [`decode_frame_imbe`] (`0x1030bfd0`) |
/// | non-zero | AMBE half rate, 49 payload bits | [`decode_frame_ambe`] (`0x1030ba70`) |
///
/// That word is not an incidental flag: [`imbe_bit_alloc`]
/// writes `0` into it and [`ambe_bit_alloc`] writes `1`, so the
/// descriptor the stream's `init` built *is* the mode selector, and it stays
/// consistent because each decoder rebuilds the descriptor for its own codec on
/// every frame. This is the only place in the decode path where the mode of
/// SPEC §1 is read.
///
/// The two decoders do not take the same arguments, and the selector simply
/// carries the union: `rate` (the AMBE sub-mode index) is dropped on the IMBE
/// path, and `voffs`, `mant` and `exp` are dropped on the AMBE path. Nothing is
/// translated or defaulted — a dead argument is never touched.
///
/// Two further arguments the caller passes are dead in *both* directions: the
/// second and fourth are the literal `0x44` byte size of the `cur` and `synth`
/// blocks (see [`PARAM_BLOCK_BYTES`]), pushed by `0x1030d5d0` and never
/// forwarded. They are omitted here rather than modelled as ignored parameters;
/// the differential test passes garbage in those slots to show the omission is
/// sound.
///
/// `allow_high` is likewise forwarded to the AMBE decoder (where it gates
/// [`fundamental_from_lag`]'s high-fundamental range) and to the
/// IMBE decoder's last parameter, which its body never reads. The caller
/// derives it as `flags & 0x80`.
///
/// **`allow_high` is dead through this path**, and the port keeps it only
/// because the original passes it. Its one consumer is the `w0 > 0x6bca` clamp
/// in [`fundamental_from_log2`], and the AMBE lag field is
/// `b1 = 7` bits in every sub-mode (SPEC §3.1): the largest fundamental 128
/// lags can produce is `0x6657`, below the clamp. `selector_allow_high_is_dead`
/// proves both halves of that — the arithmetic over the whole lag domain, and
/// bit-exact agreement of real frames decoded under both values of the flag.
/// A mutation that replaces it with a constant therefore survives; the branch
/// is genuinely unreachable, not untested.
///
/// Returns the chosen decoder's status verbatim, zero-extended from 16 bits:
/// 0 on success, 4 for a rejected frame, or the tone decoder's own status.
#[allow(clippy::too_many_arguments)]
pub fn decode_frame(
    cur: &mut [i16],
    synth: &mut [i16],
    prev: &mut [i16],
    bits: &mut [i16],
    rate: i16,
    voffs: i16,
    ctrl: &[i16],
    alloc: &mut [i16; 13],
    mant: i16,
    exp: i16,
    allow_high: i16,
    t: &FrameTables<'_>,
) -> i16 {
    if alloc[0] == 0 {
        decode_frame_imbe(
            cur,
            synth,
            prev,
            bits,
            voffs,
            ctrl,
            alloc,
            mant,
            exp,
            t.nibbles,
            t.gain64,
            t.hoc_weight,
            t.hoc_loading,
            t.codewords,
            t.blob,
        )
    } else {
        decode_frame_ambe(
            cur,
            synth,
            prev,
            bits,
            rate,
            ctrl,
            alloc,
            allow_high,
            t.submode,
            t.voicing_patterns,
            t.gain32,
            t.block_lens,
            t.prba9,
            t.prba7,
            t.coef_books,
            t.blob,
        )
    }
}

// ======================= from aL8b1 =======================

/// `+0x470` — the **hold** parameter template: the block copied into `cur`
/// when a frame is lost but the loss is not yet judged unrecoverable.
pub const S_TEMPLATE_HOLD: usize = 0x470 / 2;

/// `+0x4f8` — the **give-up** parameter template, used when the decoder
/// raised bit 3, when three consecutive frames have failed
/// (`status & 0xe == 0xe`), or when the caller forces a mute with flag 8.
pub const S_TEMPLATE_GIVEUP: usize = 0x4f8 / 2;

/// `+0x580` — the previous frame's parameter block, 0x43 words, with the
/// saturating frame counter of [`bump_frame_counter`] at its word 0x43.
pub const S_PREV_AL8B1: usize = 0x580 / 2;

/// `+0x614` — the 52-byte control block of SPEC §2 (FEC error counts).
pub const S_CTRL: usize = 0x614 / 2;

/// `+0x646` — the last word of that control block, which is **not** advisory:
/// it is OR-ed into the caller's flags on entry, so the control block can
/// force mute, repeat or the cached-tone path from outside.
pub const S_CTRL_FLAGS: usize = 0x646 / 2;

/// `+0x79c`, `+0x79e` — the cached channel-unit length and band level replayed
/// when the caller sets flag `0x4000`.
pub const S_TONE_RATE: usize = 0x79c / 2;

/// `+0x79e` — see [`S_TONE_RATE`].
pub const S_TONE_LEVEL: usize = 0x79e / 2;

/// `+0x7a0` — the 13-word bit-allocation descriptor of SPEC §3; word 0 is the
/// codec selector read by [`decode_frame`].
pub const S_ALLOC: usize = 0x7a0 / 2;

/// `+0x7ba` — the **frame-history shift register**: shifted left one place on
/// every entry and OR-ed with 1 whenever the frame was *not* decoded, so its
/// low bits are a run of recent failures.
pub const S_STATUS_AL8B1: usize = 0x7ba / 2;

/// `+0x7bc` — a counter cleared on every successfully dispositioned frame.
pub const S_CLEARED: usize = 0x7bc / 2;

/// `+0x7c0`, `+0x7c2` — the (mantissa, exponent) pair the frame decoders take
/// as the running level estimate.
pub const S_MANT: usize = 0x7c0 / 2;

/// `+0x7c2` — see [`S_MANT`].
pub const S_EXP: usize = 0x7c2 / 2;

/// `+0x7c4` — the 0x12-word signalling-sequence tracker of
/// [`tone_seq_track`].
pub const S_TONE_SEQ: usize = 0x7c4 / 2;

/// Words of the control block handed to [`decode_frame`] (52 bytes).
pub const CTRL_WORDS: usize = 0x34 / 2;

/// Frame decoded; `cur` holds ordinary speech parameters.
pub const R_SPEECH: i16 = 2;

/// Frame decoded as silence.
pub const R_SILENCE: i16 = 0;

/// Frame is a **signalling tone**, not speech — either the tone decoder said
/// so or the signalling-sequence tracker rebuilt one. Returned as `0x8002`,
/// i.e. with the high bit set.
pub const R_TONE: i16 = 0x8002u16 as i16;

/// Frame was **not** decoded: `cur` was filled from one of the two templates
/// and the failure was recorded in the status shift register.
pub const R_REPEAT: i16 = 0x20;

/// `0x1030d5d0` — decode one 20 ms frame's parameters and decide what the
/// frame *is*.
///
/// This is the decoder's frame sequencer. `dec_synth` calls it once per frame,
/// on `half = 0`, with `cur` the fresh 68-byte parameter block and `synth` the
/// block at `state + 1000` that both half-frames are synthesised from
/// (SPEC §7.1). Everything else it needs — the previous frame, the control
/// block, the allocation vector, the level estimate and the tone tracker —
/// lives at fixed offsets in `state`, listed above.
///
/// The body is four stages:
///
/// 1. **Flags.** `flags |= state[S_CTRL_FLAGS]`, so the control block can
///    force a disposition, and the status shift register is shifted up one
///    place to make room for this frame's verdict.
///
/// 2. **Decode**, unless `flags & 0xc` asks for a forced repeat (`4`) or a
///    forced mute (`8`). [`decode_frame`] is given `cur`, `synth`, the
///    previous block, the payload and the allocation vector; a non-zero
///    status splits on **bit 3**: set is a hard failure that goes straight to
///    the give-up template, clear only disqualifies the frame from the
///    `cur[0]` disposition below. Only the second is reachable — see the
///    coverage notes.
///
/// 3. **Signalling sequence.** `cur[1]` — the tone symbol the decoder wrote,
///    `0xff` when the frame is not a tone — is pushed through
///    [`tone_seq_track`], which may *substitute* a scripted symbol while its
///    countdown runs. If it substituted (or if `flags & 0x4000` replays the
///    cached tone instead), the frame is rebuilt as a tone:
///    [`pitch_config_init`] lays the tone's band configuration into `cur` at
///    the mean of the two band levels the symbol names, and success — a return
///    of **0**, not a non-zero — is what makes this a tone frame, reported as
///    [`R_TONE`]. A rejected channel-unit length (return 4) simply falls
///    through to the ordinary disposition.
///
/// 4. **Disposition.** With neither forced flag and neither repeat nor mute
///    raised, `cur[0]` — the frame class the decoder wrote — selects the
///    return: 1 speech, 2 silence, 3 tone. Class 1 is refined once more: on an
///    **IMBE** stream (`alloc[0] == 0`) a `cur[1]` other than `0xff` means the
///    frame really is a tone after all, so it returns [`R_TONE`] rather than
///    [`R_SPEECH`]. Anything else — a forced flag, a raised repeat/mute, or an
///    unrecognised class — copies a template into `cur` and returns
///    [`R_REPEAT`]: the *give-up* template if the decoder raised bit 3, the
///    caller forced a mute, or the status register shows three consecutive
///    failures (`status & 0xe == 0xe`), otherwise the *hold* template. `cur` is
///    promoted into `synth` and the failure is recorded as `status |= 1`.
///
/// Both exits bump the previous block's frame counter; the decoded exits also
/// clear `state[S_CLEARED]`.
///
/// Coverage notes — every line is bit-exact against the reference, but four
/// details are transcribed rather than *distinguished* by any input, and two
/// of them are dead code in the original:
///
/// * **`hard_fail` is unreachable.** [`decode_frame`] returns only 0 or 4 on
///   every path of both codecs, so bit 3 of its status is never set and the
///   give-up template is never selected by *that* route (it is still reached
///   through `flags & 8` and through the run-length test, both covered).
///   Swapping the template on this branch survives mutation.
/// * **The `cur[1] != 0xff` refinement of frame class 1 is unreachable.**
///   Class 1 is only ever written by a decoder that also writes `cur[1] =
///   0xff`; the one thing that could change it, the IMBE signalling-tone
///   detector, cannot be made to accept through the bit stream (see the
///   matching note on `decode_frame_imbe`). The conjunction is therefore
///   verified *false* on both codecs — dropping the `cur[1]` test is caught,
///   because it makes IMBE return [`R_TONE`] — but dropping the `alloc[0]`
///   test survives, and the [`R_TONE`] outcome of this branch is unverified.
/// * **`flags & 0x80` is dead.** It is forwarded as [`decode_frame`]'s
///   `allow_high`, which that function's own tests show no input makes
///   observable.
/// * **The two band-count lookups are the same lookup** in the reachable
///   domain. [`tone_seq_track`] only ever substitutes symbol 20 or 21, whose
///   frame-size class is 0, and [`band_count_for_mode`] ignores its third
///   argument for class 0 — so the `1` and the `0` select the same word and
///   the mean is of a word with itself. The two calls are kept because the
///   original makes them; the distinction would only appear for a symbol in
///   `0x80..=0xa3`, which the tracker cannot produce.
///
/// `state` must be at least `0x3f4` words (`0x7e8` bytes). `flags` is the
/// 16-bit value the caller pushes — the original reads the argument slot as a
/// word and reuses the same slot for the return value.
#[allow(clippy::too_many_arguments)]
pub fn decode_frame_params(
    bits: &mut [i16],
    rate: i16,
    flags: u16,
    voffs: i16,
    cur: &mut [i16],
    synth: &mut [i16],
    state: &mut [i16],
    t: &FrameTables<'_>,
) -> i16 {
    // 1. flags, and the status shift register.
    let flags = flags | state[S_CTRL_FLAGS] as u16;
    state[S_STATUS_AL8B1] = state[S_STATUS_AL8B1].wrapping_mul(2);

    // 2. decode, unless a disposition was forced.
    // The decoder's status splits on bit 3: set is an immediate give-up,
    // clear only disqualifies the frame from the `cur[0]` disposition.
    let mut hard_fail = false;
    let mut soft_fail = false;
    if flags & 0xc == 0 {
        let mut alloc = [0i16; 13];
        alloc.copy_from_slice(&state[S_ALLOC..S_ALLOC + 13]);
        let mant = state[S_MANT];
        let exp = state[S_EXP];
        let status = {
            let (head, tail) = state.split_at_mut(S_CTRL);
            decode_frame(
                cur,
                synth,
                &mut head[S_PREV_AL8B1..],
                bits,
                rate,
                voffs,
                &tail[..CTRL_WORDS],
                &mut alloc,
                mant,
                exp,
                (flags & 0x80) as i16,
                t,
            ) as u16
        };
        state[S_ALLOC..S_ALLOC + 13].copy_from_slice(&alloc);
        if status != 0 {
            if status & 8 == 0 {
                soft_fail = true;
            } else {
                hard_fail = true;
            }
        }
    }

    // 3. signalling sequence. The original zero-extends `cur[1]` into a
    // 32-bit stack slot and passes its address; the tracker writes only the
    // low word back, so the value stays zero-extended.
    let mut sym = cur[1];
    let substituted = tone_seq_track(&mut sym, &mut state[S_TONE_SEQ..], 0);

    let tone_cfg: Option<(i16, i16)> = if flags & 0x4000 != 0 {
        Some((state[S_TONE_RATE], state[S_TONE_LEVEL]))
    } else if substituted != 0 {
        // Two band indices are named by the same symbol under two different
        // sub-selectors, and the tone is placed at the mean of their levels.
        let unit = sym;
        let cls = frame_size_class(unit) as i32;
        let hi = band_count_for_mode(cls, unit, 1, t.blob) as u16 as i16;
        let lo = band_count_for_mode(cls, unit, 0, t.blob) as u16 as i16;
        let a = cur[(8 + hi as i32) as usize] as i32;
        let b = cur[(8 + lo as i32) as usize] as i32;
        Some((unit, ((b + a) >> 1) as i16))
    } else {
        None
    };

    if let Some((unit, level)) = tone_cfg {
        if pitch_config_init(cur, unit, level, level, 0, t.blob) == 0 {
            copy_params_reset_phase(synth, cur);
            return finish_decoded(state, R_TONE);
        }
    }

    // 4. disposition.
    let template = if flags & 8 != 0 {
        S_TEMPLATE_GIVEUP
    } else {
        if flags & 4 == 0 {
            if hard_fail {
                state_copy(cur, synth, state, S_TEMPLATE_GIVEUP);
                return finish_undecoded(state);
            }
            if !soft_fail {
                match cur[0] {
                    1 => {
                        let r = if state[S_ALLOC] == 0 && cur[1] != 0xff {
                            R_TONE
                        } else {
                            R_SPEECH
                        };
                        return finish_decoded(state, r);
                    }
                    2 => return finish_decoded(state, R_SILENCE),
                    3 => return finish_decoded(state, R_TONE),
                    _ => {}
                }
            }
        }
        // The low byte of the status register: three consecutive undecoded
        // frames.
        if state[S_STATUS_AL8B1] as u16 as u8 & 0xe == 0xe {
            S_TEMPLATE_GIVEUP
        } else {
            S_TEMPLATE_HOLD
        }
    };

    state_copy(cur, synth, state, template);
    finish_undecoded(state)
}

/// Copy `state[at..]` into `cur`, promote `cur` into `synth`, and record the
/// failure in the status shift register.
pub fn state_copy(cur: &mut [i16], synth: &mut [i16], state: &mut [i16], at: usize) {
    copy_params_reset_phase(cur, &state[at..]);
    copy_params_reset_phase(synth, cur);
    state[S_STATUS_AL8B1] |= 1;
}

/// Tail of the two exits that produced usable parameters.
pub fn finish_decoded(state: &mut [i16], ret: i16) -> i16 {
    state[S_CLEARED] = 0;
    bump_frame_counter(&mut state[S_PREV_AL8B1..]);
    ret
}

/// Tail of the exit that filled `cur` from a template.
pub fn finish_undecoded(state: &mut [i16]) -> i16 {
    bump_frame_counter(&mut state[S_PREV_AL8B1..]);
    R_REPEAT
}

// ======================= from aL9b1 =======================

/// `+0x3e8` — the per-frame parameter block held in the state, in `i16` words.
/// This is the block half-frame 1 is synthesised from and the one
/// [`decode_frame_params`] treats as `cur`.
pub const S_FRAME_PARAMS: usize = 0x3e8 / 2;

/// `+0x170` — the half-frame length the callees read back, hard-wired to 0x50
/// (80 samples) on every entry regardless of what the caller asked for.
pub const S_NOMINAL_N: usize = 0x170 / 2;

/// A parameter block is 0x44 words; the original's scratch copy is the
/// 0x88-byte stack local `local_88`, and `fill16(local_88, 0, 0x44)` zeroes
/// exactly that.
pub const PARAM_WORDS: usize = 0x44;

/// The half-frame length is clamped to `[0x4c, 0x54]` = 76…84 samples
/// (SPEC §6) before anything else happens, and the clamped value — not the
/// requested one — is what both callees see.
pub fn clamp_half_frame(n: i16) -> i16 {
    if n > 0x54 {
        0x54
    } else if n < 0x4c {
        0x4c
    } else {
        n
    }
}

/// Every constant table `dec_synth` has to forward. `frame.blob` is the
/// `0x1037f978` region, which doubles as the 512-entry cosine table the
/// synthesis hub wants, so it is not repeated.
pub struct DecTables<'a> {
    /// The frame decoder's codebooks; see [`FrameTables`].
    pub frame: FrameTables<'a>,
    /// `0x10381c18` — 10 Hilbert / differentiating taps (SPEC §5).
    pub hilbert: &'a [i16],
    /// `0x10381c30` — 81 × `round(32767·sqrt(i/80))`.
    pub sqrt_tbl: &'a [i16],
}

/// `0x10309dd0` — `dec_synth(pcm, n, bits, rate, flags, half, state)`:
/// synthesise one 10 ms half-frame of PCM.
///
/// The public decode entry point. A 20 ms frame is two calls, `half = 0` then
/// `half = 1`; the payload is consumed by the first and ignored by the second.
///
/// 1. Clamp `n` to `[76, 84]` and publish the nominal half-frame length 0x50
///    at `+0x170`.
/// 2. `half == 0`: zero a scratch parameter block, run
///    [`decode_frame_params`] (`0x1030d5d0`) — which decodes the payload into
///    the state's block at `+0x3e8` and mirrors it into the scratch copy, or
///    substitutes a hold / give-up template when the frame is unusable — then
///    synthesise from the scratch copy. The return value is that function's
///    **frame disposition**: speech, silence, signalling tone or repeat.
/// 3. `half != 0`: synthesise straight from the state's block, and return 0.
///    Nothing is decoded, so a caller cannot learn a disposition from the
///    second half-frame.
///
/// `pcm` receives `n` samples; `bits` is one payload bit per word, MSB first
/// (SPEC §2). `rate` is the channel-unit / sub-mode selector, `flags` the
/// forcing word — bit 2 repeat, bit 3 mute, bit 14 replay the cached tone —
/// which is OR-ed with the control block's own flags word inside.
///
/// `state` must be at least [`DEC_STATE_WORDS`] long and zero-initialised by
/// `dec_init` (`0x10309cd0`) before the first call.
#[allow(clippy::too_many_arguments)]
pub fn dec_synth(
    pcm: &mut [i16],
    n: i16,
    bits: &mut [i16],
    rate: i16,
    flags: u16,
    half: i16,
    state: &mut [i16],
    t: &DecTables<'_>,
) -> i16 {
    let n = clamp_half_frame(n);
    state[S_NOMINAL_N] = 0x50;

    // The state-resident block, lifted out so it can be passed alongside the
    // state itself (see the module note on aliasing).
    let mut frame = [0i16; PARAM_WORDS];
    frame.copy_from_slice(&state[S_FRAME_PARAMS..S_FRAME_PARAMS + PARAM_WORDS]);

    if half == 0 {
        // `fill16(local_88, 0, 0x44)`.
        let mut scratch = [0i16; PARAM_WORDS];
        let ret = decode_frame_params(
            bits,
            rate,
            flags,
            n,
            &mut frame,
            &mut scratch,
            state,
            &t.frame,
        );
        state[S_FRAME_PARAMS..S_FRAME_PARAMS + PARAM_WORDS].copy_from_slice(&frame);
        dec_synth_subframe(
            &mut scratch,
            pcm,
            n,
            0,
            state,
            t.hilbert,
            t.frame.blob,
            t.sqrt_tbl,
        );
        ret
    } else {
        dec_synth_subframe(
            &mut frame,
            pcm,
            n,
            half,
            state,
            t.hilbert,
            t.frame.blob,
            t.sqrt_tbl,
        );
        state[S_FRAME_PARAMS..S_FRAME_PARAMS + PARAM_WORDS].copy_from_slice(&frame);
        0
    }
}
