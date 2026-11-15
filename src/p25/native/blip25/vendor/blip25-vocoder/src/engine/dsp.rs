//! Fixed-point DSP layer: primitives, block-floating-point arithmetic, FFTs and
//! the shared kernels the decoder is built from.
//!
//! Ported from the reverse-engineered original; each item keeps the address it
//! came from. Where two distinct helpers would otherwise share a name, each
//! carries a `_<group>` suffix naming the section below that it appears in.
#![allow(dead_code, clippy::all)]

// ======================= from prim =======================

/// `0x1030b2b0` — advance a 16-bit LCG in place.
///
/// Multiplier 0xAD, increment 0x3619, wrapping at 16 bits. Used to drive the
/// unvoiced (noise) branch of synthesis, which is why decoded noise is
/// reproducible frame-to-frame rather than truly random.
#[inline(never)]
pub fn rand_step(seed: &mut i16) {
    *seed = seed.wrapping_mul(0xADu16 as i16).wrapping_add(0x3619);
}

/// `0x1030b620` — fill `n` 16-bit words with `val`.
///
/// Returns the number of words written. The original returns a pointer one
/// past the end and treats `n` as signed: a non-positive count writes nothing.
#[inline(never)]
pub fn fill16(dst: &mut [i16], val: i16, n: i16) -> usize {
    if n <= 0 {
        return 0;
    }
    let n = n as usize;
    dst[..n].fill(val);
    n
}

/// `0x10310220` — one conditional-subtract division step (`SUBC`).
///
/// Computes `acc - (divisor << 15)`; on a non-negative result the accumulator
/// takes that difference shifted left with a 1 shifted in, otherwise it is
/// restored and shifted left with a 0. Sixteen iterations produce a 16-bit
/// quotient in the low half.
pub fn subc(divisor: i16, acc: &mut i32) {
    let diff = acc.wrapping_sub((divisor as i32) << 15);
    *acc = if diff < 0 {
        acc.wrapping_mul(2)
    } else {
        diff.wrapping_mul(2).wrapping_add(1)
    };
}

// ======================= from batch1 =======================

/// The library's `norm` (leading-redundant-sign-bit count), reproduced with its
/// degenerate cases intact.
///
/// For a non-zero `x` this is the left shift that moves the most significant
/// significant bit into bit 30, i.e. `0x1e - msb(|x|)`. Two inputs fall through
/// the original's `if (mag != 0)` guard with the search variable still at 31 and
/// so yield **-1**, not a sane shift: `x == 0` and `x == -1`. Callers of this
/// helper reproduce whichever of those the original happened to guard against,
/// so the raw value is returned rather than clamped.
#[inline(never)]
pub fn norm_shift_batch1(x: u32) -> i32 {
    let mag = if (x as i32) < 0 { !x } else { x };
    0x1e - top_bit_batch3(mag)
}

/// The library's exponent-align shift: left for a positive count, arithmetic
/// right for a negative one, and a full sign-fill once the count passes -30.
/// The shift amount is masked to 5 bits exactly as the x86 shift instructions
/// do, which matters because callers pass unclamped exponent differences.
pub fn shift_i32_batch1(v: i32, sh: i16) -> i32 {
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

/// Saturating 32-bit subtract, in the original's overflow-detect form
/// (`(~b ^ d) & (d ^ a) < 0`), saturating towards the sign of the minuend.
pub fn sat_sub_batch1(a: i32, b: i32) -> i32 {
    let d = a.wrapping_sub(b);
    if ((!b ^ d) & (d ^ a)) < 0 {
        if a < 0 {
            i32::MIN
        } else {
            i32::MAX
        }
    } else {
        d
    }
}

/// `0x10320530` — MSVC's `__allshr`: arithmetic right shift of a 64-bit value.
///
/// Register-argument helper (`edx:eax` value, `cl` count), not a cdecl
/// function. A count of 64 or more yields the sign fill; a count of 32..63
/// shifts the high half down into the low half.
pub fn allshr(v: i64, count: u8) -> i64 {
    let hi = (v >> 32) as i32;
    let lo = v as u32;
    let fill = hi >> 31;
    if count >= 0x40 {
        return ((fill as u32 as u64) << 32 | fill as u32 as u64) as i64;
    }
    if count < 0x20 {
        let n = count & 0x1f;
        let new_lo = if n == 0 {
            lo
        } else {
            lo >> n | (hi as u32) << (32 - n)
        };
        let new_hi = hi >> n;
        return ((new_hi as u32 as u64) << 32 | new_lo as u64) as i64;
    }
    let new_lo = hi >> (count & 0x1f);
    ((fill as u32 as u64) << 32 | new_lo as u32 as u64) as i64
}

/// `0x1031ecc0` — population count over the low `n` bits of a word.
///
/// The word is shifted arithmetically, so for a negative `x` and an `n` past
/// bit 31 the sign bit keeps contributing. A non-positive `n` counts nothing.
pub fn popcount_low(x: u32, n: u16) -> i32 {
    let mut acc = 0i32;
    if (n as i16) > 0 {
        let mut x = x as i32;
        for _ in 0..n {
            acc += x & 1;
            x >>= 1;
        }
    }
    acc
}

/// `0x1030cdb0` — classify a channel-unit length into one of five frame
/// classes.
///
/// The comparisons are unsigned-after-bias, so exactly these windows map:
/// `5..=0x7a` → 0, `0x80..=0x8f` → 1, `0x90..=0x9f` → 2, `0xa0..=0xa3` → 3,
/// everything else → 4. Returned in `al`; Ghidra types it `char`.
#[inline(never)]
pub fn frame_size_class(x: i16) -> i8 {
    let u = x as u16;
    if u.wrapping_sub(5) < 0x76 {
        return 0;
    }
    if u.wrapping_sub(0x80) < 0x10 {
        return 1;
    }
    if u.wrapping_sub(0x90) < 0x10 {
        return 2;
    }
    3 + (u.wrapping_sub(0xa0) > 3) as i8
}

/// `0x10316350` — sum `n` words into a 32-bit accumulator, normalise it, and
/// return the sum's block exponent.
///
/// `*out` receives the top 16 bits of the left-normalised sum and the return
/// value is the incoming exponent adjusted by the normalisation, plus 16 for
/// the mantissa's new radix point. `param_3` in the decompilation is the
/// incoming exponent (an `int`, not a `short`).
///
/// A sum of exactly -1 hits [`norm_shift_batch1`]'s degenerate case: the shift becomes
/// 0xffff, the mantissa is shifted by 31 and the exponent returned is 17 below
/// the input. That is the original's behaviour, reproduced deliberately.
pub fn sum_norm(out: &mut i16, v: &[i16], exp: i32, n: u16) -> i32 {
    let mut sum: u32 = 0;
    if (n as i16) > 0 {
        for &x in &v[..n as usize] {
            sum = sum.wrapping_add(x as i32 as u32);
        }
    }
    let sh: u32 = if sum == 0 {
        0
    } else {
        norm_shift_batch1(sum) as u32 & 0xffff
    };
    *out = ((sum << (sh & 0x1f)) >> 16) as i16;
    exp.wrapping_sub(sh as i32).wrapping_add(0x10)
}

/// `0x10312570` — expand 32 packed dibits into two words of doubled dibits.
///
/// Each 2-bit symbol of `bits` is written twice into the output, so a 16-bit
/// run of dibits becomes a 32-bit word in which every symbol occupies four
/// bits. The low 16 bits of `bits` produce `out0`, the high 16 `out1`, and both
/// are emitted most-significant-symbol-first. This is how a packed voicing
/// decision is stretched to the 4-bit-per-band form the rest of the DSP uses.
#[inline(never)]
pub fn expand_dibits(out0: &mut u32, out1: &mut u32, bits: u32) {
    let mut src = bits as i32;
    for w in 0..2 {
        let mut acc = 0u32;
        let mut b = 30i32;
        for _ in 0..8 {
            let d = (src & 3) as u32;
            src >>= 2;
            acc |= (d << (b & 31)) | (d << ((b - 2) & 31));
            b -= 4;
        }
        if w == 0 {
            *out0 = acc;
        } else {
            *out1 = acc;
        }
    }
}

/// `0x10313e60` — pick a pitch-refinement offset from a pair of voicing masks.
///
/// `0xaaaaaaaa` selects the odd bit of every dibit, i.e. the "voiced" flag of
/// each band. With no voiced band in `mask_a` the result is zero; if `a` is
/// positive and `mask_b` also has a voiced band the result is `a - 256*b`;
/// otherwise it is the low byte of `b` scaled by 128. All three results are
/// truncated to 16 bits.
#[inline(never)]
pub fn voiced_offset(out: &mut i16, mask_a: u32, a: i16, mask_b: u32, b: i16) {
    if mask_a & 0xaaaa_aaaa == 0 {
        *out = 0;
        return;
    }
    if a > 0 && (mask_b & 0xaaaa_aaaa) != 0 {
        *out = a.wrapping_sub(b.wrapping_mul(256));
        return;
    }
    *out = (b as i8 as i16).wrapping_mul(128);
}

/// `0x10317520` — block exponent of a 16-bit vector, from its peak magnitude.
///
/// Returns the *negated* normalisation shift of `max|v[i]| << 16`, so the
/// result is 0 for a vector already using the full range and -14 for one whose
/// peak is 1. `-0x8000` is folded to `0x7fff` before the maximum is taken, so
/// the peak is never itself an overflowing negation. An all-zero vector gives 0.
pub fn norm_exp_absmax(v: &[i16], n: u16) -> i32 {
    let mut m = 0i16;
    if (n as i16) > 0 {
        for &x in &v[..n as usize] {
            let a = if x < 0 {
                if x == i16::MIN {
                    0x7fff
                } else {
                    -x
                }
            } else {
                x
            };
            if m < a {
                m = a;
            }
        }
    }
    let x = (m as i32) << 16;
    if x != 0 {
        -((norm_shift_batch1(x as u32) as u32 & 0xffff) as i32)
    } else {
        0
    }
}

/// `0x10319f80` — block exponent of the largest value in `[from, to)` whose
/// tag matches.
///
/// Walks two parallel arrays; only entries where `tags[i] == tag` are
/// considered, and because the running maximum starts at zero only positive
/// values can win. Returns the negated normalisation shift of that maximum in
/// the high half of a word, or 0 if nothing qualified.
pub fn norm_exp_max_tagged(vals: &[i16], tags: &[i16], from: i16, to: i16, tag: i16) -> i32 {
    let mut best: u32 = 0;
    if from < to {
        let count = to.wrapping_sub(from) as u16 as usize;
        for i in (from as usize)..(from as usize + count) {
            let scaled = ((vals[i] as i32) << 16) as u32;
            if tags[i] == tag && (best as i32) < (scaled as i32) {
                best = scaled;
            }
        }
    }
    if best != 0 {
        -((norm_shift_batch1(best) as u32 & 0xffff) as i32)
    } else {
        0
    }
}

/// `0x103124b0` — expand a voicing-decision codebook index into a band mask.
///
/// For `idx < 0x38` the answer is a straight table lookup of one of 56
/// pre-baked patterns. Above that, `idx - 0x38` is treated as `nbits` raw bits
/// and each bit is replicated across the eight output dibits according to the
/// allocation table at `0x10381920` (`nbits` is clamped to 8).
///
/// The return packs two halves: the high 16 bits are the freshly built pattern,
/// the low 16 a combination of it with the previous frame's pattern
/// (`prev & 0x5555`) — agreement is kept and disagreement is smeared one dibit
/// left, which is the "repeat the previous voicing decision" rule.
///
/// Domain: `nbits <= 0` or `3 <= nbits`; `1` and `2` index before the table in
/// the original and are not reachable from the codec.
pub fn voicing_pattern(idx: i16, prev: u32, nbits: i16) -> u32 {
    if idx <= 0x37 {
        return tables::VOICING_LUT[idx as usize];
    }
    let nb = if nbits > 8 { 8 } else { nbits };
    let mut pat: u16 = 0;
    let mut s: i16 = 0;
    while s < nb {
        debug_assert!(
            nb >= 3,
            "voicing_pattern: nbits 1..2 reads before the table"
        );
        let rep = tables::BIT_ALLOC[(nb as usize - 3) * 8 + s as usize];
        if rep > 0 {
            let bit = (((idx.wrapping_sub(0x38) as i32) >> (s as u32 & 0x1f)) & 1) as u16;
            for _ in 0..rep as u16 {
                pat = (((pat as i16) >> 2) as u16) | (bit << 14);
            }
        }
        s += 1;
    }
    let x = (prev & 0x5555) ^ pat as u32;
    (x.wrapping_mul(4) & x) | (prev & 0x5555 & pat as u32) | ((pat as i16 as i32) << 16) as u32
}

/// `0x1031a060` — one complex butterfly pass over a Hermitian-symmetric
/// spectrum.
///
/// `a` walks forwards and `b` walks backwards through the same buffer, two
/// words (one complex value) at a time, with the twiddle index advancing by
/// `stride` per butterfly. The real part combines as `(x/2 ± rot)` and the
/// imaginary part as `(rot ± y/2)` — note the reversed sign on `b`'s imaginary
/// output, which is what makes this the conjugate-symmetric merge stage of a
/// real FFT rather than a plain radix-2 butterfly.
///
/// Twiddles come from the 512-entry cosine table; the sine pointer is that
/// table offset by 128 words. Indices used are `k*stride` for `k = 1..=n`, so
/// `n * stride` must stay below 384.
pub fn fft_butterfly_pass(buf: &mut [i16], a_off: usize, b_off: usize, stride: i16, n: u16) {
    if (n as i16) <= 0 {
        return;
    }
    let (mut pa, mut pb) = (a_off as isize, b_off as isize);
    let mut ti = 0isize;
    for _ in 0..n {
        ti += stride as isize;
        let c = tables::COS512[ti as usize] as i32;
        let s = tables::COS512[128 + ti as usize] as i32;

        // Read order is the original's: `b`'s real part is re-read after `a`'s
        // real part has been written, so an aliased `pa == pb` would see the
        // new value. The codec never aliases them, but the order is kept.
        let half = (buf[pa as usize] as i32).wrapping_shl(16) >> 1;
        let rot = c
            .wrapping_mul(buf[pb as usize] as i32)
            .wrapping_sub(s.wrapping_mul(buf[(pb + 1) as usize] as i32));
        buf[pa as usize] = ((half.wrapping_add(rot) as u32) >> 16) as i16;
        let br = buf[pb as usize] as i32;
        buf[pb as usize] = ((half.wrapping_sub(rot) as u32) >> 16) as i16;

        let rot2 = s
            .wrapping_mul(br)
            .wrapping_add(c.wrapping_mul(buf[(pb + 1) as usize] as i32));
        let half2 = (buf[(pa + 1) as usize] as i32).wrapping_shl(16) >> 1;
        buf[(pa + 1) as usize] = ((half2.wrapping_add(rot2) as u32) >> 16) as i16;
        buf[(pb + 1) as usize] = ((rot2.wrapping_sub(half2) as u32) >> 16) as i16;

        pa += 2;
        pb -= 2;
    }
}

/// `0x1030a7a0` — `2^x`, split into an integer exponent and a Q31 mantissa.
///
/// The integer part of `x` (its high 16 bits) plus one is written to `exp`, and
/// the fractional part drives a degree-5 minimax polynomial whose coefficients
/// are the powers of `ln 2` in Q16: `0x58b9 = 22713 ≈ ln2 * 2^15`,
/// `0x3d7f ≈ (ln2)^2/2`, and so on. The result is `2^frac / 2` in Q31, hence
/// the `+ 0x40000000` and the exponent bias of one.
///
/// Every stage truncates its accumulator (`& 0xffffe000`, `& 0xffffc000`,
/// `& 0xffff8000`), shedding progressively fewer bits as the term shrinks; that
/// truncation is part of the result, not a rounding artefact.
pub fn pow2_frac(x: i32, exp: &mut i16) -> i32 {
    *exp = ((x >> 16) as i16).wrapping_add(1);
    let t = ((x as u32 & 0xffff) >> 1) as i32;

    let step = |acc: i32, mul: i32, sh: u32, mask: u32, add: i32| -> i16 {
        ((((acc.wrapping_mul(mul).wrapping_add(0x8000)) >> sh) & mask as i32).wrapping_add(add)
            >> 16) as i16
    };

    let r = step(t, 0xaec4, 3, 0xffff_e000, 0x4ecb_0000u32 as i32);
    let r = step(
        (r as i32).wrapping_mul(t).wrapping_mul(2),
        1,
        3,
        0xffff_e000,
        0x38d6_0000,
    );
    let r = step(
        (r as i32).wrapping_mul(t).wrapping_mul(2),
        1,
        2,
        0xffff_c000,
        0x3d7f_8000,
    );
    let r = step(
        (r as i32).wrapping_mul(t).wrapping_mul(2),
        1,
        1,
        0xffff_8000,
        0x58b9_0000,
    );
    ((((r as i32)
        .wrapping_mul(t)
        .wrapping_mul(2)
        .wrapping_add(0x8000))
        >> 1)
        & 0xffff_8000u32 as i32)
        .wrapping_add(0x4000_0000)
}

/// `0x1030d1b0` — impose a decaying skirt around a spectral peak.
///
/// Starting from `level` at the bins either side of `peak` and falling by `dec`
/// per bin outwards, each of the 56 bins is raised to the skirt if it sits
/// below it. The peak bin itself is skipped. The running level is held in Q16
/// with a saturating subtract, truncated back to 16 bits each step, and floored
/// at -0x77ff — the same floor that is applied to the incoming `level`.
///
/// Domain: `0 <= peak < 56`; the original indexes unguarded.
pub fn spread_peak(arr: &mut [i16], peak: i16, level: i16, dec: i16) {
    const FLOOR: i16 = -0x77ff;
    let start = if level < FLOOR { FLOOR } else { level };

    let decay = |cur: i16| -> i16 {
        let d = sat_sub_batch1((cur as i32).wrapping_shl(16), (dec as i32).wrapping_shl(16));
        let next = (d >> 16) as i16;
        if next < FLOOR {
            FLOOR
        } else {
            next
        }
    };

    let mut cur = start;
    let mut i = peak - 1;
    while i >= 0 {
        if arr[i as usize] < cur {
            arr[i as usize] = cur;
        }
        cur = decay(cur);
        i -= 1;
    }

    let mut cur = start;
    let first = peak.wrapping_add(1);
    if first < 0x38 {
        for i in first..0x38 {
            if arr[i as usize] < cur {
                arr[i as usize] = cur;
            }
            cur = decay(cur);
        }
    }
}

/// `0x1030b490` — subtract two mantissa/exponent numbers and renormalise:
/// `(m2@e2) - (m1@e1)`.
///
/// Both operands are shifted into a common exponent one above the larger of the
/// two (the guard bit that makes the subtraction overflow-free), the difference
/// is renormalised, and the resulting exponent is written to `exp`. Shortcuts:
/// if `m1` is zero or is more than 31 binary points below `m2`, the answer is
/// `m2` unchanged; if `m2` is negligible instead, the answer is `-m1`
/// (saturated at `0x7fff`); an exactly cancelling difference returns mantissa
/// and exponent both zero.
pub fn sub_norm(m1: i16, e1: i16, m2: i16, e2: i16, exp: &mut i16) -> i16 {
    addsub_norm(m1, e1, m2, e2, exp, true)
}

/// The body shared by [`add_norm`] and [`sub_norm`].
///
/// The two differ only in the sign with which the aligned first operand enters
/// the sum, and in the pass-through case, where the subtract negates (with the
/// `i16::MIN` saturation the original has). Everything else — the two guards,
/// the alignment to `max(exp) + 1`, the renormalise and the zero-mantissa
/// tail — is common. The alignment shift is spelled `shift32_al2b2` here;
/// `sub_norm` used `shift_i32_batch1`, which is the same function.
fn addsub_norm(a: i16, a_exp: i16, b: i16, b_exp: i16, out_exp: &mut i16, sub: bool) -> i16 {
    if a != 0 && (b_exp as i32) - (a_exp as i32) < 0x20 {
        if b == 0 || (a_exp as i32) - (b_exp as i32) > 0x1f {
            *out_exp = a_exp;
            return if !sub {
                a
            } else if a == i16::MIN {
                0x7fff
            } else {
                -a
            };
        }
        let m = (if a_exp <= b_exp { b_exp } else { a_exp }).wrapping_add(1);
        let va = shift32_al2b2((a as i32).wrapping_mul(0x10000), a_exp.wrapping_sub(m)) as u32;
        let vb = shift32_al2b2((b as i32).wrapping_mul(0x10000), b_exp.wrapping_sub(m)) as u32;
        let sum = if sub {
            vb.wrapping_sub(va)
        } else {
            vb.wrapping_add(va)
        };
        let nrm = if sum == 0 {
            0i16
        } else {
            norm_exp_batch3(sum) as i16
        };
        let r = ((sum << (nrm as u32 & 0x1f)) >> 16) as i16;
        if r == 0 {
            *out_exp = 0;
            return 0;
        }
        *out_exp = m.wrapping_sub(nrm);
        return r;
    }
    if b == 0 {
        *out_exp = 0;
        return 0;
    }
    *out_exp = b_exp;
    b
}

/// `0x1031a870` — in-place bit-reversal permutation for an FFT of `size`
/// points.
///
/// The swap lists are stored as *incremental* index deltas: two running
/// pointers start at the base of the buffer, each entry advances them by its
/// delta pair, and the two words at the resulting positions are exchanged —
/// then both pointers step one further and a second exchange is made. Data is
/// interleaved `(real, imag)`, which is why swaps come in adjacent pairs.
///
/// Sizes 8, 16, 32 and 64 select their own table; anything else falls to the
/// 128-point table, which needs a 256-word buffer.
pub fn bit_reverse(buf: &mut [i16], size: i16) {
    let tbl: &[i16] = match size {
        8 => &tables::BITREV8,
        16 => &tables::BITREV16,
        32 => &tables::BITREV32,
        64 => &tables::BITREV64,
        _ => &tables::BITREV128,
    };
    if tbl[0] <= 0 {
        return;
    }
    let (mut pa, mut pb) = (0isize, 0isize);
    for k in 0..tbl[0] as u16 as usize {
        let ia = pa + tbl[1 + 2 * k] as isize;
        pa = ia + 1;
        let ib = pb + tbl[2 + 2 * k] as isize;
        pb = ib + 1;
        buf.swap(ia as usize, ib as usize);
        buf.swap(pa as usize, pb as usize);
    }
}

// ======================= from batch2 =======================

/// Shift a 32-bit accumulator by a signed amount, TI-C5x style: a positive
/// count shifts left (modulo 32), a negative count shifts arithmetic-right,
/// and a count below -30 collapses to a full sign fill (`>> 31`). Used by the
/// block-exponent alignment routines below.
pub fn shr_or_shl(x: i32, sh: i16) -> u32 {
    shift_i32_batch1(x, sh) as u32
}

/// `0x10317060` — gather eight dibits from a word, reversing their order.
///
/// Takes bits `[1:0]`, `[5:4]`, `[9:8]`, … (every fourth bit-pair, low to
/// high) of `x` and packs them into a 16-bit result at positions 14, 12, 10,
/// … 0 — i.e. it reverses the eight 2-bit groups. The reference returns this in the
/// low half of `eax`; the high half is a constant `0xffff` left over from the
/// loop counter (`-2`), which is reproduced here so the full return matches.
pub fn gather_dibits(mut x: i32) -> u32 {
    let mut shift: i32 = 14;
    let mut out: u16 = 0;
    for _ in 0..8 {
        let lo = (x as u16) & 3;
        x >>= 4;
        out |= lo << (shift & 0x1f);
        shift -= 2;
    }
    // eax = CONCAT22((short)((uint)iVar3 >> 16), out); iVar3 ends at -2.
    0xffff_0000u32 | out as u32
}

/// `0x10311e00` — in-place saturating vector subtract, `a[i] -= b[i]`.
///
/// Difference is clamped to `[-0x7fff, 0x7fff]` (note the asymmetric negative
/// limit). `n <= 0` is a no-op.
pub fn vsub_sat(a: &mut [i16], b: &[i16], n: u16) {
    if (n as i16) > 0 {
        for i in 0..n as usize {
            let mut v = a[i] as i32 - b[i] as i32;
            if v >= 0x8000 {
                v = 0x7fff;
            } else if v < -0x7fff {
                v = -0x7fff;
            }
            a[i] = v as i16;
        }
    }
}

/// `0x10310150` — normalisation shift count of a 64-bit value `{hi:lo}`.
///
/// Returns how far the (magnitude of the) 64-bit number must be shifted to
/// normalise it, biased by -9, scanning down from bit 39. `{0:0}` returns 0.
pub fn norm64(lo: u32, hi: u32) -> i32 {
    if lo == 0 && hi == 0 {
        return 0;
    }
    let mut p1 = lo;
    let mut p2 = hi;
    if (p2 as i32) < 1 && (p2 as i32) < 0 {
        p1 = !p1;
        p2 = !p2;
    }
    let mut u2: u32 = 0;
    let mut u1: u32 = 0x80;
    let mut i: i32 = 0;
    loop {
        if (u2 & p1) != 0 || (u1 & p2) != 0 {
            break;
        }
        u2 = (u2 >> 1) | (u1 << 0x1f);
        i += 1;
        u1 = ((u1 as i32) >> 1) as u32;
        if !((i as i16) < 0x28) {
            break;
        }
    }
    i - 9
}

/// `0x1030b3d0` — shift `x` to a Q15-ish scale (by `sh - 15`) and round to a
/// 16-bit magnitude, preserving sign and saturating.
///
/// Positive net shift is a left shift; negative is arithmetic-right (full sign
/// fill below -30). The result keeps only the top 16 bits (`& 0xffff0000`),
/// with `0x80000000` saturating to `0x7fffffff`.
pub fn shift_round16_batch2(x: i32, sh: i16) -> u32 {
    let s = sh - 0xf;
    let mut u = shr_or_shl(x, s);
    if (u as i32) < 0 {
        if u == 0x8000_0000 {
            u = 0x7fff_ffff;
        } else {
            u = u.wrapping_neg();
        }
        if (u & 0xffff_0000) == 0x8000_0000 {
            return 0x7fff_ffff;
        }
        return (u & 0xffff_0000).wrapping_neg();
    }
    u & 0xffff_0000
}

/// `0x1030a5a0` — interpolated lookup into the 512-entry cosine table at
/// `0x1037f978`, indexed by a Q15 argument.
///
/// The argument is scaled by `2^10`, its magnitude taken, and the top bits
/// index the table while the low 16 bits linearly interpolate to the next
/// entry (wrapping mod 512). Returns the Q15 result in the low 16 bits of an
/// `i32`. Pass `table` = the 512 `i16` at `0x1037f978`.
pub fn cos_lookup(x: i16, table: &[i16]) -> i32 {
    let mut u = ((x as i32) << 16) >> 6;
    if u < 0 {
        if (u as u32) == 0x8000_0000 {
            u = 0x7fff_ffff;
        } else {
            u = u.wrapping_neg();
        }
    }
    let uu = u as u32;
    let idx0 = (uu >> 16) as i16 as usize;
    let idx1 = ((((u >> 16) as u32) & 0xffff).wrapping_add(1) & 0x1ff) as usize;
    let frac = (uu & 0xffff) as i32;
    (table[idx0] as i32 * (0x10000 - frac) + table[idx1] as i32 * frac) >> 16
}

/// `0x1031a140` — radix-2 butterfly, variant A.
///
/// Reads a forward pair `p1[i1], p1[i1+1]` and a backward pair
/// `p2[i2], p2[i2+1]` (i2 stepping down by 2 each iteration) and writes their
/// half-sum/half-difference combinations back. `n <= 0` is a no-op.
pub fn butterfly_a(p1: &mut [i16], p2: &mut [i16], p2_start: usize, n: u16) {
    butterfly(p1, p2, p2_start, n, false);
}

/// The body shared by [`butterfly_a`] and [`butterfly_b`]: `flip` swaps the
/// operand order of the two imaginary outputs, which is the only difference
/// between the variants. The subtraction is done before the `>> 1`, so the two
/// orders are not interchangeable and the flag cannot be folded into a negate.
fn butterfly(p1: &mut [i16], p2: &mut [i16], p2_start: usize, n: u16, flip: bool) {
    if (n as i16) > 0 {
        let mut i1 = 0usize;
        let mut i2 = p2_start;
        for _ in 0..n {
            let s1 = p2[i2] as i32;
            let s2 = p2[i2 + 1] as i32;
            let s3 = p1[i1] as i32;
            let s4 = p1[i1 + 1] as i32;
            let (d1, d2) = if flip {
                (s2 - s4, s3 - s1)
            } else {
                (s4 - s2, s1 - s3)
            };
            p1[i1] = ((s1 + s3) >> 1) as i16;
            p1[i1 + 1] = (d1 >> 1) as i16;
            p2[i2] = ((s4 + s2) >> 1) as i16;
            p2[i2 + 1] = (d2 >> 1) as i16;
            i2 = i2.wrapping_sub(2);
            i1 += 2;
        }
    }
}

/// `0x10319ff0` — radix-2 butterfly, variant B.
///
/// As [`butterfly_a`] but with a different sign pattern on the imaginary
/// outputs (`p1[i1+1] = (s2-s4)/2`, `p2[i2+1] = (s3-s1)/2`).
pub fn butterfly_b(p1: &mut [i16], p2: &mut [i16], p2_start: usize, n: u16) {
    butterfly(p1, p2, p2_start, n, true);
}

/// `0x10311a20` — copy an MBE parameter block (voiced case) or clear it.
///
/// When `src[0] == 1` the header words, harmonic count `src[2]`, the first
/// `src[2]` harmonic amplitudes, and a tail padded up to 0x38 with the last
/// amplitude are all copied; `p3` lands in `dst[0x45]`. Otherwise the block is
/// cleared. `dst`/`src` must be ≥0x46 words.
pub fn copy_params_batch2(dst: &mut [i16], src: &[i16], p3: i16) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[3] = src[3];
    if src[0] == 1 {
        dst[2] = src[2];
        dst[6] = src[6];
        dst[4] = src[4];
        dst[5] = src[5];
        dst[0x45] = p3;
        dst[7] = src[7];
        let cnt = src[2];
        if cnt > 0 {
            for i in 0..cnt as usize {
                dst[8 + i] = src[8 + i];
            }
        }
        let uv1 = src[2] as u16;
        if (uv1 as i16) < 0x38 {
            let fillval = src[src[2] as usize + 7];
            let count = 0x38u32.wrapping_sub(uv1 as u32) & 0xffff;
            let mut j = uv1 as usize + 8;
            for _ in 0..count {
                dst[j] = fillval;
                j += 1;
            }
        }
        dst[0x43] = 0;
    } else {
        dst[4] = 0;
        dst[5] = 0;
        dst[0x45] = 0;
        dst[7] = 0;
        dst[0x43] = 0;
    }
}

/// `0x103193d0` — pitch/energy candidate refinement.
///
/// Depending on the voicing flags `p8`/`p9`, picks between two pitch estimates
/// `p6`/`p7` (or a doubled/halved variant, flagging the choice in the three
/// output words), forms an energy term, adds the bias `param_1`, and returns it
/// block-normalised. `out_r` receives the chosen pitch; `out_f3/4/5` the flags.
#[inline(never)]
pub fn refine_pitch(
    param_1: i32,
    out_r: &mut i16,
    out_f3: &mut i16,
    out_f4: &mut i16,
    out_f5: &mut i16,
    p6: i16,
    p7: i16,
    p8: i16,
    p9: i16,
    p10: i16,
) -> i32 {
    *out_r = p6;
    *out_f3 = 0;
    *out_f4 = 0;
    *out_f5 = 0;

    let mut param_6 = p6;
    let mut iv5: i32;
    let mut uv2: u32 = 0;
    let mut go_final = false;

    if p8 == 0 {
        if p9 == 0 {
            iv5 = 0;
            go_final = true;
        } else {
            *out_r = p7;
            param_6 = p7;
            iv5 = param_6 as i32 * p10 as i32;
        }
    } else if p9 == 0 {
        iv5 = param_6 as i32 * p10 as i32;
    } else {
        let mut param_7 = p7;
        let ip7 = p7 as i32;
        let ip6 = p6 as i32;
        if param_6 < ((ip7.wrapping_mul(0x9998) as u32 >> 16) as i16)
            && ((ip7.wrapping_mul(0x6666) as u32 >> 16) as i16) < param_6
            && param_6 < 0x4000
        {
            param_7 = p7 >> 1;
            *out_f4 = 1;
        } else if ((ip6.wrapping_mul(0x6aaa) as u32 >> 16) as i16) < param_7
            && param_7 < ((ip6.wrapping_mul(0xa000u32 as i32) as u32 >> 16) as i16)
            && param_7 < 0x4000
        {
            param_7 = (ip7.wrapping_shl(0x11) as u32 >> 16) as i16;
            *out_f5 = 1;
        }
        *out_f3 = ((param_7 as i32)
            .wrapping_mul(0x10000)
            .wrapping_add(ip6.wrapping_mul(-0x10000)) as u32
            >> 16) as i16;
        iv5 = ((((((param_7 as i32) << 16) >> 1).wrapping_add(ip6.wrapping_mul(0x10000) >> 1))
            as u32
            >> 16) as i16 as i32)
            .wrapping_mul(p10 as i32);
    }

    if !go_final {
        uv2 = (iv5.wrapping_mul(2) >> 4).wrapping_add(param_1) as u32;
        if uv2 == 0 {
            iv5 = 0;
            uv2 = 0xf;
        } else {
            let nrm = (norm_shift_batch1(uv2) as u32) & 0xffff;
            iv5 = uv2.wrapping_shl(nrm & 0x1f) as i32;
            uv2 = 0xfu32.wrapping_sub(nrm) & 0xffff;
        }
    }

    let diff = uv2.wrapping_sub(0xf);
    let s4 = diff as i16;
    // The original masks the count out of `diff`'s low byte; that agrees with
    // `s4` modulo 32, so this is just the library's signed shift.
    shift_i32_batch1(iv5, s4)
}

/// `0x10311b00` — unpack `p2-1` nibbles from the packed table at `0x10381428`.
///
/// A triangular index `((p2-9)*(p2+6))/2` selects a starting 16-bit word and
/// nibble within it; the routine then streams 4-bit values low-to-high,
/// reloading successive table words. This is a bit-allocation / quantiser
/// table read. `table` = the `i16` at `0x10381428`; `out` receives `p2-1`
/// nibble values. Valid for `p2` in the range the codec actually uses (≥ ~9,
/// so the index and output length are non-negative).
pub fn unpack_nibbles(out: &mut [u16], p2: i16, table: &[i16]) {
    let mut s8 = p2 - 2;
    let uv5 = ((p2 as i32 - 9).wrapping_mul(p2 as i32 + 6) >> 1) as u32;
    let uv3_init = ((uv5 as u16 as i16) >> 2) as i16;
    let mut iv2 = (uv5 & 0xffff).wrapping_add((uv3_init as u32).wrapping_mul(0xffff_fffc)) as i32;
    let mut iv7 = uv3_init as i32 + 1;
    let shift0 = (((iv2 as u8) & 7) as u32) << 2;
    let mut uv3: u16 = ((table[uv3_init as usize] as i32) >> shift0) as u16;
    let mut oi = 0usize;
    while s8 >= 0 {
        let uv4 = uv3 & 0xf;
        uv3 = ((uv3 as i16) >> 4) as u16;
        iv2 += 1;
        out[oi] = uv4;
        oi += 1;
        s8 -= 1;
        if (iv2 as i16) > 3 {
            let s6 = iv7 as i16;
            iv7 += 1;
            uv3 = table[s6 as usize] as u16;
            iv2 = 0;
        }
    }
}

// ======================= from batch3 =======================

/// Index of the highest set bit, or 31 for zero — the original's open-coded
/// `for (i = 31; v >> i == 0; i--)` loop, which never runs for `v == 0`.
pub fn top_bit_batch3(v: u32) -> i32 {
    // `leading_zeros` is 32 for zero, and `& 31` folds that back to 0 so the
    // result is 31 — exactly what the open-coded loop leaves behind when it
    // never runs. Identical for every input, and one instruction.
    31 - ((v.leading_zeros() & 31) as i32)
}

/// Block exponent: how far left `x` must be shifted for its magnitude to fill
/// bit 30. Zero and −1 both come out as −1, matching the original.
pub fn norm_exp_batch3(x: u32) -> i32 {
    norm_shift_batch1(x)
}

/// The library's variable shift. Negative `s` shifts right arithmetically and
/// saturates the *count* at 31; positive `s` shifts left and simply drops the
/// bits that fall off the top.
pub fn shift32_al2b2(v: i32, s: i16) -> i32 {
    shift_i32_batch1(v, s)
}

/// 32-bit saturating add, exactly as the reference open-codes it: on overflow the
/// result is `0x7fffffff` when the first operand was non-negative and
/// `0x80000000` when it was negative.
pub fn sat_add32(a: i32, b: i32) -> i32 {
    let r = a.wrapping_add(b);
    if ((r ^ a) & (r ^ b)) < 0 {
        (0x7fffffffu32).wrapping_add((a < 0) as u32) as i32
    } else {
        r
    }
}

/// High half of a 32-bit accumulator as a Q15 word.
pub fn high16_batch3(x: i32) -> i16 {
    (((x as u32) >> 16) as u16) as i16
}

/// `0x10381370` — 64-entry monotonically increasing scalar codebook searched by
/// [`quantize_codebook64`]. The spacing (dense in the middle, coarse at the
/// ends) is that of a log-domain quantiser.
pub const GAIN_CODEBOOK: [i16; 64] = [
    -9917, -9102, -8311, -7440, -6597, -5828, -5081, -4272, -3370, -2903, -2583, -2305, -1962,
    -1601, -1138, -711, -302, 57, 433, 795, 1132, 1510, 1909, 2333, 2705, 3038, 3376, 3689, 3979,
    4339, 4754, 5129, 5435, 5695, 5991, 6300, 6596, 6969, 7342, 7751, 8101, 8511, 8835, 9102, 9375,
    9698, 10055, 10415, 10762, 11084, 11403, 11752, 12123, 12468, 12863, 13239, 13615, 13997,
    14382, 14770, 15301, 15849, 16640, 17809,
];

/// `0x1037f978` — one full period of `round(32767 * cos(pi * k / 256))`, 512
/// entries. The butterfly reads cosines at `TWIDDLE[j]` and sines at
/// `TWIDDLE[128 + j]` (`cos(x - pi/2) = sin x`, negated), i.e. `0x1037fa78` in
/// the decompilation is just this table offset by a quarter period.
pub const TWIDDLE: [i16; 512] = [
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

/// `0x10311a00` — advance a saturating frame counter held at state `+0x86`.
///
/// The counter stops at 100, so callers can use it as "at least N frames since
/// the event" without it ever wrapping. Ghidra types the field `short`; the
/// comparison is signed.
pub const FRAME_COUNTER_WORD: usize = 0x86 / 2;

#[inline(never)]
pub fn bump_frame_counter(state: &mut [i16]) {
    let c = &mut state[FRAME_COUNTER_WORD];
    if *c < 100 {
        *c += 1;
    }
}

/// `0x10310250` — fold `n` one-bit-per-word values into a single word, MSB
/// first.
///
/// This is the inverse of the wrapper's bit unpacking: the channel bits arrive
/// as an array of 0/1 `i16`, and quantiser indices are reassembled from them
/// here. The original ORs the whole word rather than masking bit 0, so a
/// non-boolean input contaminates the result; that is reproduced.
/// `n` is tested as signed, so a non-positive count yields 0.
pub fn pack_bits_i16(bits: &[i16], n: i16) -> u16 {
    let mut acc: u16 = 0;
    if n > 0 {
        for &b in &bits[..n as usize] {
            acc = acc.wrapping_mul(2) | b as u16;
        }
    }
    acc
}

/// `0x10310250` — see [`pack_bits_i16`]; this is the `u16`-payload spelling.
pub fn pack_bits(bits: &[u16], n: i16) -> u16 {
    let mut acc: u16 = 0;
    if n > 0 {
        for &b in &bits[..n as usize] {
            acc = acc.wrapping_mul(2) | b;
        }
    }
    acc
}

/// `0x10317280` — `dst[i] = high16_batch3(src[i] * coef * 2)`, i.e. a Q15 scale of a
/// vector.
///
/// The doubling is the usual fractional-multiply correction (Q15 × Q15 → Q30,
/// shifted up one to Q31). It wraps rather than saturates, so the single input
/// pair (−32768, −32768) folds to 0. The original computes `src - dst` once and
/// indexes both from the same cursor, so in-place use is supported; being
/// element-wise, the Rust two-slice form is equivalent.
pub fn scale_vec(dst: &mut [i16], src: &[i16], n: i16, coef: i16) {
    if n > 0 {
        for i in 0..n as usize {
            let p = (src[i] as i32).wrapping_mul(coef as i32).wrapping_mul(2);
            dst[i] = high16_batch3(p);
        }
    }
}

/// `0x1030b5e0` — copy `n` words. Returns the number copied (the original
/// returns `dst + n`, used to chain fills). `n` is signed: non-positive copies
/// nothing.
pub fn copy16_batch3(dst: &mut [i16], src: &[i16], n: i16) -> usize {
    if n <= 0 {
        return 0;
    }
    let n = n as usize;
    dst[..n].copy_from_slice(&src[..n]);
    n
}

/// `0x1030aec0` — scale `x` by `2^(exp - 15)` and round the result *up* to a
/// whole multiple of 65536, saturating.
///
/// Returned with the low 16 bits cleared, so the caller reads the high half as
/// an integer count: this is `ceil()` in the block-floating-point domain, used
/// where a length or harmonic count must not be under-estimated. Rounding up is
/// done by adding 0xffff before truncating; only that add is saturated, the
/// shift itself is not.
pub fn scale_ceil16(x: i32, exp: i16) -> i32 {
    let s = exp.wrapping_sub(0xf);
    let v = shift32_al2b2(x, s);
    let r = v.wrapping_add(0xffff);
    let r = if ((r ^ v) & r) < 0 {
        (0x7fffffffu32).wrapping_add((v < 0) as u32) as i32
    } else {
        r
    };
    (r as u32 & 0xffff_0000) as i32
}

/// `0x10318aa0` — block exponent of the largest unflagged entry of `vals`.
///
/// Scans `vals[i]` for which `flags[i] == 0`, keeps the largest as a 32-bit
/// value in the high half, and returns the *negated* normalisation shift of
/// that peak (0 if nothing qualified). A negative return is the exponent the
/// caller then applies to bring the block into range. Because the running
/// maximum starts at 0, negative peaks never win, so the result is always
/// `-(30 - msb)` for a positive peak.
pub fn peak_exp_unflagged(vals: &[i16], flags: &[i16], n: i16) -> i32 {
    let mut best: i32 = 0;
    if n > 0 {
        for i in 0..n as usize {
            let v = (vals[i] as i32).wrapping_mul(0x10000);
            if flags[i] == 0 && best < v {
                best = v;
            }
        }
    }
    if best == 0 {
        return 0;
    }
    let m = if best < 0 {
        !(best as u32)
    } else {
        best as u32
    };
    let i = top_bit_batch3(m);
    ((0x1eu32.wrapping_sub(i as u32) & 0xffff) as i32).wrapping_neg()
}

/// `0x1030b430` — signed 32/16 division, magnitude-first.
///
/// Both operands are made positive (with −2^31 / −2^15 clamped to the positive
/// maximum rather than overflowing), divided as C truncating division, then the
/// sign is reapplied from the XOR of the originals. `|num| < |den|` short-cuts
/// to 0 rather than trusting the divide. **A zero divisor faults**; the caller
/// is responsible.
pub fn div32(num: i32, den: i16) -> i32 {
    div32_common(num, den, 0)
}

/// The body shared by [`div32`] and [`div32_half`]: `q_shift` is the extra
/// right shift applied to the magnitude quotient before the sign goes back on.
fn div32_common(num: i32, den: i16, q_shift: u32) -> i32 {
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
    let q = if n < d { 0 } else { (n / d) >> q_shift };
    if sign < 0 {
        -q
    } else {
        q
    }
}

/// `0x1030a600` — as [`div32`] but the quotient is halved (arithmetic shift
/// right by one) before the sign is applied, giving a Q14-ish result one bit
/// below the plain quotient.
pub fn div32_half(num: i32, den: i16) -> i32 {
    div32_common(num, den, 1)
}

/// `0x103101b0` — saturating variable shift.
///
/// Right shifts (negative count) behave like [`shift32_al2b2`]. Left shifts first
/// test whether any significant bit would be shifted out — the mask
/// `-1 << (31 - s)` must be all-zero for a positive input, or all-ones for a
/// negative one — and return ±INT_MAX/MIN instead of wrapping. This is the
/// C5x `SFTS`-with-saturation.
pub fn shl_sat(x: i32, s: i16) -> i32 {
    if s < 0 {
        return if s < -0x1e {
            x >> 31
        } else {
            x >> ((-s) as u8 as u32 & 0x1f)
        };
    }
    let b = s as u8;
    let mask = (0xffff_ffffu32 << ((0x1fu32.wrapping_sub(b as u32)) & 0x1f)) as i32;
    let mut kept = mask & x;
    if x < 0 {
        kept = kept.wrapping_sub(mask);
    }
    if kept != 0 {
        return (0x7fffffffu32).wrapping_add((x < 0) as u32) as i32;
    }
    ((x as u32) << (b as u32 & 0x1f)) as i32
}

/// `0x10317200` — scale a vector by a Q15 coefficient and requantise it to a
/// new block exponent in one pass.
///
/// `dst[i] = shift(src[i] * coef * 2, exp_in - exp_out) >> 16`. Note the
/// deep-right-shift case is `>> 31` of the *product* truncated to 16 bits, not
/// `>> 31` then `>> 16`, so it yields 0 or −1 — the sign of the product rather
/// than zero. As with [`scale_vec`], in-place use is supported.
#[inline(never)]
pub fn scale_vec_shift(dst: &mut [i16], src: &[i16], n: i16, coef: i16, exp_in: i16, exp_out: i16) {
    if n <= 0 {
        return;
    }
    let s = exp_in.wrapping_sub(exp_out);
    for i in 0..n as usize {
        let p = (src[i] as i32).wrapping_mul(coef as i32).wrapping_mul(2);
        // The deep-right arm is `(p >> 31) as i16`, which is 0 or -1 — the
        // same two values `high16_batch3(p >> 31)` produces, so all three arms
        // are one shift.
        dst[i] = high16_batch3(shift_i32_batch1(p, s));
    }
}

/// Multiply a 32-bit accumulator by a Q15 coefficient, keeping 32 bits.
///
/// The reference open-codes this as `high16_batch3(v) * (2c) + ((low16(v) * c) >> 15)`,
/// which is the exact 32×16 fractional product truncated back to 32 bits.
pub fn q15mul32(v: i32, c: i32) -> i32 {
    (high16_batch3(v) as i32)
        .wrapping_mul(c.wrapping_mul(2))
        .wrapping_add((((v as u32) & 0xffff).wrapping_mul(c as u32) as i32) >> 15)
}

/// `0x1030b2d0` — square root of a block-floating-point value, in place on the
/// exponent.
///
/// The mantissa is normalised so bit 30 is set, giving `m` in [0.5, 1) as Q15,
/// then a two-term polynomial `((c1*m + c0)*m + k)` approximates `sqrt(m)`. An
/// odd exponent is absorbed by an extra multiply by 0xb504 (= 46340/65536 =
/// 1/sqrt(2)) and the exponent becomes `(e + 1) >> 1`. Result is returned in
/// the high half with the low 16 bits cleared; `x == 0` returns 0 and leaves
/// the exponent alone.
pub fn sqrt_norm(x: i32, exp: &mut i16) -> i32 {
    if x == 0 {
        return 0;
    }
    let n = norm_exp_batch3(x as u32);
    let e = (*exp).wrapping_sub(n as i16) as u16;
    let m = high16_batch3(((x as u32) << (n as u32 & 0x1f)) as i32) as i32;

    let t = high16_batch3(
        m.wrapping_mul(-0x198a)
            .wrapping_add(0x38a5_8000u32 as i32)
            .wrapping_mul(2)
            .wrapping_add(0x8000),
    ) as i32;
    let mut acc = (t
        .wrapping_mul(m)
        .wrapping_add(0x141f_8000)
        .wrapping_mul(2)
        .wrapping_add(0x8000) as u32
        & 0xffff_0000) as i32;
    if e & 1 != 0 {
        acc = (high16_batch3(acc) as i32).wrapping_mul(0xb504);
    }
    acc = (acc.wrapping_add(0x8000) as u32 & 0xffff_0000) as i32;
    *exp = ((((e as i16) as i32) + 1) >> 1) as i16;
    acc
}

/// `0x10318130` — leaky accumulate of `add` into the block-floating-point value
/// `(*val, *exp)`.
///
/// `*val` is scaled by 0xf334/2^16 (0.95) and `add` by 0xcccc/2^16 (0.8) with a
/// further 4-bit attenuation, the two are aligned to the larger exponent,
/// summed and renormalised. A zero sum, or an exponent that has fallen below
/// −16, is pinned to 0x7fff with exponent −17 — the "value is effectively
/// infinite / uninitialised" sentinel. This is a background-level tracker with
/// a 0.95 forget factor.
pub fn track_level(val: &mut i16, exp: &mut i16, add: i16, add_exp: i16) {
    let e0 = *exp;
    let m = if add_exp <= e0 { e0 } else { add_exp };

    let a = shift32_al2b2((*val as i32).wrapping_mul(0xf334), e0.wrapping_sub(m));
    let b = shift32_al2b2(
        (add as i32).wrapping_mul(0xcccc),
        add_exp.wrapping_sub(m).wrapping_sub(4),
    );
    let sum = a.wrapping_add(b);

    let nrm = if sum == 0 {
        0i16
    } else {
        norm_exp_batch3(sum as u32) as i16
    };
    let mut ex = m.wrapping_sub(nrm);
    let mut v = ((sum as u32) << (nrm as u32 & 0x1f)) as i32;
    if v == 0 || ex < -0x10 {
        v = 0x7fff_0000;
        ex = -0x11;
    }
    *val = high16_batch3(v);
    *exp = ex;
}

/// `0x10312330` — nearest-neighbour search of the 64-entry [`GAIN_CODEBOOK`].
///
/// Replaces `*x` with the closest codebook entry and returns its index. The
/// distance is `(x - c)^2` computed on saturating 32-bit differences, so a
/// difference that saturates still compares sanely. Ties go to the lower index
/// (the comparison is strict). Note the entry-0 distance is computed without
/// the `-0x8000` guard the loop applies, so a difference of exactly −32768
/// against entry 0 wraps to a large negative metric and wins; that quirk is
/// reproduced.
pub fn quantize_codebook64(x: &mut i16) -> i16 {
    let a = (*x as i32).wrapping_mul(0x10000);

    let d0 = high16_batch3(sat_add32(
        a,
        (GAIN_CODEBOOK[0] as i32).wrapping_mul(-0x10000),
    )) as i32;
    let mut best = d0.wrapping_mul(d0).wrapping_mul(2);
    let mut best_i: i16 = 0;

    for i in 1..64usize {
        let d = high16_batch3(sat_add32(
            a,
            (GAIN_CODEBOOK[i] as i32).wrapping_mul(-0x10000),
        ));
        let metric = if d == i16::MIN {
            i32::MAX
        } else {
            (d as i32).wrapping_mul(d as i32).wrapping_mul(2)
        };
        if metric < best {
            best = metric;
            best_i = i as i16;
        }
    }
    *x = GAIN_CODEBOOK[best_i as usize];
    best_i
}

/// `0x1030a660` — divide and renormalise: returns the Q15 mantissa of
/// `num / den` and writes its exponent to `out_exp`.
///
/// The numerator's exponent `num_exp` is carried in; if the quotient would
/// overflow 16 bits (`|den| << 16 <= |num|`) the numerator is pre-shifted right
/// and the exponent bumped. `exp_bias` is subtracted from the result exponent,
/// letting the caller fold in a fixed Q-point change. A zero mantissa forces
/// the exponent to 0. **A zero divisor faults.**
pub fn div_norm(num: i32, num_exp: i16, den: i16, exp_bias: i16, out_exp: &mut i16) -> i16 {
    let sign = (den as i32) ^ num;
    let mut n = if num < 0 {
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

    let mut e = num_exp;
    if d.wrapping_mul(0x10000) <= n {
        n >>= 1;
        e = e.wrapping_add(1);
    }
    let mut q = ((n / d) >> 1) as i16;
    if sign < 0 {
        q = q.wrapping_neg();
    }

    let v = (q as i32).wrapping_mul(0x10000);
    let nrm = if v == 0 {
        0i16
    } else {
        norm_exp_batch3(v as u32) as i16
    };
    let m = high16_batch3(((v as u32) << (nrm as u32 & 0x1f)) as i32);
    if m == 0 {
        *out_exp = 0;
        return 0;
    }
    *out_exp = e.wrapping_sub(nrm).wrapping_sub(exp_bias);
    m
}

/// `0x10312620` — remap a packed voicing field onto eight fixed frequency
/// bands.
///
/// For each of eight band centres (0x0c00, 0x1c00 … 0x7c00 in Q15 of half the
/// sample rate) it works out which coded band that centre falls in — the
/// `log2`-ish chain `norm`, then a multiply by 0xb86c with the exponent as the
/// shift, is a reciprocal of the fundamental `w0` — clamps the index into the
/// available range, and shifts the selected bit of `bits` into the result, MSB
/// first.
///
/// The return is `undefined4` in Ghidra and really is 32 bits wide: the low 16
/// are the eight collected bits, and bit 16 carries the doubled accumulator's
/// overflow, so callers that only want the bits must mask. `nbands` is the
/// number of coded bands.
pub fn map_voicing_bits(bits: i16, nbands: i32, w0: i16) -> u32 {
    let mut acc: u32 = 0;
    let mut ctr: i32 = 7;
    let mut limit: i32 = 7;
    let mut centre: i32 = 0x0c00_0000;
    let mut doubled: i32;
    let mut last: u16;

    loop {
        let u = (high16_batch3(centre) as i32)
            .wrapping_mul(w0 as i32)
            .wrapping_sub(0xece6)
            .wrapping_mul(0x10);

        let mut idx: i32 = if u < 0 {
            0
        } else {
            let e = if u == 0 {
                0u32
            } else {
                0x1eu32.wrapping_sub(top_bit_batch3(u as u32) as u32) & 0xffff
            };
            let m = high16_batch3(((u as u32) << (e & 0x1f)) as i32) as i32;
            let sh = ((0xfu32.wrapping_sub(e) & 0xffff) as i32).wrapping_sub(0x13);
            shift32_al2b2(m.wrapping_mul(0xb86c), sh as i16)
        };

        idx = nbands.wrapping_sub(idx >> 16).wrapping_sub(1);
        if (idx as i16 as i32) < limit {
            idx = ctr;
        }
        ctr -= 1;
        centre = centre.wrapping_add(0x1000_0000);
        doubled = (acc as i32).wrapping_mul(2);
        limit -= 1;
        last = ((((bits as i32) >> (idx as u32 & 0x1f)) & 1) as u16) | (doubled as u16);
        acc = last as u32;

        if (ctr as u16) >= 0x8000 {
            break;
        }
    }
    ((high16_batch3(doubled) as u16 as u32) << 16) | last as u32
}

/// `0x10319a80` — accumulate a linearly-faded copy of `src` into `dst`.
///
/// Each sample is `src[i] * (1 - g)` where `g` is the Q15 high half of a phase
/// accumulator that advances by `step * 2^step_exp` every sample; the product
/// is formed as `x - x*g` with the 32×16 fractional multiply, then added into
/// `dst[i]` with 32-bit saturation. The accumulator itself saturates too, so
/// once the ramp reaches full scale it stays there. This is the overlap-add
/// cross-fade between the previous and current synthesis frame.
pub fn ramp_mix(dst: &mut [i32], src: &[i32], step: i16, step_exp: i16, n: i16) {
    if n <= 0 {
        return;
    }
    let base = (step as i32).wrapping_mul(0x10000);
    let mut acc: i32 = 0;
    for i in 0..n as usize {
        let x = src[i];
        let g = high16_batch3(acc) as i32;
        let y = (high16_batch3(x) as i32)
            .wrapping_mul(g)
            .wrapping_mul(-2)
            .wrapping_sub(((((x as u32) & 0xffff).wrapping_mul(g as u32)) as i32) >> 15)
            .wrapping_add(x);
        dst[i] = sat_add32(y, dst[i]);
        acc = sat_add32(acc, shift32_al2b2(base, step_exp));
    }
}

/// `0x1030a310` — add two block-floating-point values, returning the mantissa
/// and writing the exponent.
///
/// If either operand is zero, or the exponents differ by 32 or more (one term
/// is entirely below the other's LSB), the dominant operand is passed through
/// untouched. Otherwise both are aligned to `max(exp) + 1` — the extra bit is
/// the guard against the sum overflowing — added, and renormalised. A sum whose
/// mantissa rounds to zero returns exponent 0.
pub fn add_norm(a: i16, a_exp: i16, b: i16, b_exp: i16, out_exp: &mut i16) -> i16 {
    addsub_norm(a, a_exp, b, b_exp, out_exp, false)
}

/// `0x1031a710` — one radix-2 butterfly stage over interleaved pairs.
///
/// Processes four points per iteration (`ceil(n/2)` iterations), combining the
/// running spectrum in `sum` with the rotated pair in `dif`:
///
/// ```text
/// t          = c*dif[k] - s*dif[k+1]     (complex rotate, Q15)
/// sum'[k]    = sum[k]/2 + t
/// dif'[k]    = sum[k]/2 - t
/// ```
///
/// with the imaginary half computed the same way from `c*dif[k+1] + s*dif[k]`.
/// The `>> 1` on the incoming term is the per-stage scaling that keeps the
/// transform from overflowing. `stride` steps the twiddle table, so it encodes
/// which stage of the FFT this is; each iteration consumes two twiddles.
/// Twiddles come from [`TWIDDLE`] — cosines at `j`, sines at `128 + j`.
///
/// Callers must keep `128 + (2*ceil(n/2) - 1) * stride` inside the table.
pub fn fft_butterfly_stage(sum: &mut [i16], dif: &mut [i16], stride: i16, n: i16) {
    radix2_stage(sum, dif, stride, n, true);
}

/// The body shared by [`fft_butterfly_stage`] and [`fft_stage`].
///
/// The two stages are the same butterfly and differ only in which 16 bits of
/// the 32-bit result they keep: `scaled` keeps bits 16..31, the plain stage
/// bits 15..30 (the `wrapping_mul(2)`). One body serves both.
fn radix2_stage(xe: &mut [i16], xo: &mut [i16], stride: i16, n: i16, scaled: bool) {
    if n <= 0 {
        return;
    }
    let iters = ((((n as u16).wrapping_sub(1)) >> 1).wrapping_add(1)) as usize;
    let st = stride as i32;
    let mut j: i32 = 0;
    // `(x * 2) >> 16` truncated to `i16` is exactly `x >> 15` truncated to
    // `i16`, so the two scalings collapse to one variable shift.
    let sh = if scaled { 16u32 } else { 15 };

    let mut b = 0usize;
    for _ in 0..iters {
        for _ in 0..2 {
            let c = TWIDDLE[j as usize] as i32;
            let s = TWIDDLE[(j + 128) as usize] as i32;

            let yr = xo[b] as i32;
            let yi = xo[b + 1] as i32;
            let t = c.wrapping_mul(yr).wrapping_sub(s.wrapping_mul(yi));
            let h = ((xe[b] as i32).wrapping_mul(0x10000)) >> 1;
            xe[b] = (h.wrapping_add(t) >> sh) as i16;
            xo[b] = (h.wrapping_sub(t) >> sh) as i16;

            let u = c.wrapping_mul(yi).wrapping_add(s.wrapping_mul(yr));
            let h2 = ((xe[b + 1] as i32).wrapping_mul(0x10000)) >> 1;
            xe[b + 1] = (h2.wrapping_add(u) >> sh) as i16;
            xo[b + 1] = (h2.wrapping_sub(u) >> sh) as i16;

            j += st;
            b += 2;
        }
    }
}

pub fn resonator3(dst: &mut [i32], src: &[i32], state: &mut [i32], n: i16) {
    if n <= 0 {
        return;
    }
    for i in 0..n as usize {
        let x = src[i];

        // The compiler inlines a 64-bit accumulator with two saturating
        // stores; reproduced here with i64 directly.
        let acc1 = (q15mul32(state[1], 0x78ab) as i64) - (state[0] as i64) + (x as i64);
        let sat1 = acc1.clamp(i32::MIN as i64, i32::MAX as i64) as i32;

        state[1] = sat1;
        let s2 = state[2];
        state[0] = x;
        let s4 = state[4];
        let s3 = state[3];
        state[2] = sat1;
        state[4] = s2;

        let acc2 = acc1 + (s4 as i64) - 2 * (s2 as i64) + 2 * (q15mul32(s3, 0x7c20) as i64);

        let s5 = state[5];
        state[5] = s3;
        let acc3 = acc2 - (q15mul32(s5, 0x78af) as i64);
        let out = acc3.clamp(i32::MIN as i64, i32::MAX as i64) as i32;

        state[3] = out;
        dst[i] = out;
    }
}

// ======================= from batch4 =======================

/// Variable left shift with the x86 5-bit count mask.
#[inline]
pub fn shl_var(x: i32, n: i16) -> i32 {
    x.wrapping_shl(n as u32 & 0x1f)
}

/// Variable arithmetic right shift with the x86 5-bit count mask, given a
/// *negative* nominal shift `n`; counts past -30 collapse to a sign fill.
#[inline(never)]
pub fn shr_neg(x: i32, n: i16) -> i32 {
    if n < -0x1e {
        x >> 31
    } else {
        x >> (n.wrapping_neg() as u32 & 0x1f)
    }
}

/// Signed shift by `n` (left if non-negative), matching the original's
/// three-way branch.
#[inline(never)]
pub fn shift_var(x: i32, n: i16) -> i32 {
    if n < 0 {
        shr_neg(x, n)
    } else {
        shl_var(x, n)
    }
}

/// `bsr`: index of the most significant set bit. Undefined for 0 in hardware;
/// callers always guard.
#[inline]
pub fn msb(x: u32) -> i32 {
    31 - x.leading_zeros() as i32
}

/// The library's normalisation shift: `0x1e - msb(|x|)`, i.e. the left shift
/// that brings `x` to Q30. Zero yields 0 here; callers special-case it.
#[inline]
pub fn norm_shift_batch4(x: i32) -> i16 {
    if x == 0 {
        return 0;
    }
    let m = if x < 0 { !(x as u32) } else { x as u32 };
    (0x1e - msb(m)) as i16
}

/// ETSI `L_sub`, written the way the original does (negate then add, with the
/// complement of the subtrahend standing in for its sign).
#[inline]
pub fn l_sub(a: i32, b: i32) -> i32 {
    let neg = b.wrapping_neg();
    let r = a.wrapping_add(neg);
    if (!b ^ r) & (r ^ a) < 0 {
        if a < 0 {
            i32::MIN
        } else {
            i32::MAX
        }
    } else {
        r
    }
}

/// `0x1030af10` — take the 16-bit mantissa of `x` at exponent `exp`, left in
/// the high half of the return value (the low 16 bits are zeroed).
///
/// The nominal shift is `exp - 15`; a shift below -30 collapses to a sign
/// fill. Ghidra's `uint` return is really a Q31 accumulator whose low word has
/// been discarded, so this is "round `x` down to 16-bit precision at exponent
/// `exp`".
pub fn hi16_at_exp(x: i32, exp: i16) -> i32 {
    let s = exp.wrapping_sub(0xf);
    let v = if s < 0 { shr_neg(x, s) } else { shl_var(x, s) };
    (v as u32 & 0xffff_0000) as i32
}

/// `0x1030b230` — the residual of `x` below the 16-bit mantissa at exponent
/// `exp`, returned scaled by `exp`.
///
/// Takes the top 16 bits of `x` at exponent `exp`, truncating *towards zero*
/// (that is what the `+ 0x10000` on a negative product does), shifts that back
/// to `x`'s scale, subtracts it, and rescales the remainder by `exp`. Used to
/// split a wide accumulator into mantissa and residual for extended-precision
/// multiplies.
pub fn frac_at_exp(x: i32, exp: i16) -> i32 {
    let s = exp.wrapping_sub(0xf);
    let hi = if s < 0 {
        if s < -0x1e {
            (x >> 31) as i16
        } else {
            ((x >> (s.wrapping_neg() as u32 & 0x1f)) >> 16) as i16
        }
    } else {
        (shl_var(x, s) >> 16) as i16
    };

    let mut v = (hi as i32) << 16;
    if v < 0 {
        v = v.wrapping_add(0x10000);
    }
    let back = (0xfi16).wrapping_sub(exp);
    v = if back < 0 {
        shr_neg(v, back)
    } else {
        shl_var(v, back)
    };

    let rem = x.wrapping_sub(v);
    if exp < 0 {
        shr_neg(rem, exp)
    } else {
        shl_var(rem, exp)
    }
}

/// Polynomial coefficients for [`log2_q`], read from `.data` at `0x1037f948`.
/// `LOG2_C[0]` is `sqrt(2)/2` in Q15 — the point the mantissa is centred on.
pub const LOG2_C: [i16; 6] = [23170, -23636, 22285, -23636, 16714, -16384];

/// `0x1030b100` — base-2 logarithm of `x * 2^-exp`, as a Q16.15 result.
///
/// `x` is normalised to Q30, the mantissa is offset by `sqrt(2)/2` and fed to a
/// degree-4 minimax polynomial (coefficients in `.data`, see [`LOG2_C`]); the
/// exponent `exp - norm` is returned in the high 16 bits and the fractional
/// part in the low 15. Both the polynomial's intermediate roundings
/// (`+ 0x8000`, then masking off the low bits) and its final arithmetic
/// `>> 15` are reproduced exactly.
pub fn log2_q(x: i16, exp: i16) -> i32 {
    let acc = (x as i32) << 16;
    let n = if acc == 0 { 0 } else { norm_shift_batch4(acc) };

    let d = ((shl_var(acc, n).wrapping_sub((LOG2_C[0] as i32) << 16)) >> 16) as i16 as i32;

    let mut t = (LOG2_C[1] as i32).wrapping_mul(d);
    t = (t.wrapping_mul(2).wrapping_add(0x8000) & 0xffff_0000u32 as i32)
        .wrapping_add((LOG2_C[2] as i32) << 16);
    let mut t = ((t >> 16) as i16 as i32).wrapping_mul(d);
    t = (t.wrapping_mul(2).wrapping_add(0x8000) & 0xffff_0000u32 as i32)
        .wrapping_add((LOG2_C[3] as i32) << 16);
    let mut t = ((t >> 16) as i16 as i32).wrapping_mul(d);
    t = ((t.wrapping_mul(2).wrapping_add(0x8000) >> 1) & 0xffff_8000u32 as i32)
        .wrapping_add((LOG2_C[4] as i32) << 16);
    let t = ((t >> 16) as i16 as i32).wrapping_mul(d);
    let t = (t.wrapping_mul(8).wrapping_add(0x20000) & 0xfffc_0000u32 as i32)
        .wrapping_add((LOG2_C[5] as i32) << 16);

    (t >> 15).wrapping_add((exp.wrapping_sub(n) as i32) << 16)
}

/// `0x1031a520` — block exponent of `n` interleaved complex words, based on
/// the largest squared magnitude.
///
/// The running maximum starts at `0x1000000` rather than 0, which floors the
/// reported exponent at 5 and keeps very quiet spectra from being scaled up
/// into noise. The return is `norm(max) - 1`, so the result is one bit shy of
/// full scale to leave headroom.
pub fn cplx_block_exp(x: &[i16], n: i16) -> i32 {
    let mut max = 0x0100_0000i32;
    if n > 0 {
        for i in 0..n as usize {
            let re = x[2 * i] as i32;
            let im = x[2 * i + 1] as i32;
            let v = re.wrapping_mul(re).wrapping_add(im.wrapping_mul(im));
            if max < v {
                max = v;
            }
        }
    }
    if max == 0 {
        return -1;
    }
    ((norm_shift_batch4(max) as u16) as i32) - 1
}

/// `0x1031fc50` — codebook search: index of the entry minimising the summed
/// squared error against `target`.
///
/// `book` is scanned `cand` times with a word stride of `stride`; each
/// candidate's error is `sum(2 * (target[i] - book[i])^2)` computed on
/// saturating 32-bit accumulators with the difference taken in the high half
/// (so it clamps to a 16-bit range), and `(-32768)^2` is forced to
/// `0x7fffffff`. Ties keep the earliest index.
pub fn vq_search(target: &[i16], book: &[i16], n: i16, cand: i16, stride: i16) -> i16 {
    if cand <= 0 {
        return 0;
    }
    let mut best = 0i16;
    let mut min = i32::MAX;
    for k in 0..cand {
        let mut acc = 0i32;
        if n > 0 {
            let off = (k as isize) * (stride as isize);
            for i in 0..n as isize {
                let a = (target[i as usize] as i32) << 16;
                let b = (book[(off + i) as usize] as i32) << 16;
                let d = (l_sub(a, b) >> 16) as i16;
                let sq = if d == i16::MIN {
                    i32::MAX
                } else {
                    (d as i32).wrapping_mul(d as i32).wrapping_mul(2)
                };
                acc = acc.saturating_add(sq);
            }
        }
        if acc < min {
            min = acc;
            best = k;
        }
    }
    best
}

/// `0x10310280` — unpack the low `n` bits of `word` into one bit per output
/// element, most significant first.
///
/// `dst[n-1]` receives bit 0, `dst[n-2]` bit 1 and so on. The source is
/// shifted *arithmetically* as a 16-bit value, so asking for more than 16 bits
/// replicates the sign. Returns the number of words written (the original
/// returns a pointer one past the end).
pub fn unpack_bits(word: i16, dst: &mut [i16], n: i16) -> i16 {
    let mut w = word;
    let mut i = n.wrapping_sub(1);
    while i >= 0 {
        dst[i as usize] = w & 1;
        w >>= 1;
        i -= 1;
    }
    n
}

/// `0x10312f30` — copy `n` words between two buffers in either direction.
///
/// `store == false` copies `src` into `dst`; `store == true` copies `dst` back
/// out into `src`. One routine serves both the load and the save of a state
/// vector. Returns the number of words moved (the original returns the `dst`
/// cursor).
pub fn copy16_batch4(dst: &mut [i16], src: &mut [i16], n: i16, store: bool) -> i16 {
    if n <= 0 {
        return 0;
    }
    for i in 0..n as usize {
        if store {
            src[i] = dst[i];
        } else {
            dst[i] = src[i];
        }
    }
    n
}

/// `0x10314870` — copy 57 words with the first sample duplicated, then hold
/// the last value out to `n`.
///
/// `dst[0] = src[0]` and `dst[i] = src[i-1]` for `i` in 1..=56, so the source's
/// 56 values become a 57-point sequence delayed by one with its left edge
/// extended; anything past index 56 repeats `src[55]`. This is the standard
/// edge extension applied to a spectral envelope before interpolation.
pub fn spread_hold(dst: &mut [i16], src: &[i16], n: i16) {
    dst[0] = src[0];
    for i in 1..=0x38usize {
        dst[i] = src[i - 1];
    }
    if n > 0x39 {
        let pad = src[0x37];
        for i in 0x39..n as usize {
            dst[i] = pad;
        }
    }
}

/// `0x10314d10` — identical to [`spread_hold`] but with the arguments in
/// (dst, n, src) order and the hold value read back from `dst[56]`.
///
/// Since `dst[56]` was just written from `src[55]`, the two produce the same
/// bytes; the reference contains both because the call sites differ.
pub fn spread_hold_alt(dst: &mut [i16], n: i16, src: &[i16]) {
    // `dst[0x38]` was just written from `src[0x37]`, so reading the pad from
    // either buffer gives the same word; one body serves both entry points.
    spread_hold(dst, src, n);
}

/// `0x103131f0` — one pass of the bit de/interleaver.
///
/// Walks a table of signed steps, advancing a cursor through `data` by each
/// step; whenever `steps[k] - off` is non-negative that position is exchanged
/// with the next word of `out` (read when `store` is false, written when it is
/// true). After `limit` steps the walk restarts from `base` with `off`
/// decremented, which is what sweeps the diagonal of the interleaving matrix.
/// Runs until the `out` cursor reaches `out_end`.
///
/// The original passes the three cursors by pointer and updates them in place;
/// they are `&mut` here, and word indices replace the raw addresses.
/// The returned value is the updated `off`.
#[allow(clippy::too_many_arguments)]
pub fn interleave_walk(
    out: &mut [i16],
    out_pos: &mut usize,
    out_end: usize,
    data: &mut [i16],
    data_pos: &mut isize,
    steps: &[i16],
    step_pos: &mut isize,
    count: &mut i16,
    base_pos: isize,
    off: i16,
    limit: i16,
    store: bool,
) -> i16 {
    let mut off = off;
    while *out_pos < out_end {
        let s = steps[*step_pos as usize].wrapping_sub(off);
        if s >= 0 {
            let idx = (*data_pos + s as isize) as usize;
            if store {
                data[idx] = out[*out_pos];
            } else {
                out[*out_pos] = data[idx];
            }
            *out_pos += 1;
        }
        let d = steps[*step_pos as usize];
        *step_pos += 1;
        *count = count.wrapping_add(1);
        *data_pos += d as isize;
        if (limit as i32) + 1 <= *count as i32 {
            *count = 1;
            off = off.wrapping_sub(1);
            *step_pos -= limit as isize;
            *data_pos = base_pos;
        }
    }
    off
}

/// `0x1030c300` — fill the IMBE frame's bit-allocation descriptor.
///
/// The 13-word block is laid out as
/// `[0]=0, [1]=1, [2]=total, [3]=0, [4]=12, [5]=8, [6..13]=per-vector counts`;
/// word 2 is filled in by summing words 4, 5 and 6..12, giving **88** — the
/// IMBE payload size. Ghidra's `undefined4` writes are pairs of `i16`.
#[inline(never)]
pub fn imbe_bit_alloc(t: &mut [i16; 13]) {
    t[0] = 0;
    t[1] = 1;
    t[3] = 0;
    t[4] = 12;
    t[5] = 8;
    t[6] = 6;
    t[7] = 0x3e;
    t[8] = 0;
    t[9] = 0;
    t[10] = 0;
    t[11] = 0;
    t[12] = 0;

    let mut total = t[5].wrapping_add(t[4]);
    t[2] = total;
    for i in 0..7 {
        total = total.wrapping_add(t[6 + i]);
        t[2] = total;
    }
}

/// `0x1030bc70` — fill the AMBE frame's bit-allocation descriptor for `mode`.
///
/// Same 13-word layout as [`imbe_bit_alloc`]. `mode` is clamped to 0..=4 and
/// stored in word 3; it progressively strips bits off the last vectors, giving
/// totals 49, 48, 47, 46 and 45 — mode 0 being AMBE half rate's 49 payload
/// bits.
pub fn ambe_bit_alloc(t: &mut [i16; 13], mode: i16) {
    let mode = mode.clamp(0, 4);
    t[3] = mode;
    t[0] = 1;
    t[1] = 1;
    t[4] = 5;
    t[5] = 7;
    t[6] = 5;
    t[7] = 9;
    t[8] = 7;
    t[9] = 5;
    t[10] = 4;
    t[11] = 4;
    t[12] = 3;
    match mode {
        1 => t[12] = 2,
        3 => {
            t[11] = 3;
            t[12] = 1;
        }
        4 => {
            t[10] = 3;
            t[11] = 3;
            t[12] = 1;
        }
        2 => t[12] = 1,
        _ => {}
    }

    let mut total = t[4].wrapping_add(t[5]);
    t[2] = total;
    for i in 0..7 {
        total = total.wrapping_add(t[6 + i]);
        t[2] = total;
    }
}

/// `0x103177d0` — expand a packed voicing decision word into one flag per
/// harmonic.
///
/// `mask` holds sixteen 2-bit band decisions (LSB = band 0), read out with an
/// arithmetic shift so bands past the top replicate the sign. `step` is the
/// harmonic spacing; the band index is the high half of a Q31 accumulator that
/// advances by `step * 4` per harmonic, clamped to 15. Each harmonic ORs its
/// band with the next one — a harmonic straddling a band boundary counts as
/// voiced if either side is — and a resulting 3 is folded back to 1. Entries
/// past `n` are zeroed out to 56.
///
/// `step` must be non-negative: a negative accumulator would index the band
/// table backwards, off the original's stack frame.
pub fn voicing_expand(dst: &mut [i16], mask: i32, step: i16, n: i16) {
    let mut bands = [0i16; 16];
    let mut m = mask;
    for b in bands.iter_mut() {
        *b = (m as u16 & 3) as i16;
        m >>= 2;
    }

    let inc = ((step as i32) << 6) >> 4;
    if n > 0 {
        let mut acc = inc;
        for i in 0..n as usize {
            let mut lo = (acc >> 16) as i16;
            if lo > 0xf {
                lo = 0xf;
            }
            let mut hi = lo.wrapping_add(1);
            let mut v = bands[lo as usize];
            if hi > 0xf {
                hi = 0xf;
            }
            v |= bands[hi as usize];
            if v > 2 {
                v = 1;
            }
            dst[i] = v;
            acc = acc.wrapping_add(inc);
        }
    }
    if n < 0x38 {
        for i in n.max(0) as usize..0x38 {
            dst[i] = 0;
        }
    }
}

/// `0x10319b80` — accumulate a Q31 sequence scaled by a linearly ramping gain.
///
/// `dst[i] += src[i] * gain`, saturating, where `gain` is the high half of a
/// phase accumulator that starts at zero and advances by `step << 16` shifted
/// by `shift` (also saturating). The multiply is the usual 32x16 split:
/// `(lo * g) >> 15` plus `hi * g * 2`. This is how one sinusoid is added into
/// the synthesis buffer with an amplitude that tracks across the frame.
pub fn mac_ramp(dst: &mut [i32], src: &[i32], step: i16, shift: i16, n: i16) {
    if n <= 0 {
        return;
    }
    let base = (step as i32) << 16;
    let mut phase = 0i32;
    for i in 0..n as usize {
        let s = src[i];
        let g = (phase >> 16) as i16 as i32;
        let lo = ((s as u32 & 0xffff).wrapping_mul(g as u32) as i32) >> 15;
        let hi = ((s >> 16) as i16 as i32).wrapping_mul(g).wrapping_mul(2);
        dst[i] = dst[i].saturating_add(lo.wrapping_add(hi));
        phase = shift_var(base, shift).saturating_add(phase);
    }
}

/// `0x1031aa20` — in-place radix-2 butterflies over pairs of complex words,
/// with an optional downscale.
///
/// Each group of four words is one butterfly on the complex pair
/// `(x[0], x[1])` / `(x[2], x[3])`: sum in place, difference into the second.
/// `mode == 0` is unscaled and wraps at 16 bits; `mode == -2` scales by 1/4 and
/// any other value by 1/2, both computed in 32 bits before truncation — the
/// block-floating-point guard against growth through the FFT stages.
/// `n` counts complex points; `ceil(n/2)` groups are processed.
pub fn bfly_pairs(x: &mut [i16], n: i16, mode: i16) {
    if n <= 0 {
        return;
    }
    let cnt = ((((n.wrapping_sub(1)) as u16) >> 1).wrapping_add(1)) as u32;
    let sh = match mode {
        -2 => 2,
        0 => 0,
        _ => 1,
    };
    let mut b = 0usize;
    for _ in 0..cnt {
        // `mode == 0` is the same expression with `sh == 0`: the sum of two
        // `i16` cannot leave the low 17 bits, so `((a + b) >> 0) as i16` is
        // exactly `a.wrapping_add(b)`. One arm covers all three modes.
        let (a0, a2) = (x[b] as i32, x[b + 2] as i32);
        x[b] = ((a0 + a2) >> sh) as i16;
        x[b + 2] = ((a0 - a2) >> sh) as i16;
        let (a1, a3) = (x[b + 1] as i32, x[b + 3] as i32);
        x[b + 1] = ((a1 + a3) >> sh) as i16;
        x[b + 3] = ((a1 - a3) >> sh) as i16;
        b += 4;
    }
}

/// `0x1031a580` — one decimation FFT stage over an even/odd pair of halves.
///
/// `xe` and `xo` are the two halves (interleaved re/im). Each butterfly halves
/// its input (`<< 16 >> 1`, i.e. a Q31 value at half scale), forms
/// `t = c*Yr - s*Yi` and `s*Yr + c*Yi`, and writes sum into `xe` and
/// difference into `xo`, keeping the high 16 bits of twice the result — so the
/// stage has unity gain with one bit of headroom per pass. `stride` is the
/// twiddle step in table entries and advances once per butterfly;
/// `ceil(n/2)` iterations handle two butterflies each.
///
/// The two halves must not overlap. The twiddle index is not wrapped, matching
/// the original's raw pointer walk.
pub fn fft_stage(xe: &mut [i16], xo: &mut [i16], stride: i16, n: i16) {
    radix2_stage(xe, xo, stride, n, false);
}

/// `0x1030c700` — renormalise the harmonic magnitude vector held in the
/// decoder state, dropping harmonics that fall past the band edge.
///
/// The state is addressed by word: `[2]` is the harmonic count, `[6]` the
/// fundamental in Q15, `[8..]` the magnitudes and `[66]` their shared block
/// exponent. Starting from `count * f0` scaled by 2^13, harmonics are peeled
/// off the top while the running frequency exceeds `0x4ccd` (0.6 of Nyquist);
/// each peeled magnitude is scaled by 1/3 (`0xd556`, verified as a positive
/// 54614 in the disassembly), individually normalised, and given its own
/// exponent, while the frequency drops by `f0/8`. The per-harmonic exponents
/// are then maximised and every magnitude is shifted down to that common
/// exponent, which is written back to `[66]`.
///
/// The count must be at least 1 — with 0 the original returns an uninitialised
/// stack word as the new exponent.
#[inline(never)]
pub fn refine_mags(state: &mut [i16]) {
    let l_raw = state[2] as u16;
    let l = l_raw as i16;
    let mut exps = [0i16; 56];
    let mut mags = [0i16; 56];

    if l > 0 {
        let e0 = state[66];
        for i in 0..l as usize {
            exps[i] = e0;
            mags[i] = state[8 + i];
        }
    }
    let u9 = l_raw;

    let f0 = state[6];
    let acc = (l as i32).wrapping_mul(f0 as i32).wrapping_mul(0x2000);
    let mut hi = (acc >> 16) as i16;
    let mut u7 = l_raw;

    while hi > 0x4ccd {
        let e0 = state[66];
        u7 = u7.wrapping_sub(1);
        let v = hi as i32;
        let m = ((v.wrapping_mul(0xd556) >> 16) as i16) as i32;
        let prod = m
            .wrapping_mul(mags[u7 as i16 as usize] as i32)
            .wrapping_mul(2);
        let sh = if prod == 0 {
            0
        } else {
            norm_shift_batch4(prod)
        };

        let next = (v << 16).wrapping_sub(((f0 as i32) << 16) >> 3);
        exps[u7 as i16 as usize] = e0.wrapping_sub(sh).wrapping_add(1);
        mags[u7 as i16 as usize] = (shl_var(prod, sh) >> 16) as i16;
        hi = (next >> 16) as i16;
    }

    // The common exponent is kept in a register; `exps[0]` itself is *not*
    // overwritten, so harmonic 0 is rescaled like every other one.
    let mut max = exps[0];
    if (u9 as i16) > 1 {
        for i in 1..u9 as usize {
            if max < exps[i] {
                max = exps[i];
            }
        }
    }

    if (u9 as i16) > 0 {
        let mut i = 0i16;
        loop {
            let v = (mags[i as usize] as u16 as i32) << 16;
            let d = exps[i as usize].wrapping_sub(max);
            let r = if d < 0 {
                if d < -0x1e {
                    mags[i as usize] >> 15
                } else {
                    ((v >> (d.wrapping_neg() as u32 & 0x1f)) >> 16) as i16
                }
            } else {
                (shl_var(v, d) >> 16) as i16
            };
            state[8 + i as usize] = r;
            i += 1;
            if i >= state[2] {
                break;
            }
        }
    }
    state[66] = max;
}

// ======================= from l1_batch1 =======================

/// Index of the highest set bit, or 31 for zero — the original's open-coded
/// `bsr`, which leaves its destination unchanged (here: 31) for a zero input.
#[inline]
pub fn top_bit_al2b5(v: u32) -> i32 {
    top_bit_batch3(v)
}

/// Block exponent: how far left `x` must be shifted for its magnitude to fill
/// bit 30. The call sites in this batch all special-case `x == 0` to a shift of
/// zero *before* calling, which is why this helper does not.
#[inline]
pub fn norm_exp_l1_batch1(x: i32) -> i32 {
    norm_shift_batch1(x as u32)
}

/// The library's variable shift, open-coded at almost every call site in this
/// batch: a non-negative count shifts left (wrapping), a count in `-30..0`
/// shifts right arithmetically, and anything below that is a full sign fill.
#[inline]
pub fn shift32_al3b2(v: i32, s: i16) -> i32 {
    shift_i32_batch1(v, s)
}

/// `0x10309d80` — `dec_ctrl`: latch the 52-byte control block into the decoder
/// state at byte offset `0x614`.
///
/// This is the public `dec_ctrl(state, ctrl)` entry point of FINDINGS §3. The
/// original copies the block in five pieces (14 words, a dword, a word, 7
/// words, two words) purely because the compiler unrolled a struct assignment;
/// the effect is a flat 26-word copy. `state` is indexed in words from the DSP
/// state base, so the destination is word `0x30a`.
pub fn dec_ctrl(state: &mut [i16], ctrl: &[i16]) {
    const D: usize = 0x614 / 2;
    copy16_batch3(&mut state[D..], &ctrl[..14], 14);
    state[D + 14] = ctrl[14];
    state[D + 15] = ctrl[15];
    state[D + 16] = ctrl[16];
    copy16_batch3(&mut state[D + 17..], &ctrl[17..24], 7);
    state[D + 24] = ctrl[24];
    state[D + 25] = ctrl[25];
}

/// `0x1030b380` — square root of a block-floating-point value, rescaled to a
/// caller-chosen exponent and rounded to 16 bits.
///
/// Wraps the layer-0 [`sqrt_norm`]: takes its mantissa/exponent result, shifts
/// it by `exp_out - target` and rounds with `+0x8000 >> 16`. The exponent is
/// passed to `sqrt_norm` through a stack slot that the callee only writes 16
/// bits of, so only the low half of `exp` is ever significant.
pub fn sqrt_to_exp(x: i32, exp: i16, target: i16) -> i32 {
    let mut e = exp;
    let m = sqrt_norm(x, &mut e);
    let s = e.wrapping_sub(target);
    shift32_al3b2(m, s).wrapping_add(0x8000) >> 16
}

/// `0x10317310` — rescale a block of words from one block exponent to another.
///
/// `dst[i] = sat_shl(src[i] << 16, e_from - e_to) >> 16`, using the saturating
/// variable shift [`shl_sat`]. The shift count is formed as a full 32-bit
/// difference but the callee only reads its low half.
pub fn scale_block_exp(dst: &mut [i16], src: &[i16], n: u16, e_from: i32, e_to: i32) {
    if (n as i16) > 0 {
        let s = e_from.wrapping_sub(e_to) as i16;
        for i in 0..n as usize {
            dst[i] = (shl_sat((src[i] as i32) << 16, s) >> 16) as i16;
        }
    }
}

/// `0x10319000` — split a value at an exponent into an integer part rounded up
/// and the (negated) fractional remainder.
///
/// `*out_int` is [`scale_ceil16`] of `x` at `exp`, i.e. `ceil(x * 2^(exp-15))`.
/// `*out_frac` is the difference between that integer and the un-rounded value,
/// scaled by 2^15 — a Q15 measure of how much the ceiling over-shot. Used where
/// a harmonic count must be an integer but the residual phase still matters.
pub fn ceil_and_residual(out_int: &mut i16, out_frac: &mut i16, x: i16, exp: i16) {
    let v = (x as i32) << 16;
    let n = (scale_ceil16(v, exp) >> 16) as i16;
    *out_int = n;
    let s = exp.wrapping_sub(0xf);
    let shifted = shift32_al3b2(v, s);
    *out_frac = ((((n as i32) << 16).wrapping_sub(shifted)).wrapping_shl(15) >> 16) as i16;
}

/// `0x103111b0` — average two spectra into the first, clamp, and hold the last
/// value out to 56 entries.
///
/// `a[i] = sat_shl((a[i] + b[i]) / 2 in Q31, 1) >> 16`, then clamped to
/// `[-0x7800, 0x77ff]` — an asymmetric limit leaving about 0.7 dB of headroom
/// below full scale. Entries `n..0x38` are filled with `a[n-1]`, the usual
/// "hold the last harmonic" extension.
///
/// `n` must be at least 1; the original reads `a[n-1]` unconditionally.
pub fn avg_and_extend(a: &mut [i16], b: &[i16], n: u16) {
    if (n as i16) > 0 {
        for i in 0..n as usize {
            let s = (((b[i] as i32) << 16) >> 1).wrapping_add(((a[i] as i32) << 16) >> 1);
            let mut v = (shl_sat(s, 1) >> 16) as i16;
            if v > 0x77ff {
                v = 0x77ff;
            } else if v < -0x7800 {
                v = -0x7800;
            }
            a[i] = v;
        }
    }
    let n = n as i16;
    let last = a[(n - 1) as usize];
    if n < 0x38 {
        for i in n as usize..0x38 {
            a[i] = last;
        }
    }
}

/// `0x10315c60` — nearest codeword by Hamming distance over the 10-bit table
/// at `0x10381a60`.
///
/// Scans entries `from..=to`, skipping any whose top six bits differ from
/// `pattern` (the table stores a 6-bit tag in those bits), and keeps the entry
/// with the fewest differing bits among the low ten. `*best` and `*best_idx`
/// are only updated on a strict improvement, so the caller seeds `*best` with
/// the incumbent distance and ties keep the earlier index.
pub fn nearest_codeword(
    pattern: u16,
    best: &mut i16,
    best_idx: &mut i16,
    from: i16,
    to: i16,
    table: &[u16],
) {
    if from > to {
        return;
    }
    let mut i = from;
    loop {
        let x = table[i as u16 as usize] ^ pattern;
        if (x as u32) & 0xfc00 == 0 {
            let d = popcount_low(x as u32, 10) as i16;
            if d < *best {
                *best = d;
                *best_idx = i;
            }
        }
        if i == to {
            break;
        }
        i = i.wrapping_add(1);
    }
}

/// `0x1030cc90` — build the per-harmonic voiced/unvoiced flag vector from a
/// small configuration block.
///
/// `cfg` is a 16-bit view of a control struct; `dst` is the 56-word flag array
/// the struct points at (`cfg` byte offset `0x80`), passed separately here
/// because the port cannot follow a pointer stored inside an `i16` slice.
///
/// | `cfg[0]` | action |
/// |---|---|
/// | 1 | expand the packed 2-bit band mask at `cfg[4..6]` with [`voicing_expand`] |
/// | 2 | all-unvoiced |
/// | 3 | all-unvoiced, then set one or two flags chosen by frame size |
///
/// In mode 3 the frame-size class of `cfg[1]` decides how `cfg[3]` is read:
/// class 0 uses it as a single index, classes 1–3 as a packed pair of byte
/// indices (`hi = cfg[3] >> 8`, `lo = cfg[3] & 0xff`). Class 4 does nothing.
pub fn voicing_setup(cfg: &[i16], dst: &mut [i16]) {
    match cfg[0] {
        1 => {
            let mask = (cfg[4] as u16 as u32) | ((cfg[5] as u16 as u32) << 16);
            voicing_expand(dst, mask as i32, cfg[6], cfg[2]);
        }
        2 => {
            fill16(dst, 0, 0x38);
        }
        3 => {
            fill16(dst, 0, 0x38);
            let class = frame_size_class(cfg[1]) as i32;
            if class == 0 {
                dst[cfg[3] as usize] = 1;
            } else if ((class - 1) as u32) < 3 {
                let u = cfg[3] as u16;
                dst[(u >> 8) as usize] = 1;
                dst[(u & 0xff) as usize] = 1;
            }
        }
        _ => {}
    }
}

/// `0x103125d0` — look up a pre-baked voicing pattern and expand it to dibits.
///
/// For `mode == 1` the index `(idx << (7 - sh)) as i16` selects a 32-bit
/// pattern from the table at `0x10381720` — all-`0x5` nibbles with voiced bands
/// punched in — which [`expand_dibits`] then stretches to two words of 4-bit
/// symbols. Any other mode clears both outputs.
///
/// The shift count is a full 32-bit `7 - sh` masked to five bits by the
/// hardware, and the shift itself is 16-bit, so a count of 16 or more yields
/// index 0.
pub fn voicing_from_table(
    mode: i16,
    out0: &mut u32,
    out1: &mut u32,
    idx: i16,
    sh: i32,
    table: &[u32],
) {
    if mode == 1 {
        let c = (7i32.wrapping_sub(sh)) as u32 & 0x1f;
        let i = (((idx as u16 as u32) << c) as u16) as i16;
        expand_dibits(out0, out1, table[i as usize]);
    } else {
        *out1 = 0;
        *out0 = 0;
    }
}

/// `0x103179d0` — ten-tap antisymmetric (Hilbert / differentiating) filter over
/// a spectrum, sliding one word per output.
///
/// The taps at `0x10381c18` are `18411/(2k+1)` for `k = 0..9`, i.e. the
/// truncated impulse response of a Hilbert transformer. Each output is
/// `sum_k 2·t_k·(x[b+2+2k] − x[b−2k])` accumulated in a wrapping 32-bit
/// register, then reduced by [`frac_at_exp`] at exponent 1. `b` starts at word
/// `0x13` of `src` and advances by one per output, so `src` is read over words
/// `1 ..= 0x32 + n`.
///
/// Outputs `n..0x38` are zeroed.
pub fn hilbert_diff(out: &mut [i16], src: &[i16], n: u16, taps: &[i16]) {
    if (n as i16) > 0 {
        let mut base = 0x13usize;
        for i in 0..n as usize {
            let mut acc = 0i32;
            for k in 0..10 {
                let t = taps[k] as i32;
                let hi = src[base + 2 + 2 * k] as i32;
                let lo = src[base - 2 * k] as i32;
                acc = acc
                    .wrapping_add(hi.wrapping_mul(t).wrapping_mul(2))
                    .wrapping_sub(lo.wrapping_mul(t).wrapping_mul(2));
            }
            out[i] = (frac_at_exp(acc, 1) >> 16) as i16;
            base += 1;
        }
    }
    let n = n as i16;
    fill16(&mut out[n as usize..], 0, 0x38i16.wrapping_sub(n));
}

/// `0x10313540` — convert a frequency to a harmonic index and return the
/// residual frequency error.
///
/// `1/L` is formed once as `div_norm(0x41000000, 8, L)`. When `enable` is set,
/// `(w/2 + 0x20800000)` is divided by it to give a harmonic number, which is
/// clamped into `[0, min(L-1, cap)]` and stored to `*index`. The return value
/// is `(index · 2 · (1/L)) − 1/L` re-rounded to 16 bits — the offset of that
/// harmonic's centre from the queried frequency.
///
/// With `enable == 0` the stored index is left alone and only the return value
/// is recomputed from it.
///
/// `l` must be non-zero: [`div_norm`] faults on a zero divisor.
pub fn harmonic_index(index: &mut i16, l: i16, w: i16, cap: i16, enable: i16) -> i32 {
    let lv = (l as i32) << 16;
    let n1 = if lv == 0 {
        0u16
    } else {
        norm_exp_l1_batch1(lv) as u16
    };
    let mut e1: i16 = 0;
    let recip = div_norm(
        0x4100_0000,
        8,
        (((lv as u32) << (n1 as u32 & 0x1f)) as i32 >> 16) as i16,
        (0xfi32.wrapping_sub(n1 as i32)) as i16,
        &mut e1,
    );
    let recip_u = recip as u16;

    if enable != 0 {
        let v = (((w as i32) << 16) >> 1).wrapping_add(0x2080_0000);
        let n2 = if v == 0 {
            0u16
        } else {
            norm_exp_l1_batch1(v) as u16
        };
        let mut e2: i16 = 0;
        let r = div_norm(
            ((v as u32) << (n2 as u32 & 0x1f)) as i32,
            (8i32.wrapping_sub(n2 as i32)) as i16,
            recip_u as i16,
            e1,
            &mut e2,
        );
        let sh = shift32_al3b2((r as i32) << 16, e2.wrapping_sub(0xf));
        let mut d = (sh >> 16) as i16;
        if (d as i32) > (l as i32) - 1 {
            d = ((l as i32) - 1) as i16;
        } else if d < 0 {
            d = 0;
        }
        if d > cap {
            d = cap;
        } else if d < 0 {
            d = 0;
        }
        *index = d;
    }

    let base = shift32_al3b2(-0x4100_0000, (-7i16).wrapping_sub(e1));
    let t = ((((*index as i32) << 17).wrapping_add(0x10000)) >> 16) as i16;
    let acc = base.wrapping_add(
        (t as i32)
            .wrapping_mul(recip_u as i16 as i32)
            .wrapping_mul(2),
    );
    shift32_al3b2(acc, e1.wrapping_add(7)).wrapping_add(0x8000) >> 16
}

/// `0x10319e30` — accumulate one harmonic's contribution into the synthesis
/// buffer.
///
/// A 32-bit phase accumulator advances by `step << 5` (the decompilation's
/// `(step << 16) >> 11`) per sample. Its top byte indexes a 257-entry waveform
/// table and the next 15 bits linearly interpolate to the following entry —
/// exactly the "phase masked to 9 bits, cosine table" scheme of SPEC §7.1, one
/// octave finer.
///
/// When `taper_step != 0` the sample is additionally scaled by
/// `(0x8000 − cos(taper_phase)) / 2`, a raised-cosine fade whose own phase
/// advances by `taper_step` per sample — this is the overlap-add ramp.
///
/// The result is shifted by the block exponent `shift` and **saturating**-added
/// into `dst` from index `*cursor`, which is advanced by `n`. Note the
/// saturation limit takes its sign from the *addend*, not from the running
/// sum: `0x7fffffff + (v < 0)`.
#[allow(clippy::too_many_arguments)]
pub fn synth_harmonic(
    phase: &mut i32,
    table: &[i16],
    step: i16,
    taper_phase: i32,
    cursor: &mut usize,
    dst: &mut [i32],
    shift: i16,
    taper_step: i16,
    n: u16,
    cos_tbl: &[i16],
) {
    if (n as i16) <= 0 {
        return;
    }
    let inc = ((step as i32) << 16) >> 11;
    let mut tp = taper_phase;
    let mut p = *cursor;

    for _ in 0..n {
        let acc = *phase;
        *phase = acc.wrapping_add(inc);

        let hi = ((acc as u32) >> 16) as u16;
        let idx = (hi & 0xff) as usize;
        let frac = ((acc.wrapping_sub((hi as i16 as i32) << 16)).wrapping_shl(15) >> 16) as i16;

        let t0 = ((table[idx] as i32) << 16) >> 1;
        let t1 = ((table[idx + 1] as i32) << 16) >> 1;
        let d = (t1.wrapping_sub(t0) >> 16) as i16;
        let mut v = t0.wrapping_add((d as i32).wrapping_mul(frac as i32).wrapping_mul(2));

        if taper_step != 0 {
            let c = cos_lookup(((tp as i32) << 10 >> 16) as i16, cos_tbl) as i16;
            let g = ((0x8000i32.wrapping_sub(c as i32)).wrapping_shl(15) >> 16) as i16;
            let lo = (v as u32 & 0xffff) as i32;
            let vhi = ((v >> 16) as i16) as i32;
            v = (lo.wrapping_mul(g as i32) >> 15)
                .wrapping_add(vhi.wrapping_mul(g as i32).wrapping_mul(2));
            tp = tp.wrapping_add((taper_step as i32) << 16);
        }

        let v = shift32_al3b2(v, shift);
        let cur = dst[p];
        let r = cur.wrapping_add(v);
        dst[p] = if ((r ^ cur) & (r ^ v)) < 0 {
            (0x7fff_ffffu32).wrapping_add((v < 0) as u32) as i32
        } else {
            r
        };
        p += 1;
    }
    *cursor = p;
}

// ======================= from l1_batch2 =======================

/// `0x1e - msb(|x|)` for the **inlined** block-exponent sequence
///
/// ```text
/// mov eax,x ; test x,x ; jns L ; not eax ; L: bsr eax,eax ; mov r,0x1e ; sub r,eax
/// ```
///
/// Note this is *not* the same degenerate case as the library's out-of-line
/// `norm` helper (SPEC §4.1), which leaves its search variable at 31 and so
/// reports −1. Here `bsr` leaves its destination unchanged when the source is
/// zero, and the destination **is** the source — so a zero magnitude gives a
/// bit index of 0 and a shift of 30, not −1. A zero magnitude is only
/// reachable for `x == -1`; every call site guards `x == 0` separately.
pub fn norm_shift_l1_batch2(x: u32) -> i32 {
    0x1e - top_bit_l1_batch5(x)
}

/// The block-exponent helper as the originals spell it inline: zero maps to a
/// shift of zero, everything else to [`norm_shift_l1_batch2`] truncated to 16 bits.
pub fn norm_exp_l1_batch2(x: u32) -> u32 {
    nrm(x) as u32
}

pub fn high16_al2b1(x: i32) -> i16 {
    high16_batch3(x)
}

/// `0x1030a890` — `2^(x · 2^(in_exp-15))`, requantised to `out_exp`.
///
/// `x` is promoted to Q31, aligned by `in_exp - 15`, fed to
/// [`pow2_frac`], and the resulting (mantissa, exponent) pair is shifted to the
/// caller's requested exponent and rounded with `+ 0x8000` before the final
/// `>> 16`. The whole thing is the library's scalar exponential.
pub fn pow2_scaled(x: i16, in_exp: i16, out_exp: i16) -> i32 {
    let v = shift_i32_batch1((x as i32) << 16, in_exp.wrapping_sub(0xf));
    let mut e: i16 = 0;
    let m = pow2_frac(v, &mut e);
    shift_i32_batch1(m, e.wrapping_sub(out_exp)).wrapping_add(0x8000) >> 16
}

/// `0x1030bd40` — write the default contents of the 146-byte gain/AGC control
/// block.
///
/// Four 32-bit fields, then 56 zero words, then `0x38003333`, two zero dwords
/// and three zero words. `st` is the block viewed as `i16`; it must be at least
/// 73 words long. The interesting immediates are `0x00ff0001` (a pair of byte
/// flags plus a 16-bit limit), `0x0000000f`, `0x00004027` and `0x38003333`.
pub fn init_agc_config(st: &mut [i16]) {
    fn wr32(st: &mut [i16], w: usize, v: u32) {
        st[w] = v as u16 as i16;
        st[w + 1] = (v >> 16) as u16 as i16;
    }
    wr32(st, 0, 0x00ff_0001);
    wr32(st, 2, 0x0000_000f);
    wr32(st, 4, 0);
    wr32(st, 6, 0x0000_4027);
    fill16(&mut st[8..], 0, 0x38);
    wr32(st, 64, 0x3800_3333);
    wr32(st, 66, 0);
    wr32(st, 68, 0);
    fill16(&mut st[70..], 0, 3);
}

/// `0x1030cd50` — harmonic/band count for a frame, by mode.
///
/// Mode 0 computes it arithmetically: `ceil(a · 0xa000 · 2^-19) - 1`, floored
/// at 0 (`0xa000 · 2^-19 = 1/12.8`, i.e. the 8 kHz half-band divided by the
/// fundamental). Modes 1..3 instead read a table — which turns out to be the
/// shared cosine table at `0x1037f978` indexed from word 300 — at `b + 2·a`,
/// minus one. Any other mode returns 0.
///
/// `cos_tbl` is the 512-word table at `0x1037f978`.
pub fn band_count_for_mode(mode: i32, a: i16, b: i16, cos_tbl: &[i16]) -> u32 {
    if mode == 0 {
        let r = scale_ceil16((a as i32).wrapping_mul(0xa000), 0xc).wrapping_sub(0x10000);
        if r < 0 {
            return 0;
        }
        return (r >> 16) as u16 as u32;
    }
    if (mode as u32).wrapping_sub(1) < 3 {
        let idx = 300 + b as i32 + (a as i32) * 2;
        return cos_tbl[idx as usize].wrapping_sub(1) as u16 as u32;
    }
    0
}

/// `0x103148e0` — the Q16 ratio `num / den`.
///
/// `den` is block-normalised first, the quotient is taken with
/// [`div32_half`] after a conditional pre-halving of the numerator (so the
/// 16-bit quotient never overflows), and the result is shifted back by the
/// accumulated exponent. Equal operands short-circuit to exactly `0x10000`
/// (1.0). **A zero `den` faults inside the divide.**
pub fn ratio_q16(num: i16, den: i16) -> i32 {
    if num == den {
        return 0x10000;
    }
    let d32 = (den as i32) << 16;
    let n = norm_exp_l1_batch2(d32 as u32);
    let dn = ((d32 as u32) << (n & 0x1f)) as i32;
    let mut acc = (num as i32) << 16;
    let mut e = -4i32;
    if high16_al2b1(dn) <= num {
        acc >>= 1;
        e = -3;
    }
    let q = div32_half(acc, high16_al2b1(dn)) as i16;
    let v = (q as i32) << 16;
    let sh = (e - ((0u32.wrapping_sub(n).wrapping_sub(4) & 0xffff) as i32) - 0xf) as i16;
    shift_i32_batch1(v, sh)
}

/// `0x103173e0` — copy `n` words while rescaling each by `2^sh`.
///
/// A zero shift degenerates to a plain [`copy16_batch3`]. Otherwise each word is
/// promoted to Q31, shifted (right past −30 becoming a sign fill), and the high
/// half stored. As everywhere in this library the count is signed, so a
/// non-positive `n` writes nothing — and in the negative-shift arm the original
/// falls out without even copying.
pub fn shift_copy16(dst: &mut [i16], src: &[i16], n: u16, sh: i16) {
    // The three arms are one expression: `shift_i32_batch1` already saturates
    // the right-shift count at 31, and a zero shift leaves the high half equal
    // to the source word, which is the plain copy.
    if (n as i16) <= 0 {
        return;
    }
    for i in 0..n as usize {
        dst[i] = high16_al2b1(shift_i32_batch1((src[i] as i32) << 16, sh));
    }
}

/// `0x10319080` — `a + 1/d`, split into an integer part and a Q15 remainder.
///
/// `d` is block-normalised, `0x40000000 @ exp 1` (i.e. 1.0) is divided by it
/// with [`div_norm`], the reciprocal is added to the block-floating value
/// `(a, a_exp)`, and the sum is then split: `int_out` is its integer part
/// (via [`hi16_at_exp`], which truncates towards −∞) and `frac_out` the
/// leftover fraction scaled to Q15. This is how a pitch period is turned into
/// an integer lag plus a fractional phase.
///
/// **A zero `d` faults inside the divide.**
pub fn add_recip_split(int_out: &mut i16, frac_out: &mut i16, a: i16, a_exp: i16, d: i16) {
    let v = (d as i32) << 16;
    let n = norm_exp_l1_batch2(v as u32);
    let mut e1: i16 = 0;
    let q = div_norm(
        0x4000_0000,
        1,
        high16_al2b1(((v as u32) << (n & 0x1f)) as i32),
        (-4i32 - n as i32) as i16,
        &mut e1,
    );
    let mut e2: i16 = 0;
    let m = add_norm(a, a_exp, q, e1, &mut e2);
    let acc = (m as i32) << 16;
    let hi = high16_al2b1(hi16_at_exp(acc, e2));
    *int_out = hi;
    let shifted = shift_i32_batch1(acc, e2.wrapping_sub(0xf));
    *frac_out = high16_al2b1(shifted.wrapping_sub((hi as i32) << 16).wrapping_shl(0xf));
}

/// `0x10311910` — resample a 56-entry envelope onto `n` output points by
/// linear interpolation, returning the summed output.
///
/// The phase increment is `(0x40000/n) · ((phase & 0x7f) << 9) / 2^11`, held in
/// Q16 so the high word is the table index and the low word the interpolation
/// fraction. Each output is `t[i-1]·(1-f) + t[i]·f`, formed as
/// `2·(t[i]·f + t[i-1]·2^15) - 2·f·t[i-1]`; the index saturates at `0x37` and an
/// index of exactly zero passes `t[0]` through unscaled. The return value
/// accumulates each output shifted down by 6, i.e. the mean energy of the
/// resampled envelope.
///
/// Domain: the phase must stay non-negative — a negative high word is
/// zero-extended into the table index by the original and reads out of bounds.
pub fn resample_lerp(out: &mut [i16], n: u16, table: &[i16], phase: i16) -> i32 {
    let d = div32_half(0x8_0000, n as i16) as i16 as i32;
    let t = ((((phase as i32) << 0x19) >> 0x10) as i16) as i32;
    let step = d.wrapping_mul(t).wrapping_mul(2) >> 0xc;
    if (n as i16) <= 0 {
        return 0;
    }

    let mut acc = 0i32;
    let mut ph = step;
    for slot in out.iter_mut().take(n as usize) {
        let hi = high16_al2b1(ph);
        let v = if hi >= 0 && hi < 0x37 && hi == 0 {
            (table[0] as i32) << 16
        } else {
            let idx = if hi < 0x37 { hi as u16 as usize } else { 0x37 };
            let f = ((ph >> 1) as u32 & 0x7fff) as i32;
            let prod = f.wrapping_mul(table[idx - 1] as i32).wrapping_mul(2);
            let r = if prod == i32::MIN {
                i32::MAX
            } else {
                prod.wrapping_neg()
            };
            r.wrapping_add(
                (table[idx] as i32)
                    .wrapping_mul(f)
                    .wrapping_add((table[idx - 1] as i32) << 0xf)
                    .wrapping_mul(2),
            )
        };
        ph = ph.wrapping_add(step);
        *slot = high16_al2b1(v);
        acc = acc.wrapping_add(v >> 6);
    }
    acc
}

/// `0x103163b0` — `dst[i] = 2^(src[i] / 1024)`, all brought to one block
/// exponent.
///
/// Each input is scaled by `2^6` and fed to [`pow2_frac`], whose 16-bit
/// mantissa lands in `dst` and whose exponent goes to a scratch array. The
/// largest exponent becomes the block exponent, written to `*out_exp`, and
/// every mantissa is shifted down to it. This is the inverse of the library's
/// log-domain spectral representation.
///
/// Domain: `n` in `1 ..= 56` — the scratch exponent array is 56 words, and with
/// `n <= 0` the original reads it uninitialised.
pub fn pow2_vector(dst: &mut [i16], out_exp: &mut i16, src: &[i16], n: u16) {
    fill16(dst, 0, n as i16);
    let mut exps = [0i16; 56];
    if (n as i16) > 0 {
        for i in 0..n as usize {
            dst[i] = high16_al2b1(pow2_frac(((src[i] as i32) << 16) >> 10, &mut exps[i]));
        }
    }
    let mut mx = exps[0];
    if (n as i16) > 1 {
        for &e in exps.iter().take(n as usize).skip(1) {
            if mx < e {
                mx = e;
            }
        }
    }
    if (n as i16) > 0 {
        for i in 0..n as usize {
            // All three arms collapse to one shift: for `sh < -30` the
            // original's `dst[i] >> 15` is the sign fill, which is exactly the
            // high half of `x >> 31`.
            let x = ((dst[i] as u16 as u32) << 16) as i32;
            let sh = exps[i].wrapping_sub(mx);
            dst[i] = high16_al2b1(shift_i32_batch1(x, sh));
        }
    }
    *out_exp = mx;
}

/// `0x1031a1b0` — one merge stage of the real-input FFT, in place.
///
/// Handles the DC/Nyquist pair specially (three variants, selected by
/// `stage_flag` and `variant`), runs the appropriate half-sum/half-difference
/// butterfly over the interior via [`butterfly_a`]/[`butterfly_b`], applies the
/// twiddled conjugate-symmetric merge [`fft_butterfly_pass`] with stride
/// `0x200 >> log2n`, and finally halves the centre bin and negates-and-halves
/// its imaginary partner. Returns `count + 1` — the stage counter the caller
/// threads through.
///
/// `buf` must hold at least `2^log2n + 2` words. The forward and backward
/// halves of the butterfly are provably disjoint (the forward run ends below
/// `2^log2n / 2` and the backward run starts above it), which is why the buffer
/// can be split for the borrow checker without changing behaviour.
pub fn fft_merge_stage(
    buf: &mut [i16],
    count: i32,
    log2n: i16,
    variant: i16,
    stage_flag: i16,
) -> i32 {
    let size = (1u16 << (log2n as u32 & 0x1f)) as i16;
    let n = ((size >> 2).wrapping_sub(1)) as u16;
    let stride = ((0x200i32 >> (log2n as u32 & 0x1f)) as u16) as i16;
    let sz = size as usize;

    if stage_flag == 0 {
        let a = ((buf[1] as i32) << 16) >> 1;
        let b = ((buf[0] as i32) << 16) >> 1;
        buf[0] = high16_al2b1(b.wrapping_add(a));
        buf[1] = high16_al2b1(b.wrapping_sub(a));
    } else if variant == 0 {
        let a = ((buf[0] as i32) << 16) >> 1;
        let b = ((buf[1] as i32) << 16) >> 1;
        buf[0] = high16_al2b1(b.wrapping_add(a));
        buf[sz] = high16_al2b1(a.wrapping_sub(b));
        buf[sz + 1] = 0;
        buf[1] = 0;
    } else {
        let a = ((buf[0] as i32) << 16) >> 2;
        let b = ((buf[sz] as i32) << 16) >> 2;
        buf[0] = high16_al2b1(b.wrapping_add(a));
        buf[1] = high16_al2b1(a.wrapping_sub(b));
    }

    let half = ((size as i32) >> 1) as usize;
    {
        let (front, back) = buf.split_at_mut(half);
        let p2_start = sz - 2 - half;
        if variant == 0 {
            butterfly_a(&mut front[2..], back, p2_start, n);
        } else {
            butterfly_b(&mut front[2..], back, p2_start, n);
        }
    }
    fft_butterfly_pass(buf, 2, sz - 2, stride, n);

    buf[half] = (((buf[half] as i32) << 16) >> 17) as i16;
    let x = (buf[half + 1] as i32) << 16;
    buf[half + 1] = if x == i32::MIN {
        (i32::MAX >> 17) as i16
    } else {
        (x.wrapping_neg() >> 17) as i16
    };
    count.wrapping_add(1)
}

/// `0x10317b80` — split a harmonic into its voiced and unvoiced gains.
///
/// Given a total amplitude `amp @ 2^amp_exp` and a cross term `cross @
/// 2^cross_exp`, forms the sum and difference of their energies at a common
/// exponent `2·amp_exp + 1`, divides the fixed constant `0x3cd013a9` by
/// `(difference · amp · gain)` and scales the sum and the cross term by that
/// reciprocal. The two results come back as block-floating pairs:
/// `(*v_mant, *v_exp)` for the voiced gain and `(*u_mant, *u_exp)` for the
/// unvoiced one. A difference that cancels exactly yields
/// `(0, 0)` and `(0x7fff, 0)`.
///
/// **The divisor must be non-zero**: if `(difference·amp)·gain` is zero the
/// original normalises zero to zero and divides by it.
pub fn voicing_gains(
    u_mant: &mut i16,
    u_exp: &mut i16,
    v_mant: &mut i16,
    v_exp: &mut i16,
    amp: i16,
    amp_exp: i16,
    cross: i16,
    cross_exp: i16,
    gain: i16,
) {
    let a = amp as i32;
    let common = amp_exp.wrapping_mul(2).wrapping_add(1) as u16;
    let e2 = shift_i32_batch1(
        a.wrapping_mul(a).wrapping_mul(2),
        amp_exp.wrapping_mul(2).wrapping_sub(common as i16),
    );

    let (sum, dif);
    if cross == 0 {
        dif = high16_al2b1(e2);
        sum = dif;
    } else {
        let e3 = shift_i32_batch1(
            (cross as i32).wrapping_mul(cross as i32).wrapping_mul(2),
            cross_exp.wrapping_mul(2).wrapping_sub(common as i16),
        );
        sum = high16_al2b1(e3.wrapping_add(e2));
        dif = high16_al2b1(e2.wrapping_sub(e3));
        if dif == 0 {
            *u_mant = 0;
            *u_exp = 0;
            *v_mant = 0x7fff;
            *v_exp = 0;
            return;
        }
    }

    let den = (high16_al2b1((dif as i32).wrapping_mul(a).wrapping_mul(2)) as i32)
        .wrapping_mul(gain as i32)
        .wrapping_mul(2);
    let nd = norm_exp_l1_batch2(den as u32) as i16;
    let recip = div32_half(
        0x3cd0_13a9,
        high16_al2b1(((den as u32) << (nd as u32 & 0x1f)) as i32),
    ) as i16;
    let e_common = nd
        .wrapping_sub(common as i16)
        .wrapping_sub(amp_exp)
        .wrapping_add(4);

    let pv = (sum as i32).wrapping_mul(recip as i32).wrapping_mul(2);
    let nv = norm_exp_l1_batch2(pv as u32) as i16;
    *v_exp = -nv;
    let mv = high16_al2b1(((pv as u32) << (nv as u32 & 0x1f)) as i32);
    *v_mant = mv;
    if mv != 0 {
        *v_exp = v_exp.wrapping_add(common as i16).wrapping_add(e_common);
    }

    let pu = ((high16_al2b1(
        (high16_al2b1((cross as i32).wrapping_mul(a).wrapping_mul(2)) as i32)
            .wrapping_mul(recip as i32)
            .wrapping_mul(2),
    ) as i32)
        .wrapping_shl(0xf)) as u32;
    let nu = norm_exp_l1_batch2(pu) as i16;
    *u_exp = -nu;
    let mu = high16_al2b1((pu << (nu as u32 & 0x1f)) as i32);
    *u_mant = mu;
    if mu != 0 {
        *u_exp = u_exp
            .wrapping_add(amp_exp)
            .wrapping_add(e_common)
            .wrapping_add(cross_exp)
            .wrapping_add(2);
    }
}

// ======================= from l1_batch3 =======================

/// Block exponent `0x1e - bsr(|x|)`.
///
/// Reproduces the documented quirk: for `x == -1` the magnitude search falls
/// through with its index still at 31, so the answer is `-1` rather than a
/// meaningful shift. Callers guard `x == 0` themselves, exactly as the
/// original does.
#[inline]
pub fn norm_e(x: i32) -> i32 {
    norm_shift_batch1(x as u32)
}

/// `norm_e` in the "u16 field" form the callers use: `0x1e - bsr & 0xffff`,
/// with a zero input short-cut to 0.
#[inline(never)]
pub fn norm_u16(x: i32) -> u32 {
    if x == 0 {
        0
    } else {
        (norm_e(x) as u32) & 0xffff
    }
}

/// MSVC `__allshl` (`0x10305930`): logical left shift of a 64-bit value held
/// in `edx:eax` by `cl`. Not in the ported map — it is a compiler runtime
/// routine, given here because two functions in this batch call it directly.
pub fn allshl_l1_batch3(v: i64, count: u8) -> i64 {
    allshl_l1_batch4(v, count)
}

/// `0x1030c8b0` — copy a decoder parameter block, resetting the phase
/// accumulator.
///
/// The block is 0x43 words: a header of eight words, a 0x38-word harmonic
/// vector at word 8, and a tail. Words 0x40/0x41 (a 32-bit running phase or
/// sample counter) are cleared rather than copied; word 0x42 is copied.
/// Used when a frame's parameters are promoted to "previous frame" state.
pub fn copy_params_reset_phase(dst: &mut [i16], src: &[i16]) {
    // The tail words 0x40..=0x42 are disjoint from the 0..0x40 that
    // `copy_params_l1_batch3` writes, so the order of the two halves is free.
    dst[0x40] = 0;
    dst[0x41] = 0;
    dst[0x42] = src[0x42];
    copy_params_l1_batch3(dst, src);
}

/// `0x1031f990` — copy a decoder parameter block: the eight-word header plus
/// the 0x38-word harmonic vector at word 8.
///
/// Same shape as [`copy_params_reset_phase`] without the phase reset or the
/// word-0x42 tail, i.e. only the 0x40 words that describe the spectrum.
#[inline(never)]
pub fn copy_params_l1_batch3(dst: &mut [i16], src: &[i16]) {
    dst[..4].copy_from_slice(&src[..4]);
    dst[4] = src[4];
    dst[5] = src[5];
    dst[6] = src[6];
    dst[7] = src[7];
    copy16_batch3(&mut dst[8..], &src[8..], 0x38);
}

/// Byte table at `0x1037fea8` — two 17-entry reference sequences used by
/// [`tone_seq_track`], one per mode. Entries are the symbol codes 20/21/22
/// (the trailing zeros at 17..19 are padding between the two rows).
pub const TONE_SEQ: [u8; 0x25] = [
    21, 20, 20, 21, 21, 20, 20, 20, 21, 21, 22, 22, 22, 22, 22, 21, 21, 0, 0, 0, 21, 20, 20, 20,
    20, 20, 21, 21, 22, 22, 21, 21, 22, 22, 22, 21, 21,
];

/// `0x1030d2c0` — track a 17-symbol signalling sequence and drive its
/// countdown.
///
/// `state[0]` is the countdown, `state[1..0x12]` a 17-deep history of received
/// symbols (shifted down by one each call) and `state[0x11]` the newest, taken
/// from `sym`. Mode 1 matches against the first row of [`TONE_SEQ`], any other
/// mode against the row at offset 0x14.
///
/// While the countdown is running and the incoming symbol is in 19..=22, the
/// caller's symbol is *overwritten* with a scripted replacement — `0x15` minus
/// one bit of the mask `0x02b4e4c3` selected by the countdown — and the
/// countdown decrements; the function returns 1 to say it substituted.
/// Otherwise the history is scored against the reference: two or fewer
/// mismatches arms the countdown at 0x20. A symbol of 0xff clears the
/// countdown and is not scored.
///
/// (In the binary the mask is assembled at run time from the four bytes at
/// `0x1037fed0` and XORed with `0x3c670ce3`; the value is `0x02b4e4c3`.)
pub fn tone_seq_track(sym: &mut i16, state: &mut [i16], mode: i16) -> i32 {
    let base = if mode == 1 { 0usize } else { 0x14 };

    state.copy_within(2..2 + 0x10, 1);
    state[0x11] = *sym;
    let s = *sym;
    if s == 0xff {
        state[0] = 0;
        return 0;
    }
    if state[0] > 0 && s > 0x12 && s < 0x17 {
        let sh = (0x20i32 - state[0] as i32) as u32 & 0x1f;
        *sym = 0x15 - ((0x02b4_e4c3i32 >> sh) & 1) as i16;
        state[0] -= 1;
        return 1;
    }
    let mut miss = 0x11i16;
    state[0] = 0;
    for k in 0..0x11usize {
        if (state[1 + k] as u16) == TONE_SEQ[base + k] as u16 {
            miss -= 1;
        }
    }
    if miss < 2 {
        state[0] = 0x20;
    }
    0
}

/// `0x10310f10` — harmonic count `L` from the fundamental.
///
/// `L = floor(242588 / w0)` with a correction: if the reconstructed
/// `(2L+1) * w0` exceeds `0x80000` the count is recomputed as
/// `floor(0x80000 / w0 / 2)` (via `((q << 16) - 1) >> 17`), which is the
/// "no harmonic past 3.8 kHz" rule. Clamped to the codec's range 9..=56.
///
/// `w0` must be non-zero — the division faults otherwise.
pub fn harmonic_count(w0: i16) -> i16 {
    let mut l = div32(0x3b39c, w0) as i16;
    if (l.wrapping_mul(2).wrapping_add(1) as i32) * (w0 as i32) > 0x7ffff {
        let q = div32(0x80000, w0) as i16;
        l = (((q as i32).wrapping_mul(0x10000).wrapping_sub(1)) >> 17) as i16;
    }
    if l > 0x38 {
        return 0x38;
    }
    if l < 9 {
        return 9;
    }
    l
}

/// `0x10311e50` — `a[i] = a[i] + b[i] + k`, in Q17 headroom, clamped to the
/// log-magnitude range.
///
/// Each term is taken to 32 bits and pre-scaled by 1/4 so the three-way sum
/// cannot overflow, the sum is shifted back with saturation, and the result is
/// clamped to `[-0x7800, 0x77ff]` — the same asymmetric envelope the rest of
/// the log-magnitude path uses.
pub fn vec_add3_clamp(a: &mut [i16], b: &[i16], n: u16, k: i16) {
    if (n as i16) <= 0 {
        return;
    }
    for i in 0..n as usize {
        let s = (((a[i] as i32) << 16) >> 2)
            .wrapping_add(((b[i] as i32) << 16) >> 2)
            .wrapping_add(((k as i32) << 16) >> 2);
        let v = high16_al2b1(shl_sat(s, 2));
        a[i] = if v >= 0x7800 {
            0x77ff
        } else if v < -0x7800 {
            -0x7800
        } else {
            v
        };
    }
}

/// `0x10314d80` — normalised first moment of a vector: `sum(w[i]*v[i]) / n^2`
/// with the antisymmetric ramp `w[i] = n + 1 - 2*(i+1)`.
///
/// The weights run from `n-1` down to `-(n-1)` in steps of two, so a flat
/// vector gives zero and the result measures the vector's tilt. Both the
/// numerator and `n^2` are renormalised before a single divide, and the
/// quotient is shifted back by the difference of their exponents; the result
/// is a Q16-ish 32-bit slope.
///
/// Domain: `n >= 1` — `n == 0` divides by zero.
#[inline(never)]
pub fn weighted_slope(v: &[i16], n: i16) -> i32 {
    let mut acc = 0i32;
    let mut i = 0i16;
    if n > 0 {
        while i < n {
            i += 1;
            let w = high16_al2b1(
                ((n.wrapping_add(1) as i32).wrapping_mul(0x10000))
                    .wrapping_add((i as i32).wrapping_mul(-0x20000)),
            );
            acc = acc.wrapping_add(
                (w as i32)
                    .wrapping_mul(v[(i - 1) as usize] as i32)
                    .wrapping_mul(2),
            );
        }
    }
    let na = norm_u16(acc);
    let sq = (n as i32).wrapping_mul(n as i32).wrapping_mul(2);
    let ns = norm_u16(sq);
    let q = div32_half(
        (((acc as u32) << (na & 0x1f)) as i32) >> 1,
        ((((sq as u32) << (ns & 0x1f)) as i32) >> 16) as i16,
    ) as i16;
    let qh = (q as i32) << 16;
    let s = ((0x13u32.wrapping_sub(na) & 0xffff) as i32 - 0x1f) + ns as i32;
    let sw = s as i16;
    if sw >= 0 {
        (((qh as u32) << ((sw as u8) & 0x1f) as u32) as i32) >> 16
    } else if sw < -0x1e {
        qh >> 31
    } else {
        (qh >> ((sw as u8).wrapping_neg() & 0x1f) as u32) >> 16
    }
}

/// `0x10317480` — convert a 32-bit accumulator vector to Q15 words at a
/// common exponent.
///
/// Each word is shifted by `exp_src - exp_dst` with saturation, rounded with
/// `+ 0x8000` (also saturating) and truncated to its high half. This is the
/// standard "collapse a block-floating-point accumulator to Q15" step.
pub fn round_block_i32(dst: &mut [i16], exp_dst: i32, src: &[i32], exp_src: i32, n: u16) {
    if (n as i16) <= 0 {
        return;
    }
    let sh = (exp_src.wrapping_sub(exp_dst)) as i16;
    for i in 0..n as usize {
        let v = shl_sat(src[i], sh);
        let s = v.wrapping_add(0x8000);
        let s = if ((s ^ v) & s) < 0 {
            (0x7fff_ffffu32).wrapping_add((v < 0) as u32) as i32
        } else {
            s
        };
        dst[i] = high16_al2b1(s);
    }
}

/// Read a 32-bit state field held at word `w` of an `i16`-addressed state
/// block (little-endian, low word first).
#[inline]
pub fn rd32_l1_batch3(s: &[i16], w: usize) -> i32 {
    ((s[w] as u16 as u32) | ((s[w + 1] as u16 as u32) << 16)) as i32
}

#[inline]
pub fn wr32_l1_batch3(s: &mut [i16], w: usize, v: i32) {
    s[w] = v as i16;
    s[w + 1] = (v >> 16) as i16;
}

/// `0x103170a0` — dot product of two Q15 vectors with a 64-bit accumulator,
/// returned as a normalised 32-bit mantissa.
///
/// Products are formed as `a*b*2` (wrapping at 32 bits, as the original does)
/// and accumulated in 64 bits, so no dynamic range is lost across a long
/// vector. [`norm64`] then gives the shift that normalises the sum; the
/// negated shift is written to `out_exp` and the shifted low 32 bits are
/// returned. `n` is signed — a non-positive count yields `(0, 0)`.
pub fn dot_norm64(out_exp: &mut i16, b: &[i16], a: &[i16], n: u16) -> i32 {
    let mut acc: i64 = 0;
    if (n as i16) > 0 {
        for i in 0..n as usize {
            let p = (a[i] as i32).wrapping_mul(b[i] as i32).wrapping_mul(2);
            acc = acc.wrapping_add(p as i64);
        }
    }
    let sh = (norm64(acc as u32, (acc >> 32) as u32) as u16) as i16;
    let v = if sh >= 0 {
        allshl_l1_batch3(acc, sh as u8)
    } else {
        allshr(acc, (-(sh as i32)) as u32 as u8)
    };
    *out_exp = sh.wrapping_neg();
    v as i32
}

/// `0x10318330` — window one half-frame out of the analysis ring and refill it
/// with noise.
///
/// Writes `2*n` windowed samples: the first `n` are `ring[i] * win[i]` (rising
/// edge), the second `n` are `ring[i] * win[n-i]` (falling edge, the window
/// read backwards), and the rest of `out` up to `total` words is zeroed. In
/// between, the 0x54-word ring is shifted down by `n` and the vacated tail is
/// refilled from the LCG at `seed` — this is the unvoiced excitation, which is
/// why the noise is reproducible.
///
/// `win` must hold `n+1` words. Domain: `1 <= n <= 0x54`, `total >= 2*n`.
pub fn window_overlap(
    out: &mut [i16],
    seed: &mut i16,
    ring: &mut [i16],
    win: &[i16],
    n: u16,
    total: i16,
) {
    let cnt = n as i16;
    if cnt > 0 {
        for i in 0..n as usize {
            out[i] = high16_al2b1((ring[i] as i32).wrapping_mul(win[i] as i32).wrapping_mul(2));
        }
    }
    let ni = cnt as i32;
    let mut kept = 0i16;
    let mut refill = true;
    if 0x54 - ni > 0 {
        let mut j = 0i32;
        loop {
            kept += 1;
            ring[j as usize] = ring[(j + ni) as usize];
            j = kept as i32;
            if j >= 0x54 - ni {
                break;
            }
        }
        if kept > 0x53 {
            refill = false;
        }
    }
    if refill {
        let mut p = kept as usize;
        let mut left = (0x54i16.wrapping_sub(kept)) as u16;
        loop {
            rand_step(seed);
            ring[p] = *seed;
            p += 1;
            left = left.wrapping_sub(1);
            if left == 0 {
                break;
            }
        }
    }
    if cnt > 0 {
        for i in 0..n as usize {
            out[n as usize + i] = high16_al2b1(
                (ring[i] as i32)
                    .wrapping_mul(win[n as usize - i] as i32)
                    .wrapping_mul(2),
            );
        }
    }
    let tail = (total as i32).wrapping_sub((n as i32).wrapping_mul(2)) as i16;
    fill16(&mut out[2 * n as usize..], 0, tail);
}

/// `0x10319570` — advance the interpolated pitch state by one half-frame.
///
/// The state holds a 32-bit phase/period accumulator at word 0xa8. It is
/// doubled; while its high half is still positive one unit (`0x10000`) is
/// consumed and the remainder divided by `w` to give the fractional position,
/// cached as mantissa/exponent at words 0xaa/0xab. `io` carries the companion
/// accumulator and is advanced by twice the change in the primary one.
///
/// `dbl_sat` is doubled with saturation, `dbl_wrap` doubled with wraparound —
/// the pair is a magnitude and a phase, which is why only one of them clamps.
///
/// Domain: `w != 0` whenever the divide is reached.
pub fn pitch_advance(
    out: &mut i32,
    io: &mut i32,
    dbl_sat: &mut i16,
    dbl_wrap: &mut i16,
    w: i16,
    st: &mut [i16],
) {
    let io0 = *io;
    let prev = rd32_l1_batch3(st, 0xa8);
    let mut v = prev.wrapping_mul(2);
    if high16_al2b1(v) > 0 {
        v = v.wrapping_sub(0x10000);
        let neg = if v == i32::MIN { i32::MAX } else { -v };
        let u = neg >> 1;
        let nu = norm_u16(u);
        let wq = (w as i32) << 16;
        let nw = norm_u16(wq);
        let mut ex = 0i16;
        let q = div_norm(
            ((u as u32) << (nu & 0x1f)) as i32,
            (0x0fu32.wrapping_sub(nu) & 0xffff) as i16,
            high16_al2b1(((wq as u32) << (nw & 0x1f)) as i32),
            (0u32.wrapping_sub(4).wrapping_sub(nw)) as u16 as i16,
            &mut ex,
        );
        st[0xab] = ex;
        st[0xaa] = q;
    }
    wr32_l1_batch3(st, 0xa8, v);
    *out = v;
    *dbl_wrap = high16_al2b1(((*dbl_wrap as i32) << 0x11) as i32);
    *dbl_sat = high16_al2b1(shl_sat((*dbl_sat as i32) << 16, 1));
    *io = rd32_l1_batch3(st, 0xa8).wrapping_add((io0.wrapping_sub(prev)).wrapping_mul(2));
}

/// `0x1031a370` — the complete in-place radix-2 FFT.
///
/// `buf` holds `2^log2n` interleaved `(re, im)` pairs. Before every other pass
/// the block exponent is re-measured with [`cplx_block_exp`]; a block that
/// still has headroom (`exp > 1`) runs the unscaled stage [`fft_stage`],
/// otherwise the halving stage [`fft_butterfly_stage`] runs and the block
/// exponent is credited. The credit is accumulated as the 16-bit pattern
/// `0xffff` per scaled pass — reproduced literally, because it is added into a
/// 32-bit accumulator without sign extension — and subtracted from `exp_in` to
/// give the returned block exponent.
///
/// Structure: [`bit_reverse`], then [`bfly_pairs`] for the first (twiddle-free)
/// pass, then `log2n - 1` twiddled passes with the span doubling and the
/// twiddle stride halving from 0x80.
///
/// Domain: `2 <= log2n <= 7` — the bit-reversal tables cover up to 128 points.
pub fn fft(buf: &mut [i16], exp_in: i32, log2n: i16) -> i32 {
    let size = (1i32 << ((log2n as u8) & 0x1f)) as i16;
    let mut be = cplx_block_exp(buf, size) as i16;
    let mut acc: u32 = 0;
    if be < 2 {
        acc = 0xffff_ffff;
    }
    if be < 0 {
        acc = acc.wrapping_sub(1);
    }
    bit_reverse(buf, size);
    bfly_pairs(buf, size, acc as i16);
    acc &= 0xffff;

    let mut quarter = size >> 2;
    let mut span = 2i32;
    let mut pass = 1i16;
    let mut stride = 0x80i16;
    if log2n > 1 {
        loop {
            be = be.wrapping_sub(2);
            if (pass & 1) == 0 {
                be = cplx_block_exp(buf, size) as i16;
            }
            let scaled = be <= 1;
            let half = (span as i16) as i32;
            if quarter > 0 {
                let mut off = 0usize;
                for _ in 0..quarter as u16 {
                    let (lo, hi) = buf[off..].split_at_mut((half * 2) as usize);
                    if scaled {
                        fft_butterfly_stage(lo, hi, stride, span as i16);
                    } else {
                        fft_stage(lo, hi, stride, span as i16);
                    }
                    off += (half * 4) as usize;
                }
            }
            span = span.wrapping_mul(2);
            pass = pass.wrapping_add(1);
            acc = acc.wrapping_add(if scaled { 0xffff } else { 0 });
            stride >>= 1;
            quarter >>= 1;
            if pass >= log2n {
                break;
            }
        }
    }
    exp_in.wrapping_sub(acc as i32)
}

// ======================= from l1_batch4 =======================

/// `shift_i32_batch1` followed by the library's round-to-`i16` tail
/// (`+ 0x8000`, then `>> 16`). The original leaves the result in `eax`, so the
/// full 32-bit value is returned; callers generally keep only `ax`.
#[inline(never)]
pub fn shift_round16_l1_batch4(v: i32, sh: i16) -> i32 {
    shift_i32_batch1(v, sh).wrapping_add(0x8000) >> 16
}

pub fn high16_al2b3(v: i32) -> i16 {
    high16_batch3(v)
}

#[allow(dead_code)]
/// Saturating 32-bit add, ETSI `L_add` overflow rule, saturating by the sign of
/// the *first* operand.
pub fn sat_add_l1_batch4(a: i32, b: i32) -> i32 {
    sat_add32(a, b)
}

/// `0x1030ce00` — clamp a level in dB to `[-60, +3]` and map it to a linear
/// Q15-ish gain word: `(clamped * 0x1542c00 + 0x6bca01ff) >> 16`, i.e. about
/// `340.17 * dB + 27594`.
///
/// A leaf reached only from the `init` closure, so it is reproduced privately
/// here; it is covered by the differential test of [`gain_for_rate`], its only
/// caller in scope.
pub fn db_to_gain(db: i16) -> i16 {
    let v = if db > 3 {
        3i32
    } else if db < -0x3c {
        -0x3c
    } else {
        db as i32
    };
    (v.wrapping_mul(0x0154_2c00).wrapping_add(0x6bca_01ff) >> 16) as i16
}

/// `0x1030c920` — reset the 0x86-byte adaptive-level tracker to its power-on
/// state.
///
/// Word 0 is a mode/stage marker, words 1..2 a 32-bit counter seeded to
/// `0x000e00ff`, words 4..5 a cleared 32-bit accumulator, word 6 a Q15 level
/// of `0x4000`, words 8..0x3f a 56-entry history primed to `0x6666` (0.8),
/// words 0x40..0x41 a second cleared accumulator and word 0x42 an exponent of
/// -18.
#[inline(never)]
pub fn init_level_tracker(s: &mut [i16]) {
    s[0] = 2;
    s[6] = 0x4000;
    s[7] = 0;
    s[1] = 0x00ff;
    s[2] = 0x000e;
    s[0x40] = 0;
    s[0x41] = 0;
    s[4] = 0;
    s[5] = 0;
    fill16(&mut s[8..], 0x6666, 0x38);
    s[0x42] = -0x12;
}

/// `0x1030d3c0` — map a frame-type code to a starting gain word.
///
/// `code` selects one of five classes: `5..=0x7a` (a normal speech frame),
/// `0x80..=0x8f`, `0x90..=0x9f`, `0xa0..=0xa3`, and everything else. The first
/// four all convert the dB level in `db` through [`db_to_gain`] and return 1;
/// classes 1..3 additionally subtract 0x800 (a fixed 6 dB pad for the
/// non-speech frame types). The last class writes 0 and returns 0, i.e.
/// "no gain available for this frame type".
pub fn gain_for_rate(out: &mut i16, code: u16, db: i16) -> i32 {
    let class: i16 = if code.wrapping_sub(5) < 0x76 {
        0
    } else if code.wrapping_sub(0x80) < 0x10 {
        1
    } else if code.wrapping_sub(0x90) < 0x10 {
        2
    } else {
        3 + (code.wrapping_sub(0xa0) > 3) as i16
    };
    if class != 0 {
        if (class as i32 - 1) < 3 {
            let g = db_to_gain(db) as i32;
            *out = (g.wrapping_mul(0x10000).wrapping_sub(0x07ff_0001) >> 16) as i16;
            return 1;
        }
        *out = 0;
        return 0;
    }
    *out = db_to_gain(db);
    1
}

/// `0x103102e0` — read `n` payload bits through a moving cursor.
///
/// A thin wrapper over `pack_bits` (`0x10310250`) that also advances the
/// caller's cursor by `n` words, so the bit-field parser can walk the
/// one-bit-per-`i16` payload array field by field. `pos` is the cursor in
/// **words**; the original keeps a byte pointer and adds `2 * (i16)n`.
pub fn read_field_l1_batch4(bits: &[u16], pos: &mut i32, n: i16) -> u16 {
    let v = pack_bits(&bits[*pos as usize..], n);
    *pos = pos.wrapping_add(n as i32);
    v
}

/// `0x10310f80` — derive the pitch-search step and the harmonic-count limit
/// from a pitch period.
///
/// `word 6` gets `0x800000 / (4 * period + 0x9e)` halved — the fundamental in
/// the engine's Q units — and `word 2` gets `(period / 4 + 10) * 0.925`
/// clamped to the codec's harmonic range `[9, 56]`. Note `0xece8` is the
/// unsigned Q16 fraction 0.9254, not a negative Q15.
#[inline(never)]
pub fn set_pitch_params(cfg: &mut [i16], period: i16) {
    let den = period.wrapping_mul(4).wrapping_add(1).wrapping_add(0x9d);
    cfg[6] = div32_half(0x0080_0000, den) as i16;
    let l = ((period >> 2).wrapping_add(10) as i32).wrapping_mul(0xece8);
    let mut l = (l >> 16) as i16;
    if l > 0x38 {
        cfg[2] = 0x38;
        return;
    }
    if l < 9 {
        l = 9;
    }
    cfg[2] = l;
}

/// `0x10317780` — leak a tracked level towards a new observation, but only
/// once the voicing pattern is dense enough.
///
/// If at least 8 of the 16 even-numbered bits of `flags` are set, the state
/// becomes `0.9 * state + 0.1 * obs` (`0xe666` and `0xcccc >> 3` are unsigned
/// Q16 fractions); otherwise it is held. The 8-of-16 test uses
/// `popcount_low` (`0x1031ecc0`) over the masked word.
pub fn smooth_level_if_voiced(state: u16, obs: i16, flags: u32) -> u16 {
    if popcount_low(flags & 0x5555_5555, 0x20) >= 8 {
        let a = (obs as i32).wrapping_mul(0xcccc) >> 3;
        let b = (state as i16 as i32).wrapping_mul(0xe666);
        ((a.wrapping_add(b) >> 16) as u32 & 0xffff) as u16
    } else {
        state
    }
}

/// `0x103196a0` — the Q15 ratio `num / den`, both given as plain `i16`.
///
/// Both operands are normalised to Q30 with a -4 exponent bias, divided by
/// `div_norm` (`0x1030a660`), and the quotient is brought back to Q15 by the
/// resulting exponent less 15, with round-to-nearest. **A zero `den` faults**
/// inside `div_norm`, exactly as in the original.
#[inline(never)]
pub fn ratio_q15(num: i16, den: i16) -> i32 {
    let dv = (den as i32) << 16;
    let nd = if dv == 0 {
        0u16
    } else {
        (norm_shift_batch1(dv as u32) as u16) & 0xffff
    };
    let dm = ((dv.wrapping_shl(nd as u32 & 0x1f)) >> 16) as i16;
    let d_bias = (0u16.wrapping_sub(4).wrapping_sub(nd)) as i16;

    let nv = (num as i32) << 16;
    let nn = if nv == 0 {
        0u16
    } else {
        (norm_shift_batch1(nv as u32) as u16) & 0xffff
    };
    let n_exp = (0u16.wrapping_sub(4).wrapping_sub(nn)) as i16;

    let mut e: i16 = 0;
    let q = div_norm(nv.wrapping_shl(nn as u32 & 0x1f), n_exp, dm, d_bias, &mut e);
    shift_round16_l1_batch4((q as i32) << 16, e.wrapping_sub(0xf))
}

/// `0x10312400` — affine rescale of a vector: `v[i] = coef * v[i] + bias`.
///
/// `bias` is `-(2 * coef * (num / (n << 9)))`, i.e. the mean the caller wants
/// removed, computed once by `div32_half` (`0x1030a600`) with `0x80000000`
/// folded to `0x7fffffff`. Each element is then formed in Q31, added to the
/// bias, and passed through `shl_sat(_, 1)` (`0x103101b0`) so the result
/// saturates rather than wrapping. Used to mean-remove and gain-scale a
/// spectral-amplitude vector in place.
pub fn scale_bias_vec(v: &mut [i16], num: i32, n: i16, coef: i16) {
    let den = ((n as i32) << 25) >> 16;
    let q = div32_half(num, den as i16);
    let p = (q as i16 as i32).wrapping_mul(coef as i32).wrapping_mul(2);
    let bias = (if p == i32::MIN {
        i32::MAX
    } else {
        p.wrapping_neg()
    }) >> 1;
    if n > 0 {
        for x in v[..n as u16 as usize].iter_mut() {
            let t = ((*x as i32).wrapping_mul(coef as i32).wrapping_mul(2) >> 1).wrapping_add(bias);
            *x = high16_al2b3(shl_sat(t, 1));
        }
    }
}

/// `0x10314e60` — blend a log-domain energy with a fixed floor to get a
/// voicing/gain decision level.
///
/// The floor is `0x400` unless `cfg[4]` is clear and `thresh <= 0x666`, in
/// which case it ramps as `(0x2c000000 - thresh * 0x64000) >> 16`. The energy
/// term is `log2_q(mant, exp + 30)` (`0x1030b100`) scaled up by
/// `shl_sat(_, 10)` and weighted by `0xc000` — the unsigned Q16 fraction 0.75.
/// A non-positive mantissa substitutes the constant `-10.0` (`0xfff60000`).
pub fn voicing_level(mant: i16, exp: i16, thresh: i16, cfg: &[i16]) -> i32 {
    let floor: i16 = if cfg[4] == 0 && thresh <= 0x666 {
        (0x2c00_0000i32.wrapping_sub((thresh as i32).wrapping_mul(0x64000)) >> 16) as i16
    } else {
        0x400
    };
    let l = if mant < 1 {
        0xfff6_0000u32 as i32
    } else {
        log2_q(mant, exp.wrapping_add(0x1e))
    };
    let h = high16_al2b3(shl_sat(l, 10)) as i32;
    h.wrapping_mul(0xc000).wrapping_add((floor as i32) << 16) >> 16
}

/// `0x10305930` — MSVC's `__allshl`, the mirror of [`allshr`].
/// Register-argument helper (`edx:eax` value, `cl` count); a count of 64 or
/// more clears the whole value.
pub fn allshl_l1_batch4(v: i64, count: u8) -> i64 {
    let hi = (v >> 32) as u32;
    let lo = v as u32;
    if count >= 0x40 {
        return 0;
    }
    if count < 0x20 {
        let n = count & 0x1f;
        let new_hi = if n == 0 { hi } else { hi << n | lo >> (32 - n) };
        return (((new_hi as u64) << 32) | (lo << n) as u64) as i64;
    }
    (((lo << (count & 0x1f)) as u64) << 32) as i64
}

/// `0x103820e8` — `trunc(32768 / i)` for `i = 1..56`, index 0 unused.
///
/// The reciprocal-of-harmonic-count table: `harmonic_dft` uses it both as the
/// per-bin frequency step and as the 1/N scaling of the transform. Entries past
/// 56 are zero (the codec's harmonic count is capped at 56).
pub const RECIP_N: [i16; 57] = [
    0, 32767, 16384, 10922, 8192, 6553, 5461, 4681, 4096, 3640, 3276, 2978, 2730, 2520, 2340, 2184,
    2048, 1927, 1820, 1724, 1638, 1560, 1489, 1424, 1365, 1310, 1260, 1213, 1170, 1129, 1092, 1057,
    1024, 992, 963, 936, 910, 885, 862, 840, 819, 799, 780, 762, 744, 728, 712, 697, 682, 668, 655,
    642, 630, 618, 606, 595, 585,
];

/// `0x1031f9e0` — resample a harmonic-amplitude vector through a cosine
/// transform.
///
/// The `n` input words are copied to a scratch buffer, then each of the first
/// `min(count, n)` outputs is `sum_i x[i] · cos((base + i·step) · 2π/512)`
/// accumulated in 64 bits, block-normalised, scaled by `1/n`
/// (`RECIP_N[n]`) and rounded to Q15 — with `step` growing by `1/n` per output
/// bin and `base = step_acc/2 + 0x40` (the half-bin offset). Written in place.
///
/// `cos_tbl` is the 512-word table at `0x1037f978`. `n` must be in `0..=56`:
/// the original's scratch buffer is 56 words and `RECIP_N` has no entry beyond.
pub fn harmonic_dft(buf: &mut [i16], n: u16, count: i16, cos_tbl: &[i16]) {
    let mut local = [0i16; 64];
    let ni = n as i16;
    copy16_batch3(&mut local, buf, ni);
    let recip = RECIP_N[ni as usize] as i32;
    let cnt = if ni <= count { ni } else { count };
    if cnt <= 0 {
        return;
    }
    let mut step_acc: i32 = 0;
    for k in 0..cnt as u16 as usize {
        let mut acc: i64 = 0;
        let mut phase = (step_acc >> 1).wrapping_add(0x40);
        let step = step_acc as u16 as i16 as i32;
        if ni > 0 {
            for i in 0..n as usize {
                let c = cos_tbl[((phase >> 7) & 0x1ff) as usize] as i32;
                acc = acc.wrapping_add((local[i] as i32).wrapping_mul(c).wrapping_mul(2) as i64);
                phase = phase.wrapping_add(step);
            }
        }
        let s = norm64(acc as u32, (acc >> 32) as u32) as i16;
        let r = if s < 0 {
            allshr(acc, ((s as i32).wrapping_neg() as u32 & 0xff) as u8)
        } else {
            allshl_l1_batch4(acc, (s as u32 & 0xff) as u8)
        };
        let hi = ((r as i32) >> 16) as i16 as i32;
        let t = shift_i32_batch1(
            hi.wrapping_mul(recip).wrapping_mul(2),
            (0u16.wrapping_sub(s as u16)) as i16,
        );
        buf[k] = high16_al2b3(t.wrapping_add(0x8000));
        step_acc = step_acc.wrapping_add(recip);
    }
}

/// `0x103187f0` — accumulate one lag's cross-correlation and roll the history
/// buffer.
///
/// For each of `n` lags the accumulator gets
/// `hist[i]·win[n-i]` at exponent `*hist_exp - 2` plus `win[i]·src[i]` at
/// exponent `exp_new - 2`, both through `shl_sat` (`0x103101b0`) and summed
/// with the ETSI saturating add. The history is then reloaded from the second
/// half of `src` (`hist[0..n] = src[n..2n]`), the remainder of its 0x54-word
/// span is cleared, and `*hist_exp` is updated to `exp_new`.
///
/// `win` is indexed at `n` down to 1 for the history term and 0 up to `n-1` for
/// the current term, so it must hold `n + 1` words.
pub fn correlate_and_shift(
    acc: &mut [i32],
    src: &[i16],
    exp_new: i16,
    hist: &mut [i16],
    hist_exp: &mut i16,
    win: &[i16],
    n: i16,
) {
    if n > 0 {
        for i in 0..n as u16 as usize {
            let a = hist[i];
            let b = win[n as usize - i];
            let p1 = if a == i16::MIN && b == i16::MIN {
                i32::MAX
            } else {
                (a as i32).wrapping_mul(b as i32).wrapping_mul(2)
            };
            let u2 = shl_sat(p1, hist_exp.wrapping_sub(2));

            let c = src[i];
            let d = win[i];
            let p2 = if c == i16::MIN && d == i16::MIN {
                i32::MAX
            } else {
                (d as i32).wrapping_mul(c as i32).wrapping_mul(2)
            };
            let u3 = shl_sat(p2, exp_new.wrapping_sub(2));

            let s = sat_add_l1_batch4(u2, u3);
            acc[i] = sat_add_l1_batch4(s, acc[i]);
        }
    }
    let n_us = n as u16 as usize;
    copy16_batch3(hist, &src[n_us..], n);
    fill16(&mut hist[n_us..], 0, (0x54i32 - n as i32) as i16);
    *hist_exp = exp_new;
}

// ======================= from l1_batch5 =======================

/// Index of the most significant bit of `|x|` — the `bsr` at the heart of every
/// inlined `norm` in this batch.
///
/// **`bsr` leaves its destination unchanged when the source is zero**, and the
/// destination always holds the magnitude that was just computed, so a zero
/// magnitude yields 0. The only input that reaches it is `x == -1` (whose
/// magnitude is `~x == 0`), which therefore normalises by `0x1e - 0 == 30`.
/// Ghidra renders `bsr` as a search loop initialised to `0x1f` and so reports
/// `-1` for that input; the hardware does not agree, and the difference is
/// observable — it shifts a `-1` sum by 30 instead of 31.
pub fn top_bit_l1_batch5(x: u32) -> i32 {
    let mag = if (x as i32) < 0 { !x } else { x };
    if mag == 0 {
        return 0;
    }
    31 - mag.leading_zeros() as i32
}

/// The guarded form every caller in this batch uses: zero normalises to a shift
/// of zero, everything else to `0x1e - bsr(|x|)` truncated to 16 bits.
#[inline(never)]
pub fn nrm(x: u32) -> u16 {
    if x == 0 {
        0
    } else {
        norm_shift_l1_batch2(x) as u16
    }
}

/// Top 16 bits of an accumulator as a signed word.
pub fn high16_l1_batch5(x: i32) -> i16 {
    high16_batch3(x)
}

/// `x << 16`, wrapping — the ubiquitous "promote a word to a Q31 accumulator".
pub fn up16_l1_batch5(x: i16) -> i32 {
    (x as i32).wrapping_shl(16)
}

/// `0x1030b1f0` — `log2(x · 2^-exp)` rendered at a caller-chosen Q point.
///
/// Wraps [`log2_q`] (`0x1030b100`, whose result is Q16.15), shifts it by
/// `15 - q` with saturation, then rounds half-up into the high word. The result
/// is therefore the base-2 logarithm as a 16-bit value with `q` fractional
/// bits, saturated rather than wrapped at the ends.
pub fn log2_at_q(x: i16, exp: i16, q: i32) -> i32 {
    let l = log2_q(x, exp);
    let s = shl_sat(l, (0xf - q) as i16);
    sat_add_l1_batch4(s, 0x8000) >> 16
}

/// `0x10312480` — `(n + 2) / 3`, capped at 12.
///
/// The number of 3-bit groups a field of `n` bits occupies, limited to the
/// twelve the frame layout provides.
#[inline(never)]
pub fn triads_capped(n: i16) -> i16 {
    let q = div32(n as i32 + 2, 3) as i16;
    if q > 0xc {
        0xc
    } else {
        q
    }
}

/// `0x103171d0` — arithmetic mean of `n` words.
///
/// Sums into a 32-bit accumulator (no saturation) and divides by the count with
/// [`div32`]. Ghidra types this `void`; the quotient is left in `eax` and the
/// callers use it. A non-positive `n` sums nothing, and `n == 0` faults in the
/// divide.
pub fn mean_i16(v: &[i16], n: u16) -> i32 {
    let mut acc = 0i32;
    if (n as i16) > 0 {
        for &x in &v[..n as usize] {
            acc = acc.wrapping_add(x as i32);
        }
    }
    div32(acc, n as i16)
}

/// `0x10314ee0` — `log2` of a half-scaled word, as a Q11 result.
///
/// The input is promoted as `x << 15` (half of the usual `<< 16`) and
/// normalised, so the logarithm is taken of `x/2`; the exponent handed to
/// [`log2_q`] is `5 - norm`, which folds in a further 2^10 scale. The final
/// `(r << 13) >> 17` keeps bits 4..19 of the Q16.15 logarithm, i.e. a Q11
/// signed result.
pub fn log2_q11(x: i16) -> i32 {
    let v = (x as i32).wrapping_shl(15);
    let n = nrm(v as u32);
    let l = log2_q(
        high16_l1_batch5(v.wrapping_shl(n as u32 & 0x1f)),
        (5i32 - n as i32) as i16,
    );
    (l.wrapping_shl(0xd)) >> 0x11
}

/// `0x10310310` — unpack `n` bits of `word` at a moving cursor.
///
/// A one-line wrapper over [`unpack_bits`] (`0x10310280`) that keeps the write
/// cursor in memory: the destination pointer is read from `*cursor`, `n` bits
/// are written there most significant first, and the advanced pointer is stored
/// back. This is how the frame packer walks a bit array field by field.
pub fn emit_bits(cursor: &mut usize, dst: &mut [i16], n: i16, word: i16) {
    let at = *cursor;
    let written = unpack_bits(word, &mut dst[at..], n);
    *cursor = at.wrapping_add(written as usize);
}

/// `0x1030cb40` — weight a harmonic magnitude vector by `(k·w0 / wmax)^2`.
///
/// `state[6]` (`+0x0c`) is the fundamental `w0` and `state[8..]` (`+0x10`) the
/// per-harmonic vector. `fmax` is clamped to `0x3333` (0.4 of full scale) and
/// its reciprocal taken once with [`div_norm`]; the loop then walks `k·w0` up to
/// that limit, forming `(k·w0 · 1/wmax)^2` in block floating point and scaling
/// harmonic `k-1` by it. Each accumulation of `w0` saturates.
///
/// The result is the rising `f^2` tilt applied to the spectrum before the
/// voicing decision.
///
/// Domain: `fmax >= 1` (a zero divisor faults) and `state[6] >= 1` (the loop
/// advances by `w0` and never terminates otherwise). The vector must hold
/// `ceil(fmax / w0)` words.
pub fn weight_harmonics_by_freq_sq(state: &mut [i16], fmax: i16) {
    let lim = if fmax > 0x3333 { 0x3333 } else { fmax };
    let v = up16_l1_batch5(lim);
    let n = nrm(v as u32);
    let mut e: i16 = 0;
    let recip = div_norm(
        0x4000_0000,
        1,
        high16_l1_batch5(v.wrapping_shl(n as u32 & 0x1f)),
        (-4i32 - n as i32) as i16,
        &mut e,
    );

    let mut i = 0usize;
    let mut f = state[6];
    while f < lim {
        let p = (recip as i32).wrapping_mul(f as i32).wrapping_mul(2);
        let pn = if p == 0 {
            0
        } else {
            (0x1e - top_bit_l1_batch5(p as u32)) as i16
        };
        let m = high16_l1_batch5(p.wrapping_shl(pn as u32 & 0x1f)) as i32;
        let sh = e.wrapping_sub(pn).wrapping_sub(4).wrapping_mul(2);
        let sq = m.wrapping_mul(m).wrapping_mul(2);
        let g = high16_l1_batch5(shift_i32_batch1(sq, sh));

        state[8 + i] =
            high16_l1_batch5((g as i32).wrapping_mul(state[8 + i] as i32).wrapping_mul(2));
        i += 1;
        f = high16_l1_batch5(sat_add_l1_batch4(
            up16_l1_batch5(state[6]),
            up16_l1_batch5(f),
        ));
    }
}

/// `0x1030d490` — move the "selected harmonic" marker and recompute the
/// fundamental and harmonic count.
///
/// State layout (word indices): `[1]` the pitch numerator, `[3]` the currently
/// selected harmonic index, `[2]` the harmonic count *L*, `[6]` the
/// fundamental, `[8..]` a per-harmonic value array, and a pointer at `+0x80` to
/// a parallel one-hot flag array (`flags`).
///
/// The value stored for the old index is moved to `idx`, its flag cleared and
/// the new one set, then `w0 = state[1] / (idx + 1)` is evaluated in block
/// floating point and brought to Q15 (the `+4` on the exponent is the fixed
/// `0x17 - 0xf - 4` Q-point change). *L* is raised to at least `idx + 1` and
/// clamped to the codec's range `[9, 56]`.
///
/// Domain: `0 <= idx < 56` and `0 <= state[3] < 56`; the original indexes both
/// arrays unguarded, and `idx == -1` would fault in the divide.
pub fn select_harmonic(state: &mut [i16], flags: &mut [i16], idx: i16) {
    let old = state[3] as usize;
    let held = state[8 + old];
    state[8 + old] = 0;
    flags[old] = 0;
    state[3] = idx;
    state[8 + idx as usize] = held;
    flags[state[3] as usize] = 1;

    let num = up16_l1_batch5(state[1]);
    let n1 = nrm(num as u32);
    let den = up16_l1_batch5(state[3].wrapping_add(1));
    let n2 = nrm(den as u32);

    let mut e: i16 = 0;
    let m = div_norm(
        num.wrapping_shl(n1 as u32 & 0x1f),
        (0xf - n1 as i32) as i16,
        high16_l1_batch5(den.wrapping_shl(n2 as u32 & 0x1f)),
        (0x17 - n2 as i32) as i16,
        &mut e,
    );
    e = e.wrapping_add(4);
    state[6] = high16_l1_batch5(shift_i32_batch1(up16_l1_batch5(m), e));

    if state[2] <= state[3] {
        state[2] = state[3].wrapping_add(1);
    }
    if state[2] > 0x38 {
        state[2] = 0x38;
    } else if state[2] < 9 {
        state[2] = 9;
    }
}

/// `0x103178e0` — lay a spectral envelope into the analysis history block and
/// extend it.
///
/// Word `0x13` is zeroed, `src[0..n]` is copied to words `0x14..`, and the
/// remainder out to word `0x53` is filled with a straight ramp falling by
/// `0x5c5` per bin from the last source value (a saturating subtract of
/// `0xfa3b << 16`). Words past `0x54` are then filled by walking *backwards*
/// from word `0x52`, mirroring the tail, and finally words `0x12` down to `0x00`
/// mirror words `0x14` upwards. The result is a symmetric extension on both
/// edges so the following interpolation never reads past the data.
///
/// Domain: `1 <= n <= 0x40`; the state must hold `0x68` words.
pub fn build_spectrum_history(state: &mut [i16], src: &[i16], n: i16) {
    state[0x13] = 0;
    copy16_batch3(&mut state[0x14..], src, n);

    let mut acc = up16_l1_batch5(src[(n - 1) as usize]);
    let first = n.wrapping_add(1);
    if first < 0x41 {
        let count = (0x41u32.wrapping_sub(n as u32 + 1) & 0xffff) as usize;
        let mut at = 0x13 + first as usize;
        for _ in 0..count {
            acc = sat_add_l1_batch4(acc, 0xfa3b_0000u32 as i32);
            state[at] = high16_l1_batch5(acc);
            at += 1;
        }
    }

    let end = n as i32 + 0x14;
    if 0x41 < end {
        let mut read = 0x52usize;
        let mut i = 0x41i32;
        let mut s = 0x41i16;
        loop {
            let v = state[read];
            read -= 1;
            s = s.wrapping_add(1);
            state[0x13 + i as usize] = v;
            i = s as i32;
            if i >= end {
                break;
            }
        }
    }

    for j in 0..0x13usize {
        state[0x12 - j] = state[0x14 + j];
    }
}

/// `0x10318960` — resample a `m+1`-point curve onto `n+1` points by linear
/// interpolation.
///
/// The step `m/n` is formed once with [`div_norm`] and brought to Q16, then
/// walked as a 32-bit accumulator: its high half indexes the source and bits
/// 1..15 are the interpolation fraction. Each output is
/// `high16_l1_batch5(2·(src[i]·(0x8000 - f) + src[i+1]·f))`, i.e. a Q15 crossfade. The
/// endpoints are copied exactly rather than interpolated.
///
/// Domain: `n >= 1`; `n == 0` or `-1` gives a zero divisor. `src` must hold at
/// least `m + 2` words — the last interior step can index `src[m+1]`.
#[inline(never)]
pub fn resample_linear(out: &mut [i16], n: i16, src: &[i16], m: i16) {
    let vn = up16_l1_batch5(n);
    let nn = if vn != 0 { nrm(vn as u32) } else { 0 };
    let vm = up16_l1_batch5(m);
    let nm = nrm(vm as u32);

    let mut e: i16 = 0;
    let q = div_norm(
        vm.wrapping_shl(nm as u32 & 0x1f),
        (nm as i16).wrapping_neg(),
        high16_l1_batch5(vn.wrapping_shl(nn as u32 & 0x1f)),
        (nn as i16).wrapping_neg(),
        &mut e,
    );
    let step = shift_i32_batch1(up16_l1_batch5(q), e.wrapping_sub(0xf));

    out[0] = src[0];
    if n > 1 {
        let mut acc = step;
        for i in 1..n as usize {
            let idx = high16_l1_batch5(acc) as i32;
            let f = ((acc >> 1) as u16 & 0x7fff) as i16 as i32;
            let a = src[idx as usize] as i32;
            let b = src[(idx + 1) as usize] as i32;
            out[i] = high16_l1_batch5(
                b.wrapping_mul(f)
                    .wrapping_add(a.wrapping_mul(0x8000))
                    .wrapping_mul(2)
                    .wrapping_add(a.wrapping_mul(f).wrapping_mul(-2)),
            );
            acc = acc.wrapping_add(step);
        }
    }
    out[n as usize] = src[m as usize];
}

/// `0x10312cb0` — interleave (or de-interleave) a frame's quantiser fields.
///
/// The channel bits are held one per `i16`. In the transmit direction
/// (`store != 0`) the fields are gathered from `buf` into a local frame image
/// and written back; in the receive direction (`store == 0`) `buf` is copied
/// into the image first and the fields are scattered out of it. The image
/// orders the nine fields as `b1, b0, b2, b3, b4, b5, b6, b7, b8` — note `b1`
/// leads, not `b0`.
///
/// The permutation is two-pass and MSB-first: the leading 4 bits of `b0..b5`
/// (8 of `b3`) and the leading 3 of `b6`, `b7` and 1 of `b8` are moved first, in
/// field order, and only then the remainders. That places every field's most
/// significant bits at the front of the channel stream, where the transport's
/// stronger FEC sits.
///
/// `cfg` is the allocation vector the codec builds at `init` (`state+0x7a0`):
/// `cfg[2]` is the total bit count and `cfg[4..13]` are `b0..b8`.
///
/// Domain: the AMBE allocations (49, 48, 47, 46, 45 bits). The IMBE vector
/// (`12, 8, 6, 62` with five trailing zeros) makes the fixed 35-bit first pass
/// exceed the 88-bit payload and reads uninitialised stack.
pub fn frame_bit_permute(buf: &mut [i16], cfg: &[i16], store: i16) {
    let mut img = [0i16; 192];
    let st = store != 0;
    if !st {
        copy16_batch3(&mut img, buf, cfg[2]);
    }

    let (b0, b1, b2, b3) = (cfg[4], cfg[5], cfg[6], cfg[7]);
    let (b4, b5, b6, b7, b8) = (cfg[8], cfg[9], cfg[10], cfg[11], cfg[12]);

    // The eighteen moves as `(image offset, word count)`. Pass 1 is the leading
    // bits of every field in image order; pass 2 the remainders, in a
    // deliberately different order. Building the schedule and running it
    // through one loop is the same eighteen moves in the same order, and saves
    // the eighteen inlined copies of the call setup.
    let p1 = b1;
    let p2 = p1.wrapping_add(b0);
    let p3 = p2.wrapping_add(b2);
    let p4 = p3.wrapping_add(b3);
    let p5 = p4.wrapping_add(b4);
    let p6 = p5.wrapping_add(b5);
    let p7 = p6.wrapping_add(b6);
    let q = b3.wrapping_add(b2).wrapping_add(b0).wrapping_add(b1);
    let q1 = q.wrapping_add(b4);
    let q2 = q1.wrapping_add(b5);
    let q3 = q2.wrapping_add(b6);
    let sched: [(i16, i16); 18] = [
        (0, 4),
        (p1, 4),
        (p2, 4),
        (p3, 8),
        (p4, 4),
        (p5, 4),
        (p6, 3),
        (p7, 3),
        (p7.wrapping_add(b7), 1),
        (b1.wrapping_add(4), b0 - 4),
        (b0.wrapping_add(b1).wrapping_add(4), b2 - 4),
        (4, b1 - 4),
        (q.wrapping_add(4), b4 - 4),
        (
            q.wrapping_sub(b3).wrapping_add(8),
            (b3 as u16).wrapping_sub(8) as i16,
        ),
        (q1.wrapping_add(4), b5 - 4),
        (q2.wrapping_add(3), b6 - 3),
        (q3.wrapping_add(3), b7 - 3),
        (b7.wrapping_add(q3).wrapping_add(1), b8 - 1),
    ];

    let mut cur = 0usize;
    for &(off, n) in sched.iter() {
        let moved = copy16_batch4(&mut buf[cur..], &mut img[off as usize..], n, st);
        cur = cur.wrapping_add(moved as usize);
    }

    if st {
        copy16_batch3(buf, &img, cfg[2]);
    }
}

/// `0x10319760` — solve `y·(2·ac + k·y) = m` for each of `count` harmonics.
///
/// For harmonic `i` the driving term is `m = (i·2^16 - x0)·c`, and the answer is
/// written in block-floating form as
/// `y = m / (ac + sqrt((ac)^2 + k·m))` — the numerically stable root of the
/// quadratic, formed with [`add_norm`], [`sqrt_norm`] and [`div_norm`] so every
/// intermediate keeps its own exponent. Results at or above `c` are replaced by
/// `c` itself, normalised, which is the saturation of the ramp.
///
/// `out[i]` receives the mantissa and `exps[i]` the exponent, both for harmonic
/// `i + 1`; a non-positive `count` writes nothing.
///
/// Domain: the denominator `ac + sqrt(...)` must be non-zero, which in practice
/// means `a` and `c` non-zero; a zero denominator faults in the divide.
#[allow(clippy::too_many_arguments)]
pub fn solve_quad_ramp(
    out: &mut [i16],
    exps: &mut [i16],
    x0: i32,
    a: i16,
    k: i16,
    c: i16,
    count: u16,
) {
    if (count as i16) <= 0 {
        return;
    }
    let a2 = (a as i32).wrapping_mul(a as i32).wrapping_mul(2);
    let c32 = (c as i32).wrapping_mul(0x10000);
    let ac = (a as i32).wrapping_mul(c as i32).wrapping_mul(2);
    let c2 = (c as i32).wrapping_mul(c as i32).wrapping_mul(2);

    let mut i: i16 = 0;
    for slot in 0..count as usize {
        i = i.wrapping_add(1);
        let t = (i as i32).wrapping_mul(0x10000).wrapping_sub(x0);
        let n1 = nrm(t as u32);
        let u = (high16_l1_batch5(t.wrapping_shl(n1 as u32 & 0x1f)) as i32)
            .wrapping_mul(c as i32)
            .wrapping_mul(2);
        let n2 = nrm(u as u32);
        let e_m = ((0xfu32.wrapping_sub(n1 as u32) & 0xffff) as i32)
            .wrapping_add(0x10i32.wrapping_sub(n2 as i32));

        let n3 = nrm(a2 as u32);
        let n4 = nrm(c2 as u32);
        let prod = (high16_l1_batch5(c2.wrapping_shl(n4 as u32 & 0x1f)) as i32)
            .wrapping_mul(high16_l1_batch5(a2.wrapping_shl(n3 as u32 & 0x1f)) as i32)
            .wrapping_mul(2);
        let n5 = nrm(prod as u32);
        let m = high16_l1_batch5(u.wrapping_shl(n2 as u32 & 0x1f)) as i32;

        let mut e = ((n3 as u32).wrapping_neg().wrapping_sub(8) & 0xffff)
            .wrapping_add(0x1eu32.wrapping_sub(n4 as u32))
            .wrapping_sub(n5 as u32) as i16;

        let kw = (k as i32).wrapping_mul(m).wrapping_mul(2);
        let n6 = nrm(kw as u32);
        let s = add_norm(
            high16_l1_batch5(prod.wrapping_shl(n5 as u32 & 0x1f)),
            e,
            high16_l1_batch5(kw.wrapping_shl(n6 as u32 & 0x1f)),
            e_m.wrapping_sub(n6 as i32).wrapping_sub(4) as i16,
            &mut e,
        );

        let r = sqrt_norm(up16_l1_batch5(s), &mut e);
        let n7 = if r != 0 { nrm(r as u32) } else { 0 };
        e = e.wrapping_sub(n7 as i16);

        let n8 = nrm(ac as u32);
        let g = add_norm(
            high16_l1_batch5(ac.wrapping_shl(n8 as u32 & 0x1f)),
            (0xbu32.wrapping_sub(n8 as u32)) as i16,
            high16_l1_batch5(r.wrapping_shl(n7 as u32 & 0x1f)),
            e,
            &mut e,
        );

        let mut v = div_norm(m.wrapping_mul(0x10000), e_m as i16, g, e, &mut e);
        let scaled = shift_i32_batch1(up16_l1_batch5(v), e.wrapping_sub(0xf));
        if scaled.wrapping_add((c as i32).wrapping_mul(-0x10000)) >= 0 {
            let n9 = nrm(c32 as u32);
            e = (0xfu32.wrapping_sub(n9 as u32) & 0xffff) as i16;
            v = high16_l1_batch5(c32.wrapping_shl(n9 as u32 & 0x1f));
        }
        out[slot] = v;
        exps[slot] = e;
    }
}

pub mod tables {
    #[allow(unused_imports)]
    pub use crate::engine::tables::{COS512, TWIDDLE32 as DFT32_TW};

    /// `0x10381920` — dibit allocation for `voicing_pattern`: six rows of eight,
    /// indexed by `nbits - 3`. Each row says how many of the eight output dibits
    /// each of the `nbits` input bits owns; every row sums to 8.
    pub static BIT_ALLOC: [i16; 48] = [
        3, 2, 3, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 1, 2, 1, 2, 0, 0, 0, 1, 1, 2, 1, 1, 2,
        0, 0, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    ];

    /// `0x10381980` — the 56 pre-baked voicing patterns used for index < 0x38.
    pub static VOICING_LUT: [u32; 56] = [
        0x00020000, 0x00080000, 0x000a0000, 0x00200000, 0x002a0000, 0x00aa0000, 0x02000000,
        0x02800000, 0x00008000, 0x00000280, 0x00000200, 0x00000aa8, 0x00000800, 0x00000a00,
        0x00000a80, 0x000002a0, 0x00000002, 0x0000002a, 0x00000008, 0x00000020, 0x0000000a,
        0x000000aa, 0x00000080, 0x000000a0, 0x80000000, 0x0a800000, 0xa8000000, 0x02a00000,
        0xa0000000, 0xaa000000, 0x02aa0000, 0x0a000000, 0xaaaa0000, 0x00800000, 0x0aaa0000,
        0xaa800000, 0x08000000, 0xaaa00000, 0x2aaa0000, 0xaaa80000, 0x0000aaaa, 0x0000aa80,
        0x0000a800, 0x00002aaa, 0x0000a000, 0x000002aa, 0x00000aaa, 0x0000aaa0, 0xaaaaaaaa,
        0xaaaa0aaa, 0xa800a800, 0x02aa02aa, 0x0aaa0aaa, 0xaaaaa800, 0xa000a000, 0x08000800,
    ];

    /// `0x10381cd8` — incremental swap list for the 8-point bit reversal.
    pub static BITREV8: [i16; 5] = [2, 2, 8, 3, 3];

    /// `0x10381ce4` — incremental swap list for the 16-point bit reversal.
    pub static BITREV16: [i16; 13] = [6, 2, 16, 1, -9, 1, 15, 3, -5, 3, 7, 7, -3];

    /// `0x10381d00` — incremental swap list for the 32-point bit reversal.
    pub static BITREV32: [i16; 25] = [
        12, 2, 32, 1, -17, 1, 31, 3, -9, 1, -17, 1, 31, 3, -21, 3, 15, 3, -9, 3, 15, 7, -11, 7, 7,
    ];

    /// `0x10381d38` — incremental swap list for the 64-point bit reversal.
    pub static BITREV64: [i16; 57] = [
        28, 2, 64, 1, -33, 1, 63, 1, -81, 1, 63, 1, -33, 1, 63, 3, -41, 1, -33, 1, 63, 3, -17, 1,
        -33, 1, 63, 3, -53, 3, 31, 3, -17, 1, -33, 1, 63, 3, -41, 3, 31, 3, -17, 3, 31, 7, -27, 3,
        -17, 3, 31, 7, -9, 7, 15, 15, -5,
    ];

    /// `0x10381db0` — incremental swap list for the 128-point bit reversal.
    pub static BITREV128: [i16; 113] = [
        56, 2, 128, 1, -65, 1, 127, 1, -161, 1, 127, 1, -65, 1, 127, 3, -81, 1, -65, 1, 127, 1,
        -161, 1, 127, 1, -65, 1, 127, 3, -105, 1, -65, 1, 127, 3, -33, 1, -65, 1, 127, 3, -81, 1,
        -65, 1, 127, 3, -33, 1, -65, 1, 127, 3, -117, 3, 63, 3, -33, 1, -65, 1, 127, 3, -81, 3, 63,
        3, -33, 1, -65, 1, 127, 3, -105, 3, 63, 3, -33, 3, 63, 3, -81, 3, 63, 3, -33, 3, 63, 7,
        -59, 3, -33, 3, 63, 7, -17, 3, -33, 3, 63, 7, -41, 7, 31, 7, -17, 7, 31, 15, -21, 15, 15,
    ];
}
