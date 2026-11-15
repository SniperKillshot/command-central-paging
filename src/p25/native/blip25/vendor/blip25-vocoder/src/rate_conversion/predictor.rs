// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! Cross-rate magnitude predictor (US7634399 / AMBE-3000 rate-converter §4.5).
//!
//! The rate converter maintains a dedicated prior-frame magnitude
//! predictor state, distinct from both the source decoder's intra-rate
//! predictor and the target encoder's intra-rate predictor. Each frame,
//! the converter blends `0.65 · resampled_prior + 0.35 · current`
//! before handing the magnitudes to the target encoder. When
//! `|R_prev − 1| ≤ 0.01` (the target pitch hasn't shifted more than 1%
//! frame-to-frame), the fast path passes `current` through unchanged.
//!
//! Source: the AMBE-3000 rate-converter description, §4.5 (blend rule)
//! and §5 (state table). The separation is load-bearing: sharing this
//! state with the source decoder's or the target encoder's own predictor
//! would feed each rate's prediction residual into the other's.
//!
//! ## Why this matters
//!
//! Without the cross-rate predictor, long sustained transcoded vowels
//! pick up frame-to-frame magnitude instability on the order of 50
//! frames per US7634399 col. 7 — the AR loop between source decoder
//! and target encoder is undamped. The 0.65 blend is a first-order
//! low-pass filter applied at the rate-boundary seam.

use crate::halfrate::frame::AMBE_PITCH_TABLE;
use crate::mbe_params::{MbeParams, L_MAX};

/// Weight the cross-rate magnitude blend gives the **previous** frame's
/// absolute level. **Zero: the blend is off, deliberately.**
///
/// US7634399 col. 7 line 10 specifies 0.65 here. That weight is disabled
/// because it measures as the dominant source of conversion error, not because
/// the spec is misread. The blend
/// state stores its own output, so mixing 65% of the prior frame's level into
/// each frame put loudness through a feedback loop and compressed the dynamic
/// range: regressing our gain target on the reference converter's gave a slope
/// of 0.84 instead of 1.0, making quiet frames up to +9.7 dB loud and loud
/// frames up to −1.6 dB quiet. At 0.0 the slope is 0.997 (R² 0.994), speech
/// log-spectral distortion from a native encode falls 8.54 → 6.50 dB
/// (full→half) and 8.35 → 6.44 dB (half→full) against a 6.19 dB floor, and the
/// gain field's disagreement with the reference converter falls from 77.9% of
/// frames to 35.2%. Per this project's oracle-over-spec rule, the measurement
/// wins.
///
/// **Zero is also safe on sustained material**, which the numbers above cannot
/// show on their own — they are short-clip distortion plus a three-stem oracle,
/// and neither instrument can see a slow instability. Measured by
/// `conformance/roundtrip/examples/cross_rate_stability.rs` on five sustained
/// stimuli (held vowel, slow vibrato, ±10 dB swell, 100→200 Hz sweep, long tone)
/// in both directions at 12.8 s, 40 s and 100 s: no divergence anywhere, worst
/// slow level wander 3.8 dB, and the ratio of last-quarter to first-quarter
/// wander at or below 1.05 in every run — error shrinks over a long note rather
/// than growing.
///
/// The blend cannot run away at any weight below 1, so no weight in range —
/// zero included — is itself an instability risk. Its recursion carries the §4.3.3
/// energy-preserving `+ 0.5·log₂(R_prev)` offset, so the level it converges on
/// telescopes to `√(ω̂₀(k)/ω̂₀(0))` — bounded by the pitch grid — and every
/// fast-path frame re-anchors the state to the current frame outright. The same
/// harness confirms this directly: a ρ = 0.99 control, near-pure feedback, does
/// not diverge either.
///
/// What a nonzero weight does buy is a small reduction in frame-to-frame
/// envelope jitter — excess envelope jitter 0.96 → 0.73 dB half→full on slow
/// vibrato, over the harness's 40 s run. [`CrossRateBlend::ShapeOnly`] delivers
/// that same reduction while leaving level tracking where ρ = 0 leaves it; the
/// level blend degrades it, giving excess level jitter −0.91 dB against
/// shape-only's +0.18 dB on the swell, where 0.36 dB is the ρ = 0 value and
/// negative means the conversion is smoother than the audio it is tracking. So a
/// shape-only weight is the form to reach for if envelope jitter ever needs
/// damping; a return to 0.65 as a *level* blend is not.
pub const RHO_CROSS_RATE: f64 = 0.0;

/// How the cross-rate predictor mixes the prior frame into the current one.
///
/// The two variants differ in what the mix is allowed to move. [`Self::Level`]
/// is the US7634399 form: a weight on the prior frame's resampled magnitudes,
/// absolute level included, which is what put loudness through a feedback loop
/// and is why [`RHO_CROSS_RATE`] ships at 0. [`Self::ShapeOnly`] mixes the
/// prior frame's spectral *shape* and then restores the frame's own level, so it
/// can damp envelope jitter without touching the gain the target quantizer
/// encodes.
///
/// Shipped behaviour is [`Self::default`], which is `Level(RHO_CROSS_RATE)` —
/// inert. The non-default variants exist so a harness can A/B them; see
/// `conformance/roundtrip/examples/cross_rate_stability.rs`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum CrossRateBlend {
    /// Weight on the prior frame's resampled magnitudes, level included.
    Level(f64),
    /// Weight on the prior frame's spectral shape, with the frame's own level
    /// restored afterwards.
    ///
    /// "Level" here is the mean `log₂` magnitude across the frame's harmonics.
    /// That is exactly the half-rate gain `γ̃`: `m_tilde_to_lambda` differs from
    /// `log₂ M̃_l` only by a per-harmonic constant that depends on the voicing
    /// flag and `ω̂₀`, and the blend changes neither, so holding the mean
    /// `log₂ M̃_l` holds `γ̃` to the bit. The full-rate gain is a different
    /// functional of the same magnitudes, so there this is a close proxy rather
    /// than an identity.
    ShapeOnly(f64),
}

impl CrossRateBlend {
    /// Weight this mode puts on the prior frame. Zero means the mix is inert
    /// and the slow path is skipped entirely.
    #[inline]
    pub fn rho(self) -> f64 {
        match self {
            Self::Level(rho) | Self::ShapeOnly(rho) => rho,
        }
    }
}

impl Default for CrossRateBlend {
    fn default() -> Self {
        Self::Level(RHO_CROSS_RATE)
    }
}

/// Threshold on `|R_prev − 1|` below which the blend falls through to
/// the current frame's magnitudes unchanged (§4.5 fast path).
/// US7634399 col. 5 line 50 + col. 7 line 12.
pub(crate) const R_FAST_PATH_THRESHOLD: f64 = 0.01;

/// Minimum magnitude used when taking `log2` (guards against `log2(0)`).
/// Matches the convention in
/// the AMBE-3000 Decoder Implementation Spec §5.4.
const LOG2_FLOOR_INPUT: f64 = 1e-10;

/// Dedicated predictor state for one cross-rate conversion direction.
/// See the AMBE-3000 Rate Converter Implementation Spec §5 table.
#[derive(Clone, Debug)]
pub struct CrossRatePredictorState {
    /// Previous-frame target-rate magnitudes `M̃_l_B(−1)` for
    /// `l = 1..=56`, 1-indexed (slot `[0]` unused, per the rest of
    /// the codebase convention). Init 1.0 per §5.
    m_tilde_prev: [f32; L_MAX as usize + 1],
    /// Previous-frame target-rate harmonic count `L̂_B(−1)`. Init 30.
    l_prev: u8,
    /// Previous-frame target-rate fundamental `ω̂₀_B(−1)`. Init
    /// `AMBE_PITCH_TABLE[30].omega_0` per §5 Annex-L-row-30 rule.
    omega_0_prev: f32,
}

impl CrossRatePredictorState {
    /// Cold-start per §5 init column.
    pub fn new() -> Self {
        Self {
            m_tilde_prev: [1.0; L_MAX as usize + 1],
            l_prev: 30,
            omega_0_prev: AMBE_PITCH_TABLE[30].omega_0,
        }
    }

    /// Access the stored prior-frame magnitudes (slot `[0]` is unused).
    #[cfg(test)]
    pub(crate) fn m_tilde_prev(&self) -> &[f32; L_MAX as usize + 1] {
        &self.m_tilde_prev
    }

    /// Access the stored prior-frame harmonic count.
    #[cfg(test)]
    pub(crate) fn l_prev(&self) -> u8 {
        self.l_prev
    }

    /// Access the stored prior-frame fundamental.
    #[cfg(test)]
    pub(crate) fn omega_0_prev(&self) -> f32 {
        self.omega_0_prev
    }

    /// Reset to cold-start values. Called on extended mute / invalid
    /// sequences per §6.3 of the rate-converter spec — the spec says
    /// "state is stale", but empirically resetting cleanly avoids
    /// stuck-predictor artifacts when the source recovers.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for CrossRatePredictorState {
    fn default() -> Self {
        Self::new()
    }
}

/// Mean `log₂` magnitude across a frame's harmonics — the frame's level in the
/// domain the target quantizer's gain lives in. Empty input reports 0.0, which
/// makes the shape-only renormalisation a no-op rather than a NaN.
fn mean_log2(m: &[f32]) -> f64 {
    if m.is_empty() {
        return 0.0;
    }
    let sum: f64 = m
        .iter()
        .map(|&v| f64::from(v).max(LOG2_FLOOR_INPUT).log2())
        .sum();
    sum / m.len() as f64
}

/// Apply the cross-rate magnitude blend per §4.5 in-place on
/// `params.amplitudes_slice()` and advance `state` to reflect this
/// frame as `(−1)` for next-frame blending.
///
/// `target_omega_0` and `target_l` are the post-pitch-snap fundamental
/// and harmonic count the target encoder will actually commit to
/// (i.e. the Annex-L-dequantized `ω̂₀_B(0)` for half-rate targets). The
/// caller is responsible for supplying these — they differ from
/// `params.omega_0()` / `params.harmonic_count()` when the pitch grid
/// transform changes them.
///
/// Returns an updated `MbeParams` carrying the blended amplitudes.
/// Voicing passes through unchanged for harmonics the source carries;
/// harmonics past the source's last one inherit the source edge
/// harmonic's voicing, matching the magnitude the extension band holds.
pub fn blend(
    params: &MbeParams,
    target_omega_0: f32,
    target_l: u8,
    state: &mut CrossRatePredictorState,
) -> MbeParams {
    blend_with(
        params,
        target_omega_0,
        target_l,
        state,
        CrossRateBlend::default(),
    )
}

/// [`blend`] with the mix mode injected.
///
/// Exists so the slow path stays testable and so a stability harness can A/B a
/// nonzero weight. [`RHO_CROSS_RATE`] ships at 0.0, which short-circuits that
/// path entirely, so a test calling [`blend`] cannot distinguish correct
/// resampling arithmetic from none at all — it would pass with the whole branch
/// deleted. Tests drive this directly at a nonzero weight so the resample and
/// its Eq. 186/187 edge holds stay guarded for whenever a nonzero weight is
/// wanted.
pub(crate) fn blend_with(
    params: &MbeParams,
    target_omega_0: f32,
    target_l: u8,
    state: &mut CrossRatePredictorState,
    mode: CrossRateBlend,
) -> MbeParams {
    debug_assert!(target_l as usize <= L_MAX as usize);

    let omega_0_curr = f64::from(target_omega_0);
    let omega_0_prev = f64::from(state.omega_0_prev);
    // Guard: omega_0_prev is init'd to a positive Annex L entry and
    // updated only from target_omega_0 (> 0), so division is safe.
    let r_prev = omega_0_curr / omega_0_prev.max(1e-20);

    let l_curr = target_l;
    let amps_curr = params.amplitudes_slice();
    let l_source = amps_curr.len().min(l_curr as usize);
    let mut blended = [0f32; L_MAX as usize];
    // For target harmonics l ≤ min(L_source, L_target), seed with the
    // source magnitude (§4.3.1 fast path: no resampling).
    blended[..l_source].copy_from_slice(&amps_curr[..l_source]);
    // For l > L_source, hold the source's last real magnitude rather
    // than leaving the slot at zero.
    //
    // The edge hold is load-bearing, not cosmetic. An exact zero
    // reaches the target quantizer's `m_tilde_to_lambda`, which maps
    // `m ≤ 0` to Λ = −1000; because γ is the *mean* of Λ across the
    // frame, a single such slot out of ~30 drags the frame's gain index
    // to the codebook floor. Holding the edge value is the same
    // out-of-range convention BABA-A Eq. 187 uses for the prior frame
    // and that `DecoderState::prev_lambda_at` already implements.
    if let Some(&edge) = amps_curr.get(l_source.wrapping_sub(1)) {
        for slot in blended[l_source..l_curr as usize].iter_mut() {
            *slot = edge;
        }
    }

    // ρ = 0 short-circuits the whole slow path: its result is scaled by
    // ρ, so at ρ = 0 every value it computes is discarded. That saves the work
    // on the ~56% of frames that take this branch, and closes a NaN path —
    // `0.0 * f32::INFINITY` is NaN, so a non-finite prior magnitude would
    // otherwise poison a frame whose blend is supposed to be inert.
    let rho = mode.rho();
    if rho > 0.0 && (r_prev - 1.0).abs() > R_FAST_PATH_THRESHOLD {
        // Shape-only mode needs the frame's level *before* the mix, taken from
        // the current-frame magnitudes the ρ = 0 path would have emitted.
        let level_before = match mode {
            CrossRateBlend::ShapeOnly(_) => Some(mean_log2(&blended[..l_curr as usize])),
            CrossRateBlend::Level(_) => None,
        };
        // Slow path: resample prior magnitudes onto current target
        // grid via log-domain linear interpolation per §4.3.2, with
        // §4.3.3 energy-preserving offset `+ 0.5·log₂(R_prev)`.
        let energy_offset = 0.5 * r_prev.log2();
        let mut log_m_prev = [0.0_f64; L_MAX as usize + 1];
        for l in 1..=state.l_prev as usize {
            let m = f64::from(state.m_tilde_prev[l]).max(LOG2_FLOOR_INPUT);
            log_m_prev[l] = m.log2();
        }

        for l_b in 1..=l_curr as usize {
            // Map target harmonic l_b to its fractional position on the
            // prior-frame grid. The spec's R is the source→target pitch
            // shift `ω̂₀_B(0) / ω̃₀_A`; the resample inside the predictor
            // instead runs against the prior *target* frame, so the ratio
            // is `R_prev = ω̂₀_B(0) / ω̂₀_B(−1)` and the prior-frame
            // harmonic lining up with l_b sits at `l_b · R_prev`.
            let l_prev = (state.l_prev as usize).max(1);
            let src_pos = (l_b as f64) * r_prev;
            let prev_log = if src_pos < 1.0 {
                // Below the prior grid — hold the first prior harmonic
                // (BABA-A Eq. 186: Λ̃₀(−1) = Λ̃₁(−1)).
                log_m_prev[1]
            } else if src_pos > l_prev as f64 {
                // Above the prior grid — hold the last prior harmonic
                // (BABA-A Eq. 187: Λ̃_l(−1) = Λ̃_L̃(−1)(−1) for l > L̃(−1)).
                log_m_prev[l_prev]
            } else {
                let pos_lo = src_pos.floor() as usize;
                let pos_hi = (pos_lo + 1).min(l_prev);
                let frac = src_pos - pos_lo as f64;
                let v_lo = log_m_prev[pos_lo];
                let v_hi = log_m_prev[pos_hi];
                v_lo + frac * (v_hi - v_lo)
            };
            let prev_resampled_log = prev_log + energy_offset;
            // 0.65 · prior_resampled + 0.35 · current, in linear domain.
            let prior_linear = (2.0_f64).powf(prev_resampled_log) as f32;
            let curr = blended[l_b - 1];
            blended[l_b - 1] = rho as f32 * prior_linear + (1.0 - rho as f32) * curr;
        }

        // Shape-only: put the frame back on its own level. A single log-domain
        // offset is a per-frame gain, so it cannot alter the shape the mix just
        // produced — only where that shape sits.
        if let Some(before) = level_before {
            let after = mean_log2(&blended[..l_curr as usize]);
            let scale = (2.0_f64).powf(before - after) as f32;
            for slot in blended[..l_curr as usize].iter_mut() {
                *slot *= scale;
            }
        }
    }
    // Fast path (|R−1| ≤ 0.01): blended already holds current amps; no
    // mix. State still advances (next frame's R_prev is computed
    // against this frame's ω̂₀_B regardless of whether the blend fired).

    // Build result MbeParams. Extend voicing to cover the target
    // harmonic range: harmonics past source L_A inherit the source's
    // last real harmonic's voicing.
    //
    // The flag has to agree with the magnitude the slot carries.
    // `m_tilde_to_lambda` scales an *unvoiced* harmonic by
    // 0.2046/√ω̂₀ (≈ 0.53) before taking log₂, so an extension slot
    // holding the edge magnitude but flagged unvoiced reports a Λ about
    // 5 dB above the edge harmonic it copied. γ is the *mean* of Λ over
    // the frame, so a mismatched fill flag moves the frame's gain index
    // on its own. Inheriting keeps the pair (magnitude, voicing)
    // self-consistent with the harmonic being extended.
    let voiced_src = params.voiced_slice();
    let mut voiced_out = [false; L_MAX as usize];
    let l_voiced = voiced_src.len().min(l_curr as usize);
    voiced_out[..l_voiced].copy_from_slice(&voiced_src[..l_voiced]);
    if let Some(&edge_voiced) = voiced_src.get(l_voiced.wrapping_sub(1)) {
        for slot in voiced_out[l_voiced..l_curr as usize].iter_mut() {
            *slot = edge_voiced;
        }
    }
    let result = MbeParams::new(
        target_omega_0,
        l_curr,
        &voiced_out[..l_curr as usize],
        &blended[..l_curr as usize],
    )
    .expect("blend result has valid MbeParams constraints");

    // Advance state for next frame: store the blended (post-§4.5)
    // magnitudes as M̃_l_B(−1). Spec §4.5 describes this as "previous
    // frame's reconstructed magnitudes after target rate quantization";
    // storing the blend output is a close approximation (the target
    // encoder's quantizer is near-lossless on natural-speech
    // magnitudes, so pre/post-quant blend differs only at sub-bit
    // levels). A future pass could tighten this by storing the target
    // encoder's dequantize output instead.
    state.m_tilde_prev.fill(0.0);
    state.m_tilde_prev[0] = 1.0;
    state.m_tilde_prev[1..(l_curr as usize + 1)]
        .copy_from_slice(&blended[..((l_curr as usize - 1) + 1)]);
    state.l_prev = l_curr;
    state.omega_0_prev = target_omega_0;

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params(omega_0: f32, l: u8, amps_fill: f32) -> MbeParams {
        let voiced = vec![true; l as usize];
        let amps = vec![amps_fill; l as usize];
        MbeParams::new(omega_0, l, &voiced, &amps).expect("test params valid")
    }

    /// Build params with a distinct magnitude per harmonic, so an edge hold is
    /// distinguishable from a zero fill and from a flat fill.
    fn make_ramp(omega_0: f32, l: u8) -> MbeParams {
        let voiced = vec![true; l as usize];
        let amps: Vec<f32> = (0..l).map(|i| 10.0 + f32::from(i)).collect();
        MbeParams::new(omega_0, l, &voiced, &amps).expect("test params valid")
    }

    #[test]
    fn extension_band_holds_the_edge_magnitude_never_zero() {
        // The case every other test in this module misses: the target wants MORE
        // harmonics than the source carries. The two codecs' Annex L grids
        // disagree on the harmonic count for the same audio, so this is the
        // common case in real conversion, not a corner.
        //
        // Slots past the source's last harmonic must carry the edge magnitude.
        // Leaving them at zero is not a harmless gap: the target quantizer's
        // `m_tilde_to_lambda` maps m ≤ 0 to Λ = −1000, and γ is the *mean* of Λ
        // over the frame, so one zero slot out of ~30 drags the frame's gain
        // index to the codebook floor. That is a ~40 dB error from an empty slot.
        let target_omega = 0.15_f32;
        let mut state = CrossRatePredictorState::new();
        state.omega_0_prev = target_omega; // R = 1 → fast path, isolates the fill
        state.l_prev = 24;
        let params = make_ramp(target_omega, 18);
        let out = blend(&params, target_omega, 24, &mut state);
        let amps = out.amplitudes_slice();

        // Source harmonics pass through untouched.
        for i in 0..18 {
            assert_eq!(amps[i], 10.0 + i as f32, "harmonic {i} must pass through");
        }
        // Extension band holds the source's last real magnitude, 10 + 17 = 27.
        for i in 18..24 {
            assert_eq!(
                amps[i], 27.0,
                "harmonic {i} is past the source grid: must hold the edge, not zero"
            );
            assert_ne!(amps[i], 0.0, "harmonic {i} must never be exactly zero");
        }
    }

    #[test]
    fn rho_cross_rate_is_zero_and_disables_the_blend() {
        // Guards the shipped value, which no other test does: the arithmetic
        // test above derives its expectation from this constant and so passes at
        // any weight. Flipping ρ back to 0.65 costs ~2 dB of speech quality, so
        // it must not happen with a green suite.
        assert_eq!(
            RHO_CROSS_RATE, 0.0,
            "the cross-rate blend is off deliberately; see the constant's docs"
        );
        // Behavioural half: with the prior frame at a wildly different level and
        // ω₀ far enough off to select the slow path, the output must still be
        // exactly the current frame's magnitudes. Any prior-frame leakage fails.
        let mut state = CrossRatePredictorState::new();
        state.omega_0_prev = 0.14;
        state.l_prev = 20;
        for l in 1..=20 {
            state.m_tilde_prev[l] = 1.0;
        }
        let params = make_params(0.15, 20, 400.0);
        let out = blend(&params, 0.15, 20, &mut state);
        for (i, &a) in out.amplitudes_slice().iter().take(20).enumerate() {
            assert_eq!(
                a, 400.0,
                "harmonic {i}: prior frame leaked into the output at ρ = 0"
            );
        }
    }

    /// Prior state holding a steeply tilted envelope at a level far below the
    /// current frame's, with `ω_prev` off `target_omega` by enough to select the
    /// slow path. Both level and shape differ, so a mode that moves only one of
    /// them is distinguishable from a mode that moves both.
    fn tilted_prior_state(target_omega: f32) -> CrossRatePredictorState {
        let mut state = CrossRatePredictorState::new();
        state.omega_0_prev = target_omega / 1.0714; // |R − 1| ≈ 7% → slow path
        state.l_prev = 20;
        for l in 1..=20 {
            state.m_tilde_prev[l] = 10.0 * (1.0 + l as f32);
        }
        state
    }

    #[test]
    fn shape_only_blend_holds_the_frame_level_and_still_moves_the_shape() {
        // The alternative to a level blend: mix the prior frame's spectral shape
        // in, then put the frame back on its own level. Level is the mean log₂
        // magnitude, which is the half-rate γ̃ exactly (see CrossRateBlend docs),
        // so "holds the level" is checkable to floating-point tolerance rather
        // than as a trend.
        let target_omega = 0.15_f32;
        let flat = 400.0_f32;
        let mut state = tilted_prior_state(target_omega);
        let params = make_params(target_omega, 20, flat);
        let out = blend_with(
            &params,
            target_omega,
            20,
            &mut state,
            CrossRateBlend::ShapeOnly(0.65),
        );
        let amps = &out.amplitudes_slice()[..20];

        let level = mean_log2(amps);
        let want = f64::from(flat).log2();
        assert!(
            (level - want).abs() < 1e-5,
            "shape-only moved the frame level: mean log₂ {level} vs {want} \
             ({:.3} dB)",
            6.0206 * (level - want)
        );

        // And it must actually have done something: the current frame is flat,
        // so any shape at all in the output came from the prior frame.
        let lo = amps.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = amps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            hi / lo > 1.2,
            "shape-only left the flat input flat (ratio {:.4}); the mix did not run",
            hi / lo
        );
    }

    #[test]
    fn level_blend_moves_the_frame_level_where_shape_only_does_not() {
        // Pins the distinction between the two modes rather than each in
        // isolation: on identical state and input, Level pulls the frame's level
        // toward the prior frame's and ShapeOnly does not. Collapsing ShapeOnly
        // into Level — or dropping the renormalisation — fails here.
        let target_omega = 0.15_f32;
        let flat = 400.0_f32;
        let want = f64::from(flat).log2();

        let mut state_level = tilted_prior_state(target_omega);
        let params = make_params(target_omega, 20, flat);
        let level_out = blend_with(
            &params,
            target_omega,
            20,
            &mut state_level,
            CrossRateBlend::Level(0.65),
        );
        let level_after = mean_log2(&level_out.amplitudes_slice()[..20]);

        let mut state_shape = tilted_prior_state(target_omega);
        let shape_out = blend_with(
            &params,
            target_omega,
            20,
            &mut state_shape,
            CrossRateBlend::ShapeOnly(0.65),
        );
        let shape_after = mean_log2(&shape_out.amplitudes_slice()[..20]);

        assert!(
            want - level_after > 0.5,
            "Level(0.65) should pull the frame level down toward the prior \
             frame's: {level_after} vs {want}"
        );
        assert!(
            (shape_after - want).abs() < 1e-5,
            "ShapeOnly(0.65) must leave the level alone: {shape_after} vs {want}"
        );
    }

    #[test]
    fn default_blend_mode_is_the_shipped_level_weight() {
        // The mode plumbing must not have moved shipped behaviour. Two halves:
        // the default *is* Level(RHO_CROSS_RATE), and `blend` — the entry point
        // the converters call — is bit-identical to that default.
        assert_eq!(
            CrossRateBlend::default(),
            CrossRateBlend::Level(RHO_CROSS_RATE)
        );
        assert_eq!(CrossRateBlend::default().rho(), RHO_CROSS_RATE);

        let target_omega = 0.15_f32;
        let params = make_params(target_omega, 20, 400.0);
        let mut a = tilted_prior_state(target_omega);
        let mut b = tilted_prior_state(target_omega);
        let via_blend = blend(&params, target_omega, 20, &mut a);
        let via_mode = blend_with(&params, target_omega, 20, &mut b, CrossRateBlend::default());
        assert_eq!(
            &via_blend.amplitudes_slice()[..20],
            &via_mode.amplitudes_slice()[..20],
            "blend() must equal blend_with(default)"
        );
        assert_eq!(a.l_prev(), b.l_prev());
        assert_eq!(a.omega_0_prev(), b.omega_0_prev());
    }

    #[test]
    fn extension_band_voicing_matches_the_magnitude_it_carries() {
        // The extension band holds the edge *magnitude*; its flag has to agree.
        // `m_tilde_to_lambda` divides an unvoiced harmonic by ≈0.53 before
        // log₂, so a slot carrying the edge magnitude while flagged unvoiced
        // reports Λ ≈ 5 dB above the harmonic it copied — and γ is the mean of
        // Λ, so the fill's own flag moves the frame's gain index.
        //
        // Checked in both directions so an unconditional force cannot pass:
        // a voiced edge must extend voiced, an unvoiced edge unvoiced.
        let target_omega = 0.15_f32;
        for edge_voiced in [true, false] {
            let source_l: usize = 18;
            let target_l: u8 = 24;
            let mut state = CrossRatePredictorState::new();
            state.omega_0_prev = target_omega; // R = 1 → fast path, isolates the fill
            state.l_prev = target_l;
            // Alternate voicing below the edge so the test also pins that the
            // in-range flags still pass through untouched.
            let voiced: Vec<bool> = (0..source_l)
                .map(|i| {
                    if i + 1 == source_l {
                        edge_voiced
                    } else {
                        i % 2 == 0
                    }
                })
                .collect();
            let amps: Vec<f32> = (0..source_l).map(|i| 10.0 + i as f32).collect();
            let params = MbeParams::new(
                target_omega,
                u8::try_from(source_l).expect("source_l fits u8"),
                &voiced,
                &amps,
            )
            .expect("test params valid");
            let out = blend(&params, target_omega, target_l, &mut state);
            let out_voiced = out.voiced_slice();
            let out_amps = out.amplitudes_slice();

            for i in 0..source_l {
                assert_eq!(
                    out_voiced[i], voiced[i],
                    "edge_voiced={edge_voiced}: in-range harmonic {i} voicing changed"
                );
            }
            for i in source_l..target_l as usize {
                assert_eq!(
                    out_amps[i],
                    amps[source_l - 1],
                    "edge_voiced={edge_voiced}: harmonic {i} must hold the edge magnitude"
                );
                assert_eq!(
                    out_voiced[i], edge_voiced,
                    "edge_voiced={edge_voiced}: harmonic {i} carries the edge magnitude, \
                     so it must carry the edge voicing too"
                );
            }
        }
    }

    #[test]
    fn no_blended_harmonic_is_zero_across_every_grid_mismatch() {
        // Sweep the whole (source L, target L) space rather than one pair: any
        // exact zero reaching the target quantizer costs the frame its gain, so
        // the invariant is "never zero", for all mismatches in both directions.
        let target_omega = 0.15_f32;
        for source_l in [9u8, 12, 20, 30, 44, L_MAX] {
            for target_l in [9u8, 12, 20, 30, 44, L_MAX] {
                for r_prev in [1.0_f32, 0.85, 1.2] {
                    let mut state = CrossRatePredictorState::new();
                    state.omega_0_prev = target_omega / r_prev;
                    state.l_prev = source_l;
                    for l in 1..=source_l as usize {
                        state.m_tilde_prev[l] = 5.0 + l as f32;
                    }
                    let params = make_ramp(target_omega, source_l);
                    let out = blend(&params, target_omega, target_l, &mut state);
                    for (i, &a) in out
                        .amplitudes_slice()
                        .iter()
                        .take(target_l as usize)
                        .enumerate()
                    {
                        assert!(
                            a > 0.0 && a.is_finite(),
                            "source_l={source_l} target_l={target_l} r={r_prev}: \
                             harmonic {i} is {a}, which the target quantizer maps to Λ = −1000"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn prior_frame_resample_holds_at_both_grid_edges() {
        // The prior-frame half of the same defect. When the resampled position
        // falls outside the prior frame's grid, BABA-A Eq. 186/187 hold the
        // first / last prior harmonic; the code this replaced substituted a
        // numerical floor, which is the same log(0) trap one step removed.
        let target_omega = 0.15_f32;
        // r_prev far from 1 forces the slow path and pushes high harmonics off
        // the top of the prior grid.
        let mut state = CrossRatePredictorState::new();
        state.omega_0_prev = target_omega / 1.5;
        state.l_prev = 10;
        for l in 1..=10 {
            state.m_tilde_prev[l] = 100.0;
        }
        let params = make_params(target_omega, 30, 100.0);
        // Nonzero weight required: the shipped ρ = 0 skips the resample, and
        // this test was measured to pass with both edge holds deleted before it
        // was pointed at the core directly.
        let out = blend_with(
            &params,
            target_omega,
            30,
            &mut state,
            CrossRateBlend::Level(0.65),
        );
        // Prior grid is flat at 100 and the current frame is flat at 100, so
        // every blended harmonic must land near 100 even where the resample
        // ran off the end of the prior grid. A floor substitution there would
        // collapse the prior contribution and pull these far below 100.
        for (i, &a) in out.amplitudes_slice().iter().take(30).enumerate() {
            assert!(
                a > 50.0,
                "harmonic {i} is {a}: the prior-frame edge hold collapsed"
            );
        }
    }

    #[test]
    fn cold_start_uses_annex_l_row_30() {
        let state = CrossRatePredictorState::new();
        assert_eq!(state.l_prev(), 30);
        let expected_omega = AMBE_PITCH_TABLE[30].omega_0;
        assert_eq!(state.omega_0_prev(), expected_omega);
        for l in 0..=L_MAX as usize {
            assert_eq!(state.m_tilde_prev()[l], 1.0);
        }
    }

    #[test]
    fn fast_path_passes_current_unchanged_when_r_within_one_percent() {
        // Set state ω_prev to match the target exactly → R = 1.0 → fast path.
        let target_omega = 0.15_f32;
        let mut state = CrossRatePredictorState::new();
        state.omega_0_prev = target_omega;
        state.l_prev = 20;
        for l in 1..=20 {
            state.m_tilde_prev[l] = 42.0;
        }
        let params = make_params(target_omega, 20, 100.0);
        let out = blend(&params, target_omega, 20, &mut state);
        for (i, &a) in out.amplitudes_slice().iter().enumerate().take(20) {
            assert_eq!(
                a, 100.0,
                "harmonic {i}: fast-path should pass current through"
            );
        }
        // State advanced.
        assert_eq!(state.l_prev, 20);
        assert_eq!(state.omega_0_prev, target_omega);
        for l in 1..=20 {
            assert_eq!(state.m_tilde_prev[l], 100.0);
        }
    }

    #[test]
    fn slow_path_blend_weights_follow_rho_at_matching_grids() {
        // Force slow path by setting ω_prev outside the ±1% band
        // around target. Since both grids are "the same" conceptually
        // (l_prev = l_curr, prior M = M_prior, current M = M_curr),
        // resampling lands on integer harmonic indices and we can
        // verify the blend coefficients cleanly.
        //
        // Setup: prior all = 100, current all = 400, prior ω = 0.14,
        // current ω = 0.15 → R_prev = 0.15/0.14 ≈ 1.0714 (> 1.01).
        // Energy offset = 0.5·log₂(1.0714) ≈ 0.0498.
        // For target harmonic l_b, source pos = l_b · R_prev ≈ 1.07·l_b,
        // so for l_b = 1: src_pos ≈ 1.07, interp between log_prev[1]=log₂(100) and [2]=log₂(100),
        // = log₂(100). Add energy_offset → log₂(100·√R). prior_linear = 100·√R ≈ 103.53.
        // blended = ρ · 103.53 + (1 − ρ) · 400.
        //
        // This pins the blend *arithmetic* and cannot pin the shipped weight;
        // `rho_cross_rate_is_zero_and_disables_the_blend` does that. Both are
        // needed: this one passes at any weight, that one would pass on an
        // arithmetically broken blend.
        let mut state = CrossRatePredictorState::new();
        state.omega_0_prev = 0.14;
        state.l_prev = 20;
        for l in 1..=20 {
            state.m_tilde_prev[l] = 100.0;
        }
        let params = make_params(0.15, 20, 400.0);
        // Driven at a nonzero weight on purpose: at the shipped ρ = 0 the slow
        // path is short-circuited and this test would pass with it deleted.
        const RHO: f64 = 0.65;
        let out = blend_with(&params, 0.15, 20, &mut state, CrossRateBlend::Level(RHO));

        let r_prev: f64 = 0.15_f64 / 0.14_f64;
        let expected_prior_linear = 100.0_f64 * r_prev.sqrt();
        let expected_blend = RHO * expected_prior_linear + (1.0 - RHO) * 400.0;
        // Middle harmonics land well inside the source grid and fully
        // exercise the interp path.
        for l in 5..=15 {
            let observed = out.amplitudes_slice()[l - 1] as f64;
            let rel_err = (observed - expected_blend).abs() / expected_blend;
            assert!(
                rel_err < 0.01,
                "l={l}: observed {observed}, expected {expected_blend} (rel err {rel_err})"
            );
        }
    }

    #[test]
    fn state_advances_every_frame_regardless_of_path() {
        // Both fast and slow paths must update state so next-frame R is
        // computed against this frame's ω.
        let mut state = CrossRatePredictorState::new();
        // Frame 1: fast path (target == init).
        let omega_init = state.omega_0_prev;
        let p1 = make_params(omega_init, state.l_prev, 50.0);
        let _ = blend(&p1, omega_init, state.l_prev, &mut state);
        assert_eq!(state.omega_0_prev, omega_init);
        // Frame 2: slow path (20% pitch shift).
        let p2 = make_params(omega_init * 1.2, 25, 80.0);
        let _ = blend(&p2, omega_init * 1.2, 25, &mut state);
        assert!((state.omega_0_prev - omega_init * 1.2).abs() < 1e-6);
        assert_eq!(state.l_prev, 25);
    }

    #[test]
    fn voicing_is_preserved_across_blend() {
        // Blend touches only amplitudes; voiced_slice must match input.
        let mut state = CrossRatePredictorState::new();
        state.omega_0_prev = 0.14;
        state.l_prev = 10;
        let l: u8 = 10;
        let voiced = vec![
            true, false, true, false, true, false, true, false, true, false,
        ];
        let amps = vec![100.0_f32; l as usize];
        let params = MbeParams::new(0.15, l, &voiced, &amps).unwrap();
        let out = blend(&params, 0.15, l, &mut state);
        assert_eq!(&out.voiced_slice()[..l as usize], &voiced[..]);
    }

    #[test]
    fn reset_returns_to_cold_start() {
        let mut state = CrossRatePredictorState::new();
        state.omega_0_prev = 0.25;
        state.l_prev = 40;
        state.m_tilde_prev[1] = 999.0;
        state.reset();
        let fresh = CrossRatePredictorState::new();
        assert_eq!(state.l_prev, fresh.l_prev);
        assert_eq!(state.omega_0_prev, fresh.omega_0_prev);
        assert_eq!(state.m_tilde_prev[1], 1.0);
    }

    #[test]
    fn sustained_input_converges_toward_stable_output() {
        // Multi-frame test: feed a constant MbeParams stream with
        // varying pitch (to exercise slow path) and verify the blend
        // output converges to a stable value, matching the first-order
        // low-pass behavior the predictor implements.
        let mut state = CrossRatePredictorState::new();
        // Start at a distinct ω from init.
        state.omega_0_prev = 0.20;
        state.l_prev = 15;
        for l in 1..=15 {
            state.m_tilde_prev[l] = 0.0; // start with a far-from-target prior
        }
        let mut last_l1 = 0.0_f64;
        for frame in 0..30 {
            let p = make_params(0.15, 20, 500.0);
            let out = blend(&p, 0.15, 20, &mut state);
            let this_l1 = out.amplitudes_slice()[0] as f64;
            if frame > 5 {
                // After a few frames the blend output should be very
                // close to the input (since the fast-path threshold
                // kicks in once ω_prev == target).
                assert!(
                    (this_l1 - 500.0).abs() < 1.0,
                    "frame {frame}: blended l=1 = {this_l1}, expected ~500"
                );
            }
            last_l1 = this_l1;
        }
        assert!((last_l1 - 500.0).abs() < 1.0);
    }
}
