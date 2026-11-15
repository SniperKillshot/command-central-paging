//! Cross-rate conversion — parse at rate A, requantize, pack at rate B.
//!
//! Parameters flow through:
//!
//! ```text
//!   A-rate frame bits
//!     → parse via the source rate's wire module (decode + dequantize)
//!     → MbeParams
//!     → quantize at the destination rate
//!     → pack via the destination rate's wire module (prioritize + encode)
//!     → B-rate frame bits
//! ```
//!
//! Each converter holds two state objects: the source-rate decoder
//! state (which advances as input frames are consumed) and the
//! destination-rate encoder state (which advances as output frames are
//! produced). They are independent because the upstream radio's encoder
//! and the downstream radio's decoder evolve in lockstep with the
//! converter's encoder and decoder respectively, not with each other.
//!
//! ## Boundary behaviour
//!
//! The full-rate grid (b̂₀ ∈ [0, 207] → ω̃₀ ∈ [4π/246.5, 4π/39.5] ≈
//! [0.05098, 0.31819]) and the half-rate Annex L grid (120 entries,
//! ω̃₀ ∈ [0.05105, 0.31398]) almost but don't quite coincide at the
//! edges — full-rate extends ~0.14% past half-rate's minimum and
//! ~1.3% past its maximum. A half-rate frame at the grid endpoint,
//! round-tripped through full-rate and back, can pick up enough drift
//! at the first hop to fall outside half-rate's grid at the second.
//! The source parameters' ω̃₀ is therefore snapped to the destination
//! grid's representable range (keeping L unchanged — the boundary
//! entries share L across rates) before the destination quantizer runs,
//! so boundary frames encode to the nearest grid endpoint instead of
//! rejecting.
//!
//! ## Frame-kind handling
//!
//! Half-rate defines four pitch-index regions (§13.1 Table 14):
//! voice `[0, 119]`, erasure `[120, 123]`, silence `[124, 125]`, and
//! tone `[126, 127]`. Full-rate IMBE has no tone/silence signaling —
//! any `b̂₀ ∉ [0, 207]` is a frame-repeat trigger at the decoder
//! (§1.11). Converters map between these worlds as follows:
//!
//! | Source kind            | Emitted frame               | Encoder state |
//! |------------------------|-----------------------------|---------------|
//! | Voice                  | Voice at destination rate   | advances      |
//! | Half-rate tone         | Full-rate voice from Eq. 206–209 MBE bridge params (tone rendered as a 1–2 harmonic voiced sinusoid) | advances |
//! | Half-rate silence      | Destination-rate erasure signal | does **not** advance |
//! | Half-rate erasure      | Destination-rate erasure signal | does **not** advance |
//! | Full-rate erasure/reserved | Half-rate erasure (b̂₀=120) | does **not** advance |
//!
//! The "does not advance" rule matches §1.11 / §2.8: repeat and mute
//! preserve prior-frame parameters in the receiver, and the encoder
//! has nothing new to quantize, so its predictor state must not
//! drift.
//!
//! ## Not implemented
//!
//! - **Voicing-band normalization** and **magnitude interpolation across
//!   L mismatches** described in US7634399. This path relies on
//!   the destination quantizer's own pitch table to pick a representable
//!   `(ω₀, L)` and resamples per-harmonic amplitudes only via that
//!   choice. Bridging IMBE full-rate (L = 9..56 from a 208-entry table)
//!   to AMBE+2 half-rate (L = 9..56 from a 120-entry Annex L table) is
//!   well-behaved in practice because both share the same `(ω₀, L)`
//!   relationship; large excursions are rare.

use core::f32::consts::PI;

use crate::fullrate::dequantize::{
    band_for_harmonic, dequantize, encode_pitch as full_encode_pitch,
    pitch_decode as full_pitch_decode, quantize, vuv_band_count, DecodeError, DecoderState,
    EncodeError, PITCH_INDEX_MAX,
};
use crate::fullrate::frame::{decode_frame as decode_full, encode_frame as encode_full};
use crate::fullrate::priority::{prioritize as prioritize_full, L_MIN as FULL_L_MIN, PARAM_MAX};
use crate::halfrate::dequantize::{
    decode_to_params, encode_pitch as half_encode_pitch, encode_tone_frame_info, params_to_tone,
    quantize as quantize_half, Decoded, DecoderState as HalfDecoderState,
};
use crate::halfrate::frame::{
    decode_frame as decode_half, encode_frame as encode_half, AMBE_PITCH_TABLE, DIBITS_PER_FRAME,
};
use crate::halfrate::priority::{prioritize as prioritize_half, PARAM_COUNT};
use crate::mbe_params::{MbeParams, L_MAX};
use crate::rate_conversion::predictor::{
    blend_with as cross_rate_blend_with, CrossRateBlend, CrossRatePredictorState,
};
use crate::rate_conversion::tone_reanalysis::ToneEnvelopes;

/// Lowest ω̃₀ in the half-rate Annex L table (entry 119, L = 56).
/// Used to clamp source parameters to the half-rate grid's range
/// before handing them to the half-rate quantizer.
fn ambe_plus2_omega_min() -> f32 {
    AMBE_PITCH_TABLE[AMBE_PITCH_TABLE.len() - 1].omega_0
}

/// Highest ω̃₀ in the half-rate Annex L table (entry 0, L = 9).
fn ambe_plus2_omega_max() -> f32 {
    AMBE_PITCH_TABLE[0].omega_0
}

/// Lowest ω̃₀ emittable by the full-rate encoder (b̂₀ = 207): `4π/246.5`.
fn imbe_omega_min() -> f32 {
    4.0 * PI / (f32::from(PITCH_INDEX_MAX) + 39.5)
}

/// Highest ω̃₀ emittable by the full-rate encoder (b̂₀ = 0): `4π/39.5`.
fn imbe_omega_max() -> f32 {
    4.0 * PI / 39.5
}

/// If `params.omega_0()` falls outside `[min, max]`, return a copy with
/// ω̃₀ snapped to the nearest boundary — L, voicing, and amplitudes
/// unchanged. Returns the input cloned otherwise. Cheap; avoids an
/// allocation path on the common in-range case.
fn clamp_omega_to(params: &MbeParams, min: f32, max: f32) -> MbeParams {
    let w = params.omega_0();
    if w >= min && w <= max {
        return params.clone();
    }
    let clamped = w.clamp(min, max);
    MbeParams::new(
        clamped,
        params.harmonic_count(),
        params.voiced_slice(),
        params.amplitudes_slice(),
    )
    .expect("clamped ω₀ stays inside (0, π)")
}

/// Widen `params`' per-harmonic voicing to whole full-rate V/UV bands: a band
/// with any voiced harmonic becomes wholly voiced, and every other band stays
/// wholly unvoiced.
///
/// Full-rate `b̂₁` carries one bit per band of up to three harmonics, and
/// [`collapse_vuv`] reads each band's **first** harmonic as that band's vote.
/// Per-harmonic flags that are not already band-uniform therefore do not
/// survive the trip: [`expand_vuv`] hands the decoder band-uniform flags
/// whatever went in, so any flag on a harmonic that is not its band's first
/// one is simply discarded.
///
/// That discard is silent for a voiced spectrum, where neighbouring harmonics
/// agree anyway. It is not silent for the Eq. 206–209 tone bridge, which flags
/// `l₁`/`l₂` and nothing else: whenever `l₁` is not its band's first harmonic
/// the band votes on an unvoiced neighbour, `b̂₁` comes out zero, and the whole
/// tone is transmitted as unvoiced. The receiver then synthesises it as band
/// noise rather than as a sinusoid, which on the reference corpus reads about
/// 5.6 dB **hotter** than the tone asked for.
///
/// Widening carries no threshold and no constant. It states the bridge's own
/// voicing intent — "there is a voiced component in this band" — in the only
/// resolution `b̂₁` has.
///
/// [`collapse_vuv`]: crate::fullrate::dequantize::collapse_vuv
/// [`expand_vuv`]: crate::fullrate::dequantize::expand_vuv
fn widen_voicing_to_bands(params: &MbeParams) -> MbeParams {
    let l = params.harmonic_count();
    let k = vuv_band_count(l);
    let src = params.voiced_slice();
    let mut voiced = [false; L_MAX as usize];
    for band in 1..=k {
        let members = (1..=l).filter(|&h| band_for_harmonic(h, k) == band);
        if members.clone().any(|h| src[usize::from(h) - 1]) {
            for h in members {
                voiced[usize::from(h) - 1] = true;
            }
        }
    }
    MbeParams::new(
        params.omega_0(),
        l,
        &voiced[..usize::from(l)],
        params.amplitudes_slice(),
    )
    .expect("only the voicing flags changed")
}

/// First `b̂₀` value in the half-rate erasure range (§13.1 Table 14,
/// `[120, 123]`). Emitted by converters when the source-rate frame
/// signals an erasure / silence / reserved pitch condition.
const HALFRATE_ERASURE_B0: u16 = 120;

/// First `b̂₀` value in the full-rate reserved range (§6.1,
/// `[208, 255]`). The full-rate decoder treats any `b̂₀ ∉ [0, 207]`
/// as a frame-repeat trigger (§1.11), so a single sentinel is
/// sufficient.
const FULLRATE_ERASURE_B0: u16 = PITCH_INDEX_MAX as u16 + 1;

/// Emit a half-rate erasure frame: 36 dibits whose deprioritized
/// `b̂₀ = 120` (other parameters zero). After the channel codec pass
/// these will round-trip through `classify_frame` as
/// [`crate::halfrate::dequantize::FrameKind::Erasure`].
fn ambe_plus2_erasure_dibits() -> [u8; DIBITS_PER_FRAME] {
    let mut b = [0u16; PARAM_COUNT];
    b[0] = HALFRATE_ERASURE_B0;
    let u = prioritize_half(&b);
    encode_half(&u)
}

/// Emit a full-rate erasure frame: 72 dibits whose deprioritized
/// `b̂₀ = 208` (just past `PITCH_INDEX_MAX`). The full-rate decoder's
/// §1.11 repeat logic will substitute the previous voice frame.
fn imbe_erasure_dibits() -> [u8; 72] {
    let mut b = [0u16; PARAM_MAX];
    b[0] = FULLRATE_ERASURE_B0;
    // Prioritize uses the L-indexed map; at L=L_MIN only the fixed
    // b̂₀/b̂₁ positions receive bits (the rest of b̂ is zero), which is
    // what we want — the decoder rejects on pitch before the rest
    // matters.
    let u = prioritize_full(&b, FULL_L_MIN);
    encode_full(&u)
}

/// Error from a cross-rate conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConvertError {
    /// The source-rate dequantizer rejected the input frame.
    Decode(DecodeError),
    /// The destination-rate quantizer rejected the parameters. Only
    /// the full-rate encoder produces this; the half-rate encoder
    /// uses [`DecodeError`] for its own pitch-out-of-range case.
    Encode(EncodeError),
    /// Parameters fell outside the half-rate pitch table when packing
    /// to half rate. Re-uses [`DecodeError::BadPitch`] for symmetry
    /// with the half-rate encoder's signature.
    HalfPitchOutOfRange,
}

impl From<DecodeError> for ConvertError {
    fn from(e: DecodeError) -> Self {
        Self::Decode(e)
    }
}

impl From<EncodeError> for ConvertError {
    fn from(e: EncodeError) -> Self {
        Self::Encode(e)
    }
}

// ---------------------------------------------------------------------------
// P25 Phase 1 full-rate (IMBE) → P25 Phase 2 half-rate (AMBE+2)
// ---------------------------------------------------------------------------

/// Full-rate (144-bit IMBE) → half-rate (72-bit AMBE+2) converter.
///
/// Holds the source decoder state, the destination encoder state, and
/// the cross-rate magnitude predictor state (AMBE-3000 rate-converter
/// spec §4.5). Call [`Self::convert`] once per 20 ms frame.
#[derive(Clone, Debug)]
pub struct FullToHalfConverter {
    decoder: DecoderState,
    encoder: HalfDecoderState,
    predictor: CrossRatePredictorState,
    /// Controls whether [`Self::convert`] applies the §4.5 blend. On
    /// (spec-conformant) by default. Off for A/B validation of the
    /// predictor's smoothing effect against an unpredicted baseline.
    predictor_enabled: bool,
    /// How the §4.5 mix combines the prior frame with this one. Ships at
    /// [`CrossRateBlend::default`], which is inert.
    blend: CrossRateBlend,
}

impl Default for FullToHalfConverter {
    fn default() -> Self {
        Self {
            decoder: DecoderState::default(),
            encoder: HalfDecoderState::default(),
            predictor: CrossRatePredictorState::default(),
            predictor_enabled: true,
            blend: CrossRateBlend::default(),
        }
    }
}

impl FullToHalfConverter {
    /// Cold-start converter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert one full-rate frame (72 dibits) into one half-rate
    /// frame (36 dibits). On a full-rate erasure / reserved pitch
    /// (`b̂₀ ∈ [208, 255]`) the output is a half-rate erasure signal
    /// and encoder state is preserved (§2.8 frame-repeat semantics).
    ///
    /// Voice frames pass through the cross-rate predictor (§4.5): the
    /// target-grid magnitudes are blended with the prior target-rate
    /// frame's magnitudes before handing to the half-rate quantizer.
    pub fn convert(&mut self, dibits: &[u8; 72]) -> Result<[u8; DIBITS_PER_FRAME], ConvertError> {
        let frame = decode_full(dibits);
        let params = match dequantize(&frame.info, &mut self.decoder) {
            Ok(p) => p,
            Err(DecodeError::BadPitch) => return Ok(ambe_plus2_erasure_dibits()),
            Err(other) => return Err(ConvertError::Decode(other)),
        };
        // Full-rate IMBE has no tone signalling, so a tone arrives here as an
        // ordinary voiced frame with its energy piled onto one or two
        // harmonics. The half-rate wire does have tone frames, and re-encoding
        // a tone as a voiced spectrum instead of using one costs real spectral
        // accuracy — see [`params_to_tone`]. Emit the tone frame and leave both
        // the predictor and the encoder state untouched: a tone carries no
        // spectral envelope for them to track, and advancing them on one
        // poisons the prediction for the voice frames around it.
        if let Some((id, amplitude)) = params_to_tone(&params) {
            return Ok(encode_half(&encode_tone_frame_info(id, amplitude)));
        }
        let clamped = clamp_omega_to(&params, ambe_plus2_omega_min(), ambe_plus2_omega_max());
        // Pre-compute the target-committed (ω̂₀, L) so the predictor
        // sees what the half-rate encoder will actually emit — the
        // encoder snaps ω̂₀ to the Annex L grid, and L derives from
        // that snap. If the pitch is out of range, fall through the
        // encoder error path rather than advancing predictor state on
        // a non-committed frame.
        let blended = if self.predictor_enabled {
            match half_encode_pitch(clamped.omega_0()) {
                Some(b0) => {
                    let target_entry = AMBE_PITCH_TABLE[b0 as usize];
                    cross_rate_blend_with(
                        &clamped,
                        target_entry.omega_0,
                        target_entry.l,
                        &mut self.predictor,
                        self.blend,
                    )
                }
                None => clamped,
            }
        } else {
            clamped
        };
        let u = quantize_half(&blended, &mut self.encoder).map_err(|e| match e {
            DecodeError::BadPitch => ConvertError::HalfPitchOutOfRange,
            other => ConvertError::Decode(other),
        })?;
        Ok(encode_half(&u))
    }

    /// Borrow the source decoder state.
    pub fn decoder_state(&self) -> &DecoderState {
        &self.decoder
    }

    /// Borrow the destination encoder state.
    pub fn encoder_state(&self) -> &HalfDecoderState {
        &self.encoder
    }

    /// Borrow the cross-rate predictor state.
    pub fn predictor_state(&self) -> &CrossRatePredictorState {
        &self.predictor
    }

    /// Enable or disable the §4.5 cross-rate magnitude blend. `true`
    /// (default, spec-conformant) applies the predictor on every
    /// voice / tone frame; `false` routes parameters straight from
    /// source decoder to target encoder without the log-domain blend.
    /// Intended for A/B validation against an unpredicted baseline.
    pub fn set_predictor_enabled(&mut self, enabled: bool) {
        self.predictor_enabled = enabled;
    }

    /// Select how the §4.5 mix combines the prior frame with this one.
    ///
    /// Distinct from [`Self::set_predictor_enabled`]: the predictor still runs
    /// at [`CrossRateBlend::Level`]`(0.0)` — it holds the extension band at the
    /// source's edge magnitude and advances its state — it just does not mix.
    /// Disabling the predictor skips all of that.
    ///
    /// Ships at [`CrossRateBlend::default`]. Intended for A/B measurement; see
    /// `conformance/roundtrip/examples/cross_rate_stability.rs`.
    pub fn set_cross_rate_blend(&mut self, blend: CrossRateBlend) {
        self.blend = blend;
    }

    /// The §4.5 mix mode currently selected.
    pub fn cross_rate_blend(&self) -> CrossRateBlend {
        self.blend
    }
}

// ---------------------------------------------------------------------------
// P25 Phase 2 half-rate (AMBE+2) → P25 Phase 1 full-rate (IMBE)
// ---------------------------------------------------------------------------

/// Half-rate (72-bit AMBE+2) → full-rate (144-bit IMBE) converter.
///
/// Holds the source decoder state, the destination encoder state, and
/// the cross-rate magnitude predictor state (AMBE-3000 rate-converter
/// spec §4.5).
#[derive(Clone, Debug)]
pub struct HalfToFullConverter {
    decoder: HalfDecoderState,
    encoder: DecoderState,
    predictor: CrossRatePredictorState,
    /// Controls whether [`Self::convert`] applies the §4.5 blend. See
    /// [`FullToHalfConverter::set_predictor_enabled`] for semantics.
    predictor_enabled: bool,
    /// How the §4.5 mix combines the prior frame with this one. Ships at
    /// [`CrossRateBlend::default`], which is inert.
    blend: CrossRateBlend,
    /// The last few tone identities' measured magnitude envelopes. Pure cache —
    /// it carries no cross-frame signal, so it never affects a voice frame and
    /// re-measuring instead would give bit-identical output.
    tone_envelopes: ToneEnvelopes,
}

impl Default for HalfToFullConverter {
    fn default() -> Self {
        Self {
            decoder: HalfDecoderState::default(),
            encoder: DecoderState::default(),
            predictor: CrossRatePredictorState::default(),
            predictor_enabled: true,
            blend: CrossRateBlend::default(),
            tone_envelopes: ToneEnvelopes::default(),
        }
    }
}

impl HalfToFullConverter {
    /// Cold-start converter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert one half-rate frame (36 dibits) into one full-rate
    /// frame (72 dibits). Handles all four half-rate frame kinds:
    ///
    /// - Voice → full-rate voice (normal requantize path; §4.5 blend applied).
    /// - Tone → full-rate voice encoded from the Eq. 206–209 MBE bridge
    ///   parameters with their magnitude envelope measured through the
    ///   engine's amplitude stage (§4.5 blend applied so state stays coherent
    ///   across voice/tone transitions). Eq. 209 leaves every harmonic but
    ///   `l₁`/`l₂` at exactly zero, which is `−∞` to the full-rate quantiser's
    ///   log-magnitude DCT and pins the gain to the codebook floor; the
    ///   substitution keeps the bridge's `ω̃₀`, `L`, voicing and Eq. 209 level
    ///   and takes only the shape around the tone from the measurement. See
    ///   the `tone_reanalysis` module.
    /// - Silence / Erasure → full-rate erasure signal. Encoder and
    ///   predictor state both preserved so the downstream receiver's
    ///   §1.11 repeat logic stays consistent.
    pub fn convert(&mut self, dibits: &[u8; DIBITS_PER_FRAME]) -> Result<[u8; 72], ConvertError> {
        let frame = decode_half(dibits);
        let (params, is_tone) = match decode_to_params(&frame.info, &mut self.decoder) {
            Ok(Decoded::Voice(p)) => (p, false),
            Ok(Decoded::Tone { fields, params }) => (
                self.tone_envelopes
                    .substitute_magnitudes(fields.id, &params)
                    .unwrap_or(params),
                true,
            ),
            Ok(Decoded::Erasure) => return Ok(imbe_erasure_dibits()),
            Err(_) => return Ok(imbe_erasure_dibits()),
        };
        let clamped = clamp_omega_to(&params, imbe_omega_min(), imbe_omega_max());
        // Pre-compute the target-committed (ω̂₀, L) so the predictor
        // sees what the full-rate encoder will actually emit. The
        // full-rate encoder snaps ω̂₀ onto the Eq. 47 grid, where
        // L = ⌊4π/ω̃₀⌋ − 39.
        let blended = if self.predictor_enabled {
            match full_encode_pitch(clamped.omega_0()) {
                Some(b0) => match full_pitch_decode(b0) {
                    Some(target_entry) => cross_rate_blend_with(
                        &clamped,
                        target_entry.omega_0,
                        target_entry.l,
                        &mut self.predictor,
                        self.blend,
                    ),
                    None => clamped,
                },
                None => clamped,
            }
        } else {
            clamped
        };
        // The tone bridge flags `l₁`/`l₂` and nothing else, which `b̂₁`'s
        // one-bit-per-band resolution cannot carry. Widen on the *destination*
        // grid, after the blend has committed (ω̂₀, L) and therefore the band
        // count K.
        let blended = if is_tone {
            widen_voicing_to_bands(&blended)
        } else {
            blended
        };
        let l = blended.harmonic_count();
        let b = quantize(&blended, &mut self.encoder)?;
        let u = prioritize_full(&b, l);
        Ok(encode_full(&u))
    }

    /// Borrow the source decoder state.
    pub fn decoder_state(&self) -> &HalfDecoderState {
        &self.decoder
    }

    /// Borrow the destination encoder state.
    pub fn encoder_state(&self) -> &DecoderState {
        &self.encoder
    }

    /// Borrow the cross-rate predictor state.
    pub fn predictor_state(&self) -> &CrossRatePredictorState {
        &self.predictor
    }

    /// Enable or disable the §4.5 cross-rate magnitude blend. See
    /// [`FullToHalfConverter::set_predictor_enabled`] for semantics.
    pub fn set_predictor_enabled(&mut self, enabled: bool) {
        self.predictor_enabled = enabled;
    }

    /// Select how the §4.5 mix combines the prior frame with this one. See
    /// [`FullToHalfConverter::set_cross_rate_blend`] for semantics.
    pub fn set_cross_rate_blend(&mut self, blend: CrossRateBlend) {
        self.blend = blend;
    }

    /// The §4.5 mix mode currently selected.
    pub fn cross_rate_blend(&self) -> CrossRateBlend {
        self.blend
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mbe_params::MbeParams;

    /// Build a valid voice MbeParams that lies inside both pitch tables.
    fn sample_params(omega_0: f32) -> MbeParams {
        let l = MbeParams::harmonic_count_for(omega_0);
        let voiced: Vec<bool> = (1..=l).map(|h| h <= l / 2).collect();
        let amps: Vec<f32> = (1..=l)
            .map(|h| 100.0 * (-(h as f32) * 0.05).exp())
            .collect();
        MbeParams::new(omega_0, l, &voiced, &amps).unwrap()
    }

    fn imbe_dibits_from(params: &MbeParams, state: &mut DecoderState) -> [u8; 72] {
        let l = params.harmonic_count();
        let b = quantize(params, state).expect("quantize imbe");
        let u = prioritize_full(&b, l);
        encode_full(&u)
    }

    fn ambe_plus2_dibits_from(
        params: &MbeParams,
        state: &mut HalfDecoderState,
    ) -> [u8; DIBITS_PER_FRAME] {
        let u = quantize_half(params, state).expect("quantize ambe_plus2");
        encode_half(&u)
    }

    fn assert_valid_dibits<const N: usize>(dibits: &[u8; N]) {
        for (i, d) in dibits.iter().enumerate() {
            assert!(*d < 4, "dibit {i} = {d} is out of range");
        }
    }

    #[test]
    fn converters_construct() {
        let _ = FullToHalfConverter::new();
        let _ = HalfToFullConverter::new();
    }

    #[test]
    fn full_to_half_emits_valid_dibits() {
        let mut src_state = DecoderState::new();
        let mut conv = FullToHalfConverter::new();
        let params = sample_params(0.20);
        let input = imbe_dibits_from(&params, &mut src_state);
        let out = conv.convert(&input).expect("convert");
        assert_valid_dibits(&out);
    }

    #[test]
    fn half_to_full_emits_valid_dibits() {
        let mut src_state = HalfDecoderState::new();
        let mut conv = HalfToFullConverter::new();
        let params = sample_params(0.18);
        let input = ambe_plus2_dibits_from(&params, &mut src_state);
        let out = conv.convert(&input).expect("convert");
        assert_valid_dibits(&out);
    }

    #[test]
    fn cross_rate_round_trip_preserves_pitch_within_quantizer_grid() {
        let mut src_state = DecoderState::new();
        let mut a_to_b = FullToHalfConverter::new();
        let mut b_to_a = HalfToFullConverter::new();
        let mut sink_state = DecoderState::new();

        let params = sample_params(0.20);

        let mut last_omega = 0f32;
        for _ in 0..6 {
            let a = imbe_dibits_from(&params, &mut src_state);
            let b = a_to_b.convert(&a).expect("A→B");
            let a2 = b_to_a.convert(&b).expect("B→A");

            let frame = decode_full(&a2);
            let back = dequantize(&frame.info, &mut sink_state).expect("decode");
            last_omega = back.omega_0();
        }

        let rel = (last_omega - params.omega_0()).abs() / params.omega_0();
        assert!(rel < 0.05, "ω₀ drift {rel:.4} exceeds 5%");
    }

    #[test]
    fn half_grid_minimum_round_trips_without_rejection() {
        // Boundary case: a half-rate frame at the Annex L minimum
        // (b̂₀=119, ω₀=0.051051, L=56) comes back from full-rate as
        // b̂₀=207 → ω₀=0.050979, just below the half-rate grid minimum.
        // Without the boundary clamp FullToHalfConverter rejects it
        // with HalfPitchOutOfRange; the clamp snaps ω₀ to the
        // half-rate grid edge before quantizing.
        let mut src_state = HalfDecoderState::new();
        let l = 56u8;
        let voiced: Vec<bool> = (1..=l).map(|h| h <= l / 2).collect();
        let amps: Vec<f32> = (1..=l)
            .map(|h| 100.0 * (-(h as f32) * 0.05).exp())
            .collect();
        let params = MbeParams::new(ambe_plus2_omega_min(), l, &voiced, &amps).unwrap();

        let mut a_to_b = HalfToFullConverter::new();
        let mut b_to_a = FullToHalfConverter::new();
        for _ in 0..4 {
            let a = ambe_plus2_dibits_from(&params, &mut src_state);
            let b = a_to_b.convert(&a).expect("half → full");
            let _ = b_to_a.convert(&b).expect("full → half (was failing)");
        }
    }

    #[test]
    fn full_grid_maximum_round_trips_without_rejection() {
        // Symmetric: full-rate b̂₀=0 → ω₀=0.318194 is above the
        // half-rate grid maximum (0.313977). FullToHalfConverter
        // should clamp down to the half-rate max rather than reject.
        let mut src_state = DecoderState::new();
        let params = MbeParams::silence(); // ω₀ = 4π/39.5 = full-rate max
        let mut conv = FullToHalfConverter::new();
        for _ in 0..4 {
            let a = imbe_dibits_from(&params, &mut src_state);
            let _ = conv.convert(&a).expect("full → half at grid top");
        }
    }

    #[test]
    fn clamp_is_no_op_for_in_range_omega() {
        let params = sample_params(0.20);
        let clamped = clamp_omega_to(&params, 0.1, 0.3);
        assert_eq!(clamped.omega_0(), params.omega_0());
        assert_eq!(clamped.harmonic_count(), params.harmonic_count());
    }

    #[test]
    fn clamp_snaps_to_the_nearest_boundary() {
        let below = sample_params(0.10);
        let snapped = clamp_omega_to(&below, 0.15, 0.25);
        assert!((snapped.omega_0() - 0.15).abs() < 1e-6);
        assert_eq!(snapped.harmonic_count(), below.harmonic_count());

        let above = sample_params(0.30);
        let snapped = clamp_omega_to(&above, 0.15, 0.25);
        assert!((snapped.omega_0() - 0.25).abs() < 1e-6);
    }

    // ---- Tone / silence / erasure handling -----------------------------

    use crate::halfrate::dequantize::{classify_frame, FrameKind};

    /// Build a half-rate erasure frame's 36 dibits by deliberately
    /// prioritizing `b̂₀ = 120` with everything else zero — same path
    /// the converter uses to emit erasure on its output side.
    fn ambe_plus2_erasure_input() -> [u8; DIBITS_PER_FRAME] {
        ambe_plus2_erasure_dibits()
    }

    /// Build a half-rate tone frame's 36 dibits for tone ID 5 (first
    /// valid Annex T entry, 156.25 Hz with l₁=l₂=1). Tone frames use
    /// a different bit layout from voice (§2.10.1 Table 20), so we
    /// set the signature bits directly rather than going through the
    /// voice-path prioritizer.
    fn ambe_plus2_tone_input() -> [u8; DIBITS_PER_FRAME] {
        // Signature: û₀(11..6) = 0x3F, û₃(3..0) = 0.
        // Tone ID copy 4 in û₃(12..5) = 5 (the primary ID the
        // parser reads).
        let mut u = [0u16; 4];
        u[0] = 0x3F << 6; // signature
        u[3] = 5u16 << 5; // tone ID = 5 in copy 4 position
        encode_half(&u)
    }

    /// Build a full-rate "erasure" frame by prioritizing `b̂₀ = 208`
    /// into û and encoding. The decoder will reject with
    /// `DecodeError::BadPitch`, which the converter maps to a
    /// half-rate erasure on its output.
    fn imbe_erasure_input() -> [u8; 72] {
        imbe_erasure_dibits()
    }

    #[test]
    fn ambe_plus2_erasure_dibits_round_trip_as_erasure() {
        // Self-check: the dibits we emit must classify as Erasure
        // after the channel codec pass (otherwise receivers would
        // interpret the converter's output as voice).
        let dibits = ambe_plus2_erasure_dibits();
        let frame = decode_half(&dibits);
        assert_eq!(classify_frame(&frame.info), FrameKind::Erasure);
    }

    #[test]
    fn imbe_erasure_dibits_round_trip_as_bad_pitch() {
        // Self-check: the dibits we emit must decode to a reserved
        // `b̂₀`, which the full-rate dequantizer rejects as BadPitch —
        // the §1.11 frame-repeat trigger at the receiver.
        let dibits = imbe_erasure_dibits();
        let frame = decode_full(&dibits);
        let mut state = DecoderState::new();
        match dequantize(&frame.info, &mut state) {
            Err(DecodeError::BadPitch) => {}
            other => panic!("expected BadPitch, got {other:?}"),
        }
    }

    #[test]
    fn full_to_half_emits_erasure_on_reserved_pitch() {
        // Full-rate input signaling b̂₀ = 208 → half-rate output must
        // classify as Erasure at the receiver (not Voice, which would
        // play garbage, or Tone, which would mis-interpret the frame).
        let mut conv = FullToHalfConverter::new();
        let input = imbe_erasure_input();
        let out = conv.convert(&input).expect("convert");
        let frame = decode_half(&out);
        assert_eq!(classify_frame(&frame.info), FrameKind::Erasure);
    }

    #[test]
    fn half_to_full_emits_erasure_on_erasure_input() {
        // Half-rate erasure input → full-rate erasure output that
        // triggers frame repeat at the receiver.
        let mut conv = HalfToFullConverter::new();
        let input = ambe_plus2_erasure_input();
        let out = conv.convert(&input).expect("convert");
        let frame = decode_full(&out);
        let mut sink = DecoderState::new();
        match dequantize(&frame.info, &mut sink) {
            Err(DecodeError::BadPitch) => {}
            other => panic!("expected BadPitch on re-decode, got {other:?}"),
        }
    }

    /// Speech must never be mistaken for a tone. The detector keys on energy
    /// concentration, and a voiced frame spreads energy across many harmonics;
    /// this pins that a broad spectrum stays voice.
    #[test]
    fn full_to_half_does_not_mistake_voice_for_a_tone() {
        use crate::halfrate::dequantize::{classify_frame, FrameKind};

        let mut enc = DecoderState::new();
        let mut conv = FullToHalfConverter::new();
        for seed in 0..24u32 {
            // A decaying harmonic series — energy in every harmonic, which is
            // what a tone never has.
            let omega_0 = 0.18 + (seed as f32) * 0.002;
            let l = MbeParams::harmonic_count_for(omega_0);
            let voiced: Vec<bool> = (1..=l).map(|h| h % 3 != 0).collect();
            let amps: Vec<f32> = (1..=l)
                .map(|h| 2000.0 * (-(h as f32) * 0.08).exp() + 40.0)
                .collect();
            let params = MbeParams::new(omega_0, l, &voiced, &amps).expect("params");
            let Ok(b) = quantize(&params, &mut enc) else {
                continue;
            };
            let full = encode_full(&prioritize_full(&b, l));
            let out = conv.convert(&full).expect("convert");
            assert_ne!(
                classify_frame(&decode_half(&out).info),
                FrameKind::Tone,
                "seed {seed}: a broad voiced spectrum was classified as a tone"
            );
        }
    }

    #[test]
    fn half_to_full_encodes_tone_as_voice() {
        // Tone ID 1 → full-rate voice frame. The far-side receiver
        // will synthesize it as a 1-2-harmonic voiced sinusoid (not
        // exact to the reference's tone generator, but audible and continuous
        // rather than a dropout).
        let mut conv = HalfToFullConverter::new();
        let input = ambe_plus2_tone_input();
        let out = conv.convert(&input).expect("convert tone");
        let frame = decode_full(&out);
        let mut sink = DecoderState::new();
        let params = dequantize(&frame.info, &mut sink).expect("tone → voice");
        assert!(
            params.harmonic_count() >= 9,
            "tone yielded L={}",
            params.harmonic_count()
        );
    }

    /// A bridged tone has to come out at the level it asked for.
    ///
    /// Eq. 206–209 puts the tone's whole energy on `l₁`/`l₂` and leaves every
    /// other `M̃_l` at exactly zero. The full-rate quantiser works in `log₂ M̃`,
    /// where a zero is `−∞`, and one `−∞` in a DCT block drags the whole block
    /// with it: the transmitted gain lands on Annex E index 0 and the tone comes
    /// out tens of dB down — a DTMF digit converted half → full is inaudible.
    ///
    /// Measured against the Eq. 209 amplitude the tone frame itself carries,
    /// read back through the full-rate dequantizer, i.e. what a receiver would
    /// synthesize. Nothing here is derived from the substitution: the target is
    /// [`tone_to_mbe_params`]'s own output and the measurement is a round trip
    /// through the wire.
    #[test]
    fn half_to_full_tone_reaches_the_level_it_asked_for() {
        use crate::halfrate::dequantize::tone_to_mbe_params;
        use crate::halfrate::frame::ANNEX_T;

        let peak = |p: &MbeParams| {
            p.amplitudes_slice()
                .iter()
                .copied()
                .fold(0f32, |acc, m| if m > acc { m } else { acc })
        };

        let mut checked = 0usize;
        let mut skipped = 0usize;
        for id in 0..=254u8 {
            let Some(row) = ANNEX_T[usize::from(id)] else {
                continue;
            };
            for amplitude in [70u8, 127] {
                let Some(asked) = tone_to_mbe_params(id, amplitude) else {
                    continue;
                };
                // Eq. 207's L̃ and the full-rate grid's L(ω̃₀) do not agree
                // everywhere, and where the tone's own harmonic falls past the
                // full-rate L the component is dropped outright — a separate
                // defect of the bridge's grid, with a separate fix, and not a
                // level question. Six of 308 (I_D, A_D) pairs are affected.
                let l_full = MbeParams::harmonic_count_for(asked.omega_0());
                if row.l1 > l_full || row.l2 > l_full {
                    skipped += 1;
                    continue;
                }
                let input = encode_half(&encode_tone_frame_info(id, amplitude));
                assert_eq!(
                    classify_frame(&decode_half(&input).info),
                    FrameKind::Tone,
                    "tone {id}: the test input is not a tone frame"
                );
                let want = peak(&asked);

                let mut conv = HalfToFullConverter::new();
                let mut sink = DecoderState::new();
                let mut got = 0f32;
                // Four frames: the full-rate quantizer's magnitude predictor
                // needs a frame or two to settle, and a real tone runs for many.
                for _ in 0..4 {
                    let out = conv.convert(&input).expect("half → full on a tone");
                    let back =
                        dequantize(&decode_full(&out).info, &mut sink).expect("full-rate decode");
                    got = peak(&back);
                }
                let db = 20.0 * (got / want).log10();
                // Eq. 206–209's bare two-harmonic spectrum lands 38–41 dB down
                // here; a measured envelope's worst case over the whole table is
                // −15.7 dB. The bound separates those two populations and does
                // not sit at either.
                // Bounded on BOTH sides. The lower bound is what catches the
                // silent-tone fault; the upper one matters because the fix
                // traded "inaudible" for "sometimes too loud", and a one-sided
                // check cannot see that — on the reference corpus the envelope
                // overshoots by up to +27 dB on a frequency sweep. Over this
                // table the measured spread is −15.38 … +1.55 dB (302 pairs,
                // median −0.26, p95 +0.50), so +6 leaves ~4.5 dB of headroom
                // while still catching a gross overshoot.
                assert!(
                    db < 6.0,
                    "tone {id} at A_D {amplitude}: the converted full-rate frame peaks \
                     {db:.1} dB ABOVE the Eq. 209 amplitude it carries ({got} vs {want}) — \
                     the envelope is overshooting"
                );
                assert!(
                    db > -22.0,
                    "tone {id} at A_D {amplitude}: the converted full-rate frame peaks \
                     {db:.1} dB below the Eq. 209 amplitude it carries ({got} vs {want}) — \
                     the tone is effectively silent"
                );
                checked += 1;
            }
        }
        assert_eq!(
            (checked, skipped),
            (302, 6),
            "the Annex T sweep changed shape; the level bound was set against 302 cases"
        );
    }

    /// Every Annex T tone the bridge can carry must reach the wire with the
    /// band holding its strongest harmonic marked **voiced**.
    ///
    /// `b̂₁` is read raw off the emitted frame, never through
    /// [`expand_vuv`](crate::fullrate::dequantize::expand_vuv): that function
    /// rebuilds band-uniform per-harmonic flags whatever the wire said, so a
    /// check that reads voicing back through it cannot tell a voiced band from
    /// an unvoiced one and would pass with the widening deleted.
    ///
    /// The expected band is derived from the emitted frame's own magnitudes —
    /// which harmonic carries the tone — and not from the voicing under test.
    #[test]
    fn half_to_full_tone_voices_the_band_carrying_it() {
        use crate::fullrate::priority::deprioritize as deprioritize_full;
        use crate::halfrate::dequantize::tone_to_mbe_params;
        use crate::halfrate::frame::ANNEX_T;

        let mut checked = 0usize;
        let mut skipped = 0usize;
        let mut unvoiced = Vec::new();
        let mut wrong_band = Vec::new();
        for id in 0..=254u8 {
            let Some(row) = ANNEX_T[usize::from(id)] else {
                continue;
            };
            // Where Eq. 207's L̃ and the full-rate grid's L(ω̃₀) disagree the
            // tone's own harmonic falls past the destination L and the
            // component is dropped outright — a separate defect of the
            // bridge's grid with a separate fix. There is then no voiced
            // harmonic to widen and nothing here to assert.
            let Some(asked) = tone_to_mbe_params(id, 110) else {
                continue;
            };
            let l_full = MbeParams::harmonic_count_for(asked.omega_0());
            if row.l1 > l_full || row.l2 > l_full {
                skipped += 1;
                continue;
            }
            let input = encode_half(&encode_tone_frame_info(id, 110));
            let mut conv = HalfToFullConverter::new();
            let mut sink = DecoderState::new();
            // A run, because the frame the converter settles into is the one a
            // held tone actually transmits.
            let mut last = None;
            for _ in 0..4 {
                let out = conv.convert(&input).expect("half → full on a tone");
                let u = decode_full(&out).info;
                let params = dequantize(&u, &mut sink).expect("full-rate decode");
                last = Some((u, params));
            }
            let (u, params) = last.expect("four frames converted");
            let entry = full_pitch_decode(((u[0] >> 4) & 0xFF) as u8).expect("voice-range b̂₀");
            let b = deprioritize_full(&u, entry.l);
            let b1 = b[1];
            let k = vuv_band_count(entry.l);

            // Where the tone landed, from the magnitudes alone.
            let amps = params.amplitudes_slice();
            let (peak_idx, _) = amps[..usize::from(entry.l)]
                .iter()
                .enumerate()
                .max_by(|a, c| a.1.partial_cmp(c.1).expect("finite"))
                .expect("L >= 9");
            let peak_band = band_for_harmonic(peak_idx as u8 + 1, k);

            if b1 == 0 {
                unvoiced.push((id, row.l1, row.l2));
            } else if entry.l == asked.harmonic_count() && b1 & (1u16 << (k - peak_band)) == 0 {
                // Only where the destination grid kept the bridge's L. Where it
                // did not, the envelope was measured on one harmonic grid and
                // is being carried index-wise onto another, so the strongest
                // harmonic can land anywhere — a separate defect, and not the
                // one this test guards.
                wrong_band.push((id, peak_idx + 1, peak_band, b1));
            }
            checked += 1;
        }
        assert_eq!(
            (checked, skipped),
            (151, 3),
            "the Annex T sweep changed shape"
        );
        assert!(
            unvoiced.is_empty(),
            "{} of {checked} Annex T tones convert to a fully unvoiced full-rate frame \
             (b̂₁ = 0), so the receiver synthesises them as band noise: {:?}...",
            unvoiced.len(),
            &unvoiced[..unvoiced.len().min(8)]
        );
        assert!(
            wrong_band.is_empty(),
            "{} of {checked} Annex T tones voice a band that does not hold the tone's own \
             strongest harmonic (id, peak harmonic, its band, b̂₁): {:?}...",
            wrong_band.len(),
            &wrong_band[..wrong_band.len().min(8)]
        );
    }

    /// The audible consequence, measured on decoded audio: a converted tone
    /// must come out at the level the half-rate decoder gives the same tone
    /// frame, in **both** directions.
    ///
    /// A parameter-domain check cannot see this. The Eq. 209 amplitude survives
    /// conversion to within a fraction of a dB either way, yet a frame whose
    /// `b̂₁` collapsed to zero is synthesised as band noise instead of as a
    /// sinusoid and lands several dB hot — so the fault is only visible after
    /// synthesis, and only to a two-sided bound.
    #[test]
    fn half_to_full_tone_decodes_at_the_level_the_half_rate_decoder_gives_it() {
        use crate::halfrate::dequantize::tone_to_mbe_params;
        use crate::halfrate::frame::ANNEX_T;
        use crate::vocoder::{Rate, Vocoder};

        let level_db = |pcm: &[i16]| {
            let n = pcm.len().max(1) as f64;
            let e: f64 = pcm.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            10.0 * ((e / n) / (32768.0 * 32768.0) + 1e-14).log10()
        };

        let mut worst_hot = (f64::NEG_INFINITY, 0u8);
        let mut worst_quiet = (f64::INFINITY, 0u8);
        let mut checked = 0usize;
        let mut skipped = 0usize;
        for id in 0..=254u8 {
            let Some(row) = ANNEX_T[usize::from(id)] else {
                continue;
            };
            // Same exclusion as `half_to_full_tone_voices_the_band_carrying_it`:
            // where the tone's harmonic falls past the destination L the
            // component is dropped and the frame is tens of dB down for a
            // reason this test is not the guard for.
            let Some(asked) = tone_to_mbe_params(id, 110) else {
                continue;
            };
            let l_full = MbeParams::harmonic_count_for(asked.omega_0());
            if row.l1 > l_full || row.l2 > l_full {
                skipped += 1;
                continue;
            }
            let input = encode_half(&encode_tone_frame_info(id, 110));
            let packed_half = pack_dibits_9(&input);
            let mut half_rx = Vocoder::new(Rate::HalfRate3600x2450);
            let mut conv = HalfToFullConverter::new();
            let mut full_rx = Vocoder::new(Rate::FullRate7200x4400);
            let (mut a, mut b) = (f64::NAN, f64::NAN);
            // Six frames: both decoders interpolate, and a real tone is held.
            for _ in 0..6 {
                let half_pcm = half_rx.decode_bits(&packed_half).expect("half-rate decode");
                let full = conv.convert(&input).expect("half → full on a tone");
                let full_pcm = full_rx
                    .decode_bits(&pack_dibits_18(&full))
                    .expect("full-rate decode");
                a = level_db(&half_pcm);
                b = level_db(&full_pcm);
            }
            // The silence sentinel decodes to digital silence on both sides;
            // a ratio of two floors says nothing.
            if a < -80.0 {
                continue;
            }
            let d = b - a;
            if d > worst_hot.0 {
                worst_hot = (d, id);
            }
            if d < worst_quiet.0 {
                worst_quiet = (d, id);
            }
            checked += 1;
        }
        assert_eq!(
            (checked, skipped),
            (151, 3),
            "the Annex T sweep changed shape; the level bounds were set against 151 tones"
        );
        // Measured over these 151 tones the spread is −11.36 … +1.93 dB with
        // the band widening in place and −14.65 … +10.65 dB without it: the
        // tones whose band vote lands on an unvoiced neighbour go out fully
        // unvoiced, are synthesised as band noise, and read hot. Both bounds
        // sit between the two populations and at neither.
        assert!(
            worst_hot.0 < 5.0,
            "tone {} converts {:.2} dB HOTTER than the half-rate decoder renders the same \
             tone frame — the frame is going out unvoiced and being synthesised as noise",
            worst_hot.1,
            worst_hot.0
        );
        assert!(
            worst_quiet.0 > -13.0,
            "tone {} converts {:.2} dB quieter than the half-rate decoder renders the same \
             tone frame",
            worst_quiet.1,
            worst_quiet.0
        );
    }

    /// MSB-first pack of 36 dibits into the 9 bytes the half-rate `Vocoder`
    /// wants.
    fn pack_dibits_9(dibits: &[u8; DIBITS_PER_FRAME]) -> [u8; 9] {
        let mut out = [0u8; 9];
        for (i, &d) in dibits.iter().enumerate() {
            for bit in 0..2u32 {
                let v = (d >> (1 - bit)) & 1;
                let pos = 2 * i + bit as usize;
                out[pos / 8] |= v << (7 - (pos % 8));
            }
        }
        out
    }

    /// MSB-first pack of 72 dibits into the 18 bytes the full-rate `Vocoder`
    /// wants.
    fn pack_dibits_18(dibits: &[u8; 72]) -> [u8; 18] {
        let mut out = [0u8; 18];
        for (i, &d) in dibits.iter().enumerate() {
            for bit in 0..2u32 {
                let v = (d >> (1 - bit)) & 1;
                let pos = 2 * i + bit as usize;
                out[pos / 8] |= v << (7 - (pos % 8));
            }
        }
        out
    }

    #[test]
    fn half_to_full_erasure_is_idempotent() {
        // Per §2.8 frame-repeat semantics the encoder state must not
        // advance on erasure. Observe externally: two consecutive
        // erasure inputs produce bit-identical output frames. A state
        // advance would produce different predictor residuals even
        // with identical inputs.
        let mut conv = HalfToFullConverter::new();
        let a = conv.convert(&ambe_plus2_erasure_input()).unwrap();
        let b = conv.convert(&ambe_plus2_erasure_input()).unwrap();
        assert_eq!(a, b, "erasure output should be deterministic");
    }

    #[test]
    fn full_to_half_erasure_is_idempotent() {
        let mut conv = FullToHalfConverter::new();
        let a = conv.convert(&imbe_erasure_input()).unwrap();
        let b = conv.convert(&imbe_erasure_input()).unwrap();
        assert_eq!(a, b, "erasure output should be deterministic");
    }

    // ---- Cross-rate predictor integration (§4.5) -----------------------

    #[test]
    fn cross_rate_predictor_state_tracks_target_across_frames() {
        // Analogue of quantize_multi_frame_predictor_state_tracks_decoder
        // in fullrate::dequantize::tests, but for the rate-converter
        // predictor. After each frame, the converter's predictor state
        // should hold exactly the post-§4.5 magnitudes and the
        // target-committed (ω̂₀_B, L̂_B). Feeding a varying-pitch voice
        // sequence exercises both fast and slow paths.
        let mut src_state = DecoderState::new();
        let mut conv = FullToHalfConverter::new();
        // Start: predictor at cold-start init (Annex L row 30).
        let initial_omega_prev = conv.predictor_state().omega_0_prev();
        let initial_l_prev = conv.predictor_state().l_prev();
        assert_eq!(initial_l_prev, 30);

        // Frame 1: a voice frame. Predictor should advance.
        let p1 = sample_params(0.20);
        let a1 = imbe_dibits_from(&p1, &mut src_state);
        let _ = conv.convert(&a1).unwrap();
        // After frame 1, state's ω_prev / l_prev match the half-rate
        // encoder's committed values (not the input's ω₀).
        assert_ne!(
            conv.predictor_state().omega_0_prev(),
            initial_omega_prev,
            "predictor ω_prev should have advanced from cold-start init"
        );
        let omega_after_f1 = conv.predictor_state().omega_0_prev();
        let _l_after_f1 = conv.predictor_state().l_prev();

        // Frame 2: different pitch — forces slow path and further advance.
        let p2 = sample_params(0.25);
        let a2 = imbe_dibits_from(&p2, &mut src_state);
        let _ = conv.convert(&a2).unwrap();
        let omega_after_f2 = conv.predictor_state().omega_0_prev();
        assert!(
            (omega_after_f2 - omega_after_f1).abs() > 1e-4,
            "frame 2 should shift predictor ω_prev (observed Δ = {})",
            (omega_after_f2 - omega_after_f1).abs()
        );

        // The stored omega_0_prev must equal an Annex L grid entry
        // (the half-rate encoder snaps to that grid) — i.e. be
        // exactly representable, not drifted.
        let found = AMBE_PITCH_TABLE.iter().any(|entry| {
            (entry.omega_0 - omega_after_f2).abs() < 1e-6
                && entry.l == conv.predictor_state().l_prev()
        });
        assert!(
            found,
            "predictor ω_prev/l_prev = ({omega_after_f2}, {}) must be an AMBE_PITCH_TABLE entry",
            conv.predictor_state().l_prev(),
        );

        // Erasure frame: should not advance predictor state (§6.3).
        let e = imbe_erasure_input();
        let before_omega = conv.predictor_state().omega_0_prev();
        let before_l = conv.predictor_state().l_prev();
        let _ = conv.convert(&e).unwrap();
        assert_eq!(
            conv.predictor_state().omega_0_prev(),
            before_omega,
            "erasure must not advance predictor ω_prev"
        );
        assert_eq!(
            conv.predictor_state().l_prev(),
            before_l,
            "erasure must not advance predictor l_prev"
        );
    }

    #[test]
    fn set_predictor_enabled_false_skips_blend() {
        // With the predictor off, the converter should route parameters
        // directly from source decoder to target encoder without
        // advancing CrossRatePredictorState. Confirmed by observing
        // that predictor state stays at cold-start across a frame.
        let mut src_state = DecoderState::new();
        let mut conv = FullToHalfConverter::new();
        conv.set_predictor_enabled(false);
        let initial_omega = conv.predictor_state().omega_0_prev();
        let initial_l = conv.predictor_state().l_prev();

        let p = sample_params(0.18);
        let a = imbe_dibits_from(&p, &mut src_state);
        let _ = conv.convert(&a).unwrap();

        assert_eq!(
            conv.predictor_state().omega_0_prev(),
            initial_omega,
            "predictor should not advance when disabled"
        );
        assert_eq!(
            conv.predictor_state().l_prev(),
            initial_l,
            "predictor l_prev should not advance when disabled"
        );
    }

    /// Run a pitch-moving full-rate sequence through a converter configured with
    /// `blend` and return the emitted half-rate frames.
    fn full_to_half_run(blend: CrossRateBlend) -> Vec<[u8; DIBITS_PER_FRAME]> {
        let mut src_state = DecoderState::new();
        let mut conv = FullToHalfConverter::new();
        conv.set_cross_rate_blend(blend);
        let mut out = Vec::new();
        // A pitch that moves every frame keeps |R − 1| above the 1% fast-path
        // threshold, which is the only condition under which the mix runs at all.
        for k in 0..24u32 {
            let omega = 0.16 + 0.02 * ((k as f32) * 0.7).sin();
            let a = imbe_dibits_from(&sample_params(omega), &mut src_state);
            out.push(conv.convert(&a).expect("convert"));
        }
        out
    }

    #[test]
    fn cross_rate_blend_mode_is_wired_to_the_emitted_frames() {
        // The setter has to reach the bits, and the default has to be the
        // shipped inert mix. Three runs of the same input:
        //  - default and an explicit Level(0.0) must be bit-identical (the
        //    default is not silently something else);
        //  - Level(0.65) must differ (the mode reaches the quantizer at all);
        //  - ShapeOnly(0.65) must differ from both (the two modes are distinct
        //    all the way to the wire, not just inside the predictor).
        let default_run = full_to_half_run(CrossRateBlend::default());
        let explicit_zero = full_to_half_run(CrossRateBlend::Level(0.0));
        assert_eq!(
            default_run, explicit_zero,
            "the default mix mode must be Level(0.0)"
        );

        let level_run = full_to_half_run(CrossRateBlend::Level(0.65));
        assert_ne!(
            default_run, level_run,
            "Level(0.65) did not reach the emitted frames"
        );

        let shape_run = full_to_half_run(CrossRateBlend::ShapeOnly(0.65));
        assert_ne!(
            default_run, shape_run,
            "ShapeOnly(0.65) did not reach the emitted frames"
        );
        assert_ne!(
            level_run, shape_run,
            "ShapeOnly and Level produced identical wire frames; the level \
             restoration is not doing anything"
        );
    }

    #[test]
    fn half_to_full_honours_the_cross_rate_blend_mode() {
        // Same wiring check in the other direction, which uses a different
        // target-pitch lookup and so a separate call site.
        let run = |blend: CrossRateBlend| {
            let mut src_state = HalfDecoderState::new();
            let mut conv = HalfToFullConverter::new();
            conv.set_cross_rate_blend(blend);
            assert_eq!(conv.cross_rate_blend(), blend);
            let mut out = Vec::new();
            for k in 0..24u32 {
                let omega = 0.16 + 0.02 * ((k as f32) * 0.7).sin();
                let a = ambe_plus2_dibits_from(&sample_params(omega), &mut src_state);
                out.push(conv.convert(&a).expect("convert"));
            }
            out
        };
        let default_run = run(CrossRateBlend::default());
        assert_eq!(default_run, run(CrossRateBlend::Level(0.0)));
        assert_ne!(
            default_run,
            run(CrossRateBlend::Level(0.65)),
            "Level(0.65) did not reach the emitted frames"
        );
        assert_ne!(
            default_run,
            run(CrossRateBlend::ShapeOnly(0.65)),
            "ShapeOnly(0.65) did not reach the emitted frames"
        );
    }

    #[test]
    fn cross_rate_predictor_is_state_separate_from_decoder_and_encoder() {
        // Regression guard for the state-separation invariant.
        // After a frame, the converter's three predictor states
        // (source decoder's, cross-rate's, target encoder's) must
        // have distinct addresses — we only own one of each via the
        // three accessor methods, but confirming they're independent
        // structs is enough: the borrow-checker + type system already
        // enforce that.
        let conv = FullToHalfConverter::new();
        let _decoder = conv.decoder_state();
        let _encoder = conv.encoder_state();
        let _predictor = conv.predictor_state();
        // If this test compiles, the three accessors exist and
        // return distinct types. Runtime invariant is enforced by
        // each state's update rule not touching the others.
    }
}
