//! Unified [`Vocoder`] handle — a façade over the codec engine and the
//! rate-specific wire layers.
//!
//! One [`Vocoder`] is one channel: it owns the [`crate::engine`] state and
//! offers the same `encode` / `decode` / `reset` operations whatever [`Rate`]
//! it was built for, so calling code does not branch on the wire format.
//!
//! ## The two layers
//!
//! ```text
//!   PCM  <->  [ engine ]  <->  payload bits  <->  [ wire layer ]  <->  wire bytes
//!             analysis/synthesis    49 or 88          FEC, interleave,
//!             (bit-exact)                             priority, packing
//! ```
//!
//! The engine is codec-only: it performs no error correction and has no
//! bad-frame input. FEC, de-interleaving and erasure marking are this crate's
//! job and live in [`crate::fullrate`] / [`crate::halfrate`], which stay public
//! for consumers that need frame-by-frame wire access.
//!
//! ## Quick start
//!
//! ```rust
//! use blip25_vocoder::vocoder::{Rate, Vocoder};
//!
//! // Encode: PCM in → bits out.
//! let tx = Vocoder::new(Rate::FullRate7200x4400);
//! let pcm = vec![0i16; 160 * 10]; // any length; one frame per 20 ms
//! let frames = tx.encode(&pcm);   // Vec of 18-byte FEC frames
//! assert!(frames.iter().all(|f| f.len() == 18));
//!
//! // Decode: bits → PCM (separate channel, separate state).
//! let mut rx = Vocoder::new(Rate::FullRate7200x4400);
//! for f in &frames {
//!     assert_eq!(rx.decode_bits(f).expect("decode").len(), 160);
//! }
//! ```
//!
//! ## Rate is fixed for a handle's lifetime
//!
//! [`Rate`] is chosen at construction and cannot be changed; build a new
//! handle to switch. The codec's state — predictor history, the previous
//! frame held for concealment, the enhancement filters — is quantized on the
//! grid of the rate that produced it, so carrying it across a rate change
//! would mean reinterpreting one rate's history under another's grid. A fresh
//! handle is the honest way to express the discontinuity.
//!
//! An FNE bridging rates wants [`Transcoder`] rather than two handles: it
//! converts in the parameter domain and keeps the source and destination
//! states apart for exactly this reason.

use crate::enhancement::{self, EnhancementMode, EnhancementState};
use crate::mbe_params::MbeParams;

/// How badly the transport's FEC layer had to work on a frame, handed to the
/// codec so it can conceal errors it would otherwise be blind to.
///
/// Re-exported from the engine, where it is a typed view of the reference
/// decoder's own control block rather than an invention of this crate:
/// `error_rate` is control word `+0x24` and `lost` / `mute` are the forcing
/// bits of word `+0x32`. Pass one to [`Vocoder::decode_bits_with_status`]
/// when your own FEC layer — not this crate's — is the thing that measured
/// the channel.
pub use crate::engine::FrameStatus;

/// 8 kHz mono — the only sample rate the vocoder produces.
const SAMPLE_RATE_HZ: f32 = 8_000.0;

/// Number of i16 PCM samples per 20 ms frame at 8 kHz. Constant across every
/// supported rate.
pub const FRAME_SAMPLES: usize = 160;

/// Wire format of one channel. Selects both the mode the codec core runs (full
/// or half) and the wire layer wrapped around its payload bits.
///
/// These four are the whole P25 rate space, so this enum is deliberately **not**
/// `#[non_exhaustive]`: a consumer matches it exhaustively and the compiler
/// tells them if that ever stops being true. A fifth rate would be a breaking
/// change to this crate, which is the honest signal — a carrier whose framing
/// differs reaches the wire layers directly (see [`crate::halfrate::frame`])
/// rather than arriving here as a new variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Rate {
    /// P25 Phase 1 FDMA full rate — the vocoder TIA-102.BABA-A titles "IMBE".
    /// 18-byte FEC frame (72 dibits).
    /// 7 200 bps total / 4 400 bps voice + 2 800 bps FEC.
    FullRate7200x4400,
    /// Full-rate info-only — same codec as [`Self::FullRate7200x4400`] with the Annex H
    /// Golay/Hamming/PN FEC layer stripped. 11-byte wire frame (88 prioritized
    /// info bits packed MSB-first). 4 400 bps total (= voice). Byte layout
    /// matches JMBE / OP25 / the reference `p25_nofec` convention bit-for-bit.
    /// Use this for storage when you specifically need an info-only IMBE
    /// archive; otherwise prefer the FEC-bearing [`Self::FullRate7200x4400`] — see
    /// `docs/wire_formats_and_storage.md`.
    FullRate4400x4400,
    /// P25 Phase 2 TDMA half rate — BABA-A's "Half-Rate Vocoder", the
    /// reference's rate 33. 9-byte FEC frame (36 dibits).
    /// 3 600 bps total / 2 450 bps voice + 1 150 bps FEC. The same codec frame
    /// carries DMR Tier II/III and NXDN voice, which differ only in interleave
    /// and burst framing.
    HalfRate3600x2450,
    /// Half-rate info-only — same 49 info bits as
    /// [`Self::HalfRate3600x2450`] with the Golay/Hamming/PN FEC layer
    /// stripped. 7-byte wire frame: the 49 info bits in **natural / "AMBE_d"
    /// order** plus 7 trailing pad bits. 2 450 bps total (= voice).
    ///
    /// Byte layout is `û₀(12)‖û₁(12)‖û₂(11)‖û₃(14)` laid down sequentially,
    /// MSB-first, by [`crate::halfrate::frame::pack_natural`]. This is the
    /// 7-octet payload P25 encrypts (TIA-102.AAAD-B §1.2 — encryption sits
    /// between the vocoder and the burst, so a receiver strips FEC to reach
    /// it), and the layout mbelib, DSD, MMDVM, and an IDAS/NXDN
    /// over-the-air wire use.
    ///
    /// For storage, the FEC-bearing [`Self::HalfRate3600x2450`] is
    /// recommended — see `docs/wire_formats_and_storage.md`.
    HalfRate2450x2450,
}

impl Rate {
    /// The engine mode this rate runs: the two IMBE rates share one codec, the
    /// two AMBE+2 rates the other.
    #[inline]
    const fn engine_mode(self) -> crate::engine::Mode {
        match self {
            Rate::FullRate7200x4400 | Rate::FullRate4400x4400 => crate::engine::Mode::FullRate,
            Rate::HalfRate3600x2450 | Rate::HalfRate2450x2450 => crate::engine::Mode::HalfRate,
        }
    }

    /// Number of bytes in one wire frame at this rate. Includes FEC for the
    /// FEC-bearing variants ([`Self::FullRate7200x4400`],
    /// [`Self::HalfRate3600x2450`]) and is just the packed info bits for the
    /// no-FEC variants ([`Self::FullRate4400x4400`], [`Self::HalfRate2450x2450`]).
    #[inline]
    pub const fn fec_frame_bytes(self) -> usize {
        match self {
            Rate::FullRate7200x4400 => 18,
            Rate::FullRate4400x4400 => 11,
            Rate::HalfRate3600x2450 => 9,
            Rate::HalfRate2450x2450 => 7,
        }
    }

    /// Soft-decision channel-bit count for one frame — the number of LLRs
    /// [`Vocoder::decode_soft`] expects (one per FEC channel bit). `None` for
    /// the no-FEC info-only rates, which carry no FEC layer to soft-decode.
    pub const fn soft_frame_bits(self) -> Option<usize> {
        match self {
            Rate::FullRate7200x4400 => Some(144),
            Rate::HalfRate3600x2450 => Some(72),
            Rate::FullRate4400x4400 | Rate::HalfRate2450x2450 => None,
        }
    }

    /// PCM samples per frame (always 160 at 8 kHz / 20 ms; provided for
    /// symmetry with [`Self::fec_frame_bytes`]).
    #[inline]
    pub const fn frame_samples(self) -> usize {
        FRAME_SAMPLES
    }

    /// Number of info vectors one frame carries at this rate — the length
    /// [`Vocoder::decode_info`] expects. 4 for the half-rate AMBE+2 codec
    /// (`û₀..û₃`), 8 for full-rate IMBE (`b̂₀..b̂₇`), regardless of whether the
    /// rate carries FEC.
    #[inline]
    pub const fn info_vector_count(self) -> usize {
        match self {
            Rate::HalfRate3600x2450 | Rate::HalfRate2450x2450 => 4,
            Rate::FullRate7200x4400 | Rate::FullRate4400x4400 => 8,
        }
    }

    /// One wire frame, [`Self::fec_frame_bytes`] long, marked as an **erasure**.
    ///
    /// Feed this to [`Vocoder::decode_bits`] in place of a frame the transport
    /// lost. The engine holds the previous good frame and fades it out over
    /// successive erasures ([`FrameDisposition::is_concealed`]) — the same
    /// concealment a radio applies when the channel drops frames.
    ///
    /// Both codecs mark an erasure in-band, by placing the pitch index outside
    /// its valid range: `b0 ∈ [120, 127]` for the half-rate AMBE+2 wire (the
    /// `û₀(11..6) ∈ {0x3C, 0x3D}` erasure escape), `b0 > 207` for IMBE. This
    /// returns the maximal marker in each range, so no single bit error can
    /// turn it back into a decodable pitch. Every other field is zero: nothing
    /// downstream of the erasure test reads them.
    pub fn erasure_frame(self) -> Vec<u8> {
        match self {
            Rate::HalfRate3600x2450 => {
                ambe_plus2_pipeline::info_vec_to_fec_bytes(&ambe_erasure_info()).to_vec()
            }
            Rate::HalfRate2450x2450 => {
                crate::halfrate::frame::pack_natural(&ambe_erasure_info()).to_vec()
            }
            Rate::FullRate7200x4400 => {
                imbe_pipeline::info_vec_to_fec_bytes(&IMBE_ERASURE_INFO).to_vec()
            }
            Rate::FullRate4400x4400 => {
                imbe_pipeline::info_vec_to_bytes(&IMBE_ERASURE_INFO).to_vec()
            }
        }
    }
}

/// Half-rate pitch index marking an erasure: the top of the `[120, 127]`
/// erasure escape range (`û₀(11..6) == 0x3C`).
const AMBE_ERASURE_B0: u16 = 127;

/// The half-rate info vector carrying [`AMBE_ERASURE_B0`], every other
/// parameter zero.
fn ambe_erasure_info() -> [u16; 4] {
    let mut b = [0u16; crate::halfrate::priority::PARAM_COUNT];
    b[0] = AMBE_ERASURE_B0;
    crate::halfrate::priority::prioritize(&b)
}

/// The IMBE info vector whose pitch index decodes to 252 — outside the valid
/// `[0, 207]` range, so the frame is an erasure marker.
const IMBE_ERASURE_INFO: [u16; 8] = [0xFFF, 0, 0, 0, 0, 0, 0, 0];

/// Per-frame statistics recorded by the most recent [`Vocoder::encode_pcm`] or
/// [`Vocoder::decode_bits`] call. `None` until at least one frame has been
/// processed.
///
/// Encode-side fills only `analysis`; decode-side fills only `decode`.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FrameStats {
    /// Stats from the last encoded frame.
    pub analysis: Option<AnalysisStats>,
    /// Stats from the last decoded frame.
    pub decode: Option<DecodeStats>,
}

/// Encode-side per-frame stats.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct AnalysisStats {
    /// What the engine reported about the frame it analysed.
    pub disposition: FrameDisposition,
    /// The MBE parameters the emitted wire bits carry, recovered by running the
    /// rate's dequantizer over them. `None` when the emitted frame carries no
    /// parameter set the dequantizer accepts (an erasure or tone frame, or a
    /// payload the grid rejects).
    pub params: Option<MbeParams>,
}

/// What the engine reported about a frame, on either side.
///
/// This is a **bit word**, not an enumeration: several flags can be set at once
/// (a tone frame reports `0x8002`, tone *and* speech). The raw word is exposed
/// via [`Self::raw`] and the flags this crate has identified have predicates
/// below; unknown bits are preserved rather than discarded.
///
/// Flag meanings come from the reference's own classification and have been
/// confirmed against it — [`Self::is_speech`] is set on every frame of the
/// conformance speech corpus, and [`Self::is_concealed`] is what an in-band
/// erasure marker ([`Rate::erasure_frame`]) produces.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameDisposition(i16);

impl FrameDisposition {
    /// Set when the frame carries speech, or is a background frame whose noise
    /// floor moved. In practice this is set on essentially every frame of real
    /// audio, on both sides — it is not a voiced/unvoiced discriminator.
    pub const SPEECH: i16 = 0x0002;
    /// Decode side: the payload was rejected as unusable and the previous
    /// frame was held instead, fading to silence over the following frames.
    pub const CONCEALED: i16 = 0x0020;
    /// Encode side: the activity timer was re-armed on this frame.
    pub const ACTIVITY: i16 = 0x0400;
    /// The frame is an in-band signalling tone rather than voice.
    pub const TONE: i16 = -0x8000; // 0x8000 as i16

    /// Wrap a raw disposition word.
    #[inline]
    pub const fn from_raw(word: i16) -> Self {
        Self(word)
    }

    /// The raw disposition word, including any bits this crate does not name.
    #[inline]
    pub const fn raw(self) -> i16 {
        self.0
    }

    /// Whether a given flag is set.
    #[inline]
    pub const fn has(self, flag: i16) -> bool {
        self.0 & flag != 0
    }

    /// [`Self::SPEECH`] — speech, or a background frame whose noise floor moved.
    #[inline]
    pub const fn is_speech(self) -> bool {
        self.has(Self::SPEECH)
    }

    /// [`Self::TONE`] — an in-band signalling tone frame.
    #[inline]
    pub const fn is_tone(self) -> bool {
        self.has(Self::TONE)
    }

    /// [`Self::CONCEALED`] — the payload was unusable, so this frame's audio is
    /// a held/faded copy of the previous one rather than a decode of its own
    /// bits.
    #[inline]
    pub const fn is_concealed(self) -> bool {
        self.has(Self::CONCEALED)
    }

    /// [`Self::ACTIVITY`] — the encoder re-armed its activity timer.
    #[inline]
    pub const fn activity_rearmed(self) -> bool {
        self.has(Self::ACTIVITY)
    }

    /// Whether this frame's audio came from its own payload rather than from
    /// concealment. The inverse of [`Self::is_concealed`].
    #[inline]
    pub const fn is_decoded(self) -> bool {
        !self.is_concealed()
    }
}

/// The engine crate classifies the disposition into a small enum. It is really
/// a bit word — a tone frame reports `0x8002`, tone *and* speech — so take the
/// raw word and model the flags here rather than the classification.
#[inline]
fn disposition_of(f: crate::engine::Frame) -> FrameDisposition {
    FrameDisposition::from_raw(f.code())
}

/// Decode-side per-frame stats.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DecodeStats {
    /// FEC error count on the leading codeword (Golay c̃₀ for full-rate, AMBE+2
    /// c̃₀ for half-rate). Always 0 on the no-FEC rates, which carry no FEC.
    pub epsilon_0: u8,
    /// Total FEC error count across all coded vectors of the frame. Always 0 on
    /// the no-FEC rates.
    pub epsilon_t: u8,
    /// What the engine reported about this frame, including whether it was
    /// concealed rather than decoded.
    pub disposition: FrameDisposition,
}

/// Errors that can surface from [`Vocoder`] operations. Wraps the rate-specific
/// error types so a single `?` covers any Vocoder call.
#[derive(Debug)]
#[non_exhaustive]
pub enum VocoderError {
    /// Input PCM slice was the wrong length. Always `frame_samples()`.
    WrongPcmLength {
        /// Number of samples the channel expected.
        expected: usize,
        /// Number of samples the caller passed.
        got: usize,
    },
    /// Input bit slice was the wrong length. Always `fec_frame_bytes()`.
    WrongBitsLength {
        /// Number of bytes the channel expected.
        expected: usize,
        /// Number of bytes the caller passed.
        got: usize,
    },
    /// Info-vector slice was the wrong length. Always `info_vector_count()`.
    WrongInfoLength {
        /// Number of info vectors the channel expected.
        expected: usize,
        /// Number the caller passed.
        got: usize,
    },
    /// Wire-quantize failure during transcode (a rate grid rejected the
    /// parameters, or the predictor returned a non-finite value).
    Quantize(String),
    /// Requested transcode direction is not supported. The pair of rates has no
    /// parameter-domain converter wired up.
    UnsupportedTranscode {
        /// Rate of the input FEC frame.
        from: Rate,
        /// Rate of the output FEC frame.
        to: Rate,
    },
    /// Soft-decision decode was requested on a rate with no FEC layer to
    /// soft-decode (the no-FEC info-only rates).
    SoftUnsupported {
        /// The rate that has no FEC to soft-decode.
        rate: Rate,
    },
    /// Soft-decision LLR slice was the wrong length. Always
    /// [`Rate::soft_frame_bits`] for the rate.
    WrongSoftLength {
        /// Number of LLRs the rate expected.
        expected: usize,
        /// Number of LLRs the caller passed.
        got: usize,
    },
}

impl core::fmt::Display for VocoderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VocoderError::WrongPcmLength { expected, got } => {
                write!(f, "expected {expected} PCM samples per frame, got {got}")
            }
            VocoderError::WrongBitsLength { expected, got } => {
                write!(f, "expected {expected} FEC bytes per frame, got {got}")
            }
            VocoderError::WrongInfoLength { expected, got } => {
                write!(f, "expected {expected} info vectors per frame, got {got}")
            }
            VocoderError::Quantize(msg) => write!(f, "quantize error: {msg}"),
            VocoderError::UnsupportedTranscode { from, to } => {
                write!(f, "unsupported transcode direction: {from:?} -> {to:?}")
            }
            VocoderError::SoftUnsupported { rate } => {
                write!(
                    f,
                    "soft-decision decode not supported for {rate:?} (no FEC layer to soft-decode)"
                )
            }
            VocoderError::WrongSoftLength { expected, got } => {
                write!(f, "expected {expected} soft LLRs per frame, got {got}")
            }
        }
    }
}

impl std::error::Error for VocoderError {}

// ===========================================================================
// Engine <-> wire adapters
//
// The engine speaks payload bits: 49 for AMBE+2, 88 for IMBE, one bit per
// array slot, MSB of the channel frame first. Those bits ARE the info vectors
// û concatenated MSB-first at each rate's field widths, so the whole adapter is
// a width-driven pack/unpack. The wire layer then FEC-codes, interleaves and
// packs them into the format the rate names.
// ===========================================================================

/// Widths of the AMBE+2 half-rate info vectors, in payload-bit order.
const AMBE_INFO_WIDTHS: [u8; 4] = crate::halfrate::frame::INFO_WIDTHS;
/// Widths of the IMBE full-rate info vectors, in payload-bit order.
const IMBE_INFO_WIDTHS: [u8; 8] = crate::fullrate::frame::INFO_WIDTHS;

/// The engine's per-rate payload buffer: one bit per slot, sized for the larger
/// of the two codecs.
type Payload = [i16; 88];

/// Spread info vectors into engine payload bits.
fn info_to_payload<const N: usize>(info: &[u16; N], widths: &[u8; N], out: &mut Payload) -> usize {
    let mut at = 0usize;
    for (i, &w) in widths.iter().enumerate() {
        for k in (0..w as usize).rev() {
            out[at] = ((info[i] >> k) & 1) as i16;
            at += 1;
        }
    }
    at
}

/// Gather engine payload bits back into info vectors.
fn payload_to_info<const N: usize>(payload: &Payload, widths: &[u8; N]) -> [u16; N] {
    let mut out = [0u16; N];
    let mut at = 0usize;
    for (i, &w) in widths.iter().enumerate() {
        let mut v = 0u16;
        for _ in 0..w {
            v = (v << 1) | (payload[at] & 1) as u16;
            at += 1;
        }
        out[i] = v;
    }
    out
}

/// Clamp a frame's total FEC error count into the byte [`DecodeStats`] carries.
/// A vector that decoded as uncorrectable already reports `u8::MAX`, so the sum
/// can exceed a byte on a badly damaged frame.
#[inline]
fn saturate(total: u16) -> u8 {
    total.min(u8::MAX as u16) as u8
}

/// The engine, plus the per-rate translation between its payload bits and the
/// wire bytes the rate names.
struct Engine {
    rate: Rate,
    encoder: crate::engine::Encoder,
    decoder: crate::engine::Decoder,
}

impl Engine {
    fn new(rate: Rate) -> Self {
        let mode = rate.engine_mode();
        Self {
            rate,
            encoder: crate::engine::Encoder::new(mode),
            decoder: crate::engine::Decoder::new(mode),
        }
    }

    fn reset(&mut self) {
        self.encoder.reset();
        self.decoder.reset();
    }

    /// Analyse one 160-sample frame and wrap the engine's payload in this
    /// rate's wire format.
    fn encode(
        &mut self,
        pcm: &[i16; FRAME_SAMPLES],
    ) -> (Vec<u8>, crate::engine::Frame, EmittedInfo) {
        let mut payload: Payload = [0; 88];
        let n = self.rate.engine_mode().payload_bits();
        let disposition = self.encoder.encode(pcm, &mut payload[..n]);
        match self.rate {
            Rate::HalfRate3600x2450 | Rate::HalfRate2450x2450 => {
                let info = payload_to_info(&payload, &AMBE_INFO_WIDTHS);
                let bytes = if self.rate == Rate::HalfRate3600x2450 {
                    ambe_plus2_pipeline::info_vec_to_fec_bytes(&info).to_vec()
                } else {
                    crate::halfrate::frame::pack_natural(&info).to_vec()
                };
                (bytes, disposition, EmittedInfo::Ambe(info))
            }
            Rate::FullRate7200x4400 | Rate::FullRate4400x4400 => {
                let info = payload_to_info(&payload, &IMBE_INFO_WIDTHS);
                let bytes = if self.rate == Rate::FullRate7200x4400 {
                    imbe_pipeline::info_vec_to_fec_bytes(&info).to_vec()
                } else {
                    imbe_pipeline::info_vec_to_bytes(&info).to_vec()
                };
                (bytes, disposition, EmittedInfo::Imbe(info))
            }
        }
    }

    /// Strip this rate's wire format and synthesise the payload. `bits` is
    /// pre-validated to [`Rate::fec_frame_bytes`].
    ///
    /// `upstream` is the channel quality a caller measured with its own FEC
    /// layer; when present it wins over anything this crate's wire layer
    /// concluded, because a caller that ran the FEC itself is the only party
    /// that knows. `None` falls back to the local wire layer, which reports a
    /// clean frame at the no-FEC rates — correct only when they are carrying
    /// storage rather than a live link.
    fn decode(
        &mut self,
        bits: &[u8],
        upstream: Option<crate::engine::FrameStatus>,
    ) -> (Vec<i16>, DecodeStats) {
        let mut payload: Payload = [0; 88];
        let (n, epsilon_0, epsilon_t) = match self.rate {
            Rate::HalfRate3600x2450 => {
                let frame = crate::halfrate::frame::decode_frame(&unpack_dibits_n::<36>(bits));
                let n = info_to_payload(&frame.info, &AMBE_INFO_WIDTHS, &mut payload);
                (n, frame.errors[0], saturate(frame.error_total()))
            }
            Rate::HalfRate2450x2450 => {
                let info = crate::halfrate::frame::unpack_natural(bits);
                (
                    info_to_payload(&info, &AMBE_INFO_WIDTHS, &mut payload),
                    0,
                    0,
                )
            }
            Rate::FullRate7200x4400 => {
                let frame = crate::fullrate::frame::decode_frame(&unpack_dibits_n::<72>(bits));
                let n = info_to_payload(&frame.info, &IMBE_INFO_WIDTHS, &mut payload);
                (n, frame.errors[0], saturate(frame.error_total()))
            }
            Rate::FullRate4400x4400 => {
                let info = imbe_pipeline::bytes_to_info_vec(bits);
                (
                    info_to_payload(&info, &IMBE_INFO_WIDTHS, &mut payload),
                    0,
                    0,
                )
            }
        };
        // The engine acts on this: it drives error concealment, and a decoder
        // that always claims a clean frame has concealment switched off exactly
        // when the channel needs it. `FrameStatus::new` clamps into the live
        // 0..=15 range, so a saturated count cannot land in the engine's dead
        // band, where it would silently disable concealment on the worst frames.
        //
        // An uncorrectable leading codeword means the info bits are not
        // trustworthy at all, so the frame is reported lost — hold the previous
        // one and fade — rather than synthesised from bits the FEC could not
        // vouch for. That is the BABA-A §4.1 repeat trigger. Only the half-rate
        // wire can raise it: its `c̃₀` is extended Golay [24, 12], which detects
        // four errors and says so, while every IMBE vector is Golay [23, 12] or
        // Hamming [15, 11] — perfect codes that always return an answer.
        //
        // The no-FEC rates carry no FEC layer to have corrected anything, so
        // their frames are clean by construction.
        let status = match upstream {
            Some(supplied) => supplied,
            None if epsilon_0 == u8::MAX => crate::engine::FrameStatus::LOST,
            None => crate::engine::FrameStatus::new(u32::from(epsilon_t), false),
        };
        self.synth(&payload, n, status, epsilon_0, epsilon_t)
    }

    /// Spread already-recovered info vectors into the payload and synthesise.
    /// The wire layer is not involved: the caller's FEC layer produced these.
    fn decode_info(
        &mut self,
        info: &[u16],
        status: crate::engine::FrameStatus,
    ) -> (Vec<i16>, DecodeStats) {
        let mut payload: Payload = [0; 88];
        let n = match self.rate {
            Rate::HalfRate3600x2450 | Rate::HalfRate2450x2450 => {
                let v: [u16; 4] = core::array::from_fn(|i| info[i]);
                info_to_payload(&v, &AMBE_INFO_WIDTHS, &mut payload)
            }
            Rate::FullRate7200x4400 | Rate::FullRate4400x4400 => {
                let v: [u16; 8] = core::array::from_fn(|i| info[i]);
                info_to_payload(&v, &IMBE_INFO_WIDTHS, &mut payload)
            }
        };
        // Nothing local corrected anything, so the reported epsilons are zero;
        // the severity the caller supplied is what reaches the engine.
        self.synth(&payload, n, status, 0, 0)
    }

    /// Drive the engine and package the stats. Shared by both entry points.
    fn synth(
        &mut self,
        payload: &Payload,
        n: usize,
        status: crate::engine::FrameStatus,
        epsilon_0: u8,
        epsilon_t: u8,
    ) -> (Vec<i16>, DecodeStats) {
        let mut pcm = vec![0i16; FRAME_SAMPLES];
        let disposition = self
            .decoder
            .decode_with_status(&payload[..n], status, &mut pcm);
        (
            pcm,
            DecodeStats {
                epsilon_0,
                epsilon_t,
                disposition: disposition_of(disposition),
            },
        )
    }
}

/// The info vectors an encode call emitted, kept at their native width so
/// [`AnalysisStats`] can recover the parameters they carry.
enum EmittedInfo {
    Ambe([u16; 4]),
    Imbe([u16; 8]),
}

/// Cross-frame state the encode-side parameter readback needs. The dequantizers
/// are stateful (log-amplitude prediction, previous L / γ), so recovering the
/// parameters an emitted frame carries requires walking every frame in order —
/// exactly as a decoder would.
#[derive(Clone)] // CCP patch: for Vocoder::copy_encode_state_from
enum ParamReadback {
    Ambe(Box<crate::halfrate::dequantize::DecoderState>),
    Imbe(Box<crate::fullrate::dequantize::DecoderState>),
}

impl ParamReadback {
    fn new(rate: Rate) -> Self {
        match rate {
            Rate::HalfRate3600x2450 | Rate::HalfRate2450x2450 => {
                ParamReadback::Ambe(Box::default())
            }
            Rate::FullRate7200x4400 | Rate::FullRate4400x4400 => {
                ParamReadback::Imbe(Box::default())
            }
        }
    }

    fn read(&mut self, emitted: &EmittedInfo) -> Option<MbeParams> {
        match (self, emitted) {
            (ParamReadback::Ambe(state), EmittedInfo::Ambe(info)) => {
                crate::halfrate::dequantize::dequantize(info, state).ok()
            }
            (ParamReadback::Imbe(state), EmittedInfo::Imbe(info)) => {
                crate::fullrate::dequantize::dequantize(info, state).ok()
            }
            // Rate fixes both arms at construction, so the mismatched pairs are
            // unreachable; report "no parameters" rather than panic.
            _ => None,
        }
    }
}

// ===========================================================================
// Vocoder
// ===========================================================================

/// One codec channel: the engine plus this rate's wire layer.
///
/// Owns all per-rate state internally. One [`Vocoder`] is *both* encoder and
/// decoder for a single channel direction; consumers running bidirectional
/// voice typically allocate two — one each direction — to mirror the
/// reference's per-direction state isolation.
///
/// State is not `Sync`; one channel = one thread. State is `Send`, so the
/// channel can move between threads.
pub struct Vocoder {
    rate: Rate,
    engine: Engine,
    last_stats: FrameStats,
    /// Optional post-decode enhancement chain (biquad + compressor + boundary
    /// fade). [`EnhancementMode::None`] by default (a strict no-op), so the
    /// shipped path is sample-identical to the reference codec. This is the one
    /// standalone DSP hook kept for the future AGC / noise-removal audio chain;
    /// see [`crate::enhancement`].
    enhancement: EnhancementMode,
    enhancement_state: EnhancementState,
    /// Disposition of the PREVIOUS decoded frame, so the enhancement chain's
    /// boundary fade can arm on a `Repeat` → decoded transition. Tracked here
    /// rather than read from `last_stats` because `last_stats` has already been
    /// overwritten with the current frame by the time `apply` runs.
    prev_disposition: FrameDisposition,
    /// Decoder-side mirror used to recover the parameters an encoded frame
    /// carries, for [`AnalysisStats::params`].
    #[cfg(feature = "encode")]
    readback: ParamReadback,
}

impl Vocoder {
    /// Open a new channel at the given rate, all state cold.
    pub fn new(rate: Rate) -> Self {
        Self {
            rate,
            engine: Engine::new(rate),
            last_stats: FrameStats::default(),
            // Enhancement: NONE by default. The shipped decode path IS the
            // reference codec, sample for sample; layering a post-filter would
            // make the output diverge from it. Opt in via set_enhancement.
            enhancement: EnhancementMode::None,
            enhancement_state: EnhancementState::default(),
            prev_disposition: FrameDisposition::default(),
            #[cfg(feature = "encode")]
            readback: ParamReadback::new(rate),
        }
    }

    /// CCP patch: switch the reference engine's in-band tone detector on or off
    /// for encoding. Off by default, as upstream (which never enables it).
    /// With it on, steady signalling tones are coded as tone frames instead of
    /// being analysed as speech; speech encodes the same either way.
    /// Configuration, so it survives [`Self::reset`].
    #[cfg(feature = "encode")]
    pub fn set_tone_detection(&mut self, on: bool) {
        self.engine
            .encoder
            .set_flags(if on { crate::engine::ENC_FLAG_TONE_DETECT } else { 0 });
    }

    /// CCP patch: whether [`Self::set_tone_detection`] is on.
    #[cfg(feature = "encode")]
    pub fn tone_detection(&self) -> bool {
        self.engine.encoder.flags() & crate::engine::ENC_FLAG_TONE_DETECT != 0
    }

    /// CCP patch: take over another channel's encode-side cross-frame state
    /// (the engine encoder state and the parameter readback), so this channel
    /// carries on from the frame that one emitted. For running two encoders
    /// over the same audio and sending one frame or the other: the one whose
    /// frame was not sent copies the other's state, so both predict from what
    /// the decoder actually received. Configuration (tone detection,
    /// enhancement) is not copied. Both channels must be at the same rate.
    #[cfg(feature = "encode")]
    pub fn copy_encode_state_from(&mut self, other: &Vocoder) {
        debug_assert!(self.rate == other.rate, "rate mismatch");
        self.engine.encoder.copy_state_from(&other.engine.encoder);
        self.readback = other.readback.clone();
    }

    /// Configure the post-decoder enhancement chain. Off by default
    /// ([`EnhancementMode::None`] — spec-faithful PCM). When set to
    /// [`EnhancementMode::Classical`], decoded PCM passes through a biquad
    /// cascade + soft-knee compressor + boundary-fade chain before
    /// [`Self::decode_bits`] returns. See [`crate::enhancement`] for stage
    /// details and AIC33 mapping.
    ///
    /// Resets the chain's runtime filter state (delay lines, envelope, pending
    /// fade) so the new mode starts clean. Persistent: stays configured across
    /// [`Self::reset`].
    pub fn set_enhancement(&mut self, mode: EnhancementMode) {
        self.enhancement = mode;
        self.enhancement_state = EnhancementState::default();
    }

    /// Currently configured enhancement mode.
    #[inline]
    pub fn enhancement(&self) -> &EnhancementMode {
        &self.enhancement
    }

    /// Start a fluent builder for this rate. Equivalent to
    /// `VocoderBuilder::new(rate)`.
    #[inline]
    pub fn builder(rate: Rate) -> VocoderBuilder {
        VocoderBuilder::new(rate)
    }

    /// The rate this channel was constructed at. Cannot change for the lifetime
    /// of the channel; build a new [`Vocoder`] to switch rates, or a
    /// [`Transcoder`] to bridge two.
    #[inline]
    pub fn rate(&self) -> Rate {
        self.rate
    }

    /// Number of i16 samples consumed per [`Self::encode_pcm`] call, and
    /// produced per [`Self::decode_bits`] call.
    #[inline]
    pub fn frame_samples(&self) -> usize {
        self.rate.frame_samples()
    }

    /// Number of wire bytes per encoded frame at this rate.
    #[inline]
    pub fn fec_frame_bytes(&self) -> usize {
        self.rate.fec_frame_bytes()
    }

    /// Read the most recent frame's stats. Returns the zero-default before any
    /// frame has been processed.
    #[inline]
    pub fn last_stats(&self) -> &FrameStats {
        &self.last_stats
    }

    /// Reset all channel state, as if the handle had just been built.
    /// Returns the engine to its power-on state and clears last-frame stats.
    /// The configured enhancement mode is preserved (configuration, not channel
    /// state).
    pub fn reset(&mut self) {
        self.engine.reset();
        self.last_stats = FrameStats::default();
        self.enhancement_state = EnhancementState::default();
        self.prev_disposition = FrameDisposition::default();
        #[cfg(feature = "encode")]
        {
            self.readback = ParamReadback::new(self.rate);
        }
    }

    /// Encode one 160-sample PCM frame into one wire frame.
    ///
    /// `pcm` must be exactly [`Self::frame_samples`] samples.
    ///
    /// **Encoding carries one frame of delay**, as the reference does: analysis
    /// runs over an window that, when a frame's payload is emitted, holds the
    /// *previous* 20 ms of audio. The frame returned by the first call after
    /// construction or [`Self::reset`] therefore describes the encoder's
    /// silent pre-roll rather than the audio just passed in, and the last frame
    /// of audio stays inside the engine until [`Self::flush_encode`] pushes it
    /// out. This is the reference's own contract, not an artifact of this
    /// binding.
    #[cfg(feature = "encode")]
    pub fn encode_pcm(&mut self, pcm: &[i16]) -> Result<Vec<u8>, VocoderError> {
        if pcm.len() != self.frame_samples() {
            return Err(VocoderError::WrongPcmLength {
                expected: self.frame_samples(),
                got: pcm.len(),
            });
        }
        let frame: &[i16; FRAME_SAMPLES] = pcm.try_into().expect("length already validated");
        let (bytes, disposition, emitted) = self.engine.encode(frame);
        self.last_stats.analysis = Some(AnalysisStats {
            disposition: disposition_of(disposition),
            params: self.readback.read(&emitted),
        });
        Ok(bytes)
    }

    /// Encode one frame to **info vectors**, stopping short of any wire format.
    ///
    /// The transmit mirror of [`Self::decode_info`], and the entry point for a
    /// sender that owns everything below the codec — a subscriber radio or a
    /// dispatch console that has to encrypt. Encryption applies to these
    /// parameter bits, with the FEC wrapped around the ciphertext afterwards,
    /// so a sender needs the vectors before any packing has happened.
    ///
    /// Returns exactly [`Rate::info_vector_count`] values — 4 for the
    /// half-rate AMBE+2 codec (`û₀..û₃`), 8 for full-rate IMBE (`b̂₀..b̂₇`) —
    /// each already prioritized and masked to its field width. From here:
    /// encrypt if you need to, then hand them to your own FEC and interleave
    /// (see [`crate::halfrate::frame::encode_code_vectors`] for the shared
    /// half-rate FEC core), or serialize with
    /// [`crate::halfrate::pack_natural`].
    ///
    /// The rate still selects the codec, so encode at the rate whose codec you
    /// want; the FEC-bearing and info-only variants of one codec produce the
    /// same vectors here.
    ///
    /// Like [`Self::encode_pcm`] this carries the codec's inherent one-frame
    /// delay — the vectors returned describe the *previous* frame of audio.
    ///
    /// ```rust
    /// use blip25_vocoder::vocoder::{Rate, Vocoder};
    ///
    /// let mut tx = Vocoder::new(Rate::HalfRate3600x2450);
    /// let pcm = [0i16; 160];
    /// let info = tx.encode_info(&pcm)?;
    /// assert_eq!(info.len(), 4);
    ///
    /// // Encrypt here, then apply your protocol's FEC and interleave.
    /// let vectors: [u16; 4] = info.try_into().expect("4 at this rate");
    /// let _code = blip25_vocoder::halfrate::frame::encode_code_vectors(&vectors);
    /// # Ok::<(), blip25_vocoder::vocoder::VocoderError>(())
    /// ```
    #[cfg(feature = "encode")]
    pub fn encode_info(&mut self, pcm: &[i16]) -> Result<Vec<u16>, VocoderError> {
        if pcm.len() != self.frame_samples() {
            return Err(VocoderError::WrongPcmLength {
                expected: self.frame_samples(),
                got: pcm.len(),
            });
        }
        let frame: &[i16; FRAME_SAMPLES] = pcm.try_into().expect("length already validated");
        let (_bytes, disposition, emitted) = self.engine.encode(frame);
        self.last_stats.analysis = Some(AnalysisStats {
            disposition: disposition_of(disposition),
            params: self.readback.read(&emitted),
        });
        Ok(match emitted {
            EmittedInfo::Ambe(v) => v.to_vec(),
            EmittedInfo::Imbe(v) => v.to_vec(),
        })
    }

    /// Push the engine's one-frame delay out at end-of-stream.
    ///
    /// Feeds one frame of silence so the last real frame of audio emerges, and
    /// returns it. Call this once after the final [`Self::encode_pcm`] so the
    /// trailing 20 ms is not lost.
    #[cfg(feature = "encode")]
    pub fn flush_encode(&mut self) -> Vec<Vec<u8>> {
        let silence = [0i16; FRAME_SAMPLES];
        match self.encode_pcm(&silence) {
            Ok(b) => vec![b],
            // encode_pcm only errors on length, and the length is a constant.
            Err(_) => Vec::new(),
        }
    }

    /// Encode a whole PCM buffer to wire frames — PCM in, bits out.
    ///
    /// `pcm` may be any length; the trailing partial frame is zero-padded.
    /// Returns one wire frame per 20 ms of input plus the flush frame, so
    /// `ceil(len / 160) + 1` frames in total: the first describes the silent
    /// pre-roll and the last carries the final 20 ms out of the engine's
    /// one-frame delay.
    ///
    /// Output is byte-identical to feeding the same audio through [`LiveEncoder`]
    /// or through per-frame [`Self::encode_pcm`] calls followed by
    /// [`Self::flush_encode`] — there is one encode path, and this is a
    /// convenience wrapper on it for callers that already hold the whole buffer.
    /// Real-time callers should use [`LiveEncoder`].
    #[cfg(feature = "encode")]
    pub fn encode(&self, pcm: &[i16]) -> Vec<Vec<u8>> {
        if pcm.is_empty() {
            return Vec::new();
        }
        let mut v = Vocoder::new(self.rate);
        let mut out = Vec::with_capacity(pcm.len() / FRAME_SAMPLES + 2);
        let mut frame = [0i16; FRAME_SAMPLES];
        for chunk in pcm.chunks(FRAME_SAMPLES) {
            frame[..chunk.len()].copy_from_slice(chunk);
            frame[chunk.len()..].fill(0);
            out.push(v.encode_pcm(&frame).expect("frame is FRAME_SAMPLES long"));
        }
        out.extend(v.flush_encode());
        out
    }

    /// Decode one wire frame into PCM. `bits` must be exactly
    /// [`Self::fec_frame_bytes`] bytes; returns exactly [`Self::frame_samples`]
    /// samples. Any configured [`Self::set_enhancement`] post-filter is applied
    /// before returning.
    pub fn decode_bits(&mut self, bits: &[u8]) -> Result<Vec<i16>, VocoderError> {
        self.decode_bits_inner(bits, None)
    }

    /// Decode one wire frame, telling the codec how badly **your** FEC layer
    /// had to work on it.
    ///
    /// Use this whenever the FEC was stripped upstream — the normal case on an
    /// encrypted link, where decryption sits between the FEC layer and the
    /// codec, so the frame necessarily reaches a no-FEC rate
    /// ([`Rate::FullRate4400x4400`], [`Rate::HalfRate2450x2450`]) with its
    /// channel history already discarded.
    ///
    /// The severity is not advisory: it drives error concealment inside the
    /// codec. [`Self::decode_bits`] is this call with a clean status, which at
    /// a no-FEC rate claims every frame arrived perfect and so switches
    /// concealment off exactly when a degrading channel needs it. Pass what
    /// your FEC layer measured:
    ///
    /// - [`FrameStatus::new`] with the corrected-error count for a frame that
    ///   came through (it clamps into the codec's live `0..=15` band).
    /// - [`FrameStatus::LOST`] for one the FEC could not recover — hold the
    ///   previous frame and fade.
    ///
    /// At the FEC-bearing rates the supplied status **overrides** what this
    /// crate's own wire layer measured, on the assumption that a caller
    /// passing one knows something the local FEC decode does not.
    ///
    /// [`DecodeStats::epsilon_0`] / [`DecodeStats::epsilon_t`] continue to
    /// report what *this* crate's FEC layer did, so they stay zero at the
    /// no-FEC rates however severe the status you pass.
    ///
    /// ```rust
    /// use blip25_vocoder::vocoder::{FrameStatus, Rate, Vocoder};
    ///
    /// // FEC stripped upstream: 7-byte half-rate info frame, plus the error
    /// // count the upstream FEC layer reported for it.
    /// let mut rx = Vocoder::new(Rate::HalfRate2450x2450);
    /// let frame = [0u8; 7];
    /// let pcm = rx.decode_bits_with_status(&frame, FrameStatus::new(9, false))?;
    /// assert_eq!(pcm.len(), 160);
    ///
    /// // A frame the FEC could not recover at all.
    /// let pcm = rx.decode_bits_with_status(&frame, FrameStatus::LOST)?;
    /// assert_eq!(pcm.len(), 160);
    /// # Ok::<(), blip25_vocoder::vocoder::VocoderError>(())
    /// ```
    pub fn decode_bits_with_status(
        &mut self,
        bits: &[u8],
        status: FrameStatus,
    ) -> Result<Vec<i16>, VocoderError> {
        self.decode_bits_inner(bits, Some(status))
    }

    /// Decode from **info vectors you already recovered**, skipping the wire
    /// layer entirely.
    ///
    /// This is the entry point for a receiver whose FEC layer lives upstream —
    /// which on an encrypted link is every receiver, since decryption sits
    /// between the FEC and the codec. Your deinterleave and FEC produce the
    /// info vectors; you decrypt them; you pass them here. Nothing is packed
    /// to a wire format and unpacked again on the way in.
    ///
    /// `info` must hold exactly [`Rate::info_vector_count`] values — 4 for the
    /// half-rate AMBE+2 codec, 8 for full-rate IMBE — each within its field
    /// width. Values are used as given; no FEC, no grid validation.
    ///
    /// `status` is **required**, because a caller holding info vectors is by
    /// definition the party that ran the FEC and the only one who knows how it
    /// went. Pass [`FrameStatus::new`] with the corrected-error count,
    /// [`FrameStatus::LOST`] for a frame the FEC could not recover or that
    /// never arrived, or [`FrameStatus::CLEAN`] when it genuinely arrived
    /// intact. Severity drives concealment inside the codec; claiming CLEAN on
    /// a damaged stream switches concealment off exactly when it is needed.
    ///
    /// Call this for **every** frame slot, in order, including ones you are
    /// skipping — the codec is stateful in both spectral envelope and phase,
    /// so a missed frame desynchronises everything after it. For a slot with
    /// no usable data, pass any info and [`FrameStatus::LOST`]: the payload is
    /// ignored and the previous frame is held and faded.
    ///
    /// ```rust
    /// use blip25_vocoder::vocoder::{FrameStatus, Rate, Vocoder};
    ///
    /// let mut rx = Vocoder::new(Rate::HalfRate2450x2450);
    /// assert_eq!(Rate::HalfRate2450x2450.info_vector_count(), 4);
    ///
    /// // A frame whose FEC corrected 5 bit errors.
    /// let info = [0x123u16, 0x456, 0x1FF, 0x2ABC];
    /// let pcm = rx.decode_info(&info, FrameStatus::new(5, false))?;
    /// assert_eq!(pcm.len(), 160);
    ///
    /// // A slot with nothing usable: still call, so the codec stays in step.
    /// let pcm = rx.decode_info(&info, FrameStatus::LOST)?;
    /// assert_eq!(pcm.len(), 160);
    /// # Ok::<(), blip25_vocoder::vocoder::VocoderError>(())
    /// ```
    pub fn decode_info(
        &mut self,
        info: &[u16],
        status: FrameStatus,
    ) -> Result<Vec<i16>, VocoderError> {
        let expected = self.rate.info_vector_count();
        if info.len() != expected {
            return Err(VocoderError::WrongInfoLength {
                expected,
                got: info.len(),
            });
        }
        let (mut pcm, stats) = self.engine.decode_info(info, status);
        let prev_was_decoded = self.prev_disposition.is_decoded();
        enhancement::apply(
            &self.enhancement,
            &mut self.enhancement_state,
            &mut pcm,
            SAMPLE_RATE_HZ,
            prev_was_decoded,
        );
        self.prev_disposition = stats.disposition;
        self.last_stats.decode = Some(stats);
        Ok(pcm)
    }

    fn decode_bits_inner(
        &mut self,
        bits: &[u8],
        status: Option<FrameStatus>,
    ) -> Result<Vec<i16>, VocoderError> {
        if bits.len() != self.fec_frame_bytes() {
            return Err(VocoderError::WrongBitsLength {
                expected: self.fec_frame_bytes(),
                got: bits.len(),
            });
        }
        let (mut pcm, stats) = self.engine.decode(bits, status);
        // The fade arms on the first decoded frame AFTER a repeat, so it is
        // keyed on the previous frame's disposition, not this one's.
        let prev_was_decoded = self.prev_disposition.is_decoded();
        enhancement::apply(
            &self.enhancement,
            &mut self.enhancement_state,
            &mut pcm,
            SAMPLE_RATE_HZ,
            prev_was_decoded,
        );
        self.prev_disposition = stats.disposition;
        self.last_stats.decode = Some(stats);
        Ok(pcm)
    }

    /// Number of soft-decision LLRs [`Self::decode_soft`] expects per frame, or
    /// `None` for the no-FEC rates (no FEC layer to soft-decode).
    #[inline]
    pub fn soft_frame_bits(&self) -> Option<usize> {
        self.rate.soft_frame_bits()
    }

    /// Decode one frame from **soft-decision** channel bits (LLRs) instead of
    /// hard bytes.
    ///
    /// `llrs` holds one signed value per FEC channel bit — sign = the hard
    /// decision, magnitude = confidence — in raw frame-bit order, first
    /// transmitted bit first. Length must equal [`Self::soft_frame_bits`]
    /// (144 for [`Rate::FullRate7200x4400`], 72 for
    /// [`Rate::HalfRate3600x2450`]).
    ///
    /// This is the native LLR convention, not a vendor host-packet format:
    /// a demodulator's per-bit confidence maps straight onto it. Converting
    /// from the reference's packed 4-bit soft-decision nibbles is a
    /// chip-interop concern and lives outside this crate.
    ///
    /// Runs the crate's soft Golay/Hamming FEC (Chase-II — the ~2 dB coding
    /// gain over hard slicing), then re-encodes the recovered info to a clean
    /// wire frame and synthesizes through the shipped [`Self::decode_bits`]
    /// path — so decoder state, concealment, and enhancement behave exactly as
    /// for a clean hard decode. On error-free input the result is bit-identical
    /// to `decode_bits`; the soft path diverges only by *rescuing* frames a
    /// hard decode would corrupt.
    ///
    /// Soft decode is only meaningful for the FEC-bearing rates; the no-FEC
    /// info-only rates return [`VocoderError::SoftUnsupported`].
    ///
    /// ```rust
    /// # use blip25_vocoder::vocoder::{Rate, Vocoder};
    /// let mut rx = Vocoder::new(Rate::FullRate7200x4400);
    /// let llrs = [0i8; 144]; // one LLR per channel bit
    /// let pcm = rx.decode_soft(&llrs).unwrap();
    /// assert_eq!(pcm.len(), rx.frame_samples());
    /// ```
    pub fn decode_soft(&mut self, llrs: &[i8]) -> Result<Vec<i16>, VocoderError> {
        let expected = self
            .rate
            .soft_frame_bits()
            .ok_or(VocoderError::SoftUnsupported { rate: self.rate })?;
        if llrs.len() != expected {
            return Err(VocoderError::WrongSoftLength {
                expected,
                got: llrs.len(),
            });
        }
        // Soft-decode the FEC to recovered info vectors, then re-encode a clean
        // wire frame and run the shipped hard path (synth + conceal +
        // enhancement + state). The re-encoded frame is error-free, so the hard
        // FEC decode below is lossless.
        let fec: Vec<u8> = match self.rate {
            Rate::FullRate7200x4400 => {
                let soft: &[i8; 144] = llrs.try_into().expect("length checked above");
                let frame = crate::fullrate::frame::decode_frame_soft(soft);
                imbe_pipeline::info_vec_to_fec_bytes(&frame.info).to_vec()
            }
            Rate::HalfRate3600x2450 => {
                let soft: &[i8; 72] = llrs.try_into().expect("length checked above");
                let frame = crate::halfrate::frame::decode_frame_soft(soft);
                ambe_plus2_pipeline::info_vec_to_fec_bytes(&frame.info).to_vec()
            }
            // soft_frame_bits() returned Some(_) only for the two rates above.
            _ => unreachable!("no-FEC rates gated by soft_frame_bits()"),
        };
        self.decode_bits(&fec)
    }

    /// Encode an arbitrary-length PCM slice as a stream of frames.
    ///
    /// Returns an iterator that yields one `Result<Vec<u8>>` per frame consumed
    /// (160 samples per frame). Trailing partial frames are silently dropped —
    /// the caller is responsible for padding to a multiple of
    /// [`Self::frame_samples`] if all samples must be encoded, and for calling
    /// [`Self::flush_encode`] to recover the engine's held frame.
    ///
    /// State advances across frames just as it would with manual per-frame
    /// [`Self::encode_pcm`] calls.
    ///
    /// ```rust
    /// # use blip25_vocoder::vocoder::{Rate, Vocoder};
    /// # let pcm: Vec<i16> = vec![0; 160 * 5];
    /// let mut tx = Vocoder::new(Rate::FullRate7200x4400);
    /// let bits: Result<Vec<Vec<u8>>, _> = tx.encode_stream(&pcm).collect();
    /// assert_eq!(bits.unwrap().len(), 5);
    /// ```
    #[cfg(feature = "encode")]
    pub fn encode_stream<'a>(&'a mut self, pcm: &'a [i16]) -> EncodeStream<'a> {
        EncodeStream {
            vocoder: self,
            pcm,
            pos: 0,
        }
    }

    /// Decode an arbitrary-length wire byte slice as a stream of PCM frames.
    ///
    /// Returns an iterator that yields one `Result<Vec<i16>>` per frame consumed
    /// ([`Self::fec_frame_bytes`] per frame). Trailing partial frames are
    /// silently dropped.
    ///
    /// ```rust
    /// # use blip25_vocoder::vocoder::{Rate, Vocoder};
    /// # let bits: Vec<u8> = vec![0; 18 * 5];
    /// let mut rx = Vocoder::new(Rate::FullRate7200x4400);
    /// let pcm_frames: Result<Vec<Vec<i16>>, _> = rx.decode_stream(&bits).collect();
    /// assert_eq!(pcm_frames.unwrap().len(), 5);
    /// ```
    pub fn decode_stream<'a>(&'a mut self, bits: &'a [u8]) -> DecodeStream<'a> {
        DecodeStream {
            vocoder: self,
            bits,
            pos: 0,
        }
    }

    /// Decode an arbitrary-length soft-decision LLR slice as a stream of PCM
    /// frames — the soft-decision counterpart of [`Self::decode_stream`].
    ///
    /// Yields one `Result<Vec<i16>>` per [`Self::soft_frame_bits`] LLRs consumed
    /// ([`Self::decode_soft`] per frame); trailing partial frames are silently
    /// dropped. For the no-FEC rates (no soft frame size) the stream is empty —
    /// call [`Self::decode_soft`] directly to get the explicit
    /// [`VocoderError::SoftUnsupported`].
    ///
    /// ```rust
    /// # use blip25_vocoder::vocoder::{Rate, Vocoder};
    /// # let llrs: Vec<i8> = vec![0; 144 * 5];
    /// let mut rx = Vocoder::new(Rate::FullRate7200x4400);
    /// let frames: Result<Vec<Vec<i16>>, _> = rx.decode_stream_soft(&llrs).collect();
    /// assert_eq!(frames.unwrap().len(), 5);
    /// ```
    pub fn decode_stream_soft<'a>(&'a mut self, llrs: &'a [i8]) -> DecodeSoftStream<'a> {
        DecodeSoftStream {
            vocoder: self,
            llrs,
            pos: 0,
        }
    }
}

impl core::fmt::Debug for Vocoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Vocoder")
            .field("rate", &self.rate)
            .field("frame_samples", &self.frame_samples())
            .field("fec_frame_bytes", &self.fec_frame_bytes())
            .finish_non_exhaustive()
    }
}

/// Streaming-encode iterator returned by [`Vocoder::encode_stream`]. Yields one
/// `Result<Vec<u8>>` per 160-sample input frame; trailing partial frames are
/// silently dropped.
#[cfg(feature = "encode")]
pub struct EncodeStream<'a> {
    vocoder: &'a mut Vocoder,
    pcm: &'a [i16],
    pos: usize,
}

#[cfg(feature = "encode")]
impl Iterator for EncodeStream<'_> {
    type Item = Result<Vec<u8>, VocoderError>;
    fn next(&mut self) -> Option<Self::Item> {
        let n = self.vocoder.frame_samples();
        if self.pos + n > self.pcm.len() {
            return None;
        }
        let frame = &self.pcm[self.pos..self.pos + n];
        self.pos += n;
        Some(self.vocoder.encode_pcm(frame))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.vocoder.frame_samples();
        let left = (self.pcm.len().saturating_sub(self.pos)) / n;
        (left, Some(left))
    }
}

#[cfg(feature = "encode")]
impl ExactSizeIterator for EncodeStream<'_> {}

/// Streaming-decode iterator returned by [`Vocoder::decode_stream`]. Yields one
/// `Result<Vec<i16>>` per wire frame; trailing partial frames are silently
/// dropped.
pub struct DecodeStream<'a> {
    vocoder: &'a mut Vocoder,
    bits: &'a [u8],
    pos: usize,
}

impl Iterator for DecodeStream<'_> {
    type Item = Result<Vec<i16>, VocoderError>;
    fn next(&mut self) -> Option<Self::Item> {
        let n = self.vocoder.fec_frame_bytes();
        if self.pos + n > self.bits.len() {
            return None;
        }
        let frame = &self.bits[self.pos..self.pos + n];
        self.pos += n;
        Some(self.vocoder.decode_bits(frame))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.vocoder.fec_frame_bytes();
        let left = (self.bits.len().saturating_sub(self.pos)) / n;
        (left, Some(left))
    }
}

impl ExactSizeIterator for DecodeStream<'_> {}

/// Streaming soft-decode iterator returned by [`Vocoder::decode_stream_soft`].
pub struct DecodeSoftStream<'a> {
    vocoder: &'a mut Vocoder,
    llrs: &'a [i8],
    pos: usize,
}

impl Iterator for DecodeSoftStream<'_> {
    type Item = Result<Vec<i16>, VocoderError>;
    fn next(&mut self) -> Option<Self::Item> {
        let n = self.vocoder.soft_frame_bits()?;
        if self.pos + n > self.llrs.len() {
            return None;
        }
        let frame = &self.llrs[self.pos..self.pos + n];
        self.pos += n;
        Some(self.vocoder.decode_soft(frame))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let Some(n) = self.vocoder.soft_frame_bits() else {
            return (0, Some(0));
        };
        let left = (self.llrs.len().saturating_sub(self.pos)) / n;
        (left, Some(left))
    }
}

impl ExactSizeIterator for DecodeSoftStream<'_> {}

// ===========================================================================
// Transcoder
// ===========================================================================

/// A `(from, to)` rate pair for [`Transcoder`].
///
/// Scales O(N) for N rates instead of O(N²) enum variants. Not every
/// `(from, to)` pair has a wired parameter-domain converter; see
/// [`Transcoder::new`] for the supported set.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TranscodeDirection {
    /// Rate of the input wire frame.
    pub from: Rate,
    /// Rate of the output wire frame.
    pub to: Rate,
}

impl TranscodeDirection {
    /// Construct a direction from a `(from, to)` pair.
    #[inline]
    pub const fn new(from: Rate, to: Rate) -> Self {
        Self { from, to }
    }

    /// Bytes per input wire frame for this direction.
    #[inline]
    pub const fn input_frame_bytes(self) -> usize {
        self.from.fec_frame_bytes()
    }

    /// Bytes per output wire frame for this direction.
    #[inline]
    pub const fn output_frame_bytes(self) -> usize {
        self.to.fec_frame_bytes()
    }
}

/// Bridge P25 Phase 1 ↔ P25 Phase 2 at the wire-bits layer.
///
/// Internally the transcoder runs the parameter-domain converter (BABA-A §11 —
/// extract params from one rate's bits, then re-quantize at the other rate
/// without any PCM round-trip). Avoids the 8-kHz-PCM detour, so quality stays at
/// the parameter-extraction floor rather than going through analysis-encode →
/// synthesis → analysis-encode again. The codec engine is not involved at all.
///
/// Direction is fixed at construction. State (cross-rate predictor, last-good
/// frame for repeats) is internal and advances per call.
///
/// ```rust
/// use blip25_vocoder::vocoder::{Rate, Transcoder};
///
/// let mut tx = Transcoder::new(Rate::FullRate7200x4400, Rate::HalfRate3600x2450).unwrap();
/// let phase1_bits: [u8; 18] = [0; 18];
/// let phase2_bits = tx.transcode(&phase1_bits).unwrap();
/// assert_eq!(phase2_bits.len(), 9);
/// ```
pub struct Transcoder {
    direction: TranscodeDirection,
    full_to_half: Option<crate::rate_conversion::FullToHalfConverter>,
    half_to_full: Option<crate::rate_conversion::HalfToFullConverter>,
}

impl Transcoder {
    /// Open a new transcoder for the `(from, to)` rate pair.
    ///
    /// Cross-codec pairs (run the parameter-domain converter):
    /// - `(Rate::FullRate7200x4400, Rate::HalfRate3600x2450)`
    /// - `(Rate::HalfRate3600x2450, Rate::FullRate7200x4400)`
    ///
    /// Same-codec FEC ↔ no-FEC pairs (pure wire-layer bit shuffling, no codec or
    /// predictor state):
    /// - `(Rate::FullRate7200x4400, Rate::FullRate4400x4400)` — strip Annex H FEC
    /// - `(Rate::FullRate4400x4400, Rate::FullRate7200x4400)` — add Annex H FEC
    /// - `(Rate::HalfRate3600x2450, Rate::HalfRate2450x2450)` — strip half-rate FEC
    /// - `(Rate::HalfRate2450x2450, Rate::HalfRate3600x2450)` — add half-rate FEC
    ///
    /// Any other combination returns [`VocoderError::UnsupportedTranscode`].
    pub fn new(from: Rate, to: Rate) -> Result<Self, VocoderError> {
        let direction = TranscodeDirection { from, to };
        match (from, to) {
            (Rate::FullRate7200x4400, Rate::HalfRate3600x2450) => Ok(Self {
                direction,
                full_to_half: Some(crate::rate_conversion::FullToHalfConverter::new()),
                half_to_full: None,
            }),
            (Rate::HalfRate3600x2450, Rate::FullRate7200x4400) => Ok(Self {
                direction,
                full_to_half: None,
                half_to_full: Some(crate::rate_conversion::HalfToFullConverter::new()),
            }),
            // Same-codec FEC ↔ no-FEC pairs are stateless wire transforms; both
            // converters stay None.
            (Rate::FullRate7200x4400, Rate::FullRate4400x4400)
            | (Rate::FullRate4400x4400, Rate::FullRate7200x4400)
            | (Rate::HalfRate3600x2450, Rate::HalfRate2450x2450)
            | (Rate::HalfRate2450x2450, Rate::HalfRate3600x2450) => Ok(Self {
                direction,
                full_to_half: None,
                half_to_full: None,
            }),
            _ => Err(VocoderError::UnsupportedTranscode { from, to }),
        }
    }

    /// Direction this transcoder was opened in.
    #[inline]
    pub fn direction(&self) -> TranscodeDirection {
        self.direction
    }

    /// Transcode one input wire frame to one output wire frame.
    ///
    /// `bits` must be exactly [`TranscodeDirection::input_frame_bytes`] long;
    /// the returned `Vec` has exactly [`TranscodeDirection::output_frame_bytes`].
    pub fn transcode(&mut self, bits: &[u8]) -> Result<Vec<u8>, VocoderError> {
        let in_n = self.direction.input_frame_bytes();
        if bits.len() != in_n {
            return Err(VocoderError::WrongBitsLength {
                expected: in_n,
                got: bits.len(),
            });
        }
        match (self.direction.from, self.direction.to) {
            (Rate::FullRate7200x4400, Rate::HalfRate3600x2450) => {
                let dibits_in = unpack_dibits_n::<72>(bits);
                let dibits_out = self
                    .full_to_half
                    .as_mut()
                    .expect("constructed with this direction")
                    .convert(&dibits_in)
                    .map_err(|e| VocoderError::Quantize(format!("{e:?}")))?;
                Ok(pack_dibits_n::<36, 9>(&dibits_out).to_vec())
            }
            (Rate::HalfRate3600x2450, Rate::FullRate7200x4400) => {
                let dibits_in = unpack_dibits_n::<36>(bits);
                let dibits_out = self
                    .half_to_full
                    .as_mut()
                    .expect("constructed with this direction")
                    .convert(&dibits_in)
                    .map_err(|e| VocoderError::Quantize(format!("{e:?}")))?;
                Ok(pack_dibits_n::<72, 18>(&dibits_out).to_vec())
            }
            (Rate::FullRate7200x4400, Rate::FullRate4400x4400) => {
                Ok(imbe_pipeline::fec_to_info_bytes(bits).to_vec())
            }
            (Rate::FullRate4400x4400, Rate::FullRate7200x4400) => {
                Ok(imbe_pipeline::info_to_fec_bytes(bits).to_vec())
            }
            (Rate::HalfRate3600x2450, Rate::HalfRate2450x2450) => {
                Ok(ambe_plus2_pipeline::fec_to_info_bytes(bits).to_vec())
            }
            (Rate::HalfRate2450x2450, Rate::HalfRate3600x2450) => {
                Ok(ambe_plus2_pipeline::info_to_fec_bytes(bits).to_vec())
            }
            (from, to) => Err(VocoderError::UnsupportedTranscode { from, to }),
        }
    }

    /// Reset all transcoder state. Equivalent to opening a fresh channel.
    pub fn reset(&mut self) {
        *self = Self::new(self.direction.from, self.direction.to)
            .expect("direction was validated at construction");
    }
}

// ===========================================================================
// Live (push-driven) wrappers
// ===========================================================================

/// Push-driven encoder for live PCM that arrives in chunks of arbitrary length
/// (audio-device callbacks, network reads) rather than whole 160-sample frames.
///
/// Buffers the sub-frame residue internally and emits one wire frame per 160
/// samples it accumulates, so callers can push whatever they have and harvest
/// frames as they become available.
///
/// Latency is the codec's own one-frame delay and nothing more: a frame pushed
/// now yields the bits for the *previous* frame. There is no look-ahead window
/// on top of that.
///
/// ```rust
/// # use blip25_vocoder::vocoder::{LiveEncoder, Rate};
/// let mut enc = LiveEncoder::new(Rate::FullRate7200x4400);
/// // 256 samples (audio-device callback); not a multiple of 160.
/// let frames = enc.push(&[0i16; 256]);
/// assert_eq!(frames.len(), 1);              // one whole frame accumulated
/// assert_eq!(enc.pending_samples(), 96);    // sub-frame residue
/// // End of stream: flush completes the partial frame with silence and pushes
/// // the codec's held frame out, so the trailing audio is never lost.
/// let tail = enc.flush().unwrap();
/// assert_eq!(tail.len(), 2);                // residue frame + the held frame
/// assert_eq!(enc.pending_samples(), 0);
/// ```
#[cfg(feature = "encode")]
pub struct LiveEncoder {
    vocoder: Vocoder,
    pcm_buf: Vec<i16>,
}

#[cfg(feature = "encode")]
impl LiveEncoder {
    /// Open a new live encoder at the given rate, all state cold.
    pub fn new(rate: Rate) -> Self {
        Self {
            vocoder: Vocoder::new(rate),
            pcm_buf: Vec::new(),
        }
    }

    /// Read-only access to the underlying [`Vocoder`] (for stats / rate
    /// queries).
    #[inline]
    pub fn vocoder(&self) -> &Vocoder {
        &self.vocoder
    }

    /// Append PCM samples and emit zero or more wire frames. Per-frame errors
    /// are surfaced as `Err` entries in the returned `Vec`; the buffer drains
    /// regardless so a single bad frame doesn't stall the stream.
    pub fn push(&mut self, pcm: &[i16]) -> Vec<Result<Vec<u8>, VocoderError>> {
        self.pcm_buf.extend_from_slice(pcm);
        let n = self.vocoder.frame_samples();
        let mut out = Vec::with_capacity(self.pcm_buf.len() / n);
        while self.pcm_buf.len() >= n {
            let result = self.vocoder.encode_pcm(&self.pcm_buf[..n]);
            self.pcm_buf.drain(..n);
            out.push(result);
        }
        out
    }

    /// Samples buffered but not yet formed into a whole frame (between 0 and
    /// `frame_samples()-1` after every `push` returns). Drains to 0 after
    /// [`Self::flush`].
    #[inline]
    pub fn pending_samples(&self) -> usize {
        self.pcm_buf.len()
    }

    /// Drop any pending samples without encoding them. Useful at stream
    /// shutdown when the caller doesn't want a partial-frame flush.
    #[inline]
    pub fn discard_pending(&mut self) {
        self.pcm_buf.clear();
    }

    /// Finish the stream: encode any pending residue (zero-padded to a full
    /// frame) and push the codec's held frame out, returning every remaining
    /// wire frame in order.
    ///
    /// Call this once at end-of-stream so the trailing 20 ms of audio isn't
    /// lost; the only cost is at most one frame of zero-padding tacked onto the
    /// last word of audio when residue is present.
    pub fn flush(&mut self) -> Result<Vec<Vec<u8>>, VocoderError> {
        let n = self.vocoder.frame_samples();
        let mut out = Vec::with_capacity(2);
        if !self.pcm_buf.is_empty() {
            self.pcm_buf.resize(n, 0);
            out.push(self.vocoder.encode_pcm(&self.pcm_buf)?);
            self.pcm_buf.clear();
        }
        out.extend(self.vocoder.flush_encode());
        Ok(out)
    }

    /// Reset all state — both the inner [`Vocoder`] and the residue buffer.
    pub fn reset(&mut self) {
        self.vocoder.reset();
        self.pcm_buf.clear();
    }

    /// Configured rate.
    #[inline]
    pub fn rate(&self) -> Rate {
        self.vocoder.rate()
    }
}

/// Push-driven decoder for live wire byte streams that arrive in chunks of
/// arbitrary length (network sockets, log replays, partial reads). Mirrors
/// [`LiveEncoder`].
///
/// ```rust
/// # use blip25_vocoder::vocoder::{LiveDecoder, Rate};
/// let mut dec = LiveDecoder::new(Rate::HalfRate3600x2450);
/// let chunk: [u8; 23] = [0; 23];   // 2 full 9-byte frames + 5 byte residue
/// let frames = dec.push(&chunk);
/// assert_eq!(frames.len(), 2);
/// assert_eq!(dec.pending_bytes(), 5);
/// ```
pub struct LiveDecoder {
    vocoder: Vocoder,
    bits_buf: Vec<u8>,
}

impl LiveDecoder {
    /// Open a new live decoder at the given rate, all state cold.
    pub fn new(rate: Rate) -> Self {
        Self {
            vocoder: Vocoder::new(rate),
            bits_buf: Vec::new(),
        }
    }

    /// Read-only access to the underlying [`Vocoder`].
    #[inline]
    pub fn vocoder(&self) -> &Vocoder {
        &self.vocoder
    }

    /// Append wire bytes and emit zero or more PCM frames. Per-frame errors
    /// surface as `Err` entries in the returned `Vec`; the buffer drains
    /// regardless.
    pub fn push(&mut self, bits: &[u8]) -> Vec<Result<Vec<i16>, VocoderError>> {
        self.bits_buf.extend_from_slice(bits);
        let n = self.vocoder.fec_frame_bytes();
        let mut out = Vec::with_capacity(self.bits_buf.len() / n);
        while self.bits_buf.len() >= n {
            let result = self.vocoder.decode_bits(&self.bits_buf[..n]);
            self.bits_buf.drain(..n);
            out.push(result);
        }
        out
    }

    /// Bytes currently buffered (between 0 and `fec_frame_bytes()-1` after
    /// every `push` returns).
    #[inline]
    pub fn pending_bytes(&self) -> usize {
        self.bits_buf.len()
    }

    /// Drop any pending bytes without decoding them.
    #[inline]
    pub fn discard_pending(&mut self) {
        self.bits_buf.clear();
    }

    /// Reset all state — inner [`Vocoder`] + residual byte buffer.
    pub fn reset(&mut self) {
        self.vocoder.reset();
        self.bits_buf.clear();
    }

    /// Configured rate.
    #[inline]
    pub fn rate(&self) -> Rate {
        self.vocoder.rate()
    }
}

/// Fluent builder for [`Vocoder`]. Configures the rate and the optional
/// post-decode enhancement chain in one expression instead of `new` + a setter.
///
/// ```rust
/// use blip25_vocoder::vocoder::{Rate, Vocoder};
///
/// let tx = Vocoder::builder(Rate::HalfRate3600x2450).build();
/// assert_eq!(tx.rate(), Rate::HalfRate3600x2450);
/// ```
#[derive(Clone, Debug)]
pub struct VocoderBuilder {
    rate: Rate,
    enhancement: EnhancementMode,
}

impl VocoderBuilder {
    /// Start a builder at the given rate with enhancement off.
    #[inline]
    pub fn new(rate: Rate) -> Self {
        Self {
            rate,
            enhancement: EnhancementMode::None,
        }
    }

    /// Configure the post-decode enhancement chain.
    #[inline]
    pub fn enhancement(mut self, mode: EnhancementMode) -> Self {
        self.enhancement = mode;
        self
    }

    /// Materialize the [`Vocoder`].
    pub fn build(self) -> Vocoder {
        let mut v = Vocoder::new(self.rate);
        v.set_enhancement(self.enhancement);
        v
    }
}

// ===========================================================================
// Wire-layer bit plumbing
// ===========================================================================

fn unpack_dibits_n<const N: usize>(bytes: &[u8]) -> [u8; N] {
    let mut out = [0u8; N];
    let mut bit = 0usize;
    for slot in &mut out {
        let mut d = 0u8;
        for _ in 0..2 {
            let b = (bytes[bit / 8] >> (7 - (bit % 8))) & 1;
            d = (d << 1) | b;
            bit += 1;
        }
        *slot = d;
    }
    out
}

fn pack_dibits_n<const N: usize, const B: usize>(dibits: &[u8; N]) -> [u8; B] {
    let mut out = [0u8; B];
    let mut bit = 0usize;
    for &d in dibits {
        for pos in (0..2).rev() {
            let b = (d >> pos) & 1;
            out[bit / 8] |= b << (7 - (bit % 8));
            bit += 1;
        }
    }
    out
}

/// Pack an info-bit vector MSB-first into a byte buffer. Bit `k` of `info[i]` is
/// the high-order bit of that field; remaining bytes after the last info bit are
/// left at their initial value (callers pass a zero-initialized slice when they
/// want trailing pad bits to be zero).
fn pack_info_bits<const N: usize>(info: &[u16; N], widths: &[u8; N], out: &mut [u8]) {
    let mut bit_idx = 0usize;
    for i in 0..N {
        let w = widths[i] as usize;
        let v = info[i];
        for k in (0..w).rev() {
            let b = ((v >> k) & 1) as u8;
            out[bit_idx / 8] |= b << (7 - (bit_idx % 8));
            bit_idx += 1;
        }
    }
}

fn unpack_info_bits<const N: usize>(bytes: &[u8], widths: &[u8; N], out: &mut [u16; N]) {
    let mut bit_idx = 0usize;
    for i in 0..N {
        let w = widths[i] as usize;
        let mut v = 0u16;
        for _ in 0..w {
            let b = (bytes[bit_idx / 8] >> (7 - (bit_idx % 8))) & 1;
            v = (v << 1) | u16::from(b);
            bit_idx += 1;
        }
        out[i] = v;
    }
}

/// Strip the IMBE Golay/Hamming/PN FEC wire layer: an 18-byte full-rate frame →
/// the 11-byte info-only ([`Rate::FullRate4400x4400`]) frame. Pure bit-shuffling (no
/// codec state). Public so external parity / round-trip harnesses can build the
/// info-only wire bytes that an 18-byte frame maps to.
pub fn full_rate_fec_to_info_bytes(fec_bytes: &[u8; 18]) -> [u8; 11] {
    imbe_pipeline::fec_to_info_bytes(fec_bytes)
}

mod imbe_pipeline {
    use super::{pack_dibits_n, pack_info_bits, unpack_dibits_n, unpack_info_bits};
    use crate::fullrate::frame::{decode_frame, encode_frame, INFO_WIDTHS};

    /// Info vectors → the 11-byte info-only wire frame.
    pub(super) fn info_vec_to_bytes(info: &[u16; 8]) -> [u8; 11] {
        let mut out = [0u8; 11];
        pack_info_bits(info, &INFO_WIDTHS, &mut out);
        out
    }

    /// The 11-byte info-only wire frame → info vectors.
    pub(super) fn bytes_to_info_vec(bytes: &[u8]) -> [u16; 8] {
        let mut out = [0u16; 8];
        unpack_info_bits(bytes, &INFO_WIDTHS, &mut out);
        out
    }

    /// Wire-layer transcode: 18-byte IMBE FEC frame → 11-byte info-only frame.
    pub(super) fn fec_to_info_bytes(fec_bytes: &[u8]) -> [u8; 11] {
        info_vec_to_bytes(&decode_frame(&unpack_dibits_n::<72>(fec_bytes)).info)
    }

    /// Wire-layer transcode: 11-byte IMBE info-only frame → 18-byte FEC frame.
    pub(super) fn info_to_fec_bytes(info_bytes: &[u8]) -> [u8; 18] {
        info_vec_to_fec_bytes(&bytes_to_info_vec(info_bytes))
    }

    /// Info vectors → 18-byte FEC frame.
    pub(super) fn info_vec_to_fec_bytes(info: &[u16; 8]) -> [u8; 18] {
        pack_dibits_n::<72, 18>(&encode_frame(info))
    }
}

mod ambe_plus2_pipeline {
    use super::{pack_dibits_n, unpack_dibits_n};
    use crate::halfrate::frame::{decode_frame, encode_frame, pack_natural, unpack_natural};

    /// Wire-layer transcode: 9-byte AMBE+2 FEC frame → 7-byte info-only frame.
    pub(super) fn fec_to_info_bytes(fec_bytes: &[u8]) -> [u8; 7] {
        pack_natural(&decode_frame(&unpack_dibits_n::<36>(fec_bytes)).info)
    }

    /// Wire-layer transcode: 7-byte AMBE+2 info-only frame → 9-byte FEC frame.
    pub(super) fn info_to_fec_bytes(info_bytes: &[u8]) -> [u8; 9] {
        info_vec_to_fec_bytes(&unpack_natural(info_bytes))
    }

    /// Info vectors → 9-byte FEC frame.
    pub(super) fn info_vec_to_fec_bytes(info: &[u16; 4]) -> [u8; 9] {
        pack_dibits_n::<36, 9>(&encode_frame(info))
    }
}

#[cfg(test)]
mod tests {
    /// The two ways to cover a frame the transport lost are interchangeable.
    ///
    /// A caller can either substitute [`Rate::erasure_frame`] and decode it
    /// normally, or pass the damaged frame with [`FrameStatus::LOST`]. An
    /// integrator choosing between them wants to know the choice is free, and
    /// it is: both produce byte-identical audio AND leave byte-identical
    /// decoder state, so the frames *after* the gap match too. The recovery
    /// frames are the part worth asserting — an in-band erasure is a frame the
    /// wire layer really decodes, so it could in principle advance the
    /// predictor where the out-of-band flag does not.
    ///
    /// A third option, re-sending the previous good payload, is NOT equivalent:
    /// the decoder sees a valid frame, so it neither conceals nor fades. That is
    /// a transport decision with its own trade-off and is deliberately not
    /// asserted here.
    #[test]
    fn erasure_frame_and_lost_flag_are_interchangeable() {
        for rate in [Rate::FullRate7200x4400, Rate::HalfRate3600x2450] {
            // A voiced signal, so concealment has something to hold on to.
            let pcm: Vec<i16> = (0..40 * FRAME_SAMPLES)
                .map(|i| {
                    let t = i as f64 / 8000.0;
                    ((0.6 * (2.0 * core::f64::consts::PI * 200.0 * t).sin()
                        + 0.3 * (2.0 * core::f64::consts::PI * 400.0 * t).sin())
                        * 9000.0) as i16
                })
                .collect();
            let mut tx = Vocoder::new(rate);
            let mut wire = Vec::new();
            for f in pcm.chunks_exact(FRAME_SAMPLES) {
                wire.extend(tx.encode_pcm(f).expect("frame"));
            }
            let fb = rate.fec_frame_bytes();
            let erasure = rate.erasure_frame();

            // Gap long enough to enter the fade, not just the hold.
            let gap = 12..18;
            let mut out = [Vec::new(), Vec::new()];
            for (arm, slot) in out.iter_mut().enumerate() {
                let mut rx = Vocoder::new(rate);
                for (i, f) in wire.chunks_exact(fb).enumerate() {
                    let pcm = if gap.contains(&i) {
                        if arm == 0 {
                            rx.decode_bits(&erasure).expect("erasure")
                        } else {
                            rx.decode_bits_with_status(f, FrameStatus::LOST)
                                .expect("lost")
                        }
                    } else {
                        rx.decode_bits(f).expect("good")
                    };
                    slot.push(pcm);
                }
            }
            for (i, (ea, eb)) in out[0].iter().zip(out[1].iter()).enumerate() {
                assert_eq!(
                    ea, eb,
                    "{rate:?}: frame {i} differs between the erasure and lost-flag routes\
                     (gap was {gap:?}, so frames past {} are recovery)",
                    gap.end
                );
            }
            // Guard the guard: the gap must actually have concealed something,
            // or this test would pass on two identical no-op runs.
            let gap_rms = |v: &Vec<Vec<i16>>| -> f64 {
                let s: f64 = gap
                    .clone()
                    .flat_map(|i| v[i].iter())
                    .map(|&x| f64::from(x) * f64::from(x))
                    .sum();
                (s / (gap.len() * FRAME_SAMPLES) as f64).sqrt()
            };
            let clean_rms = {
                let mut rx = Vocoder::new(rate);
                let mut v = Vec::new();
                for f in wire.chunks_exact(fb) {
                    v.push(rx.decode_bits(f).expect("good"));
                }
                gap_rms(&v)
            };
            assert!(
                gap_rms(&out[0]) < 0.9 * clean_rms,
                "{rate:?}: the gap did not fade, so this proves nothing"
            );
        }
    }

    use super::*;

    const RATES: [Rate; 4] = [
        Rate::FullRate7200x4400,
        Rate::FullRate4400x4400,
        Rate::HalfRate3600x2450,
        Rate::HalfRate2450x2450,
    ];

    /// A short sweep of tones so the encoder sees something other than silence.
    /// The encoder holds exactly one frame and no more — 20 ms, not a
    /// look-ahead window.
    ///
    /// This is a latency budget, not a detail. A radio pays encode delay on
    /// transmit and the far end pays it again on receive, so a multi-frame
    /// look-ahead is felt twice on every over. An earlier `LiveEncoder`
    /// buffered many frames to reproduce a whole-buffer analysis path
    /// causally, and that path is gone; nothing should quietly reintroduce it.
    ///
    /// Measured rather than asserted from the API: two identical silence
    /// streams are encoded side by side, one with a burst injected at frame 3,
    /// and the first output frame that differs is located. Frame 4 means one
    /// frame of delay.
    #[test]
    fn encoder_holds_exactly_one_frame() {
        let silence = [0i16; FRAME_SAMPLES];
        let burst: Vec<i16> = (0..FRAME_SAMPLES)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE_HZ;
                (12000.0 * (2.0 * core::f32::consts::PI * 700.0 * t).sin()) as i16
            })
            .collect();
        const INJECT_AT: usize = 3;

        for rate in [
            Rate::FullRate7200x4400,
            Rate::FullRate4400x4400,
            Rate::HalfRate3600x2450,
            Rate::HalfRate2450x2450,
        ] {
            // Per-frame primitive.
            let (mut quiet, mut hit) = (Vocoder::new(rate), Vocoder::new(rate));
            let mut first_diff = None;
            for n in 0..12 {
                let a = quiet.encode_pcm(&silence).expect("frame");
                let input: &[i16] = if n == INJECT_AT { &burst } else { &silence };
                let b = hit.encode_pcm(input).expect("frame");
                if a != b && first_diff.is_none() {
                    first_diff = Some(n);
                }
            }
            assert_eq!(
                first_diff,
                Some(INJECT_AT + 1),
                "{rate:?}: encode_pcm delay is not one frame"
            );

            // Chunk-driven path, which must not add buffering of its own
            // beyond the residue needed for partial frames.
            let (mut lq, mut lh) = (LiveEncoder::new(rate), LiveEncoder::new(rate));
            let (mut oa, mut ob) = (Vec::new(), Vec::new());
            for n in 0..12 {
                oa.extend(lq.push(&silence).into_iter().map(|r| r.expect("frame")));
                let input: &[i16] = if n == INJECT_AT { &burst } else { &silence };
                ob.extend(lh.push(input).into_iter().map(|r| r.expect("frame")));
            }
            let live_diff = (0..oa.len().min(ob.len())).find(|&k| oa[k] != ob[k]);
            assert_eq!(
                live_diff,
                Some(INJECT_AT + 1),
                "{rate:?}: LiveEncoder delay is not one frame"
            );
        }
    }

    /// Decode adds nothing: a frame of bits in is that frame of audio out.
    #[test]
    fn decoder_adds_no_delay() {
        for rate in [Rate::FullRate7200x4400, Rate::HalfRate3600x2450] {
            let frames = Vocoder::new(rate).encode(&program(12));
            let (mut quiet, mut hit) = (Vocoder::new(rate), Vocoder::new(rate));
            let mut first_diff = None;
            const FLIP_AT: usize = 4;
            for (n, f) in frames.iter().enumerate().take(10) {
                let a = quiet.decode_bits(f).expect("decode");
                // Damage one frame's payload and see when the audio reacts.
                let mut hurt = f.clone();
                if n == FLIP_AT {
                    hurt[2] ^= 0xFF;
                }
                let b = hit.decode_bits(&hurt).expect("decode");
                if a != b && first_diff.is_none() {
                    first_diff = Some(n);
                }
            }
            assert_eq!(
                first_diff,
                Some(FLIP_AT),
                "{rate:?}: decode reacted late, so it is buffering"
            );
        }
    }

    fn program(frames: usize) -> Vec<i16> {
        (0..frames * FRAME_SAMPLES)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE_HZ;
                let f = 220.0 + 400.0 * (t * 0.7).sin();
                (8000.0 * (2.0 * core::f32::consts::PI * f * t).sin()) as i16
            })
            .collect()
    }

    #[test]
    fn frame_sizes_are_the_documented_ones() {
        for r in RATES {
            let v = Vocoder::new(r);
            assert_eq!(v.frame_samples(), 160);
            assert_eq!(v.fec_frame_bytes(), r.fec_frame_bytes());
            assert_eq!(v.encode(&program(3))[0].len(), r.fec_frame_bytes());
            assert_eq!(r.erasure_frame().len(), r.fec_frame_bytes());
        }
    }

    #[test]
    fn whole_buffer_encode_matches_per_frame_plus_flush() {
        let pcm = program(25);
        for r in RATES {
            let whole = Vocoder::new(r).encode(&pcm);
            let mut v = Vocoder::new(r);
            let mut piecewise: Vec<Vec<u8>> = pcm
                .chunks(FRAME_SAMPLES)
                .map(|c| v.encode_pcm(c).expect("full frames"))
                .collect();
            piecewise.extend(v.flush_encode());
            assert_eq!(whole, piecewise, "{r:?}");
        }
    }

    #[test]
    fn live_encoder_matches_whole_buffer_across_ragged_chunks() {
        let pcm = program(25);
        for r in RATES {
            let whole = Vocoder::new(r).encode(&pcm);
            let mut live = LiveEncoder::new(r);
            let mut got: Vec<Vec<u8>> = Vec::new();
            // Ragged chunk sizes that are not multiples of the frame.
            let mut at = 0usize;
            for step in [37usize, 160, 1, 411, 256].iter().cycle() {
                if at >= pcm.len() {
                    break;
                }
                let end = (at + step).min(pcm.len());
                for f in live.push(&pcm[at..end]) {
                    got.push(f.expect("encode"));
                }
                at = end;
            }
            got.extend(live.flush().expect("flush"));
            assert_eq!(whole, got, "{r:?}");
        }
    }

    #[test]
    fn round_trip_produces_full_frames_of_audio() {
        let pcm = program(20);
        for r in RATES {
            let frames = Vocoder::new(r).encode(&pcm);
            let mut rx = Vocoder::new(r);
            for f in &frames {
                assert_eq!(
                    rx.decode_bits(f).expect("decode").len(),
                    FRAME_SAMPLES,
                    "{r:?}"
                );
            }
        }
    }

    #[test]
    fn reset_returns_a_channel_to_its_power_on_output() {
        let pcm = program(10);
        for r in RATES {
            let mut v = Vocoder::new(r);
            let first: Vec<Vec<u8>> = pcm
                .chunks(FRAME_SAMPLES)
                .map(|c| v.encode_pcm(c).expect("full frames"))
                .collect();
            v.reset();
            let second: Vec<Vec<u8>> = pcm
                .chunks(FRAME_SAMPLES)
                .map(|c| v.encode_pcm(c).expect("full frames"))
                .collect();
            assert_eq!(first, second, "{r:?}");
        }
    }

    #[test]
    fn erasure_frames_are_concealed_not_decoded() {
        for r in RATES {
            let mut rx = Vocoder::new(r);
            // Prime with real audio so there is something to hold onto.
            for f in Vocoder::new(r).encode(&program(5)) {
                rx.decode_bits(&f).expect("decode");
            }
            let erasure = r.erasure_frame();
            rx.decode_bits(&erasure).expect("decode");
            let d = rx.last_stats().decode.as_ref().expect("decode stats");
            assert!(d.disposition.is_concealed(), "{r:?}: {:?}", d.disposition);
        }
    }

    #[test]
    fn wrong_lengths_are_rejected() {
        let mut v = Vocoder::new(Rate::FullRate7200x4400);
        assert!(matches!(
            v.encode_pcm(&[0i16; 159]),
            Err(VocoderError::WrongPcmLength { .. })
        ));
        assert!(matches!(
            v.decode_bits(&[0u8; 17]),
            Err(VocoderError::WrongBitsLength { .. })
        ));
        assert!(matches!(
            v.decode_soft(&[0i8; 143]),
            Err(VocoderError::WrongSoftLength { .. })
        ));
        let mut n = Vocoder::new(Rate::FullRate4400x4400);
        assert!(matches!(
            n.decode_soft(&[0i8; 144]),
            Err(VocoderError::SoftUnsupported { .. })
        ));
    }

    #[test]
    fn transcode_directions_produce_the_right_frame_sizes() {
        let pairs = [
            (Rate::FullRate7200x4400, Rate::HalfRate3600x2450),
            (Rate::HalfRate3600x2450, Rate::FullRate7200x4400),
            (Rate::FullRate7200x4400, Rate::FullRate4400x4400),
            (Rate::FullRate4400x4400, Rate::FullRate7200x4400),
            (Rate::HalfRate3600x2450, Rate::HalfRate2450x2450),
            (Rate::HalfRate2450x2450, Rate::HalfRate3600x2450),
        ];
        for (from, to) in pairs {
            let mut t = Transcoder::new(from, to).expect("supported direction");
            let src = Vocoder::new(from).encode(&program(4));
            for f in &src {
                let out = t.transcode(f).expect("transcode");
                assert_eq!(out.len(), to.fec_frame_bytes(), "{from:?} -> {to:?}");
            }
        }
        assert!(Transcoder::new(Rate::FullRate4400x4400, Rate::HalfRate2450x2450).is_err());
    }

    #[test]
    fn same_codec_transcode_round_trips_exactly() {
        for (fec, info) in [
            (Rate::FullRate7200x4400, Rate::FullRate4400x4400),
            (Rate::HalfRate3600x2450, Rate::HalfRate2450x2450),
        ] {
            let mut strip = Transcoder::new(fec, info).expect("supported");
            let mut add = Transcoder::new(info, fec).expect("supported");
            for f in Vocoder::new(fec).encode(&program(6)) {
                let stripped = strip.transcode(&f).expect("strip");
                assert_eq!(add.transcode(&stripped).expect("add"), f);
            }
        }
    }

    /// The two wire formats of one codec carry the same info bits, so decoding
    /// either must produce the same audio.
    #[test]
    fn fec_and_info_only_wires_decode_identically() {
        for (fec, info) in [
            (Rate::FullRate7200x4400, Rate::FullRate4400x4400),
            (Rate::HalfRate3600x2450, Rate::HalfRate2450x2450),
        ] {
            let frames = Vocoder::new(fec).encode(&program(12));
            let mut strip = Transcoder::new(fec, info).expect("supported");
            let (mut a, mut b) = (Vocoder::new(fec), Vocoder::new(info));
            for f in &frames {
                let stripped = strip.transcode(f).expect("strip");
                assert_eq!(
                    a.decode_bits(f).expect("decode"),
                    b.decode_bits(&stripped).expect("decode"),
                    "{fec:?} vs {info:?}"
                );
            }
        }
    }

    #[test]
    fn soft_decode_of_clean_llrs_matches_hard_decode() {
        for r in [Rate::FullRate7200x4400, Rate::HalfRate3600x2450] {
            let n = r.soft_frame_bits().expect("FEC-bearing rate");
            let frames = Vocoder::new(r).encode(&program(8));
            let (mut hard, mut soft) = (Vocoder::new(r), Vocoder::new(r));
            for f in &frames {
                // Full-confidence LLRs: sign carries the hard decision.
                let llrs: Vec<i8> = (0..n)
                    .map(|i| {
                        if (f[i / 8] >> (7 - (i % 8))) & 1 == 1 {
                            127
                        } else {
                            -127
                        }
                    })
                    .collect();
                assert_eq!(
                    hard.decode_bits(f).expect("hard"),
                    soft.decode_soft(&llrs).expect("soft"),
                    "{r:?}"
                );
            }
        }
    }

    #[test]
    fn builder_configures_the_same_channel_as_new() {
        let v = Vocoder::builder(Rate::HalfRate3600x2450)
            .enhancement(EnhancementMode::None)
            .build();
        assert_eq!(v.rate(), Rate::HalfRate3600x2450);
        assert!(matches!(v.enhancement(), EnhancementMode::None));
    }

    #[test]
    fn live_decoder_drains_ragged_chunks() {
        let frames = Vocoder::new(Rate::HalfRate3600x2450).encode(&program(9));
        let flat: Vec<u8> = frames.concat();
        let mut dec = LiveDecoder::new(Rate::HalfRate3600x2450);
        let mut got = 0usize;
        for chunk in flat.chunks(5) {
            got += dec.push(chunk).len();
        }
        assert_eq!(got, frames.len());
        assert_eq!(dec.pending_bytes(), 0);
    }

    /// This crate models the disposition as the bit word it is; the engine
    /// crate classifies it into an enum. Neither is derived from the other, so
    /// pin them together over real decodes — if either side's reading of a flag
    /// drifts, this fails rather than quietly changing what a caller sees.
    #[test]
    fn our_flags_agree_with_the_engine_classification() {
        use crate::engine::{Decoder, Frame, Mode};
        let payloads = {
            let mut v: Vec<Vec<i16>> = Vec::new();
            // Real speech bits, then a run of erasures to force concealment.
            for f in Vocoder::new(Rate::HalfRate2450x2450).encode(&program(12)) {
                v.push(payload_of(&f));
            }
            let er = Rate::HalfRate2450x2450.erasure_frame();
            for _ in 0..6 {
                v.push(payload_of(&er));
            }
            v
        };
        let mut d = Decoder::new(Mode::HalfRate);
        let mut pcm = [0i16; FRAME_SAMPLES];
        let mut seen_speech = false;
        let mut seen_concealed = false;
        for p in &payloads {
            let f = d.decode(p, &mut pcm);
            let disp = FrameDisposition::from_raw(f.code());
            match f {
                // Tone sets the speech bit too, so the engine tests it first;
                // our predicates are independent bit tests, so both read true.
                Frame::Tone(_) => assert!(disp.is_tone()),
                Frame::Repeat(_) => {
                    seen_concealed = true;
                    assert!(disp.is_concealed() && !disp.is_tone());
                }
                Frame::Speech(_) => {
                    seen_speech = true;
                    assert!(disp.is_speech() && !disp.is_tone() && !disp.is_concealed());
                }
                Frame::Silence => {
                    assert!(!disp.is_speech() && !disp.is_tone() && !disp.is_concealed())
                }
                Frame::Other(_) => {}
            }
        }
        assert!(seen_speech, "corpus should decode some speech");
        assert!(seen_concealed, "erasures should conceal");
    }

    /// An `HalfRate2450x2450` wire frame as engine payload bits.
    ///
    /// The wire carries *prioritized* info vectors, so the payload bits cannot
    /// be read off it directly — they have to come back through the info
    /// vectors, which is exactly what [`Engine::decode`] does.
    fn payload_of(frame: &[u8]) -> Vec<i16> {
        let info = crate::halfrate::frame::unpack_natural(frame);
        let mut payload: Payload = [0; 88];
        let n = info_to_payload(&info, &AMBE_INFO_WIDTHS, &mut payload);
        payload[..n].to_vec()
    }

    /// The engine acts on the FEC severity this crate reports, so a frame the
    /// FEC had to repair must not decode like one that arrived clean.
    ///
    /// Both sides here see the *same* payload bits: the damage is corrected, so
    /// the recovered info is identical to the clean info. The only difference is
    /// that the FEC-bearing rate reports what it repaired and the info-only rate
    /// has no FEC layer and reports clean. If the severity were not reaching the
    /// engine, the two would produce identical audio.
    #[test]
    fn repaired_frames_conceal_where_clean_ones_do_not() {
        let frames = Vocoder::new(Rate::FullRate7200x4400).encode(&program(20));
        let mut strip =
            Transcoder::new(Rate::FullRate7200x4400, Rate::FullRate4400x4400).expect("wired");
        let (mut repaired, mut clean) = (
            Vocoder::new(Rate::FullRate7200x4400),
            Vocoder::new(Rate::FullRate4400x4400),
        );
        let (mut differed, mut severity_seen) = (0usize, 0u8);
        for f in &frames {
            // Damage bits far enough apart that the Annex-H interleave spreads
            // them across separate code vectors, each within Golay's reach.
            let mut hurt = f.clone();
            for bit in [5usize, 47, 96] {
                hurt[bit / 8] ^= 1 << (7 - (bit % 8));
            }
            let a = repaired.decode_bits(&hurt).expect("decode");
            let stats = repaired
                .last_stats()
                .decode
                .as_ref()
                .expect("stats")
                .clone();
            severity_seen = severity_seen.max(stats.epsilon_t);
            // The same recovered info, carried on a wire with no FEC to report.
            let b = clean
                .decode_bits(&strip.transcode(&hurt).expect("strip"))
                .expect("decode");
            if a != b {
                differed += 1;
            }
        }
        assert!(
            severity_seen >= 2,
            "damage should report severity, got {severity_seen}"
        );
        assert!(
            differed > 0,
            "severity reached the engine on no frame — is it still being passed?"
        );
    }

    /// The no-FEC rates are the live receive path on an encrypted link, where
    /// decryption sits between the FEC layer and the codec and the channel
    /// history arrives out-of-band. A caller that supplies it must reach the
    /// engine's concealment; if the status were dropped, every frame here
    /// would decode identically to the clean path.
    #[test]
    fn supplied_status_reaches_the_engine_on_no_fec_rates() {
        for rate in [Rate::FullRate4400x4400, Rate::HalfRate2450x2450] {
            let frames = Vocoder::new(rate).encode(&program(20));
            let (mut quiet, mut told) = (Vocoder::new(rate), Vocoder::new(rate));
            let mut differed = 0usize;
            for f in &frames {
                let a = quiet.decode_bits(f).expect("decode");
                let b = told
                    .decode_bits_with_status(f, FrameStatus::new(15, false))
                    .expect("decode");
                if a != b {
                    differed += 1;
                }
            }
            assert!(
                differed > 0,
                "{rate:?}: supplied severity changed nothing — is it still \
                 reaching the engine?"
            );

            // LOST must hold-and-fade rather than synthesise the payload, so a
            // frame decoded as lost cannot match the same bits decoded normally.
            let (mut plain, mut lost) = (Vocoder::new(rate), Vocoder::new(rate));
            for f in frames.iter().take(5) {
                plain.decode_bits(f).expect("decode");
                lost.decode_bits(f).expect("decode");
            }
            let target = &frames[5];
            assert_ne!(
                plain.decode_bits(target).expect("decode"),
                lost.decode_bits_with_status(target, FrameStatus::LOST)
                    .expect("decode"),
                "{rate:?}: FrameStatus::LOST did not conceal"
            );
        }
    }

    /// `decode_info` must be exactly the pack/unpack round-trip it exists to
    /// remove — a consumer switching to it may not hear a difference.
    #[test]
    fn decode_info_matches_the_wire_roundtrip_it_replaces() {
        for rate in [Rate::FullRate4400x4400, Rate::HalfRate2450x2450] {
            let frames = Vocoder::new(rate).encode(&program(20));
            let (mut viawire, mut direct) = (Vocoder::new(rate), Vocoder::new(rate));
            for f in &frames {
                let info: Vec<u16> = match rate {
                    Rate::HalfRate2450x2450 => crate::halfrate::frame::unpack_natural(f).to_vec(),
                    _ => imbe_pipeline::bytes_to_info_vec(f).to_vec(),
                };
                assert_eq!(info.len(), rate.info_vector_count());
                assert_eq!(
                    viawire.decode_bits(f).expect("wire"),
                    direct.decode_info(&info, FrameStatus::CLEAN).expect("info"),
                    "{rate:?}: decode_info diverged from the wire path"
                );
            }
        }
    }

    /// The severity a caller supplies must reach the engine on the info path
    /// too, and a wrong-length slice must be an error rather than a panic.
    #[test]
    fn decode_info_carries_status_and_validates_length() {
        let rate = Rate::HalfRate2450x2450;
        let frames = Vocoder::new(rate).encode(&program(20));
        let (mut quiet, mut told) = (Vocoder::new(rate), Vocoder::new(rate));
        let mut differed = 0usize;
        for f in &frames {
            let info = crate::halfrate::frame::unpack_natural(f);
            let a = quiet.decode_info(&info, FrameStatus::CLEAN).expect("clean");
            let b = told
                .decode_info(&info, FrameStatus::new(15, false))
                .expect("severe");
            if a != b {
                differed += 1;
            }
        }
        assert!(differed > 0, "supplied severity never reached the engine");

        let mut rx = Vocoder::new(rate);
        assert!(matches!(
            rx.decode_info(&[0; 3], FrameStatus::CLEAN),
            Err(VocoderError::WrongInfoLength {
                expected: 4,
                got: 3
            })
        ));
        assert!(rx.decode_info(&[0; 8], FrameStatus::CLEAN).is_err());
    }

    /// A half-rate frame whose leading codeword is beyond repair must be
    /// concealed, not synthesised from bits the FEC could not vouch for.
    #[test]
    fn unrecoverable_half_rate_frames_are_reported_lost() {
        let frames = Vocoder::new(Rate::HalfRate3600x2450).encode(&program(20));
        let mut rx = Vocoder::new(Rate::HalfRate3600x2450);
        // Prime with clean frames so there is something to hold onto.
        for f in frames.iter().take(10) {
            rx.decode_bits(f).expect("decode");
        }
        // c̃₀ is extended Golay [24, 12]: it detects four errors and refuses to
        // guess. The Annex-S interleave puts its bits at every fourth dibit.
        let mut wrecked = frames[10].clone();
        for dibit in [0usize, 4, 8, 12, 16, 20] {
            let bit = 2 * dibit;
            wrecked[bit / 8] ^= 1 << (7 - (bit % 8));
        }
        rx.decode_bits(&wrecked).expect("decode");
        let d = rx.last_stats().decode.as_ref().expect("stats");
        assert_eq!(d.epsilon_0, u8::MAX, "c0 should be uncorrectable");
        assert!(
            d.disposition.is_concealed(),
            "an unrecoverable frame must conceal, got {:?}",
            d.disposition
        );
    }

    #[test]
    fn streams_yield_one_item_per_whole_frame() {
        let pcm = program(7);
        let mut tx = Vocoder::new(Rate::FullRate7200x4400);
        assert_eq!(tx.encode_stream(&pcm).count(), 7);
        // A trailing partial frame is dropped, not encoded.
        let mut tx = Vocoder::new(Rate::FullRate7200x4400);
        assert_eq!(tx.encode_stream(&pcm[..7 * FRAME_SAMPLES - 3]).count(), 6);
    }
}
