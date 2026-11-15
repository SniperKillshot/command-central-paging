//! P25 Phase 2 TDMA half-rate vocoder wire format.
//!
//! Per TIA-102.BABA-A §13–§17 + Annexes L–T (originally the BABA-1
//! addendum, consolidated into BABA-A in 2014). The half-rate wire is
//! the AMBE+2 codec — BABA-A renames it "Half-Rate Vocoder" to avoid
//! the reference's trademark, but the bit rate (3,600 bps total = 2,450 voice +
//! 1,150 FEC), 72-bit frame size, and parameter structure are
//! AMBE+2 in everything but name. The reference vocoder's rate index 33
//! is documented as "interoperable with APCO Project 25 half-rate with
//! FEC."
//!
//! Pipeline:
//!
//! ```text
//!   36 dibits → frame::decode_frame → Frame { info: [u16; 4], errors }
//!             → priority::deprioritize → b̂₀..b̂₈
//!             → dequantize → MbeParams
//! ```
//!
//! Bit-by-bit symmetric on the encode side via [`frame::encode_frame`],
//! [`priority::prioritize`], and [`dequantize::quantize`].

pub mod dequantize;
pub mod frame;
pub mod priority;

/// Per-field extractors for diagnostic / integration use. Both return the
/// nine deprioritized parameters `[b̂₀..b̂₈]` (pitch, V/UV, gain, PRBA, HOC)
/// — [`fields_from_natural`] from a 7-byte FEC-free half-rate frame,
/// [`fields_from_fec`] from a 9-byte FEC frame (runs the full Golay decode
/// first). Use these instead of flat-slicing the 49-bit prioritized vector,
/// which mixes parameters per slice.
pub use frame::{fields_from_fec, fields_from_natural};

/// The 7-byte serialization of `û₀..û₃`.
///
/// A half-rate frame stripped of FEC carries 49 info bits laid down in
/// **natural / AMBE_d** order — `û₀(12)‖û₁(12)‖û₂(11)‖û₃(14)` sequentially,
/// MSB-first, with bits 49..55 as zero pad. This is what
/// [`crate::vocoder::Rate::HalfRate2450x2450`] emits, what the P25
/// encryption boundary operates on (TIA-102.AAAD-B §1.2: 49 bits per frame,
/// 7 octets × 18 frames per 360 ms), and what mbelib, DSD, MMDVM, and an
/// IDAS / NXDN over-the-air wire use.
///
/// Pair [`unpack_natural`] with [`frame::decode_code_vectors`] to reach a
/// protocol-agnostic half-rate source whose FEC was stripped upstream.
pub use frame::{pack_natural, unpack_natural};
