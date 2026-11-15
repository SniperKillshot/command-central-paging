//! # blip25-vocoder
//!
//! The P25 voice codec: four wire rates across two codecs — full rate
//! (P25 Phase 1 FDMA, `FullRate7200x4400` on air and `FullRate4400x4400`
//! info-only) and half rate (P25 Phase 2 TDMA, `HalfRate3600x2450` on air and
//! `HalfRate2450x2450` info-only) — plus parametric conversion between them.
//! Encode and decode route through a private codec core, bit-exact with the
//! reference software vocoder, wrapped here in the P25 wire formats.
//!
//! ## The two layers
//!
//! ```text
//!   PCM  <->  [ codec core ]  <->  payload bits  <->  [ wire layers ]  <->  wire bytes
//!             analysis/synthesis   49 or 88           FEC, interleave,
//!             (bit-exact)                             priority, packing
//! ```
//!
//! The codec core is codec-only — it performs no error correction; the one
//! channel-quality input it takes is the [`vocoder::FrameStatus`] the wire
//! layers derive, which drives its concealment. Everything else between the
//! payload bits and the air interface
//! is the rest of the crate: framing, FEC, de-interleaving, bit prioritization,
//! erasure marking, and parameter-domain rate conversion.
//!
//! The core is a private module: it is not published separately, is not a
//! dependency, and is not part of the public API. The only thing it contributes
//! to the public surface is the re-exported [`vocoder::FrameStatus`].
//!
//! The core is **patent-encumbered** and derived from the reference software
//! vocoder by reverse engineering; the wire layers are spec-derived from
//! TIA-102.BABA / BABA-A. See [`PATENT_NOTICE.md`] and [`ATTRIBUTION.md`]
//! before use.
//!
//! ## Quick start
//!
//! Most consumers should use the [`vocoder::Vocoder`] façade:
//!
//! ```rust
//! use blip25_vocoder::vocoder::{Rate, Vocoder};
//!
//! // Open a P25 Phase 1 full-rate channel.
//! let mut tx = Vocoder::new(Rate::FullRate7200x4400);
//! let pcm: [i16; 160] = [0; 160];
//! let bits = tx.encode_pcm(&pcm).unwrap();
//! assert_eq!(bits.len(), 18);
//!
//! let mut rx = Vocoder::new(Rate::FullRate7200x4400);
//! let out = rx.decode_bits(&bits).unwrap();
//! assert_eq!(out.len(), 160);
//! ```
//!
//! Three conveniences around the per-frame primitive:
//!
//! - [`vocoder::Vocoder::encode_stream`] / [`vocoder::Vocoder::decode_stream`]
//!   — slice → `Iterator<Item = Result<…>>`, drops trailing partial frames.
//! - [`vocoder::LiveEncoder`] / [`vocoder::LiveDecoder`] — chunk-driven
//!   with internal residue buffer for audio-callback / socket use.
//! - [`vocoder::VocoderBuilder`] — fluent configuration of the optional
//!   post-decode enhancement chain.
//!
//! See [`vocoder`] for the full API and [`INTEGRATION.md`] for the
//! layer boundaries a consumer is expected to own.
//!
//! ## Module organization
//!
//! See [`DESIGN.md`] at the repository root for the architectural
//! model. The public API is a façade over three orthogonal axes joined
//! at a common parameter type:
//!
//! - [`vocoder`] — the per-channel façade. **Recommended entry point.**
//! - [`mbe_params`] — the parameter model, the interchange type used by
//!   the wire and rate-conversion layers.
//! - **Wire formats**, one module per protocol-rate combination:
//!   [`fullrate`] (P25 Phase 1, 144-bit) and [`halfrate`] (P25 Phase 2,
//!   72-bit). Future protocols become sibling modules; DMR and NXDN already
//!   carry the same half-rate codec frame under a different interleave and
//!   enter through [`halfrate::frame`].
//! - [`rate_conversion`] — parameter-domain bits-to-bits conversion,
//!   a peer of the codec and wire layers, not a sub-concern of either.
//!
//! FEC primitives shared across the wire layers live in [`fec`], and the
//! optional post-decode filter chain in [`enhancement`]. `src/generated/` is
//! not a module: it holds the Annex tables (B–T), transcribed from the
//! standard and `include!`d by the wire layers that index them.
//!
//! ## Cargo features
//!
//! - `encode` (on by default) — the encode public API:
//!   [`vocoder::Vocoder::encode_pcm`], [`vocoder::LiveEncoder`], and the encode
//!   streaming iterators.
//! - `decode` (on by default) — the decode public API, the shared frame and
//!   parameter types, and the wire and rate-conversion layers. It stands alone,
//!   so `--no-default-features --features decode` builds a receive-only crate.
//! - `serde` (off by default) — derive `Serialize` / `Deserialize` on
//!   the diagnostic types in [`vocoder`] ([`vocoder::Rate`],
//!   [`vocoder::FrameStats`], [`vocoder::AnalysisStats`],
//!   [`vocoder::FrameDisposition`], [`vocoder::DecodeStats`]) plus
//!   [`mbe_params::MbeParams`]. Useful for shipping stats / params over an
//!   RPC layer (gRPC / protobuf / WS) without hand-rolled converters.
//!
//! `serde` is the only optional dependency and the only thing that pulls
//! anything in; with it off, this crate has no runtime dependencies at all.
//!
//! [`DESIGN.md`]: https://github.com/openBLIP25/blip25-vocoder/blob/main/DESIGN.md
//! [`INTEGRATION.md`]: https://github.com/openBLIP25/blip25-vocoder/blob/main/INTEGRATION.md
//! [`PATENT_NOTICE.md`]: https://github.com/openBLIP25/blip25-vocoder/blob/main/PATENT_NOTICE.md
//! [`ATTRIBUTION.md`]: https://github.com/openBLIP25/blip25-vocoder/blob/main/ATTRIBUTION.md

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
// Doc comments cross-reference crate-internal (`pub(crate)`) helpers by name;
// those links are intentional and resolve within the source. The valuable
// `broken_intra_doc_links` lint stays on to catch genuinely dead links.
#![allow(rustdoc::private_intra_doc_links)]
// A decode-only build (`--no-default-features --features decode`) gates out the
// encode public API; the encode-only helpers it leaves behind are then unused.
// Silence that dead-code noise only in the decode-only configuration.
#![cfg_attr(not(feature = "encode"), allow(dead_code))]

// The codec core. Private: the supported entry point is [`vocoder::Vocoder`],
// and the only thing from here that reaches the public API is the re-exported
// `FrameStatus`. See `src/engine/mod.rs` for why it stays `no_std`-clean.
mod engine;

mod bits;
pub mod fec;

pub mod mbe_params;

pub mod fullrate;
pub mod halfrate;

pub mod rate_conversion;

pub mod enhancement;
pub mod vocoder;
