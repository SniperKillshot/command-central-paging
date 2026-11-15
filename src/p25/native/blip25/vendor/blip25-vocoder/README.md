# blip25-vocoder

A Rust implementation of the P25 voice codec, bit-exact with the reference
vocoder.

The `Vocoder` API exposes four wire rates across the two P25 rates — **full
rate** (Phase 1, on-air 7200×4400 and info-only 4400×4400) and **half rate**
(Phase 2, on-air 3600×2450 and info-only 2450×2450) — plus parametric
conversion between them. TIA-102.BABA-A titles these vocoders IMBE and AMBE+2
respectively; this crate names them by rate, since one engine runs both.

The half-rate frame is not P25-specific. DMR Tier II/III and NXDN
(4800 / "EHR") carry the **same** 49→72-bit codec frame — same Golay pair,
same PN scramble, same bit prioritization — and differ only in the interleave
and burst framing that sit above it. `halfrate::frame::encode_code_vectors` and
`decode_code_vectors` are that shared core; the P25 entry points are thin
Annex-S adapters over them, and a DMR or NXDN consumer supplies its own
(de)interleave and calls the core directly. D-STAR uses first-generation AMBE
rather than the half-rate codec here and is not implemented.

> **Patent notice.** This source code is provided for research and
> interoperability study only. The half-rate (AMBE+2) implementation
> unavoidably reads on the claims of **US8359197**, active until
> **2028-05-20**. See [`PATENT_NOTICE.md`](./PATENT_NOTICE.md) for the full
> list, comparable-project survey, and project policy.

> **Trademarks.** IMBE, AMBE, and AMBE+2 are trademarks of Digital Voice
> Systems, Inc.; NXDN and IDAS are trademarks of Icom Incorporated (NXDN
> jointly with JVC KENWOOD). This project is not affiliated with or endorsed by
> any of them, and uses those names only to identify the systems it
> interoperates with. See [`ATTRIBUTION.md`](./ATTRIBUTION.md#trademarks).

## Install

```toml
[dependencies]
blip25-vocoder = "1.0"
```

Requires Rust 1.85 or later. With default features the crate has no runtime
dependencies at all; see [Cargo features](#cargo-features).

## Quick start

```rust
use blip25_vocoder::vocoder::{Rate, Vocoder};

// Open a P25 Phase 1 (full-rate IMBE) channel.
let mut tx = Vocoder::new(Rate::FullRate7200x4400);
let pcm: [i16; 160] = [0; 160];
let bits = tx.encode_pcm(&pcm).unwrap();    // 18-byte FEC frame

let mut rx = Vocoder::new(Rate::FullRate7200x4400);
let pcm = rx.decode_bits(&bits).unwrap();   // 160 samples
```

One frame is always 160 `i16` samples — 20 ms of 8 kHz mono, at every rate
(`vocoder::FRAME_SAMPLES`). What changes with the rate is the wire frame;
`Rate::fec_frame_bytes` gives its length.

Three runnable examples:

```bash
cargo run --release --example vocoder_demo      # full API walkthrough
cargo run --release --example vocoder_bench     # throughput micro-benchmark
cargo run --release --example foreign_protocol  # DMR / NXDN / IDAS integration
```

## The two layers

```text
  PCM  <->  [ codec core ]  <->  payload bits  <->  [ wire layers ]  <->  wire bytes
            analysis/synthesis   49 or 88           FEC, interleave,
            (bit-exact)                             priority, packing
```

The codec core is codec-only: it performs no error correction, and the one
channel-quality input it takes is the `FrameStatus` the wire layers derive,
which drives its concealment. Everything else between the payload bits and the
air interface is the rest of the crate — framing, FEC, de-interleaving, bit
prioritization, erasure marking, and parameter-domain rate conversion.

That split is also the API's two entry altitudes. `Vocoder` spans both layers
and is what most callers want. The wire modules (`halfrate::frame`,
`halfrate::dequantize` and their `fullrate` counterparts) expose the lower layer
on its own, for a carrier that lays out post-FEC bits differently. That pair is
the carrier-agnostic seam, and `encode_info` / `decode_info` is the boundary
between them: parameter vectors in and out, wire layer skipped entirely.

The codec core is a **private** module. It is not published separately, is not a
dependency, and is not part of the public API; the only thing it contributes to
the public surface is the re-exported `FrameStatus`.

## The four rates

`Rate` selects both the mode the codec core runs (full or half) and the wire
layer wrapped around its payload bits.

| `Rate` | Wire frame | Info vectors | Soft bits | What it is for |
|---|---:|---:|---:|---|
| `FullRate7200x4400` | 18 bytes (144 bits) | 8 (`b̂₀..b̂₇`) | 144 | P25 Phase 1 FDMA on air, Annex H FEC included |
| `FullRate4400x4400` | 11 bytes (88 bits) | 8 | — | Full-rate payload with Annex H stripped; byte layout matches the JMBE / OP25 / `p25_nofec` convention |
| `HalfRate3600x2450` | 9 bytes (72 bits) | 4 (`û₀..û₃`) | 72 | P25 Phase 2 TDMA on air, Annex S interleave included; the same codec frame DMR and NXDN carry |
| `HalfRate2450x2450` | 7 bytes (49 bits + 7 pad) | 4 | — | Half-rate payload in natural / "AMBE_d" order — what P25 encrypts (TIA-102.AAAD-B §1.2) and what mbelib, DSD, MMDVM and an IDAS/NXDN wire use |

The two info-only rates carry no FEC layer, so they have no soft-decision form
and `Rate::soft_frame_bits` returns `None` for them. For storage, prefer the
FEC-bearing rates — see
[`docs/wire_formats_and_storage.md`](./docs/wire_formats_and_storage.md).

`Rate::erasure_frame` builds a wire frame marked as an erasure, to hand
`decode_bits` in place of a frame the transport lost. Both rates mark an
erasure in-band by placing the pitch index outside its valid range, so the
decoder holds the last good frame and fades it out across successive erasures.

## Streaming and the rest of the façade

`Vocoder` is one handle per channel direction, enum-dispatched with no `dyn`.
Around the per-frame primitive:

- `encode_stream` / `decode_stream` — slice → `Iterator<Item = Result<…>>`,
  dropping a trailing partial frame.
- `LiveEncoder` / `LiveDecoder` — chunk-driven, with an internal residue
  buffer, for audio-callback and socket use. `push` returns whatever frames the
  new chunk completed; `flush` drains the tail.
- `decode_soft` / `decode_stream_soft` — soft-decision decode from per-bit LLRs
  (`&[i8]`), for receivers that can surface demodulator confidence.
  `soft_frame_bits` gives the expected count. Worth roughly 2 dB.
- `decode_bits_with_status` / `decode_info` — take a `FrameStatus` for the case
  where your own FEC layer, not this crate's, is what measured the channel.
- `encode_info` / `decode_info` — parameter vectors in and out, skipping the
  wire layer, for a sender or receiver that owns its own FEC and encryption.
- `Transcoder` — P25 Phase 1 ↔ Phase 2 at the wire-bit layer, plus the
  same-codec FEC ↔ no-FEC pairs.
- `last_stats` — `FrameStats` for the most recent frame: the `FrameDisposition`
  word (speech / tone / concealed / activity) and, encode-side, the `MbeParams`
  the emitted bits carry.
- `set_enhancement` and `VocoderBuilder` — configure the optional post-decode
  filter chain, **off by default**, because enabling it is a deviation from the
  reference.

See [`INTEGRATION.md`](./INTEGRATION.md) for the layer boundaries a consumer is
expected to own.

## Getting bits in and out, by protocol

Two of the four supported wire formats *are* air-interface formats, so P25 hands
its frames straight to `Vocoder`. DMR and NXDN carry the same codec frame under
a different interleave, so they enter one layer down.

| protocol | voice frames per burst | frame on air | how it enters |
|---|---:|---|---|
| **P25 Phase 1** (FDMA) | 9 per LDU | 18 bytes, 144 bits with Annex-H FEC | `Rate::FullRate7200x4400` — direct |
| **P25 Phase 2** (TDMA) | 2 per slot | 9 bytes, 72 bits with Annex-S interleave | `Rate::HalfRate3600x2450` — direct |
| **DMR** Tier II/III | 3 per burst | 9 bytes, 72 bits, DMR interleave | your deinterleave → code-vector core |
| **NXDN / IDAS** | 2 per burst | 9 bytes, 72 bits, sequential | your deinterleave → code-vector core |

### P25 — direct

The wire format is implemented, interleave included. Hand it the frame:

```rust
use blip25_vocoder::vocoder::{Rate, Vocoder};

let mut rx = Vocoder::new(Rate::FullRate7200x4400);   // Phase 1, 18-byte frames
let pcm = rx.decode_bits(&frame)?;                    // 160 samples

let mut tx = Vocoder::new(Rate::FullRate7200x4400);
let frame = tx.encode_pcm(&pcm)?;
```

Phase 2 is the same with `Rate::HalfRate3600x2450` and 9-byte frames.

### DMR and NXDN / IDAS — one layer down

The 49→72-bit voice-frame FEC belongs to the **codec**, not to any protocol:
the same Golay pair, the same `û₀`-seeded PN scramble, the same bit
prioritization carry P25 Phase 2, DMR and NXDN alike. Only the interleave and
the burst framing differ. So you supply the (de)interleave; the crate supplies
everything below it.

```rust
use blip25_vocoder::halfrate::frame::{decode_code_vectors, encode_code_vectors};
use blip25_vocoder::vocoder::{FrameStatus, Rate, Vocoder};

// Receive: your deinterleave lands the burst's 72 bits in c₀..c₃.
let frame = decode_code_vectors(c);                   // Golay + PN, both codes
let mut rx = Vocoder::new(Rate::HalfRate3600x2450);
let pcm = rx.decode_info(
    &frame.info,
    FrameStatus::new(frame.error_total().into(), false),  // drives concealment
)?;

// Transmit: parameters out, your interleave takes it from c₀..c₃.
let mut tx = Vocoder::new(Rate::HalfRate3600x2450);
let info: [u16; 4] = tx.encode_info(&pcm)?.try_into().unwrap();
let c = encode_code_vectors(&info);
```

`c₀..c₃` are 24, 23, 11 and 14 bits, LSB-aligned in `u32`, highest index
transmitted first. `decode_code_vectors_soft` takes the same four vectors as
per-bit confidence if your demodulator has it.

Three things worth knowing before you wire this up:

- **`decode_frame` / `encode_frame` are the P25 adapter**, not the shared path.
  They prepend Annex-S. A foreign protocol wants the `*_code_vectors` pair.
- **Pass your FEC's error count** to `decode_info`. It drives concealment inside
  the codec; claiming a clean frame on a damaged stream switches concealment off
  exactly when the channel needs it.
- **If you move parameter bits as bytes, use `pack_natural` / `unpack_natural`**
  — `û₀‖û₁‖û₂‖û₃` laid down sequentially, which is what mbelib, DSD, MMDVM,
  dvmhost and an IDAS/NXDN over-the-air wire use, and what P25 encrypts
  (TIA-102.AAAD-B §1.2). `Rate::HalfRate2450x2450` emits it directly.

[`examples/foreign_protocol.rs`](./examples/foreign_protocol.rs) is this path end
to end — both directions, the soft path, injected bit errors, and the byte
layout printed out.

### Encrypted links

Encryption applies to the parameter bits, with the FEC wrapped around the
ciphertext, so decryption sits between your FEC layer and the codec. That is
what `encode_info` / `decode_info` are for: they skip the wire layer entirely,
so nothing is packed and unpacked on the way through.

## Cargo features

| feature | default | what it gates |
|---|---|---|
| `encode` | on | the encode public API — `Vocoder::encode*`, `LiveEncoder`, the encode streaming iterators |
| `decode` | on | the decode public API — `Vocoder::decode*`, `Rate`, the shared frame and parameter types, the wire and rate-conversion layers |
| `serde` | off | `Serialize` / `Deserialize` derives on the diagnostic types (`Rate`, `FrameStats`, `AnalysisStats`, `FrameDisposition`, `DecodeStats`) and on `MbeParams` |

`decode` stands alone, so `--no-default-features --features decode` builds a
receive-only crate. `serde` is the only optional dependency and the only thing
that pulls anything in; the derives let an RPC layer ship stats and parameters
without a hand-rolled converter.

## Architecture

Three orthogonal axes meeting at one interchange type:

```text
                    ┌──────────────┐
   wire  ──────────▶│  MbeParams   │
   fullrate         │              │
   halfrate         │  ω₀          │
                    │  L           │
                    │  voiced[]    │
                    │  amplitudes[]│
                    └──────┬───────┘
                           │
                    rate_conversion
                    bits → params → bits, no PCM
```

- **`vocoder`** — the façade; owns all per-rate state.
- **`fullrate` / `halfrate`** — one module per protocol-rate combination:
  deframe, FEC, deinterleave, dequantize.
- **`mbe_params`** — the interchange type: fundamental frequency, harmonic
  count, per-harmonic voicing and spectral amplitudes.
- **`rate_conversion`** — a *peer* of the wire and codec layers, not a decoder
  afterthought; converts in the parameter domain, never touching PCM.
- **`fec`** — Golay / Hamming, hard and soft decision.
- **`enhancement`** — the optional post-decode filter chain.
- **`bits`** — private. The bit-map primitive the wire layers and the generated
  tables share.
- **`engine`** — private. The codec core.

`src/generated/` is not a module: it holds the Annex tables (B–T), transcribed
from the standard and `include!`d by the wire layers that index them.

Relative size is a fair map of where the difficulty lives: the wire, parameter
and API layers together run to roughly 22,600 lines; the codec core alone is
roughly 24,200.

### Inside the codec core

```text
encode.rs   12,072 lines   ← analysis, quantization, packing
decode.rs    5,743         ← unpacking, parameter reconstruction, synthesis
dsp.rs       4,362         ← the fixed-point primitives both sides share
tables.rs      622         ← constants recovered from the binary
```

The asymmetry is the shape of the problem: analysis has to *decide* things —
pitch, voicing, how to spend the bit budget — while synthesis follows
instructions. Encode carries roughly twice the code for that reason.

There is one core, not two. Both codecs are the same fixed-point engine selected
by a `Mode` flag, differing in payload width (88 bits full rate, 49 half rate)
and in which tables they index. It is allocation-free and `core`-only:
`conformance/no-std-guard` mounts `src/engine/mod.rs` as the root of its own
`#![no_std]` crate so that property keeps being enforced at compile time.
`decode.rs`, `dsp.rs`, `encode.rs` and `tables.rs` are a byte-identical port of
the reference vocoder, which is what lets the conformance tests demand exact
integer equality on captured frames — and why those files are not refactored for
style.

### The path through

```text
encode:  &[i16; 160]
           → Vocoder::encode_pcm
           → Encoder (Mode::FullRate | Mode::HalfRate)
           → [analysis: pitch → voicing → amplitude → VQ]
           → info vectors  ── encode_info returns here ──▶
           → [prioritize → FEC → interleave → pack]
           → Vec<u8>

decode:  &[u8]
           → Vocoder::decode_bits  (length-checked first)
           → [unpack → deinterleave → FEC → deprioritize]
           → info vectors  ◀── decode_info enters here ──
           → Decoder (+ FrameStatus: severity drives concealment)
           → [tone classify → dequantize → OLA synth]
           → enhancement (no-op by default)
           → Vec<i16>
```

The codec core's correctness target is the reference vocoder's output, not a
fidelity score. A change that scores better on a spectral or perceptual metric
while moving the output away from the reference is a regression, so the
listening and metric harnesses locate divergence rather than authorize it.

## Testing

`cargo test` is meaningful on a bare checkout and is what CI gates on. Nothing
in it requires external material.

| Test | Needs | What it establishes |
|---|---|---|
| `roundtrip` | nothing | property tests over the pipeline, including the rate-conversion distortion net |
| `no_panic_garbage` | nothing | the public API never panics on hostile or malformed input |
| `stream_soak` | nothing (Linux measurement) | `LiveEncoder`, `LiveDecoder` and `decode_stream` hold bounded memory over a long run |
| `golden_corpus` | ships with its fixture | output drift, and cross-platform bit-exactness |
| `dvsi_gold` | `tests/dvsi_gold.bin` | conformance against the reference vocoder, end to end through the wire layer |
| engine golden | `tests/engine_golden.bin` | function-level bit-exactness of the codec core |

Two distinctions are worth holding onto:

- **`golden_corpus` is a drift detector, not an oracle.** Every byte of its
  fixture was produced by this crate, so it can only detect *change*. What makes
  it valuable is that the fixture is frozen bytes: it cannot move with the host,
  so a float divergence on aarch64, macOS or Windows fails loudly instead of
  shipping. A test that compared the façade against the core would stay green
  through the same divergence, because both sides would move together.
- **`dvsi_gold` and the engine golden carry the correctness claim.** Both replay
  captures taken from the reference vocoder — data that is not ours to
  redistribute, so neither fixture ships in the published crate. When a fixture
  is absent the test prints what is missing and passes, so a fresh checkout is
  never a red suite and a contributor never needs the corpus to validate their
  work.

Every entry point that takes untrusted input is fuzzed — `decode_bits`,
`decode_soft`, `encode_pcm` and the transcoder (`fuzz/`, which sits outside the
workspace because it requires nightly). The workspace also carries
`conformance/roundtrip`, a `publish = false` harness whose examples measure
rate-conversion quality and bit-exactness against the reference corpus, and
`conformance/no-std-guard`. See [`RELEASING.md`](./RELEASING.md) for the release
process.

## Provenance

Provenance splits by **layer**, not by codec — the two codecs are close to
symmetric:

| Layer | Provenance |
|---|---|
| Wire framing, FEC, bit interpretation (both codecs) | **Spec** — TIA-102.BABA / BABA-A |
| Quantizer data indexed by the wire layer | **Spec** — the published Annex tables |
| The codec engine — analysis, synthesis, and its own tables (shared) | **Reverse-engineered** |

**Both codecs' wire layers are spec-derived.** Deframing, FEC, and bit
interpretation come from TIA-102.BABA / BABA-A; the IMBE half additionally
draws on the published fixed-point reference and ITU-T G.191 basic operators.

**Both codecs' audio comes from a reverse-engineered core.** AMBE+2 and IMBE
are not two engines but one, selected by a mode flag. It was recovered from a
compiled image of the DVSI reference software vocoder — x86 disassembly
transliterated function by function to fixed-point Rust, constants recovered
from the binary's data section, correctness established by differential testing
at exact integer equality. It is not a copy of that vendor's source, which this
project never obtained or read; it *is* a derivation of its behavior, and it
reads on the active patents above.

[`ATTRIBUTION.md`](./ATTRIBUTION.md) is the authoritative statement of what
is original work and what is derived, and states it in both directions —
neither overstating originality nor understating the derivation.

## Further reading

- [`PATENT_NOTICE.md`](./PATENT_NOTICE.md) — the patent list, the
  comparable-project survey, and project policy.
- [`ATTRIBUTION.md`](./ATTRIBUTION.md) — what is original, what is derived, and
  the trademark statement.
- [`DESIGN.md`](./DESIGN.md) — the architectural rationale behind the module
  layout and the two entry altitudes.
- [`INTEGRATION.md`](./INTEGRATION.md) — the layer boundaries a consumer owns.
- [`docs/codec_family_explainer.md`](./docs/codec_family_explainer.md) — the MBE
  codec family, and the wire-format-versus-implementation distinction that
  explains why open-source P25 audio and a commercial radio's differ.
- [`docs/wire_formats_and_storage.md`](./docs/wire_formats_and_storage.md) —
  what each `Rate`'s byte layout means and which one to store.
- [`CHANGELOG.md`](./CHANGELOG.md) and [`RELEASING.md`](./RELEASING.md).

## License

MIT. See [`LICENSE`](./LICENSE).
