# Vendored blip25-vocoder

`vendor/blip25-vocoder` is [blip25-vocoder](https://github.com/openBLIP25/blip25-vocoder)
1.0.0 from crates.io (MIT, `LICENSE` there). Its tests, examples and test data
are left out, and its `Cargo.toml` has their targets removed.

Two local changes, each marked `CCP patch` in the source. Neither changes
any engine code, and with both left at their defaults the output is
upstream's bit for bit.

## 1. Opt-in tone detection

- `src/engine/mod.rs`: `Encoder` gains a `flags` word, passed to `enc_frame`
  where upstream hardcodes `0`. Also added: `set_flags`, `flags`, and
  `ENC_FLAG_TONE_DETECT` (`0x1000`, the master enable the reference engine's
  in-band tone detector tests in `tone_class_half_frame`).
- `src/vocoder.rs`: `Vocoder::set_tone_detection(bool)` and
  `Vocoder::tone_detection()`.

Upstream never enables the detector. The stock analysis codes a steady sine
above about 405 Hz as noise or buzz, so alert and paging tones come out
mangled. With the detector on they are coded as clean lines, but snapped to
a raster that is a few percent off below 405 Hz. Speech encodes identically
either way.

## 2. Encoder state hand-over

- `src/engine/mod.rs`: `Encoder::copy_state_from`.
- `src/vocoder.rs`: `Vocoder::copy_encode_state_from` (engine encoder state
  plus the parameter readback), with `#[derive(Clone)]` on `ParamReadback`.

`src/lib.rs` runs a stock encoder and a tone-detecting twin over the same
audio and picks a frame per frame. IMBE codes amplitudes relative to the
previous frame, so after each choice the encoder whose frame was not sent
takes over the other's state. Without that, switching overshoots by up to
about 7 dB and clicks.

## Updating

Copy the new release's `src/`, `Cargo.toml`, `LICENSE` and `README.md` over
`vendor/blip25-vocoder`, re-apply the changes above (search for
`CCP patch`), strip the `[[example]]`/`[[test]]` targets and
dev-dependencies from `Cargo.toml`, rebuild, and run `tests/test_blip25.py`.
The golden hashes there will change if the release changes the codec's
output.
