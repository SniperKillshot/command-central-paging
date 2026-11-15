//! Bit-exactness conformance against the reference implementation.
//!
//! `tests/engine_golden.bin` is captured from the reference vocoder image
//! executing under emulation. It freezes the oracle's answers so the port can be
//! proven bit-exact on any host — no emulator, no reference binary, no x86.
//!
//! Format (little-endian):
//!
//! ```text
//! magic   "MBEGOLD\x01"
//! u32     n_cases
//! per case:
//!     u8  kind        0 = decode, 1 = encode
//!     u8  mode        1 = half rate, 0 = full rate
//!     u16 n_in        input element count
//!     u16 n_out       output element count
//!     i16 input [n_in]
//!     i16 output[n_out]
//! ```
//!
//! Cases run consecutively against one codec instance, so the replay has to
//! walk them in order — cross-frame state is part of what is under test. The
//! capture is three runs per mode, in file order:
//!
//! ```text
//!   400 decode  (random payloads, one decoder instance)
//!   400 encode  (synthetic programme, a fresh encoder)
//!   884 encode  (real speech, ANOTHER fresh encoder)
//! ```
//!
//! The two encode runs came from separate DLL instances, so the replay must
//! re-init at the boundary; the encoder is stateful and carries a one-frame
//! delay, and carrying state across the seam would corrupt every frame after
//! it. [`SYNTH_FRAMES`] is that boundary.
//!
//! This harness needs file I/O, so it is compiled only under `cfg(test)`
//! and excluded from the `no_std` guard build. The codec itself is `no_std`.

// This harness walks two arrays in step and names the position in its
// output, so the index is the point rather than an artefact of the loop.
#![allow(clippy::needless_range_loop)]

use std::time::Instant;

use super::{Decoder, Encoder, Mode, FRAME_SAMPLES};

/// The fixture is a capture of the reference vocoder's own behaviour and is
/// **not tracked** — it is not ours to redistribute. Absent, every test here
/// reports what is missing and passes, so a fresh checkout is not a red suite.
/// Present, this is the function-level proof that the port is exact.
fn golden() -> Option<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/engine_golden.bin");
    match std::fs::read(&path) {
        Ok(b) => Some(b),
        Err(_) => {
            eprintln!(
                "SKIP: {} is absent. This is the reference-conformance check; it \
                 needs a capture that is not ours to redistribute.",
                path.display()
            );
            None
        }
    }
}

struct Case<'a> {
    kind: u8,
    mode: u8,
    input: &'a [u8],
    output: &'a [u8],
}

impl Case<'_> {
    fn input_i16(&self, i: usize) -> i16 {
        i16::from_le_bytes([self.input[2 * i], self.input[2 * i + 1]])
    }
    fn output_i16(&self, i: usize) -> i16 {
        i16::from_le_bytes([self.output[2 * i], self.output[2 * i + 1]])
    }
    fn n_in(&self) -> usize {
        self.input.len() / 2
    }
    fn n_out(&self) -> usize {
        self.output.len() / 2
    }
}

/// Walk the file. Every read is byte-at-a-time little-endian, so the harness is
/// endian- and alignment-agnostic: it gives the same answer on armv7, aarch64
/// and a big-endian host.
fn cases(blob: &[u8]) -> Vec<Case<'_>> {
    assert_eq!(&blob[..8], b"MBEGOLD\x01", "bad magic");
    let n = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]) as usize;
    let mut at = 12;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let kind = blob[at];
        let mode = blob[at + 1];
        let n_in = u16::from_le_bytes([blob[at + 2], blob[at + 3]]) as usize;
        let n_out = u16::from_le_bytes([blob[at + 4], blob[at + 5]]) as usize;
        at += 6;
        let input = &blob[at..at + 2 * n_in];
        at += 2 * n_in;
        let output = &blob[at..at + 2 * n_out];
        at += 2 * n_out;
        out.push(Case {
            kind,
            mode,
            input,
            output,
        });
    }
    assert_eq!(at, blob.len(), "trailing bytes in golden.bin");
    out
}

fn mode_of(m: u8) -> Mode {
    match m {
        1 => Mode::HalfRate,
        0 => Mode::FullRate,
        other => panic!("unknown mode {other}"),
    }
}

/// Replay every decode case in the file and demand exact equality.
#[test]
fn decode_is_bit_exact() {
    let Some(blob) = golden() else { return };
    let all = cases(&blob);
    let mut decoders: [Option<Decoder>; 2] = [None, None];
    let mut pcm = [0i16; FRAME_SAMPLES];
    let mut bits = [0i16; 128];

    let mut n_run = [0usize; 2];
    for (idx, c) in all.iter().enumerate() {
        if c.kind != 0 {
            continue;
        }
        let mode = mode_of(c.mode);
        let d = decoders[c.mode as usize].get_or_insert_with(|| Decoder::new(mode));
        assert_eq!(c.n_in(), d.payload_bits(), "case {idx}: payload width");
        assert_eq!(c.n_out(), FRAME_SAMPLES, "case {idx}: output width");

        for i in 0..c.n_in() {
            bits[i] = c.input_i16(i);
        }
        d.decode(&bits[..c.n_in()], &mut pcm);

        for i in 0..FRAME_SAMPLES {
            let want = c.output_i16(i);
            if pcm[i] != want {
                let ctx: Vec<(usize, i16, i16)> = (i.saturating_sub(2)..(i + 3).min(FRAME_SAMPLES))
                    .map(|k| (k, pcm[k], c.output_i16(k)))
                    .collect();
                panic!(
                    "case {idx} ({:?}, frame {} of this mode): first divergence at sample {i}: \
                     got {}, want {}\n  (index, got, want) around it: {:?}",
                    mode, n_run[c.mode as usize], pcm[i], want, ctx
                );
            }
        }
        n_run[c.mode as usize] += 1;
    }

    // Guard against silent shrinkage: a truncated or mis-parsed golden file
    // would otherwise leave this test green while checking almost nothing.
    assert_eq!(n_run[1], 400, "half-rate decode case count changed");
    assert_eq!(n_run[0], 400, "full-rate decode case count changed");
    println!(
        "decode conformance: {} half rate + {} full rate = {} frames, all bit-exact",
        n_run[1],
        n_run[0],
        n_run[0] + n_run[1]
    );
}

/// Reset must reproduce the power-on state exactly: replaying the first frames
/// of a mode after `reset()` gives the same PCM as a fresh instance.
#[test]
fn reset_matches_power_on() {
    let Some(blob) = golden() else { return };
    let all = cases(&blob);
    for m in [1u8, 0u8] {
        let first: Vec<&Case<'_>> = all
            .iter()
            .filter(|c| c.kind == 0 && c.mode == m)
            .take(8)
            .collect();
        let mut d = Decoder::new(mode_of(m));
        let mut pcm = [0i16; FRAME_SAMPLES];
        for c in &first {
            let bits: Vec<i16> = (0..c.n_in()).map(|i| c.input_i16(i)).collect();
            d.decode(&bits, &mut pcm);
        }
        d.reset();
        for c in &first {
            let bits: Vec<i16> = (0..c.n_in()).map(|i| c.input_i16(i)).collect();
            d.decode(&bits, &mut pcm);
            for i in 0..FRAME_SAMPLES {
                assert_eq!(pcm[i], c.output_i16(i), "reset diverged at sample {i}");
            }
        }
    }
}

/// `decode_packed` must agree with `decode` on the same payload.
#[test]
fn packed_matches_unpacked() {
    let Some(blob) = golden() else { return };
    let all = cases(&blob);
    for m in [1u8, 0u8] {
        let mut a = Decoder::new(mode_of(m));
        let mut b = Decoder::new(mode_of(m));
        let (mut pa, mut pb) = ([0i16; FRAME_SAMPLES], [0i16; FRAME_SAMPLES]);
        for c in all.iter().filter(|c| c.kind == 0 && c.mode == m).take(32) {
            let bits: Vec<i16> = (0..c.n_in()).map(|i| c.input_i16(i)).collect();
            let mut packed = vec![0u8; c.n_in().div_ceil(8)];
            for (i, &v) in bits.iter().enumerate() {
                if v & 1 != 0 {
                    packed[i >> 3] |= 1 << (7 - (i & 7));
                }
            }
            a.decode(&bits, &mut pa);
            b.decode_packed(&packed, &mut pb);
            assert_eq!(pa, pb);
        }
    }
}

/// The tables the DSP layer keeps privately must agree with the extracted
/// `const` tables where they overlap — the aliasing in `dsp::tables` is only
/// sound if the two really are the same bytes.
#[test]
fn table_aliases_agree() {
    // COS512 is the first 512 words of the 0x1037f978 blob.
    assert_eq!(
        crate::engine::tables::COS512[..],
        crate::engine::tables::BLOB[..512]
    );
}

/// Throughput, printed with `--nocapture`. Not a pass/fail gate — it asserts
/// only that decoding is faster than real time, which even a debug build is.
#[test]
fn throughput() {
    let Some(blob) = golden() else { return };
    let all = cases(&blob);
    for (m, name) in [(1u8, "half rate"), (0u8, "full rate")] {
        let payloads: Vec<Vec<i16>> = all
            .iter()
            .filter(|c| c.kind == 0 && c.mode == m)
            .map(|c| (0..c.n_in()).map(|i| c.input_i16(i)).collect())
            .collect();
        let mut d = Decoder::new(mode_of(m));
        let mut pcm = [0i16; FRAME_SAMPLES];

        // Warm up, then time enough frames that the clock is not the limit.
        for p in &payloads {
            d.decode(p, &mut pcm);
        }
        let reps = 25usize;
        let n = payloads.len() * reps;
        let t0 = Instant::now();
        for _ in 0..reps {
            for p in &payloads {
                d.decode(p, &mut pcm);
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        let fps = n as f64 / dt;
        // One frame is 20 ms of audio.
        println!(
            "{name}: {n} frames in {:.3} s = {:.0} frames/s = {:.0}x realtime \
             ({:.1} us/frame)",
            dt,
            fps,
            fps * 0.020,
            1e6 / fps
        );
        assert!(fps * 0.020 > 1.0, "{name} decode is slower than real time");
    }
}

/// The decoder is a plain value with no interior pointers: it can live on the
/// stack, in a `static`, or anywhere else. This pins its footprint.
#[test]
fn decoder_footprint() {
    let n = std::mem::size_of::<Decoder>();
    println!("size_of::<Decoder>() = {n} bytes");
    assert!(n <= 4096, "decoder grew to {n} bytes");
}

// ===========================================================================
// Encoder
// ===========================================================================

/// Encode cases per mode in the synthetic-programme run, i.e. the index at
/// which the capture switched to a fresh DLL instance fed real speech.
const SYNTH_FRAMES: usize = 400;
/// Encode cases per mode in the real-speech run.
const SPEECH_FRAMES: usize = 884;

/// Replay every encode case in the file and demand exact equality.
///
/// Both modes, both runs — synthetic programme and DVSI conformance-vector
/// speech — with a re-init at the run boundary, exactly as the capture did.
#[test]
fn encode_is_bit_exact() {
    let Some(blob) = golden() else { return };
    let all = cases(&blob);
    let mut payload = [0i16; 128];
    let mut pcm = [0i16; FRAME_SAMPLES];

    let mut n_run = [[0usize; 2]; 2]; // [mode][run]
    for m in [1u8, 0u8] {
        let mut e = Encoder::new(mode_of(m));
        for (k, c) in all
            .iter()
            .enumerate()
            .filter(|(_, c)| c.kind == 1 && c.mode == m)
            .enumerate()
        {
            let (idx, c) = c;
            // The capture switched to a fresh instance here.
            if k == SYNTH_FRAMES {
                e.reset();
            }
            let run = usize::from(k >= SYNTH_FRAMES);

            assert_eq!(c.n_in(), FRAME_SAMPLES, "case {idx}: input width");
            assert_eq!(c.n_out(), e.payload_bits(), "case {idx}: payload width");

            for i in 0..FRAME_SAMPLES {
                pcm[i] = c.input_i16(i);
            }
            let nb = c.n_out();
            e.encode(&pcm, &mut payload[..nb]);

            for i in 0..nb {
                let want = c.output_i16(i);
                if payload[i] != want {
                    let ctx: Vec<(usize, i16, i16)> = (i.saturating_sub(2)..(i + 3).min(nb))
                        .map(|j| (j, payload[j], c.output_i16(j)))
                        .collect();
                    panic!(
                        "case {idx} ({:?}, {} run, frame {} of that run): first divergence at \
                         payload bit {i}: got {}, want {}\n  (bit, got, want) around it: {:?}",
                        mode_of(m),
                        if run == 0 { "synthetic" } else { "speech" },
                        k - run * SYNTH_FRAMES,
                        payload[i],
                        want,
                        ctx
                    );
                }
            }
            n_run[usize::from(m == 1)][run] += 1;
        }
    }

    // Anti-vacuity: a truncated or mis-parsed file must fail loudly.
    for (mi, name) in [(1usize, "half rate"), (0usize, "full rate")] {
        assert_eq!(n_run[mi][0], SYNTH_FRAMES, "{name} synthetic encode count");
        assert_eq!(n_run[mi][1], SPEECH_FRAMES, "{name} speech encode count");
    }
    let total: usize = n_run.iter().flatten().sum();
    println!(
        "encode conformance: {} frames all bit-exact \
         ({SYNTH_FRAMES} synthetic + {SPEECH_FRAMES} speech, per mode)",
        total
    );
}

/// `Encoder::reset` must reproduce the power-on state exactly.
#[test]
fn encoder_reset_matches_power_on() {
    let Some(blob) = golden() else { return };
    let all = cases(&blob);
    for m in [1u8, 0u8] {
        let first: Vec<&Case<'_>> = all
            .iter()
            .filter(|c| c.kind == 1 && c.mode == m)
            .take(8)
            .collect();
        let mut e = Encoder::new(mode_of(m));
        let mut payload = [0i16; 128];
        let mut pcm = [0i16; FRAME_SAMPLES];
        for _ in 0..2 {
            for c in &first {
                for i in 0..FRAME_SAMPLES {
                    pcm[i] = c.input_i16(i);
                }
                let nb = c.n_out();
                e.encode(&pcm, &mut payload[..nb]);
                for i in 0..nb {
                    assert_eq!(payload[i], c.output_i16(i), "reset diverged at bit {i}");
                }
            }
            e.reset();
        }
    }
}

/// `encode_packed` must agree with `encode` on the same audio.
#[test]
fn encode_packed_matches_unpacked() {
    let Some(blob) = golden() else { return };
    let all = cases(&blob);
    for m in [1u8, 0u8] {
        let mut a = Encoder::new(mode_of(m));
        let mut b = Encoder::new(mode_of(m));
        let mut bits = [0i16; 128];
        let mut packed = [0u8; 11];
        for c in all.iter().filter(|c| c.kind == 1 && c.mode == m).take(32) {
            let pcm: Vec<i16> = (0..FRAME_SAMPLES).map(|i| c.input_i16(i)).collect();
            let nb = c.n_out();
            a.encode(&pcm, &mut bits[..nb]);
            b.encode_packed(&pcm, &mut packed[..nb.div_ceil(8)]);
            for i in 0..nb {
                assert_eq!(
                    bits[i] & 1,
                    ((packed[i >> 3] >> (7 - (i & 7))) & 1) as i16,
                    "packed bit {i}"
                );
            }
        }
    }
}

/// Encode throughput, printed with `--nocapture`.
#[test]
fn encode_throughput() {
    let Some(blob) = golden() else { return };
    let all = cases(&blob);
    for (m, name) in [(1u8, "half rate"), (0u8, "full rate")] {
        let frames: Vec<Vec<i16>> = all
            .iter()
            .filter(|c| c.kind == 1 && c.mode == m)
            .take(400)
            .map(|c| (0..FRAME_SAMPLES).map(|i| c.input_i16(i)).collect())
            .collect();
        let mut e = Encoder::new(mode_of(m));
        let mut bits = [0i16; 128];
        let nb = e.payload_bits();

        for f in &frames {
            e.encode(f, &mut bits[..nb]);
        }
        let reps = 25usize;
        let n = frames.len() * reps;
        let t0 = Instant::now();
        for _ in 0..reps {
            for f in &frames {
                e.encode(f, &mut bits[..nb]);
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        let fps = n as f64 / dt;
        println!(
            "{name} encode: {n} frames in {:.3} s = {:.0} frames/s = {:.0}x realtime \
             ({:.1} us/frame)",
            dt,
            fps,
            fps * 0.020,
            1e6 / fps
        );
        assert!(fps * 0.020 > 1.0, "{name} encode is slower than real time");
    }
}

/// The encoder is a plain value with no interior pointers. This pins its
/// footprint.
#[test]
fn encoder_footprint() {
    let n = std::mem::size_of::<Encoder>();
    println!("size_of::<Encoder>() = {n} bytes");
    assert!(n <= 8192, "encoder grew to {n} bytes");
}

/// The disposition word is a **bit word**, not an ordinal, and the public
/// wrapper has to decode it as one. This pins the mapping against the values
/// the engine actually returns — a regression here is invisible in the PCM
/// comparisons, so nothing else in the suite would catch it.
#[test]
fn frame_disposition_mapping() {
    use crate::engine::Frame;
    let Some(blob) = golden() else { return };
    let all = cases(&blob);
    let mut decoders: [Option<Decoder>; 2] = [None, None];
    let mut pcm = [0i16; FRAME_SAMPLES];
    let mut bits = [0i16; 128];
    let mut seen = std::collections::BTreeMap::new();

    for c in all.iter().filter(|c| c.kind == 0) {
        let mode = mode_of(c.mode);
        let d = decoders[c.mode as usize].get_or_insert_with(|| Decoder::new(mode));
        for i in 0..c.n_in() {
            bits[i] = c.input_i16(i);
        }
        let f = d.decode(&bits[..c.n_in()], &mut pcm);

        // Whatever it is, the variant must agree with the raw word.
        let code = f.code();
        let expect = if (code as u16) & 0x8000 != 0 {
            "tone"
        } else if (code as u16) & 0x0020 != 0 {
            "repeat"
        } else if (code as u16) & 0x0002 != 0 {
            "speech"
        } else if code == 0 {
            "silence"
        } else {
            "other"
        };
        let got = match f {
            Frame::Tone(_) => "tone",
            Frame::Repeat(_) => "repeat",
            Frame::Speech(_) => "speech",
            Frame::Silence => "silence",
            Frame::Other(_) => "other",
        };
        assert_eq!(got, expect, "disposition {code:#06x} classified as {got}");
        *seen.entry(got).or_insert(0usize) += 1;
    }

    println!("dispositions over the golden decode set: {seen:?}");
    // Speech must dominate, and silence must occur: reading the disposition
    // word as an ordinal lands speech frames in `Silence`, which fails here.
    assert!(
        seen.get("speech").copied().unwrap_or(0) > 100,
        "no speech frames seen"
    );
}

/// The status words are load-bearing, and their live ranges are narrow and
/// trap-laden. This pins the behaviour the API documents: concealment engages
/// over 0..=15, the mute flag silences, and a raw metric above 15 must be
/// clamped rather than passed through — unclamped it lands in a dead band that
/// silently turns concealment off on the worst frames.
#[test]
fn frame_status_ranges() {
    use crate::engine::FrameStatus;
    let Some(blob) = golden() else { return };
    let all = cases(&blob);
    let frames: Vec<Vec<i16>> = all
        .iter()
        .filter(|c| c.kind == 0 && c.mode == 0)
        .take(12)
        .map(|c| (0..c.n_in()).map(|i| c.input_i16(i)).collect())
        .collect();

    let run = |st: FrameStatus| -> Vec<i16> {
        let mut d = Decoder::new(Mode::FullRate);
        let mut out = Vec::new();
        let mut pcm = [0i16; FRAME_SAMPLES];
        for f in &frames {
            d.decode_with_status(f, st, &mut pcm);
            out.extend_from_slice(&pcm);
        }
        out
    };
    let rms = |v: &[i16]| {
        (v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / v.len() as f64).sqrt()
    };

    let clean = run(FrameStatus::CLEAN);
    assert_eq!(clean, run(FrameStatus::new(0, false)), "0 must equal CLEAN");

    // Concealment engages and deepens across the live band.
    let mid = run(FrameStatus::new(7, false));
    let hard = run(FrameStatus::new(15, false));
    assert_ne!(mid, clean, "error_rate 7 must change the output");
    assert!(rms(&mid) < rms(&clean), "concealment should reduce energy");
    assert!(rms(&hard) < rms(&mid), "15 should conceal harder than 7");

    // The trap: a raw metric above the live band must be clamped, not passed.
    // `new` clamps, so a huge value behaves like 15 rather than reverting.
    assert_eq!(
        run(FrameStatus::new(255, false)),
        hard,
        "255 must clamp to 15"
    );
    assert_eq!(run(FrameStatus::new(9999, false)), hard, "clamp must hold");

    // The forcing word: bit 2 repeats-and-fades, bit 3 mutes. These are
    // different, and writing 4 for "mute" would silently request the former.
    let muted = run(FrameStatus::CLEAN.with_mute(true));
    let lost = run(FrameStatus::CLEAN.with_lost(true));
    assert!(
        rms(&muted) < 2.0,
        "mute should silence, got rms {}",
        rms(&muted)
    );
    assert!(
        rms(&lost) > rms(&muted),
        "repeat should fade, not silence instantly"
    );

    println!(
        "status ranges: clean rms {:.0}, e=7 {:.0}, e=15 {:.0}, muted {:.1}",
        rms(&clean),
        rms(&mid),
        rms(&hard),
        rms(&muted)
    );
}
