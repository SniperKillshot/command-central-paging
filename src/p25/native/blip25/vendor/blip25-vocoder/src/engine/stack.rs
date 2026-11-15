//! Peak stack measurement.
//!
//! Stack, not code size, is usually the binding constraint when a codec has to
//! live on an RTOS task. This measures the real high-water mark by running the
//! codec on a thread of known size and bisecting the smallest stack it survives
//! on — a truthful answer even under inlining, unlike summing frame sizes out
//! of a disassembly.
//!
//! Each probe runs in a **child process**: Rust aborts the whole process on
//! stack overflow, so an in-process bisection would kill the test harness on
//! its first undersized attempt.

use std::env;
use std::process::{Command, Stdio};
use std::thread;

use super::{Decoder, Encoder, Mode, FRAME_SAMPLES};

const VAR_KIND: &str = "MBE_STACK_KIND";
const VAR_SIZE: &str = "MBE_STACK_SIZE";

fn exercise_decoder() {
    let mut pcm = [0i16; FRAME_SAMPLES];
    for mode in [Mode::HalfRate, Mode::FullRate] {
        let mut d = Decoder::new(mode);
        let n = d.payload_bits();
        let mut bits = [0i16; 128];
        for f in 0..8 {
            for (i, b) in bits[..n].iter_mut().enumerate() {
                *b = ((i * 7 + f * 13) & 1) as i16;
            }
            d.decode(&bits[..n], &mut pcm);
        }
    }
}

fn exercise_encoder() {
    let mut bits = [0i16; 128];
    for mode in [Mode::HalfRate, Mode::FullRate] {
        let mut e = Encoder::new(mode);
        let n = e.payload_bits();
        for f in 0..8usize {
            let mut pcm = [0i16; FRAME_SAMPLES];
            for (i, s) in pcm.iter_mut().enumerate() {
                let t = (f * FRAME_SAMPLES + i) as i32;
                *s = (((t * 37) % 4001) - 2000) as i16;
            }
            e.encode(&pcm, &mut bits[..n]);
        }
    }
}

/// Child mode: run one workload on a thread of the requested size. Exits
/// non-zero (via abort) if it does not fit.
fn child_probe() -> bool {
    let kind = env::var(VAR_KIND).unwrap();
    let size: usize = env::var(VAR_SIZE).unwrap().parse().unwrap();
    let f: fn() = if kind == "enc" {
        exercise_encoder
    } else {
        exercise_decoder
    };
    thread::Builder::new()
        .stack_size(size)
        .spawn(f)
        .unwrap()
        .join()
        .is_ok()
}

/// Re-invoke this test binary as a child with the given stack size.
fn survives(kind: &str, size: usize) -> bool {
    let exe = env::current_exe().unwrap();
    Command::new(exe)
        .arg("--exact")
        .arg("stack_high_water_mark")
        .arg("--nocapture")
        .env(VAR_KIND, kind)
        .env(VAR_SIZE, size.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn peak(kind: &str) -> usize {
    let mut lo = 8 * 1024;
    let mut hi = 8 * 1024 * 1024;
    assert!(survives(kind, hi), "{kind}: did not survive 8 MB");
    while hi - lo > 4 * 1024 {
        let mid = lo + (hi - lo) / 2;
        if survives(kind, mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

#[test]
fn stack_high_water_mark() {
    // Child invocation: do the work and let success/abort be the answer.
    if env::var(VAR_KIND).is_ok() {
        assert!(child_probe());
        return;
    }

    let d = peak("dec");
    let e = peak("enc");
    println!("peak stack (subprocess bisection, 4 KB resolution):");
    println!("  decoder: {:7} B  ({:5.1} KB)", d, d as f64 / 1024.0);
    println!("  encoder: {:7} B  ({:5.1} KB)", e, e as f64 / 1024.0);

    // Guard against unbounded growth rather than pinning an exact figure:
    // these workloads are recursion-free and fixed-size, so a large jump means
    // something started allocating per frame.
    assert!(d < 256 * 1024, "decoder stack regressed: {d} B");
    assert!(e < 512 * 1024, "encoder stack regressed: {e} B");
}
