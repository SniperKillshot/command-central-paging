//! C interface to blip25-vocoder for src/p25/blip25.py (ctypes).
//!
//! One handle is one channel direction: a full-rate (IMBE) `Vocoder` at the
//! info-only 4400x4400 rate, whose 11-byte frames are the 88 bits u0..u7 the
//! DVM FNE carries. No post-decode enhancement.
//!
//! Encoding with tone detection on runs two encoders over the same audio: the
//! stock one, and a twin with the reference engine's in-band tone detector
//! enabled (vendored crate, "CCP patch"). The stock analysis codes a steady
//! sine above ~405 Hz as noise or buzz; the detector codes it as a clean line,
//! but snaps to a raster that is a few percent off below ~405 Hz, where the
//! stock encoder does better. So per frame:
//!
//! - the frames agree (speech, silence): send it;
//! - they differ and the twin flagged a tone: send the tone frame, unless
//!   the stock frame codes a clean line closer to the input's frequency
//!   than the tone frame's line;
//! - otherwise: send the stock frame.
//!
//! A small margin keeps a tone that sits about halfway between the two lines
//! on one encoder rather than flipping frame by frame, which smears it.
//!
//! The encoder whose frame was not sent then takes over the other's state,
//! so both predict from what the decoder received (without it, switching
//! overshoots by up to ~7 dB and clicks).

use std::f64::consts::PI;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

use blip25_vocoder::mbe_params::MbeParams;
use blip25_vocoder::vocoder::{Rate, Vocoder};

/// Bump when a signature below changes; blip25.py checks it.
const API_VERSION: u32 = 2;
const FRAME_SAMPLES: usize = 160;
const FRAME_BYTES: usize = 11;
/// A "clean line": the strongest harmonic is voiced and holds this share of
/// the frame's harmonic energy.
const CLEAN_LINE_SHARE: f64 = 0.9;
/// Switch between the stock and tone lines only when the other is closer to
/// the input frequency by more than this fraction of it (0.2%).
const SWITCH_MARGIN: f64 = 0.002;
/// Input frequency search step, Hz.
const SEARCH_STEP_HZ: f64 = 0.5;

/// Which encoder the last frame came from.
const CHOICE_SAME: i32 = 0;
const CHOICE_STOCK: i32 = 1;
const CHOICE_TONE: i32 = 2;

pub struct Channel {
    stock: Vocoder,
    twin: Option<Box<Vocoder>>,
    /// The audio the next emitted frame describes (one frame of delay).
    prev_pcm: [i16; FRAME_SAMPLES],
    last_choice: i32,
    /// The tone line won the last stock-versus-tone comparison.
    tone_held: bool,
}

impl Channel {
    fn new(tone_detection: bool) -> Self {
        let twin = tone_detection.then(|| {
            let mut v = Vocoder::new(Rate::FullRate4400x4400);
            v.set_tone_detection(true);
            Box::new(v)
        });
        Channel {
            stock: Vocoder::new(Rate::FullRate4400x4400),
            twin,
            prev_pcm: [0; FRAME_SAMPLES],
            last_choice: CHOICE_SAME,
            tone_held: false,
        }
    }

    fn encode(&mut self, pcm: &[i16]) -> Option<Vec<u8>> {
        let stock_bits = self.stock.encode_pcm(pcm).ok()?;
        let Some(twin) = self.twin.as_mut() else {
            return Some(stock_bits);
        };
        let tone_bits = twin.encode_pcm(pcm).ok()?;
        let described = self.prev_pcm;
        self.prev_pcm.copy_from_slice(pcm);

        if stock_bits == tone_bits {
            self.last_choice = CHOICE_SAME;
            return Some(stock_bits);
        }
        let twin_stats = twin.last_stats().analysis.clone();
        let stock_stats = self.stock.last_stats().analysis.clone();
        let tone_flagged = twin_stats.as_ref().is_some_and(|a| a.disposition.is_tone());
        let use_tone = tone_flagged && {
            let tone_line = twin_stats.as_ref().and_then(|a| a.params.as_ref()).and_then(strongest_line);
            let stock_line = stock_stats
                .as_ref()
                .and_then(|a| a.params.as_ref())
                .and_then(strongest_line)
                .filter(|l| l.voiced && l.share >= CLEAN_LINE_SHARE);
            match (stock_line, tone_line) {
                (Some(s), Some(t)) => {
                    let f = input_frequency(&described, s.hz.min(t.hz), s.hz.max(t.hz));
                    let (dt, ds) = ((t.hz - f).abs(), (s.hz - f).abs());
                    let margin = SWITCH_MARGIN * f;
                    let tone = if self.tone_held { dt <= ds + margin } else { dt + margin < ds };
                    self.tone_held = tone;
                    tone
                }
                _ => true,
            }
        };
        if use_tone {
            self.stock.copy_encode_state_from(twin);
            self.last_choice = CHOICE_TONE;
            Some(tone_bits)
        } else {
            twin.copy_encode_state_from(&self.stock);
            self.last_choice = CHOICE_STOCK;
            Some(stock_bits)
        }
    }
}

struct Line {
    hz: f64,
    voiced: bool,
    share: f64,
}

/// The strongest harmonic of a parameter set.
fn strongest_line(p: &MbeParams) -> Option<Line> {
    let amps = p.amplitudes_slice();
    let total: f64 = amps.iter().map(|&a| (a as f64) * (a as f64)).sum();
    let (i, &a) = amps.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1))?;
    if total <= 0.0 {
        return None;
    }
    Some(Line {
        hz: (i + 1) as f64 * p.omega_0() as f64 * 8000.0 / (2.0 * PI),
        voiced: p.voiced_slice()[i],
        share: (a as f64) * (a as f64) / total,
    })
}

/// The frequency of the input's strongest component between `lo` and `hi`
/// (widened by 5% each way): the peak of its Hann-windowed power.
fn input_frequency(pcm: &[i16; FRAME_SAMPLES], lo: f64, hi: f64) -> f64 {
    let (mut best, mut best_p) = (lo, f64::MIN);
    let mut f = lo * 0.95;
    while f <= hi * 1.05 {
        let p = power_at(pcm, f);
        if p > best_p {
            (best, best_p) = (f, p);
        }
        f += SEARCH_STEP_HZ;
    }
    best
}

/// Hann-windowed power of `pcm` at `hz`.
fn power_at(pcm: &[i16; FRAME_SAMPLES], hz: f64) -> f64 {
    let n = FRAME_SAMPLES as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (k, &s) in pcm.iter().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * PI * k as f64 / (n - 1.0)).cos();
        let ph = 2.0 * PI * hz * k as f64 / 8000.0;
        re += w * s as f64 * ph.cos();
        im -= w * s as f64 * ph.sin();
    }
    re * re + im * im
}

#[no_mangle]
pub extern "C" fn ccp_blip25_version() -> u32 {
    API_VERSION
}

/// New channel, all state cold. `tone_detection` non-zero: encode with the
/// stock/tone-detector pair (see the module docs); zero: stock encoder only,
/// identical to upstream. Null on failure.
#[no_mangle]
pub extern "C" fn ccp_blip25_new(tone_detection: i32) -> *mut Channel {
    catch_unwind(|| Box::into_raw(Box::new(Channel::new(tone_detection != 0)))).unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `h` must come from `ccp_blip25_new` and not be used again.
#[no_mangle]
pub unsafe extern "C" fn ccp_blip25_free(h: *mut Channel) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// 160 samples -> 11-byte frame. 0 on success, negative on error.
/// The frame describes the previous 20 ms (the codec's one-frame delay).
///
/// # Safety
/// `pcm` must point to 160 samples and `out` to 11 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ccp_blip25_encode(h: *mut Channel, pcm: *const i16, out: *mut u8) -> i32 {
    if h.is_null() || pcm.is_null() || out.is_null() {
        return -1;
    }
    let (c, pcm, out) = (&mut *h, slice::from_raw_parts(pcm, FRAME_SAMPLES), slice::from_raw_parts_mut(out, FRAME_BYTES));
    match catch_unwind(AssertUnwindSafe(|| c.encode(pcm))) {
        Ok(Some(bits)) if bits.len() == FRAME_BYTES => {
            out.copy_from_slice(&bits);
            0
        }
        Ok(_) => -2,
        Err(_) => -3,
    }
}

/// Which encoder the last encoded frame came from: 0 both agreed (or tone
/// detection is off), 1 stock, 2 tone detector.
///
/// # Safety
/// `h` must come from `ccp_blip25_new`.
#[no_mangle]
pub unsafe extern "C" fn ccp_blip25_last_choice(h: *const Channel) -> i32 {
    if h.is_null() {
        return -1;
    }
    (*h).last_choice
}

/// 11-byte frame -> 160 samples. 0 on success, negative on error.
///
/// # Safety
/// `frame` must point to 11 bytes and `out` to 160 writable samples.
#[no_mangle]
pub unsafe extern "C" fn ccp_blip25_decode(h: *mut Channel, frame: *const u8, out: *mut i16) -> i32 {
    if h.is_null() || frame.is_null() || out.is_null() {
        return -1;
    }
    let (c, frame, out) = (&mut *h, slice::from_raw_parts(frame, FRAME_BYTES), slice::from_raw_parts_mut(out, FRAME_SAMPLES));
    match catch_unwind(AssertUnwindSafe(|| c.stock.decode_bits(frame))) {
        Ok(Ok(pcm)) if pcm.len() == FRAME_SAMPLES => {
            out.copy_from_slice(&pcm);
            0
        }
        Ok(_) => -2,
        Err(_) => -3,
    }
}
