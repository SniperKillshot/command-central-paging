//! The codec core — encoder and decoder.
//!
//! Allocation-free, dependency-free, `no_std`-clean, and bit-exact with the
//! reference software vocoder ([`golden`] replays 800 captured decode frames
//! and 2568 captured encode frames — both rates, synthetic programme and real
//! speech — and demands exact equality).
//!
//! Both rates are one fixed-point engine selected by a mode flag:
//! [`Mode::HalfRate`] is half-rate, 49 payload bits per 20 ms frame;
//! [`Mode::FullRate`] is 88 bits per frame. A frame is always 160 samples of
//! 8 kHz PCM, processed as two 10 ms half-frames.
//!
//! This module is private. It is the audio half of the crate; everything
//! between these payload bits and the air interface — framing, FEC,
//! de-interleaving, bit priority, erasure marking — is the wire layers, and
//! [`crate::vocoder::Vocoder`] is the seam a consumer uses.
//!
//! ```ignore
//! let mut d = Decoder::new(Mode::HalfRate);
//! let mut pcm = [0i16; 160];
//! d.decode(&bits, &mut pcm);      // bits: one payload bit per i16
//!
//! let mut e = Encoder::new(Mode::HalfRate);
//! let mut bits = [0i16; 49];
//! e.encode(&pcm, &mut bits);      // one frame of encoder delay
//! ```
//!
//! # `no_std` and the standalone build
//!
//! Nothing here touches `std`, and `conformance/no-std-guard` compiles this
//! module as the root of its own `#![no_std]` crate so that stays true: the
//! surrounding crate is `std`, so only that separate build can hold the line.
//! Keep `core::` paths explicit and never reach for `alloc`.
//!
//! # Completeness over reachability
//!
//! This is a whole-codec port, not the subset the wire layers happen to call.
//! Entry points the reference exposes are kept even where nothing in this
//! crate reaches them, so `dead_code` is allowed module-wide rather than
//! pruning the port down to current callers.

#![allow(dead_code)]

pub mod tables;

mod decode;
mod dsp;
mod encode;

// Host-only harnesses: both use `std` for file and process I/O. `no_std_guard`
// is set by `conformance/no-std-guard`, which mounts this file as the root of a
// `#![no_std]` crate — a test harness there would otherwise drag these in.
#[cfg(all(test, not(no_std_guard)))]
mod golden;
#[cfg(all(test, not(no_std_guard)))]
mod stack;

use decode::{dec_init, dec_synth, DecTables, FrameTables};
use dsp::dec_ctrl;
use encode::{
    enc_frame, enc_init, front_end, spectrum_harmonic_log_mags, Bases, FrontEndTables, PackTables,
    ToneTables, ENC_STATE_WORDS,
};

/// Binary-point position of the log-magnitudes [`SpectrumProbe`] reports.
///
/// Reached by name from doc links and tests rather than by a call, so the
/// unused-import lint has to be told this re-export is load-bearing.
#[allow(unused_imports)]
pub use encode::LOG_MAG_Q_SHIFT;

/// Words of decoder state (`0x800` bytes). The highest field the closure
/// touches is the signalling tracker at `+0x7c4..+0x7e8`.
const DEC_STATE_WORDS: usize = 0x400;

/// Samples of 8 kHz PCM one frame renders.
pub const FRAME_SAMPLES: usize = 160;

/// Samples per DSP call — a frame is two of them.
const HALF: usize = 80;

/// Words of payload-bit scratch handed to the synthesiser. The engine reads
/// only the payload but is free to scribble past it, so the buffer is sized as
/// the reference wrapper sizes it.
const BITS_WORDS: usize = 256;

/// Which of the two codecs a [`Decoder`] runs.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Full rate — P25 Phase 1 / FDMA, 88 payload bits per frame. TIA-102.BABA-A
    /// calls this the IMBE vocoder.
    FullRate = 0,
    /// Half rate — P25 Phase 2 / TDMA, 49 payload bits per frame. BABA-A calls
    /// this the Half-Rate Vocoder; it is the operating point the reference
    /// indexes as rate 33.
    HalfRate = 1,
}

impl Mode {
    /// Payload bits one 20 ms frame carries.
    #[inline]
    pub const fn payload_bits(self) -> usize {
        match self {
            Mode::FullRate => 88,
            Mode::HalfRate => 49,
        }
    }
}

/// What the frame decoder made of the payload it was given.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Frame {
    /// Speech. Carries the raw disposition word.
    Speech(i16),
    /// Comfort noise / silence descriptor.
    Silence,
    /// In-band signalling tone.
    Tone(i16),
    /// Unusable payload: the previous frame was held or faded.
    Repeat(i16),
    /// Anything else the engine reported.
    Other(i16),
}

/// Per-frame status from the transport layer, handed to the decoder alongside
/// the payload.
///
/// The decoder **conceals with these** — they are not advisory. A frame the FEC
/// layer had to work hard on decodes differently from a clean one, and passing
/// zeros unconditionally disables concealment exactly when it is needed.
///
/// Only two of the reference control block's 26 words do anything, and their
/// behaviour is measured from the reference implementation rather than inferred:
///
/// * `error_rate` -> word `+0x24`. Live over **0..=15**, monotonically
///   increasing concealment. **Values of 16 and above silently revert to
///   no-concealment**, so a raw error metric must be clamped, never passed
///   through — an uncorrectable frame reported as 255 would otherwise turn
///   concealment off on the worst frame in the stream. [`FrameStatus::new`]
///   clamps for you.
/// * `lost` / `mute` -> word `+0x32`, the *forcing* word. Bit 2 (value 4)
///   requests a **repeat**: the payload is ignored and the previous frame is
///   held, then faded to silence over about four frames. Bit 3 (value 8)
///   requests a **mute**: the payload is ignored and the output drops to
///   silence within one frame. Both ignore the payload; they differ in how
///   they get to silence, and bit 3 wins if both are set.
///
/// Word `+0x22` — the one an "E0/E1 pair" reading would suggest is the first
/// error count — is **inert**: no value changes a single output sample in
/// either codec, swept across its full range.
///
/// Only two of the control block's 26 words are known to matter, so this type
/// is `#[non_exhaustive]`: build it through [`FrameStatus::new`],
/// [`FrameStatus::CLEAN`] or [`FrameStatus::LOST`] and adjust with
/// [`FrameStatus::with_lost`] / [`FrameStatus::with_mute`], leaving room for a
/// third word to be identified without breaking callers.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
#[non_exhaustive]
pub struct FrameStatus {
    /// Severity of the errors the FEC layer corrected, 0 (clean) to 15 (worst).
    pub error_rate: u8,
    /// The frame was lost entirely: hold the previous one and fade out over
    /// roughly four frames. This is the usual concealment for a dropped frame.
    pub lost: bool,
    /// Force silence within one frame. Takes precedence over `lost`.
    pub mute: bool,
}

impl FrameStatus {
    /// A clean frame: no concealment, nothing forced.
    pub const CLEAN: FrameStatus = FrameStatus {
        error_rate: 0,
        lost: false,
        mute: false,
    };

    /// A frame the transport could not recover: hold and fade.
    pub const LOST: FrameStatus = FrameStatus {
        error_rate: 15,
        lost: true,
        mute: false,
    };

    /// Build from a raw error metric of any range, clamped into the live band.
    ///
    /// Use this rather than assigning `error_rate` directly unless you have
    /// already bounded the value — see the type documentation for why 16 and
    /// above are dangerous.
    #[inline]
    pub fn new(raw_error_rate: u32, mute: bool) -> FrameStatus {
        FrameStatus {
            error_rate: raw_error_rate.min(15) as u8,
            lost: false,
            mute,
        }
    }

    /// Mark the frame lost — hold the previous one and fade — keeping the
    /// severity already set. [`FrameStatus::LOST`] is this on top of the
    /// maximum severity, which is what a transport that lost a frame outright
    /// should send.
    #[inline]
    #[must_use]
    pub fn with_lost(mut self, lost: bool) -> FrameStatus {
        self.lost = lost;
        self
    }

    /// Force silence within one frame, keeping every other field. Takes
    /// precedence over `lost` at the engine.
    #[inline]
    #[must_use]
    pub fn with_mute(mut self, mute: bool) -> FrameStatus {
        self.mute = mute;
        self
    }

    /// The 26-word control block the engine expects.
    #[inline]
    fn to_ctrl(self) -> [i16; 26] {
        let mut c = [0i16; 26];
        c[0x24 / 2] = self.error_rate.min(15) as i16;
        // Forcing word: bit 2 repeat, bit 3 mute. Writing 4 for "mute" would
        // request a repeat instead — a fade over four frames, not silence.
        let mut force = 0i16;
        if self.lost {
            force |= 4;
        }
        if self.mute {
            force |= 8;
        }
        c[0x32 / 2] = force;
        c
    }
}

impl Frame {
    /// The engine reports a **bit word**, not an ordinal: `0x8000` tone,
    /// `0x0020` repeat, `0x0002` speech, and zero for silence. Tone sets the
    /// speech bit too, so it has to be tested first.
    ///
    /// The raw word is preserved in [`Frame::code`] because the encoder uses
    /// the same field to report things this enum does not name — `0x0400`,
    /// for instance, means its activity timer re-armed.
    #[inline]
    fn from_code(c: i16) -> Frame {
        let u = c as u16;
        if u & 0x8000 != 0 {
            Frame::Tone(c)
        } else if u & 0x0020 != 0 {
            Frame::Repeat(c)
        } else if u & 0x0002 != 0 {
            Frame::Speech(c)
        } else if u == 0 {
            Frame::Silence
        } else {
            Frame::Other(c)
        }
    }

    /// The raw disposition word, or 0 for silence.
    #[inline]
    pub fn code(self) -> i16 {
        match self {
            Frame::Speech(c) | Frame::Tone(c) | Frame::Repeat(c) | Frame::Other(c) => c,
            Frame::Silence => 0,
        }
    }
}

/// Every constant table the decoder reaches, wired to the `const` tables.
///
/// Materialised once per call rather than stored: it is a block of `&'static`
/// slices to `.rodata`, so building it is free after inlining and it keeps
/// [`Decoder`] to its state array.
#[inline(always)]
fn dec_tables() -> DecTables<'static> {
    DecTables {
        frame: FrameTables {
            submode: &tables::SUBMODE,
            voicing_patterns: &tables::VOICING_PATTERNS,
            gain32: &tables::GAIN32,
            block_lens: &tables::BLOCK_LENS,
            prba9: &tables::PRBA9,
            prba7: &tables::PRBA7,
            coef_books: &tables::HOC_A,
            nibbles: &tables::NIBBLES,
            gain64: &tables::GAIN64,
            hoc_weight: &tables::HOC_WEIGHT,
            hoc_loading: &tables::HOC_LOADING,
            codewords: &tables::CODEWORDS,
            blob: &tables::BLOB,
        },
        hilbert: &tables::HILBERT,
        sqrt_tbl: &tables::SQRT80,
    }
}

/// A vocoder decoder instance.
///
/// Roughly 2.6 KB, all inline — no allocation, no interior pointers, so it can
/// live on the stack, in a static, or anywhere else.
pub struct Decoder {
    state: [i16; DEC_STATE_WORDS],
    bits: [i16; BITS_WORDS],
    mode: Mode,
}

impl Decoder {
    /// Power-on reset for `mode`.
    pub fn new(mode: Mode) -> Decoder {
        let mut d = Decoder {
            state: [0; DEC_STATE_WORDS],
            bits: [0; BITS_WORDS],
            mode,
        };
        dec_init(&mut d.state, mode as i16);
        d
    }

    /// Which codec this instance runs.
    #[inline]
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Payload bits one frame consumes: 49 (half rate) or 88 (full rate).
    #[inline]
    pub fn payload_bits(&self) -> usize {
        self.mode.payload_bits()
    }

    /// Return to the power-on state, discarding all cross-frame history.
    pub fn reset(&mut self) {
        self.state = [0; DEC_STATE_WORDS];
        dec_init(&mut self.state, self.mode as i16);
    }

    /// Decode one 20 ms frame.
    ///
    /// `payload` carries one payload bit per word (value 0 or 1), MSB of the
    /// channel frame first; it must hold at least [`Decoder::payload_bits`]
    /// entries. `pcm` receives exactly [`FRAME_SAMPLES`] samples.
    ///
    /// Decoding is stateful: consecutive frames must be fed in order, and the
    /// output of a frame depends on every frame before it.
    pub fn decode(&mut self, payload: &[i16], pcm: &mut [i16]) -> Frame {
        self.decode_with_status(payload, FrameStatus::CLEAN, pcm)
    }

    /// Decode one frame, telling the codec how badly the transport's FEC layer
    /// had to work on it.
    ///
    /// This is the form to use on a real link. [`Decoder::decode`] is this with
    /// [`FrameStatus::CLEAN`], which is correct only for frames that genuinely
    /// arrived clean — it disables concealment.
    pub fn decode_with_status(
        &mut self,
        payload: &[i16],
        status: FrameStatus,
        pcm: &mut [i16],
    ) -> Frame {
        let nb = self.payload_bits();
        assert!(payload.len() >= nb, "payload too short");
        assert!(pcm.len() >= FRAME_SAMPLES, "pcm buffer too short");

        self.bits.fill(0);
        self.bits[..nb].copy_from_slice(&payload[..nb]);

        dec_ctrl(&mut self.state, &status.to_ctrl());

        let t = dec_tables();
        let (a, b) = pcm[..FRAME_SAMPLES].split_at_mut(HALF);
        let disposition = dec_synth(a, HALF as i16, &mut self.bits, 0, 0, 0, &mut self.state, &t);
        dec_synth(b, HALF as i16, &mut self.bits, 0, 0, 1, &mut self.state, &t);
        Frame::from_code(disposition)
    }

    /// Decode one frame from packed channel bytes, MSB first.
    ///
    /// `frame` must hold at least `ceil(payload_bits / 8)` bytes.
    pub fn decode_packed(&mut self, frame: &[u8], pcm: &mut [i16]) -> Frame {
        let nb = self.payload_bits();
        assert!(frame.len() * 8 >= nb, "frame too short");
        let mut bits = [0i16; 88];
        for (i, b) in bits[..nb].iter_mut().enumerate() {
            *b = ((frame[i >> 3] >> (7 - (i & 7))) & 1) as i16;
        }
        self.decode(&bits[..nb], pcm)
    }
}

// ===========================================================================
// Encoder
// ===========================================================================

/// Every constant table the encoder's analysis front end reaches.
///
/// Like [`dec_tables`] this is a block of `&'static` slices into `.rodata`,
/// built per call and free after inlining.
#[inline(always)]
fn enc_front_tables() -> FrontEndTables<'static> {
    const R: &[i16; 48] = &tables::TONE_TABLES;
    FrontEndTables {
        hamming: &tables::HAMMING100,
        wide_win: &tables::WIDE_RAMP,
        taper: &tables::TAPER32,
        interp: &tables::INTERP7X16,
        subband_win: &tables::SUBBAND_WINDOW,
        cos: &tables::COS512,
        fb_win: &tables::FB_WINDOW,
        fb_tbl: &tables::DCT160,
        tone: ToneTables {
            class: &R[0..16],
            lin_lo_a: &R[16..20],
            bin_lo_a: &R[20..24],
            lin_lo_b: &R[24..28],
            bin_lo_b: &R[28..32],
            lin_hi_a: &R[32..36],
            bin_hi_a: &R[36..40],
            lin_hi_b: &R[40..44],
            bin_hi_b: &R[44..48],
        },
    }
}

/// Every constant table the payload packer reaches.
#[inline(always)]
fn enc_pack_tables() -> PackTables<'static> {
    PackTables {
        blob: &tables::BLOB,
        partitions: &tables::PARTITIONS,
        voicing_book: &tables::VOICING_PATTERNS,
        nibbles: &tables::NIBBLES,
        gain64: &tables::GAIN64,
        gain32: &tables::GAIN32,
        hoc_weight: &tables::HOC_WEIGHT,
        hoc_loading: &tables::HOC_LOADING,
        block_lens: &tables::BLOCK_LENS,
        prba9: &tables::PRBA9,
        prba7: &tables::PRBA7,
        coef_books: &tables::HOC_A,
    }
}

/// Words of the frame tail's two per-half-frame parameter blocks. They are raw
/// stack in the reference implementation and are zeroed here every frame.
const BLK_WORDS: usize = 0x48;

/// The `unit` argument the shipped wrappers pass to `enc_frame`. It reaches
/// only the packer's sink, which no shipped path enables.
const ENC_UNIT: i16 = 0x2000;

/// A vocoder encoder instance.
///
/// Roughly 6.4 KB, all inline — no allocation, no interior pointers.
///
/// Encoding carries **one frame of delay**: analysis runs on each 10 ms
/// half-frame and the payload is emitted on the second, over an analysis ring
/// that by then holds the previous 20 ms of audio. The first frame handed to a
/// fresh instance therefore describes silence, not that frame's audio.
pub struct Encoder {
    state: [i16; ENC_STATE_WORDS],
    bits: [i16; BITS_WORDS],
    mode: Mode,
    // CCP patch: the `flags` word handed to `enc_frame`, which upstream
    // hardcodes to 0. 0 by default, so output is unchanged unless set.
    flags: u32,
}

/// CCP patch: `flags` bit for the reference engine's in-band tone detector
/// (the master enable tested in `tone_class_half_frame`).
pub const ENC_FLAG_TONE_DETECT: u32 = 0x1000;

impl Encoder {
    /// Power-on reset for `mode`.
    pub fn new(mode: Mode) -> Encoder {
        let mut e = Encoder {
            state: [0; ENC_STATE_WORDS],
            bits: [0; BITS_WORDS],
            mode,
            flags: 0,
        };
        enc_init(&mut e.state, mode as i16, 1);
        e
    }

    /// Which codec this instance runs.
    #[inline]
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Payload bits one frame produces: 49 (half rate) or 88 (full rate).
    #[inline]
    pub fn payload_bits(&self) -> usize {
        self.mode.payload_bits()
    }

    /// Return to the power-on state, discarding all cross-frame history.
    pub fn reset(&mut self) {
        self.state = [0; ENC_STATE_WORDS];
        enc_init(&mut self.state, self.mode as i16, 1);
    }

    /// CCP patch: set the `flags` word passed to every `enc_frame` call
    /// (configuration, so it survives [`Self::reset`]).
    pub fn set_flags(&mut self, flags: u32) {
        self.flags = flags;
    }

    /// CCP patch: the current `flags` word.
    pub fn flags(&self) -> u32 {
        self.flags
    }

    /// CCP patch: take over another encoder's cross-frame state, so this one
    /// continues from what that one emitted. Both must run the same mode;
    /// this encoder keeps its own `flags`.
    pub fn copy_state_from(&mut self, other: &Encoder) {
        debug_assert!(self.mode as i16 == other.mode as i16, "mode mismatch");
        self.state.copy_from_slice(&other.state);
    }

    /// Encode one 20 ms frame.
    ///
    /// `pcm` is exactly [`FRAME_SAMPLES`] samples of 8 kHz signed 16-bit audio.
    /// `payload` receives one payload bit per word (value 0 or 1), MSB of the
    /// channel frame first; it must hold at least [`Encoder::payload_bits`]
    /// entries.
    ///
    /// Encoding is stateful and delayed by one frame: consecutive frames must
    /// be fed in order.
    pub fn encode(&mut self, pcm: &[i16], payload: &mut [i16]) -> Frame {
        let nb = self.payload_bits();
        assert!(pcm.len() >= FRAME_SAMPLES, "pcm too short");
        assert!(payload.len() >= nb, "payload buffer too short");

        self.bits.fill(0);
        let fe = enc_front_tables();
        let pk = enc_pack_tables();
        let bases = Bases { state: 0, sink: 0 };

        // The frame tail's two parameter blocks are locals of the reference's
        // `0x1030ff60`; a fresh (zeroed) pair every frame is what the scrubbed
        // reference produces.
        let mut blk0 = [0i16; BLK_WORDS];
        let mut blk1 = [0i16; BLK_WORDS];

        for half in 0..2i16 {
            let at = HALF * half as usize;
            let code = enc_frame(
                &mut self.bits,
                0,
                &pcm[at..at + HALF],
                HALF as i16,
                self.flags, // CCP patch: upstream passes 0
                half,
                ENC_UNIT,
                &mut self.state,
                &mut blk0,
                &mut blk1,
                bases,
                &fe,
                &pk,
            )
            .0;
            if half == 1 {
                payload[..nb].copy_from_slice(&self.bits[..nb]);
                return Frame::from_code(code as i16);
            }
        }
        unreachable!()
    }

    /// Encode one frame into packed channel bytes, MSB first.
    ///
    /// `frame` must hold at least `ceil(payload_bits / 8)` bytes; any bits past
    /// the payload in the last byte are cleared.
    pub fn encode_packed(&mut self, pcm: &[i16], frame: &mut [u8]) -> Frame {
        let nb = self.payload_bits();
        let nbytes = nb.div_ceil(8);
        assert!(frame.len() >= nbytes, "frame buffer too short");
        let mut bits = [0i16; 88];
        let d = self.encode(pcm, &mut bits);
        frame[..nbytes].fill(0);
        for (i, &b) in bits[..nb].iter().enumerate() {
            if b & 1 != 0 {
                frame[i >> 3] |= 1 << (7 - (i & 7));
            }
        }
        d
    }
}

/// Read-only access to the encoder's amplitude-estimation stage.
///
/// The encoder answers one question per frame — *what are the harmonic
/// magnitudes at the fundamental I chose?* — and answers it only on the way to
/// a payload. A caller that already knows the fundamental it wants (a
/// parameter-domain rate converter bridging a tone frame, for instance) needs
/// the same measurement on **its** grid, with no pitch decision and no
/// quantisation in the way.
///
/// [`SpectrumProbe`] runs the analysis front end over PCM exactly as
/// [`Encoder`] does — the same DC blocker, the same 383-sample analysis ring,
/// the same 199-tap Hamming window and 256-point transform, the same sub-band
/// and tone stages — and then hands out the harmonic log magnitudes the packer
/// would have integrated out of the newest spectrum. Nothing is packed, so no
/// payload and no channel state exist here.
///
/// Magnitudes come back as base-2 logarithms at Q11: the linear magnitude of
/// word `w` is `2^(w / 2048)`, see [`LOG_MAG_Q_SHIFT`].
///
/// Analysis is delayed the same way the encoder's is — the window sits a fixed
/// distance behind the newest sample — so the ring has to be filled before a
/// measurement means anything. [`SpectrumProbe::PRIMING_HALF_FRAMES`] pushes
/// cover the window completely.
pub struct SpectrumProbe {
    state: [i16; ENC_STATE_WORDS],
    half: i16,
}

impl SpectrumProbe {
    /// Half-frame pushes after which the 199-tap analysis window holds nothing
    /// but pushed audio.
    ///
    /// The window is `ring[28..227]` of a 383-sample ring whose newest sample
    /// is at the top, so `k` pushes of [`SpectrumProbe::HALF_FRAME_SAMPLES`]
    /// leave it clean once `383 − 80k ≤ 28`, i.e. from `k = 5`. One further
    /// push settles the DC blocker's own transient out of the window.
    pub const PRIMING_HALF_FRAMES: usize = 6;

    /// Samples one [`SpectrumProbe::push_half_frame`] consumes.
    pub const HALF_FRAME_SAMPLES: usize = HALF;

    /// Harmonics [`SpectrumProbe::harmonic_log_mags`] will measure at most.
    pub const MAX_HARMONICS: usize = 56;

    /// Power-on reset for `mode`.
    pub fn new(mode: Mode) -> SpectrumProbe {
        let mut p = SpectrumProbe {
            state: [0; ENC_STATE_WORDS],
            half: 0,
        };
        enc_init(&mut p.state, mode as i16, 1);
        p
    }

    /// Push 10 ms of audio through the analysis front end.
    ///
    /// `pcm` is exactly [`SpectrumProbe::HALF_FRAME_SAMPLES`] samples of 8 kHz
    /// signed 16-bit audio. Half-frame parity alternates on its own, starting
    /// at the frame's first half, because the pitch and sub-band stages run on
    /// one parity only.
    pub fn push_half_frame(&mut self, pcm: &[i16]) {
        assert!(pcm.len() >= HALF, "pcm too short");
        front_end(
            &mut self.state,
            &pcm[..HALF],
            HALF as i16,
            0,
            self.half,
            &enc_front_tables(),
        );
        self.half ^= 1;
    }

    /// Harmonic log magnitudes of the newest spectrum at fundamental `step`.
    ///
    /// `step` is the harmonic spacing as a Q11 increment over the 129-bin
    /// spectrum — one bin is 31.25 Hz, so a fundamental of `ω₀` radians/sample
    /// is `round(ω₀ · 128/π · 2048)`. One word is written per element of `out`,
    /// whose length is the harmonic count and must not exceed
    /// [`SpectrumProbe::MAX_HARMONICS`].
    pub fn harmonic_log_mags(&self, step: i16, out: &mut [i16]) {
        assert!(step > 0, "step must be positive");
        assert!(
            !out.is_empty() && out.len() <= Self::MAX_HARMONICS,
            "1..=56 harmonics"
        );
        let n = out.len() as i16;
        spectrum_harmonic_log_mags(out, &self.state, step, n, 0);
    }
}
