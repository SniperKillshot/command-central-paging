# Quik-Call II paging over P25: what it took

Notes for an agent working on a P25 console or gateway that has to originate
Motorola Quik-Call II (QCII) two-tone pages over a P25 Phase 1 (IMBE) path.
This describes what went wrong in our build and what fixed it. File paths
refer to our codebase and are there for orientation only.

## TL;DR

1. **The IMBE encoder is what breaks pages, not the tone generator.** A
   stock IMBE analysis encoder codes a steady pure sine above roughly
   405-450 Hz as noise or buzz. Most QCII tones (the table runs from about
   280 Hz to 2.5 kHz) sit above that line, so pages came out mangled.
2. **Turn on the encoder's in-band tone detector.** We use the open
   `blip25-vocoder` Rust crate (MIT). Version 1.0.0 contains the reference
   engine's signalling-tone detector, but its encoder always passes
   `flags = 0`, so the detector never runs. Passing the tone-detect enable
   bit `0x1000` fixes tones above ~405 Hz.
3. **Don't use the detector alone.** It snaps tones to a coarse pitch raster
   and is up to ~7% off below 405 Hz, where the stock encoder is within
   ~1.7%. We run both encoders and pick per frame (details below).
4. **Hand encoder state over when switching.** IMBE codes spectral
   amplitudes relative to the previous frame. If you switch encoders without
   copying state, you get up to +7 dB level overshoot and clicks.
5. **Enable it on every encoder that puts audio on the air.** We fixed the
   console's main TX path first and missed the FNE (DVMFNE peer) TX encoder.
   Pages over FNE talkgroups stayed broken until that one was switched too.
6. **Encode once, never transcode P25 to P25.** Every IMBE decode and
   re-encode costs quality, and the re-encode is exactly where tones get
   mangled. Pass the IMBE vectors (U0-U7) through untouched between P25
   endpoints.

## 1. Why pages break on IMBE

IMBE is a model-based vocoder: a fundamental pitch (f0), voiced/unvoiced
decisions per band, and spectral amplitudes. It assumes speech. Here is what
we measured with blip25's stock encoder:

- From about 450 Hz up through at least 3.2 kHz, the analysis codes a steady
  pure tone as noise or buzz. The band holding the tone comes out unvoiced,
  or voiced as one harmonic of a low (~80 Hz) pitch with the other harmonics
  buzzing.
- One example was a two-tone page whose B tone was 660 Hz. It decoded with
  5% line purity (energy within ±5 Hz of the strongest line) and a level
  jumping ~4 dB from frame to frame. A pager decoder won't reliably accept
  that.
- Tones up to ~440 Hz code as a clean voiced line after a lead-in, but they
  fail too when the encoder starts directly on the tone.
- The stock analysis can't code a pitch above 405.1 Hz at all. From there up
  to ~431 Hz it pins the line at 405.1 Hz, which is 4-6% off.

Speech coding is fine. The problem is specific to steady single tones, which
is exactly what QCII is.

## 2. The encoder fix: two encoders and a per-frame choice

The real fix is in the vocoder binding (`packages/blip25-vocoder-py`, which
holds a vendored copy of blip25-vocoder 1.0.0 with two local patches).

### Patch 1: opt-in tone detection

- This is a per-encoder switch that passes the reference engine's tone-detect
  master enable (`0x1000`) in the `flags` argument of `enc_frame`. It defaults
  to off.
- No engine code changed. The engine already gates its whole tone path on
  that bit; upstream just hardcodes `0`.
- With the bit on, the 660 Hz tone from section 1 comes out 100% voiced
  (f0 329.9 Hz, harmonic 2 = 659.8 Hz) with 98-100% purity.
- Speech encodes bit-identically with or without the bit. Four recorded P25
  calls (2604 frames) never tripped the detector.

### The catch: the raster

Tone frames snap to the detector's raster: 31.25 Hz steps, then the IMBE f0
grid. Below ~405 Hz that's worse than the stock encoder:

- With the detector alone we measured 266 Hz at +6.8%, 296.5 Hz at -4.2%,
  330.5 Hz at +4.1% and 358.6 Hz at -4.1%.
- Adjacent QCII tones are only ~2.7% apart, so those errors matter.

### The hybrid (`ImbeVocoder(tone_detection=True)`)

The binding runs the stock encoder and a twin with the detector on, both fed
the same audio. For each frame:

- **The two frames agree** (speech or silence, so the detector didn't fire):
  emit it.
- **They differ and the twin flagged a tone:** emit the tone frame, unless the
  stock frame codes a clean line that is closer to the input than the tone
  frame's line.
  - "Clean" means the strongest harmonic is voiced and holds at least 90% of
    the harmonic energy.
  - "Closer" means more Hann-windowed power at that frequency in the input
    frame the two frames describe.
- **Otherwise:** emit the stock frame.

### Patch 2: state hand-over

After each frame, the encoder whose frame went out copies its cross-frame
state (engine state words plus the parameter-readback dequantiser state)
into the other encoder. Both then predict from what the decoder actually
received.

Without the hand-over, switching overshot by up to +7 dB. A 389 Hz tone's
onset decoded at 5400 rms against a steady 2400. With it, the switch can't be
measured: the decoded peak level and the largest sample step stay within
0.13 dB of the worse encoder alone. No hysteresis is needed.

### Measured accuracy

Tones at -21 dBFS after 0.2 s of silence, decoded by blip25. Each cell gives
the frequency error of the strongest line, then its purity.

| Input | Stock | Detector alone | Hybrid |
|---|---|---|---|
| 17 QCII tones, 282.8-433.7 Hz | -4.3..+1.3% / 0.95-1.00 | -4.2..+4.1% / 1.00 | -1.4..+3.3% / 0.95-1.00 |
| 64 QCII tones, 445.7-2468.2 Hz | -8.0..+7.4% / 0.04-0.56 | -3.2..+3.0% / 1.00 | same as detector |
| 250-404 Hz, 2 Hz steps | -1.5..+1.7% / 0.88-1.00 | -6.1..+6.8% / 1.00 | same as stock |
| 500-3300 Hz, 25 Hz steps | -8.0..+6.0% / 0.04-0.55 | -2.3..+2.9% / 1.00 | -2.3..+2.9% / 0.40-1.00 |

The hybrid has two known weak spots:

- **422 Hz.** This sits halfway between two raster lines, so the detector
  alternates between 407.6 and 435.4 Hz (+2.4%, purity 0.52).
- **3175 Hz.** The stock frames flip between a clean line and noise, so the
  output alternates between stock and tone frames (-0.4%, purity 0.40).

Cost: two encoders, about 0.1 ms per frame instead of 0.05 ms.

## 3. Where to enable it

- Use `tone_detection=True` on every encoder that puts audio on the air. In
  our build that means the console's TX pipeline, the FNE peer TX endpoint,
  and the simulator's tone-call encoder.
- Leave decode-only (RX) instances alone; the decoder renders tone frames
  fine.
- Keep the library default off, so a bare `ImbeVocoder()` still matches
  upstream bit for bit.

The mistake to avoid: when you add a new transport, grep for every encoder
constructor (`ImbeVocoder(` in our code). We shipped QCII working on the DFSI
path and broken on FNE because the FNE TX endpoint had its own
`ImbeVocoder()`. The fix was one line (`endpoints/fne_tx.py`) plus a test
asserting the flag.

## 4. Generating the page

This lives in `console_core/endpoints/pager_tones.py` and is pure code with
no I/O.

- **Format:** 8 kHz, 16-bit, 20 ms frames (160 samples), the same as the
  P25 voice path.
- **Timing:**
  - Two-tone: tone A for 1000 ms, then tone B for 3000 ms, back to back with
    no gap.
  - Single-tone (no B configured): tone A for 8000 ms.
- **Phase:** continuous across the A to B switch, so the switch is a plain
  frequency change with no click.
- **Ramps:** a 5 ms raised-cosine ramp on the start of the first tone and the
  end of the last tone only.
- **Lead-in:** silence before tone A, so the target, any repeater and the
  digital path are keyed and synced before the tone starts. It defaults to
  500 ms and can be configured up to 5000 ms; some systems needed more than
  the original limit.
- **Level:** configured per pager, default -10 dBFS peak.
  - The page goes through the target endpoint's normal TX pipeline, and that
    pipeline applies the target's own TX trim (`tx_level_db`).
  - So the generator compensates by synthesizing at
    `tone_level_db - target_tx_level_db`, capped at 0 dBFS.
  - Pages do not go through the speech AGC that levels the operator mic to
    the P25 nominal speech level (-28 dBFS RMS, from TIA-102.BABA §4). Tones
    must not be "levelled" like speech.
- **Caching:** pages are cached per (tone A, tone B, target TX level) and
  rebuilt when one of those changes.

## 5. Integration details that bit us

- **No extra signalling around a page.** One of our transports (an analog
  AllStarLink path) automatically added an MDC1200 PTT-ID burst at key-up
  and key-down. Around a page, that burst lands right before tone A and can
  upset the decoder. When the pager holds the channel, the page now keys the
  target without the PTT ID; normal operator PTT is unchanged. Check the same
  thing for any automatic ANI or talk-permit tones.
- **Arbitration:** a page keys the target resource through the same PTT
  arbiter as an operator, with its own holder type. It fails cleanly if the
  resource is busy or the operator lacks TX rights, and key-up failures go
  back to the operator who started the page.
- **Local sidetone:** each tone frame is also sent to the operators' browsers
  as audio on the target resource, so it plays through that resource's card
  at its volume and mute. This is local monitoring only; the on-air level is
  unaffected.
- **Text-to-speech paging (AutoDispatcher)** reuses the same tone frames. The
  sequence is lead-in, optional preamble, 1 s gap, tones, 500 ms gap,
  message, then repeats.

## 6. Status and open gaps

- **Field result:** live-tested a 349 / 746.8 Hz page on a DVM (FNE)
  talkgroup and it worked after the FNE encoder fix. That was one tone pair
  on one system, not a sweep against a real decoder's acceptance window.
  Errors of up to ~3.5% remain at the worst spots (see the accuracy table).
  Verify against your pagers before relying on it.
- **AMBE+2 (half rate):** P25 Phase 2 and DMR have no equivalent hybrid. The
  binding's half-rate `AmbeVocoder` has no tone-detection option, so pages
  over AMBE paths likely sound as bad as stock IMBE did. Fixing that needs
  half-rate tone-frame work in the vocoder.
- **Licensing:** blip25 and our two patches are MIT. Upstream notes that the
  half-rate path reads on US8359197 (active until 2028-05-20); the full-rate
  patents it lists have expired. Read upstream's `PATENT_NOTICE.md` before
  distributing.

## Checklist for your build

- [ ] Find every IMBE encoder that puts audio on the air, and enable tone
      detection with the stock/tone choice on each one.
- [ ] If you switch between encoders, copy state across, or expect level
      jumps and clicks.
- [ ] Pass vectors through on P25-to-P25 paths. Never decode and re-encode a
      page.
- [ ] Generate continuous-phase tones with fixed QCII timing (1 s A + 3 s B,
      or 8 s single) and ramp only the outer edges.
- [ ] Use a configurable lead-in (500 ms default, allow up to 5 s).
- [ ] Compensate the tone level for any TX trim downstream, and keep pages
      out of speech AGC.
- [ ] Suppress MDC/ANI and talk-permit bursts while a page holds the channel.
- [ ] Test a low tone (< 405 Hz), a high tone (> 450 Hz) and a pair spanning
      both, on every transport, and decode them with a real pager.
