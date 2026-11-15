"""The blip25 vocoder (src/p25/blip25.py) and how it fits the rest.

Skipped when it hasn't been built (python tools/build_native.py). Its
reference-exactness is blip25-vocoder's own claim, tested in that project
against captures from the reference vocoder; these tests check that it is
wired in correctly: same 11-byte frame layout as DVM and the independent
mbelib decoder, steady output, sane levels.
"""
import hashlib
import os
import re
import unittest

import numpy as np

from src.audio import dsp
from src.p25 import blip25, imbe, mbe_decoder
from src.p25.voice import encode_pcm, pcm_to_frames

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORCED_PYTHON = os.environ.get('CCP_PURE_PYTHON_IMBE', '').strip() not in ('', '0')


def speechlike(seconds, f0=120.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(8000 * seconds)) / 8000
    pitch = f0 * (1 + 0.15 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(pitch) / 8000
    x = sum(np.sin(k * phase) * np.exp(-((k * pitch - 700) / 600) ** 2) for k in range(1, 30))
    env = np.clip(np.sin(2 * np.pi * 2.5 * t), 0, None) ** 0.5
    x = x * env + 0.05 * rng.standard_normal(t.size)
    return dsp.to_pcm16(0.6 * x / np.max(np.abs(x)))


def integer_speech(n):
    """Buzz with a moving pitch, a vowel-like decaying ring and noise, made
    with integer arithmetic only, so it is the same on every machine."""
    out, seed, phase, ring = [], 12345, 0, 0
    for i in range(n):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        period = 50 + (i // 400) % 20                  # 160-114 Hz
        phase += 1
        if phase >= period:
            phase, ring = 0, 9000
        ring = ring * 15 // 16
        wave = ring if (phase // 3) % 2 == 0 else -ring
        out.append(wave + (seed >> 20) - 1024)
    return np.array(out, dtype=np.int16)


def encode(cls, pcm):
    enc = cls()
    return [enc.encode(pcm[i:i + 160]) for i in range(0, pcm.size - 159, 160)]


def decode(cls, frames):
    dec = cls(seed=1)
    return np.concatenate([dec.decode(f) for f in frames]) * dec.gain


def spectral_match(ref, out):
    """Best correlation of 32 ms log spectra over codec delays up to 100 ms."""
    def spectra(x):
        n = (x.size // 256) * 256
        return np.abs(np.fft.rfft(x[:n].reshape(-1, 256) * np.hanning(256), axis=1))[:, 4:110]

    best = -1.0
    for lag in range(0, 800, 20):
        a, b = ref[:out.size - lag].astype(float), out[lag:]
        A, B = spectra(a), spectra(b[:a.size])
        loud = A.sum(1) > 0.1 * A.sum(1).max()
        best = max(best, np.corrcoef(np.log(A[loud] + 1).ravel(), np.log(B[loud] + 1).ravel())[0, 1])
    return best


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


@unittest.skipUnless(blip25.available(), 'blip25 not built')
class Blip25Tests(unittest.TestCase):
    def test_frames_are_what_the_independent_decoder_expects(self):
        """mbelib's 4400 decoder (written separately) must read every frame
        and hear the same speech: the 88-bit layout matches DVM's."""
        pcm = speechlike(3.0)
        frames = encode(blip25.Blip25Encoder, pcm)
        probe = mbe_decoder.PyIMBEDecoder(seed=1)
        for f in frames:
            self.assertEqual(len(f), 11)
            self.assertTrue(probe.decode_params(f))
            probe.prev.copy_from(probe.cur)
        self.assertGreater(spectral_match(pcm, decode(mbe_decoder.PyIMBEDecoder, frames)), 0.85)

    def test_decodes_other_encoders_frames(self):
        pcm = speechlike(3.0, 150, seed=2)
        frames = encode(imbe.PyIMBEEncoder, pcm)
        out = decode(blip25.Blip25Decoder, frames)
        self.assertGreater(spectral_match(pcm, out), 0.85)

    def test_round_trip_keeps_speech_and_level(self):
        pcm = speechlike(3.0, 110, seed=3)
        out = decode(blip25.Blip25Decoder, encode(blip25.Blip25Encoder, pcm))
        self.assertGreater(spectral_match(pcm, out), 0.85)
        self.assertLess(abs(20 * np.log10(rms(out) / rms(pcm))), 3.0)

    def test_repeatable_and_unchanged(self):
        """Same audio, same frames, every run and every platform. The hash
        pins blip25-vocoder 1.0.0's output; a different crate version that
        changes it should be a deliberate update."""
        pcm = integer_speech(16000)
        a = encode(blip25.Blip25Encoder, pcm)
        self.assertEqual(a, encode(blip25.Blip25Encoder, pcm))
        digest = hashlib.sha256(b''.join(a)).hexdigest()
        self.assertEqual(digest, GOLDEN_ENCODE, digest)
        out = decode(blip25.Blip25Decoder, a).astype(np.int16)
        np.testing.assert_array_equal(out, decode(blip25.Blip25Decoder, a).astype(np.int16))
        self.assertEqual(hashlib.sha256(out.tobytes()).hexdigest(), GOLDEN_DECODE)

    def test_silence_and_bad_frames(self):
        self.assertLess(np.max(np.abs(decode(blip25.Blip25Decoder, [imbe.silence_frame()] * 10))), 500)
        dec = blip25.Blip25Decoder()
        for frame in (bytes(11), bytes([0xFF] * 11), bytes(range(11))):
            self.assertEqual(dec.decode(frame).shape, (160,))
        with self.assertRaises(ValueError):
            dec.decode(bytes(10))
        with self.assertRaises(ValueError):
            blip25.Blip25Encoder().encode([0] * 159)

    def test_numpy_and_list_input_agree(self):
        pcm = speechlike(0.3)
        a, b = blip25.Blip25Encoder(), blip25.Blip25Encoder()
        for i in range(0, pcm.size - 159, 160):
            self.assertEqual(a.encode(pcm[i:i + 160]), b.encode(pcm[i:i + 160].tolist()))

    def test_speech_is_the_same_with_tone_detection(self):
        """The detector never fires on speech, so speech frames are upstream's."""
        for pcm in (speechlike(3.0), speechlike(3.0, 210, seed=5), integer_speech(24000)):
            on = blip25.Blip25Encoder(tone_detection=True)
            frames = []
            for i in range(0, pcm.size - 159, 160):
                frames.append(on.encode(pcm[i:i + 160]))
                self.assertEqual(on.last_choice, blip25.CHOICE_SAME)
            self.assertEqual(frames, encode(lambda: blip25.Blip25Encoder(tone_detection=False), pcm))


@unittest.skipUnless(blip25.available(), 'blip25 not built')
@unittest.skipIf(FORCED_PYTHON, 'Python vocoder forced')
class ToneTests(unittest.TestCase):
    """Steady tones through the page path (encode_pcm, the default encoder):
    a low one (< 405 Hz, where the stock analysis is closest), a high one
    (> 450 Hz, which the stock analysis codes as noise without the tone
    detector) and a two-tone pair spanning both."""

    AMP = 32767 * 10 ** (-10 / 20)      # -10 dBFS peak

    def tones(self, *spans):
        """Continuous-phase tone sequence after 0.5 s of silence:
        spans = (Hz, seconds), ..."""
        freqs = np.concatenate([np.zeros(4000)] + [np.full(int(8000 * s), f) for f, s in spans])
        x = np.sin(2 * np.pi * np.cumsum(freqs) / 8000) * self.AMP
        x[:4000] = 0
        return x.astype(np.int16)

    def through_codec(self, pcm, tone_detection=True):
        enc = blip25.Blip25Encoder(tone_detection)
        frames = [enc.encode(s) for s, _ in pcm_to_frames(pcm)]
        return decode(blip25.Blip25Decoder, frames)

    def line(self, out, start_s, seconds):
        seg = out[int(8000 * start_s):int(8000 * (start_s + seconds))]
        X = np.abs(np.fft.rfft(seg * np.hanning(seg.size), 8 * seg.size)) ** 2
        f = np.fft.rfftfreq(8 * seg.size, 1 / 8000)
        k = int(np.argmax(X))
        return f[k], X[np.abs(f - f[k]) <= 5].sum() / X.sum()

    def check(self, out, hz, start_s, seconds):
        peak, purity = self.line(out, start_s, seconds)
        self.assertLess(abs(peak - hz) / hz, 0.02, f'{hz} Hz came out at {peak:.1f} Hz')
        self.assertGreater(purity, 0.9, f'{hz} Hz purity {purity:.2f}')

    def test_low_tone(self):
        self.check(self.through_codec(self.tones((349.0, 2.0))), 349.0, 1.0, 1.0)

    def test_high_tones(self):
        for hz in (660.0, 746.8, 1000.0, 1500.0, 2468.2):
            self.check(self.through_codec(self.tones((hz, 2.0))), hz, 1.0, 1.0)

    def test_two_tone_pair_spanning_both(self):
        """Quik-Call II timing: tone A 1 s, then tone B 3 s, no gap."""
        out = self.through_codec(self.tones((349.0, 1.0), (746.8, 3.0)))
        self.check(out, 349.0, 0.75, 0.6)
        self.check(out, 746.8, 2.0, 2.0)

    def test_high_tone_needs_the_detector(self):
        """Guards the patch: without tone detection the same tone is mush."""
        _, purity = self.line(self.through_codec(self.tones((746.8, 2.0)), tone_detection=False), 1.0, 1.0)
        self.assertLess(purity, 0.6)

    def test_no_level_jump_at_tone_onset(self):
        """Switching encoders hands the state over: the onset is no louder
        than the steady tone."""
        for hz in (389.0, 445.7, 660.0, 1000.0):
            pcm = np.concatenate([speechlike(1.0) // 3, self.tones((hz, 2.0))[4000:]])
            out = decode(blip25.Blip25Decoder, encode_pcm(pcm))[8000:]
            blocks = np.sqrt(np.mean(out[:out.size // 160 * 160].reshape(-1, 160) ** 2, axis=1))
            steady = np.median(blocks[40:90])
            self.assertLess(20 * np.log10(blocks[2:40].max() / steady), 1.5, f'{hz} Hz')

    def test_every_encoder_that_transmits_detects_tones(self):
        """Every encoder in the app puts audio on the air, so each must be the
        default (tone-detecting) one: no encoder class built directly, no
        tone_detection=False outside blip25.py."""
        self.assertTrue(imbe.IMBEEncoder().tone_detection)
        built = re.compile(r'\b(\w*Encoder)\(([^)]*)\)')
        for folder, _, files in os.walk(os.path.join(ROOT, 'src')):
            for name in files:
                if not name.endswith('.py') or name in ('blip25.py', 'imbe.py'):
                    continue
                with open(os.path.join(folder, name), encoding='utf-8') as f:
                    for cls, args in built.findall(f.read()):
                        if cls.startswith('_'):
                            continue
                        self.assertEqual((cls, args), ('IMBEEncoder', ''), f'{name}: {cls}({args})')

    @unittest.skipIf(FORCED_PYTHON, 'Python vocoder forced')
    def test_it_is_the_default(self):
        self.assertIs(imbe.IMBEEncoder, blip25.Blip25Encoder)
        self.assertIs(mbe_decoder.IMBEDecoder, blip25.Blip25Decoder)


GOLDEN_ENCODE = '4e43d248c832dfa44f9c7ea1c0008c61657bbc8d97324b207b2778d75f711cb7'
GOLDEN_DECODE = '7e2300009e541429f793748c6feec34cc2da377b443554383c236266ccad243a'

if __name__ == '__main__':
    unittest.main()
