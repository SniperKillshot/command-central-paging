"""The compiled IMBE encoder must produce exactly the Python encoder's frames.

Skipped when it hasn't been built (python tools/build_native.py).
"""
import os
import unittest

import numpy as np

from src.audio import dsp
from src.audio.tones import BUILTIN, SoundLibrary
from src.p25 import imbe, imbe_native, mbe_decoder
from src.p25.voice import pcm_to_frames


def encode_all(cls, frames):
    enc = cls()
    out = []
    for samples, tones in frames:
        try:
            out.append(enc.encode(samples, tones))
        except imbe.VocoderError:
            out.append(b'error')
            enc = cls()
    return out


def speechlike(seconds, f0=120.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(8000 * seconds)) / 8000
    pitch = f0 * (1 + 0.15 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(pitch) / 8000
    x = sum(np.sin(k * phase) * np.exp(-((k * pitch - 700) / 600) ** 2) for k in range(1, 30))
    env = np.clip(np.sin(2 * np.pi * 2.5 * t), 0, None) ** 0.5
    x = x * env + 0.05 * rng.standard_normal(t.size)
    return x / np.max(np.abs(x))


@unittest.skipUnless(imbe_native.available(), 'compiled encoder not built')
class NativeMatchesPython(unittest.TestCase):
    def check(self, pcm, tones=None, name=''):
        frames = pcm_to_frames(dsp.to_pcm16(pcm) if pcm.dtype != np.int16 else pcm, tones)
        py = encode_all(imbe.PyIMBEEncoder, frames)
        nat = encode_all(imbe_native.NativeIMBEEncoder, frames)
        bad = [i for i, (a, b) in enumerate(zip(py, nat)) if a != b]
        self.assertEqual(bad, [], f'{name}: {len(bad)}/{len(py)} frames differ, first at {bad[:1]}')
        return py

    def test_speech_levels_and_pitches(self):
        for i, (f0, level) in enumerate(((90, 0.3), (140, 0.8), (220, 0.05), (300, 1.0))):
            self.check(speechlike(1.5, f0, seed=i) * level, name=f'speech f0={f0} level={level}')

    def test_noise_silence_clipping_and_dc(self):
        rng = np.random.default_rng(7)
        n = 8000
        self.check(np.zeros(n), name='silence')
        self.check(rng.standard_normal(n) * 0.3, name='noise')
        self.check(np.clip(rng.standard_normal(n) * 3, -1, 1), name='clipped noise')
        t = np.arange(n) / 8000
        self.check(np.sign(np.sin(2 * np.pi * 300 * t)), name='full-scale square')
        self.check(np.full(n, 0.5) + 0.1 * np.sin(2 * np.pi * 400 * t), name='dc offset')
        sweep = np.sin(2 * np.pi * np.cumsum(np.linspace(50, 3900, n)) / 8000)
        self.check(sweep, name='sweep')
        extremes = np.tile(np.array([32767, -32768], dtype=np.int16), n // 2)
        self.check(extremes, name='alternating extremes')

    def test_every_alert_tone_with_its_tone_track(self):
        lib = SoundLibrary('')
        for name in BUILTIN:
            audio, track = lib.render_with_tones(name)
            frames = self.check(audio, track, name=name)
            self.assertTrue(frames)

    def test_tones_between_speech(self):
        """Tone hints turn on and off mid-call (the native side keeps its own
        tone buffer only while tones are flowing)."""
        lib = SoundLibrary('')
        tone, track = lib.render_with_tones('triple_beep')
        speech = speechlike(1.0)
        gap = np.zeros(800)
        pcm = np.concatenate([speech, gap, tone, gap, speech, tone, speech])
        tr = np.concatenate([np.full((speech.size + gap.size, 2), np.nan), track,
                             np.full((gap.size + speech.size, 2), np.nan), track,
                             np.full((speech.size, 2), np.nan)])
        self.check(pcm * 0.7, tr, name='tones between speech')

    def test_numpy_and_list_input_agree(self):
        pcm = dsp.to_pcm16(speechlike(0.2))
        a, b = imbe_native.NativeIMBEEncoder(), imbe_native.NativeIMBEEncoder()
        for i in range(0, pcm.size - 159, 160):
            self.assertEqual(a.encode(pcm[i:i + 160]), b.encode(pcm[i:i + 160].tolist()))

    @unittest.skipIf(os.environ.get('CCP_PURE_PYTHON_IMBE', '').strip() not in ('', '0'), 'Python encoder forced')
    def test_it_is_the_default(self):
        self.assertIs(imbe.IMBEEncoder, imbe_native.NativeIMBEEncoder)


def frames_for(pcm, tones=None):
    frames = pcm_to_frames(dsp.to_pcm16(pcm), tones)
    enc = imbe.PyIMBEEncoder()
    return [enc.encode(s, t) for s, t in frames]


def decode(cls, frames, seed=1):
    dec = cls(seed=seed)
    return np.concatenate([dec.decode(f) for f in frames])


@unittest.skipUnless(imbe_native.available(), 'compiled decoder not built')
class NativeDecoderMatchesPython(unittest.TestCase):
    """The decoder works in floating point and draws random phases for
    unvoiced sound from its own generator, so: fully voiced audio (no
    randomness) must match to rounding, and speech must match in level."""

    def test_voiced_audio_matches_to_rounding(self):
        """Frames that are fully voiced, after a fully voiced frame, involve no
        random numbers (the tones' lead-in and padding are unvoiced)."""
        lib = SoundLibrary('')
        for name in ('alert_tone', 'eas_attention', 'wail', 'chime', 'hi_lo'):
            audio, track = lib.render_with_tones(name)
            frames = frames_for(audio, track)
            probe = mbe_decoder.PyIMBEDecoder(seed=1)
            voiced = []
            for f in frames:
                ok = probe.decode_params(f)
                voiced.append(ok and all(probe.cur.Vl[l] == 1 for l in range(1, probe.cur.L + 1)))
                probe.prev.copy_from(probe.cur)
            exact = [k for k in range(1, len(frames)) if voiced[k] and voiced[k - 1]]
            self.assertGreater(len(exact), len(frames) * 0.6, name)
            py = decode(mbe_decoder.PyIMBEDecoder, frames).reshape(-1, 160)[exact]
            nat = decode(imbe_native.NativeIMBEDecoder, frames).reshape(-1, 160)[exact]
            err = np.max(np.abs(py - nat))
            self.assertLess(err, 1e-9 * np.max(np.abs(py)), f'{name}: max difference {err}')

    def test_speech_matches_in_level(self):
        frames = frames_for(speechlike(3.0, 130, seed=3) * 0.6)
        py = decode(mbe_decoder.PyIMBEDecoder, frames).reshape(-1, 160)
        nat = decode(imbe_native.NativeIMBEDecoder, frames).reshape(-1, 160)
        rms = lambda a: np.sqrt(np.mean(a * a, axis=1)) + 1e-3
        db = 20 * np.log10(rms(nat) / rms(py))
        loud = rms(py) > 0.05 * rms(py).max()
        self.assertLess(np.median(np.abs(db[loud])), 0.5)
        self.assertGreater(np.mean(np.abs(db[loud]) < 2.0), 0.9)

    def test_silence_and_bad_frames(self):
        nat = imbe_native.NativeIMBEDecoder(seed=5)
        quiet = np.concatenate([nat.decode(imbe.silence_frame()) for _ in range(10)])
        self.assertLess(np.max(np.abs(quiet)) * 7, 400)
        garbage = bytes([0xFF] * 11)          # b0 > 207: repeated, then muted
        outs = [nat.decode(garbage) for _ in range(6)]
        self.assertTrue(np.all(outs[-1] == 0))

    def test_seeded_runs_repeat_exactly(self):
        frames = frames_for(speechlike(0.5))
        a = decode(imbe_native.NativeIMBEDecoder, frames, seed=9)
        b = decode(imbe_native.NativeIMBEDecoder, frames, seed=9)
        np.testing.assert_array_equal(a, b)

    @unittest.skipIf(os.environ.get('CCP_PURE_PYTHON_IMBE', '').strip() not in ('', '0'), 'Python decoder forced')
    def test_it_is_the_default(self):
        self.assertIs(mbe_decoder.IMBEDecoder, imbe_native.NativeIMBEDecoder)


if __name__ == '__main__':
    unittest.main()
