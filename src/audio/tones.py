"""Alert sounds: built-in generated tones plus any WAV files in the sounds folder.

Everything is produced at 8 kHz mono float in [-1, 1]. Built-in sounds also
return a tone track - the frequencies being generated at each sample - so
the P25 vocoder can code them cleanly (see IMBEEncoder._apply_tone_hint).
"""
import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np

from src.audio.dsp import RATE, read_wav, resample

logger = logging.getLogger(__name__)

LEVEL = 0.5   # -6 dBFS: loud, but leaves the vocoder headroom


def _snap(*freqs):
    """Shift tones (by < 0.4%) onto frequencies the P25 vocoder reproduces
    exactly; tones played together are snapped together."""
    from src.p25.imbe import codec_tone_frequencies
    return codec_tone_frequencies(freqs)


def _snap_track(freqs: np.ndarray) -> np.ndarray:
    """Per-sample snap for sweeps, via a 1 Hz lookup table."""
    whole = np.round(freqs).astype(int)
    table = {f: _snap(float(f))[0] for f in np.unique(whole).tolist()}
    return np.array([table[f] for f in whole.tolist()], dtype=np.float64)

Track = np.ndarray   # (n, 2) tone frequencies per sample in Hz, NaN = no tone


def _track(f1, f2=None, n=None) -> Track:
    f1 = np.asarray(f1, dtype=np.float64)
    if f1.ndim == 0:
        f1 = np.full(n, float(f1))
    tr = np.full((f1.size, 2), np.nan)
    tr[:, 0] = f1
    if f2 is not None:
        tr[:, 1] = f2
    return tr


def _fade(y: np.ndarray, ms: float = 8) -> np.ndarray:
    n = min(int(RATE * ms / 1000), y.size // 2)
    if n:
        ramp = np.linspace(0, 1, n)
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
    return y


def _from_freq_track(freqs: np.ndarray, gate: Optional[np.ndarray] = None):
    """Phase-continuous sine following a per-sample frequency track."""
    phase = 2 * np.pi * np.cumsum(freqs) / RATE
    y = np.sin(phase) * LEVEL
    tr = _track(freqs)
    if gate is not None:
        y = y * gate
        tr[gate <= 0] = np.nan
    return _fade(y), tr


def _steady(freq=1000.0, seconds=2.0):
    return _from_freq_track(np.full(int(RATE * seconds), _snap(freq)[0]))


def _dual(f1, f2, seconds):
    """Two tones at once, each at half level so the sum peaks at LEVEL."""
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    f1, f2 = _snap(f1, f2)
    y = (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)) * (LEVEL / 2)
    return _fade(y), _track(f1, f2, n)


def _alternating(f1, f2, period, seconds):
    t = np.arange(int(RATE * seconds)) / RATE
    f1, f2 = _snap(f1)[0], _snap(f2)[0]
    return _from_freq_track(np.where((t // period) % 2 == 0, f1, f2).astype(np.float64))


def _pulsed(freq, on, off, count):
    on_n, off_n = int(RATE * on), int(RATE * off)
    gate = np.concatenate([np.concatenate([_fade(np.ones(on_n), 5), np.zeros(off_n)]) for _ in range(count)])
    return _from_freq_track(np.full(gate.size, _snap(freq)[0]), gate)


def _wail(lo=650.0, hi=1050.0, cycle=1.0, seconds=4.0):
    t = np.arange(int(RATE * seconds)) / RATE
    tri = 1 - np.abs(2 * ((t / cycle) % 1) - 1)        # 0 -> 1 -> 0 each cycle
    return _from_freq_track(_snap_track(lo + (hi - lo) * tri))


def _chime():
    notes = (1046.5, 1318.5, 1568.0)
    parts, tracks = [], []
    for f in notes:
        f = _snap(f)[0]
        n = int(RATE * 0.35)
        t = np.arange(n) / RATE
        parts.append(np.sin(2 * np.pi * f * t) * np.exp(-t * 5) * LEVEL)
        tracks.append(_track(f, n=n))
    tail = int(RATE * 0.2)
    parts.append(np.zeros(tail))
    tracks.append(_track(np.nan, n=tail))
    return _fade(np.concatenate(parts)), np.concatenate(tracks)


BUILTIN: Dict[str, Dict] = {
    'alert_tone': {'label': 'Alert tone (1 kHz steady, 2 s)', 'make': lambda: _steady(1000, 2.0)},
    'hi_lo': {'label': 'Hi-lo (800/1000 Hz, 4 s)', 'make': lambda: _alternating(1000, 800, 0.5, 4.0)},
    'warble': {'label': 'Warble (fast 700/1000 Hz, 3 s)', 'make': lambda: _alternating(1000, 700, 0.08, 3.0)},
    'pulsed': {'label': 'Pulsed alert (1 kHz beeps)', 'make': lambda: _pulsed(1000, 0.2, 0.2, 6)},
    'triple_beep': {'label': 'Triple beep', 'make': lambda: _pulsed(1400, 0.15, 0.1, 3)},
    'wail': {'label': 'Wail (rising/falling, 4 s)', 'make': lambda: _wail()},
    'chime': {'label': 'Attention chime', 'make': _chime},
    # NOAA Weather Radio Warning Alarm Tone: 1050 Hz, sent for about 10 s.
    'noaa_wat': {'label': 'NOAA weather radio alert (1050 Hz, 10 s)', 'make': lambda: _steady(1050, 10.0)},
    # EAS Attention Signal: 853 Hz + 960 Hz together, 8-25 s (8 s is usual).
    'eas_attention': {'label': 'EAS attention signal (853 + 960 Hz, 8 s)', 'make': lambda: _dual(853, 960, 8.0)},
}

_SAFE = re.compile(r'^[A-Za-z0-9_\-]+$')


class SoundLibrary:
    """Built-in tones plus custom WAVs dropped into `sounds_dir`. A custom file
    named like a built-in (e.g. alert_tone.wav) replaces it."""

    def __init__(self, sounds_dir: str = 'sounds'):
        self.sounds_dir = sounds_dir

    def _custom_files(self) -> Dict[str, str]:
        found = {}
        if self.sounds_dir and os.path.isdir(self.sounds_dir):
            for fn in sorted(os.listdir(self.sounds_dir)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() == '.wav' and _SAFE.match(stem):
                    found[stem] = os.path.join(self.sounds_dir, fn)
        return found

    def list(self) -> List[Dict[str, str]]:
        custom = self._custom_files()
        out = [{'id': k, 'label': v['label'], 'source': 'built-in'} for k, v in BUILTIN.items() if k not in custom]
        out += [{'id': k, 'label': k.replace('_', ' ').capitalize(), 'source': 'file'} for k in custom]
        return out

    def exists(self, name: str) -> bool:
        return name in BUILTIN or name in self._custom_files()

    def render(self, name: str) -> np.ndarray:
        """8 kHz float audio for a sound name."""
        return self.render_with_tones(name)[0]

    def render_with_tones(self, name: str):
        """(8 kHz float audio, tone track). The track holds the generated tone
        frequencies per sample (NaN where unknown, e.g. for WAV files)."""
        custom = self._custom_files()
        if name in custom:
            x, rate = read_wav(custom[name])
            x = resample(x, rate, RATE)
            peak = np.max(np.abs(x)) if x.size else 0
            x = x * (0.9 / peak) if peak > 0.9 else x
            return x, np.full((x.size, 2), np.nan)
        if name in BUILTIN:
            return BUILTIN[name]['make']()
        raise KeyError(f"unknown sound '{name}'")
