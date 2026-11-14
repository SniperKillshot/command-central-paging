"""Small audio helpers: WAV I/O, resampling and level conditioning (numpy only)."""
import io
import wave
from typing import Tuple

import numpy as np

RATE = 8000   # P25 vocoder sample rate


def read_wav(data_or_path) -> Tuple[np.ndarray, int]:
    """Read a PCM WAV (8/16/24/32-bit, any channel count) as mono float in [-1, 1]."""
    src = io.BytesIO(data_or_path) if isinstance(data_or_path, (bytes, bytearray)) else data_or_path
    with wave.open(src, 'rb') as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128) / 128
    elif width == 2:
        x = np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        x = v.astype(np.float64) / 8388608
    elif width == 4:
        x = np.frombuffer(raw, dtype='<i4').astype(np.float64) / 2147483648
    else:
        raise ValueError(f'unsupported WAV sample width {width}')
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, rate


def wav_bytes(pcm16: np.ndarray, rate: int = RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(pcm16, dtype='<i2').tobytes())
    return buf.getvalue()


def resample(x: np.ndarray, rate_in: int, rate_out: int = RATE) -> np.ndarray:
    """Band-limited resampling via FFT. The clip is padded so the FFT's
    circular wrap-around can't smear the end into the start."""
    if rate_in == rate_out or x.size == 0:
        return x.astype(np.float64)
    pad = int(rate_in * 0.05)
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    n_in = xp.size
    n_out = int(round(n_in * rate_out / rate_in))
    X = np.fft.rfft(xp)
    keep = min(X.size, n_out // 2 + 1)
    Y = np.zeros(n_out // 2 + 1, dtype=complex)
    Y[:keep] = X[:keep]
    if rate_out < rate_in:
        # gentle roll-off over the top 8% of the new band instead of a brick wall
        taper = max(1, int(keep * 0.08))
        Y[keep - taper:keep] *= np.linspace(1.0, 0.0, taper)
    y = np.fft.irfft(Y, n_out) * (n_out / n_in)
    p_out = int(round(pad * rate_out / rate_in))
    return y[p_out:y.size - p_out]


def condition_voice(x: np.ndarray, target_rms_db: float = -18.0, peak_db: float = -1.0) -> np.ndarray:
    """Remove DC, bring speech to a consistent loudness, and keep peaks below
    full scale so the vocoder never sees clipping."""
    if x.size == 0:
        return x
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x * x))
    if rms < 1e-6:
        return x
    x = x * (10 ** (target_rms_db / 20) / rms)
    peak = np.max(np.abs(x))
    limit = 10 ** (peak_db / 20)
    if peak > limit:
        x = x * (limit / peak)
    return x


def condition_recording(x: np.ndarray, target_rms_db: float = -18.0, peak_db: float = -1.0,
                        max_gain_db: float = 24.0) -> np.ndarray:
    """Microphone audio: like condition_voice, but loudness is judged on the
    speech only (the loudest 20 ms blocks), so pauses don't make it boost
    background noise, and the boost is capped."""
    if x.size == 0:
        return x
    x = x - np.mean(x)
    # gentle high-pass (~100 Hz) - mic rumble and handling noise
    a = np.exp(-2 * np.pi * 100 / RATE)
    y = np.empty_like(x)
    prev_x = prev_y = 0.0
    for i, v in enumerate(x.tolist()):
        prev_y = a * (prev_y + v - prev_x)
        prev_x = v
        y[i] = prev_y
    block = RATE // 50
    n = y.size // block
    if n == 0:
        return y
    rms = np.sqrt(np.mean(y[:n * block].reshape(n, block) ** 2, axis=1))
    speech = np.percentile(rms, 90)
    if speech < 1e-6:
        return y
    gain = min(10 ** (target_rms_db / 20) / speech, 10 ** (max_gain_db / 20))
    y = y * gain
    peak = np.max(np.abs(y))
    limit = 10 ** (peak_db / 20)
    if peak > limit:
        y = y * (limit / peak)
    return y


def apply_gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    return x * (10 ** (gain_db / 20)) if gain_db else x


def to_pcm16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(RATE * seconds))
