"""The blip25 IMBE vocoder (src/p25/native/blip25), loaded with ctypes.

blip25-vocoder (https://github.com/openBLIP25/blip25-vocoder, MIT) is a
fixed-point Rust implementation of the P25 full-rate vocoder, bit-exact with
the DVSI reference codec. Build it with `python tools/build_native.py` (needs
Rust 1.85+). imbe.py and mbe_decoder.py use it automatically when it loads,
and fall back to the pure Python encoder/decoder otherwise; set
CCP_PURE_PYTHON_IMBE=1 to force those.

Frames are the 11-byte info-only (4400x4400) form - the 88 bits u0..u7 the
DVM FNE carries. No post-decode enhancement.

The crate is vendored in src/p25/native/blip25/vendor with two patches
(PATCHES.md there) so the encoder can use the reference engine's in-band
tone detector: the stock analysis codes a steady tone above ~405 Hz as noise
or buzz. Encoders run with it on by default - every encoder here puts audio
on the air (pages, live talk) - as a stock/tone-detector pair choosing per
frame; see src/p25/native/blip25/src/lib.rs. Speech encodes identically.
"""
import ctypes
import os
import sys
from typing import Optional, Sequence

import numpy as np

NATIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'native')
LIB_NAME = '_blip25.dll' if sys.platform == 'win32' else ('_blip25.dylib' if sys.platform == 'darwin' else '_blip25.so')
API_VERSION = 2
CHOICE_SAME, CHOICE_STOCK, CHOICE_TONE = 0, 1, 2
FRAME = 160
FRAME_BYTES = 11

_lib = None
_error: Optional[str] = None
_tried = False


def load(force: bool = False):
    """Load the library once; returns it, or None (see load_error())."""
    global _lib, _error, _tried
    if _tried and not force:
        return _lib
    _tried = True
    _lib = None
    path = os.path.join(NATIVE_DIR, LIB_NAME)
    if not os.path.exists(path):
        _error = f'{path} not built (python tools/build_native.py)'
        return None
    try:
        lib = ctypes.CDLL(path)
        lib.ccp_blip25_version.restype = ctypes.c_uint32
        if lib.ccp_blip25_version() != API_VERSION:
            _error = f'{LIB_NAME} is out of date - rebuild it (python tools/build_native.py)'
            return None
        lib.ccp_blip25_new.argtypes = [ctypes.c_int]
        lib.ccp_blip25_new.restype = ctypes.c_void_p
        lib.ccp_blip25_last_choice.argtypes = [ctypes.c_void_p]
        lib.ccp_blip25_last_choice.restype = ctypes.c_int
        lib.ccp_blip25_free.argtypes = [ctypes.c_void_p]
        lib.ccp_blip25_free.restype = None
        lib.ccp_blip25_encode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_char_p]
        lib.ccp_blip25_encode.restype = ctypes.c_int
        lib.ccp_blip25_decode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int16)]
        lib.ccp_blip25_decode.restype = ctypes.c_int
    except (OSError, AttributeError) as e:
        _error = f'could not load {path}: {e}'
        return None
    _lib = lib
    _error = None
    return lib


def available() -> bool:
    return load() is not None


def load_error() -> Optional[str]:
    load()
    return _error


class _Channel:
    def __init__(self, tone_detection: bool = False):
        lib = load()
        if lib is None:
            raise RuntimeError(_error)
        self._lib = lib
        self._h = lib.ccp_blip25_new(1 if tone_detection else 0)
        if not self._h:
            raise MemoryError('ccp_blip25_new failed')

    def __del__(self):
        h, self._h = getattr(self, '_h', None), None
        if h:
            self._lib.ccp_blip25_free(h)


class Blip25Encoder(_Channel):
    """Same interface as imbe.PyIMBEEncoder. Like the reference codec, each
    frame describes the previous 20 ms of audio (one frame of delay).

    tone_detection: run the stock/tone-detector pair (default). False gives
    the stock encoder alone, identical to upstream blip25-vocoder."""

    native = True
    tone_hints = False          # tones need no snapping or track; detected in the codec

    def __init__(self, tone_detection: bool = True):
        super().__init__(tone_detection)
        self.tone_detection = tone_detection
        self._pcm = (ctypes.c_int16 * FRAME)()
        self._out = ctypes.create_string_buffer(FRAME_BYTES)

    def encode(self, samples: Sequence[int], tones=None) -> bytes:
        """160 PCM samples -> 11-byte IMBE frame. tones is ignored."""
        from src.p25.imbe import VocoderError
        if len(samples) != FRAME:
            raise ValueError(f'IMBE frames are {FRAME} samples')
        if isinstance(samples, np.ndarray):
            ctypes.memmove(self._pcm, np.ascontiguousarray(samples, dtype=np.int16).ctypes.data, 2 * FRAME)
        else:
            self._pcm[:] = [int(s) for s in samples]
        if self._lib.ccp_blip25_encode(self._h, self._pcm, self._out) != 0:
            raise VocoderError('blip25 encoder error')
        return self._out.raw

    @property
    def last_choice(self) -> int:
        """Which encoder the last frame came from: CHOICE_SAME (they agreed,
        or tone detection is off), CHOICE_STOCK or CHOICE_TONE."""
        return self._lib.ccp_blip25_last_choice(self._h)


class Blip25Decoder(_Channel):
    """Same interface as mbe_decoder.PyIMBEDecoder.decode(). The arguments
    are accepted for compatibility; the reference decoder has no settings and
    no randomness to seed."""

    native = True
    gain = 1.0                  # output is already at full PCM scale

    def __init__(self, uvquality: int = 3, seed: Optional[int] = None):
        super().__init__()
        self._out = (ctypes.c_int16 * FRAME)()

    def decode(self, frame: bytes) -> np.ndarray:
        """One 11-byte frame -> 160 float samples (int16 scale)."""
        if len(frame) != FRAME_BYTES:
            raise ValueError('IMBE frames are 11 bytes')
        if self._lib.ccp_blip25_decode(self._h, bytes(frame), self._out) != 0:
            return np.zeros(FRAME)
        return np.frombuffer(self._out, dtype=np.int16).astype(np.float64)
