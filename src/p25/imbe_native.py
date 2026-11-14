"""The compiled IMBE encoder and decoder (src/p25/native), loaded with ctypes.

Build with `python tools/build_native.py`. The encoder produces exactly the
same frames as the pure Python one in imbe.py; the decoder matches
mbe_decoder.py to rounding (its random unvoiced phases come from its own
generator). imbe.py and mbe_decoder.py use them automatically when the
library loads; set CCP_PURE_PYTHON_IMBE=1 to force the Python versions.
"""
import ctypes
import logging
import os
import sys
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

NATIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'native')
LIB_NAME = '_imbe.dll' if sys.platform == 'win32' else ('_imbe.dylib' if sys.platform == 'darwin' else '_imbe.so')
API_VERSION = 2

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
        lib.imbe_enc_version.restype = ctypes.c_int
        if lib.imbe_enc_version() != API_VERSION:
            _error = f'{LIB_NAME} is out of date - rebuild it (python tools/build_native.py)'
            return None
        lib.imbe_enc_new.restype = ctypes.c_void_p
        lib.imbe_enc_free.argtypes = [ctypes.c_void_p]
        lib.imbe_enc_free.restype = None
        lib.imbe_enc_encode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p]
        lib.imbe_enc_encode.restype = ctypes.c_int
        lib.imbe_dec_new.argtypes = [ctypes.c_int, ctypes.c_uint64]
        lib.imbe_dec_new.restype = ctypes.c_void_p
        lib.imbe_dec_free.argtypes = [ctypes.c_void_p]
        lib.imbe_dec_free.restype = None
        lib.imbe_dec_decode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
        lib.imbe_dec_decode.restype = ctypes.c_int
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


class NativeIMBEEncoder:
    """Same interface as imbe.PyIMBEEncoder, backed by the C library."""

    native = True

    def __init__(self):
        from src.p25 import imbe
        self._imbe = imbe
        lib = load()
        if lib is None:
            raise RuntimeError(_error)
        self._lib = lib
        self._h = lib.imbe_enc_new()
        if not self._h:
            raise MemoryError('imbe_enc_new failed')
        self._pcm = (ctypes.c_int16 * imbe.FRAME)()
        self._harms = (ctypes.c_int * 8)()
        self._out = ctypes.create_string_buffer(11)
        # tone track buffer, like PyIMBEEncoder.tone_buf; only kept while tones flow
        self._tone_buf: Optional[np.ndarray] = None
        self._tone_frames_left = 0

    def __del__(self):
        h, self._h = getattr(self, '_h', None), None
        if h:
            self._lib.imbe_enc_free(h)

    def _hint(self, tones):
        imbe = self._imbe
        if tones is None and self._tone_frames_left == 0:
            return None
        if self._tone_buf is None:
            self._tone_buf = np.full((imbe.PITCH_EST_BUF_SIZE, 2), np.nan)
        hb = self._tone_buf
        keep = imbe.PITCH_EST_BUF_SIZE - imbe.FRAME
        hb[:keep] = hb[imbe.FRAME:]
        hb[keep:] = np.asarray(tones, dtype=np.float64) if tones is not None else np.nan
        # a tone stays in the analysis window for 3 more frames after the last one
        self._tone_frames_left = 4 if tones is not None and np.isfinite(hb[keep:]).any() else \
            max(0, self._tone_frames_left - 1)
        return imbe.tone_hint(hb[150])

    def encode(self, samples: Sequence[int], tones=None) -> bytes:
        imbe = self._imbe
        if len(samples) != imbe.FRAME:
            raise ValueError(f'IMBE frames are {imbe.FRAME} samples')
        if isinstance(samples, np.ndarray):
            ctypes.memmove(self._pcm, np.ascontiguousarray(samples, dtype=np.int16).ctypes.data, 2 * imbe.FRAME)
        else:
            self._pcm[:] = [int(s) for s in samples]
        hint = self._hint(tones)
        b0, n = -1, 0
        if hint is not None:
            b0, harms = hint
            n = min(len(harms), len(self._harms))
            for i in range(n):
                self._harms[i] = harms[i]
        if self._lib.imbe_enc_encode(self._h, self._pcm, b0, self._harms, n, self._out) != 0:
            raise imbe.VocoderError('IMBE encoder domain error')
        return self._out.raw


class NativeIMBEDecoder:
    """Same interface as mbe_decoder.PyIMBEDecoder.decode(), backed by C."""

    native = True

    def __init__(self, uvquality: int = 3, seed: Optional[int] = None):
        lib = load()
        if lib is None:
            raise RuntimeError(_error)
        self._lib = lib
        if seed is None:
            seed = int.from_bytes(os.urandom(8), 'little')
        self._h = lib.imbe_dec_new(uvquality, seed & 0xFFFFFFFFFFFFFFFF)
        if not self._h:
            raise MemoryError('imbe_dec_new failed')
        self._out = (ctypes.c_double * 160)()

    def __del__(self):
        h, self._h = getattr(self, '_h', None), None
        if h:
            self._lib.imbe_dec_free(h)

    def decode(self, frame: bytes) -> np.ndarray:
        """One 11-byte frame -> 160 float samples (mbelib scale)."""
        if len(frame) != 11:
            raise ValueError('IMBE frames are 11 bytes')
        self._lib.imbe_dec_decode(self._h, bytes(frame), self._out)
        return np.array(self._out, dtype=np.float64)
