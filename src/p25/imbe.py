"""P25 Phase 1 IMBE (7200x4400) voice encoder - pure Python.

A line-by-line port of the fixed-point encoder that dvmhost ships in
src/vocoder/imbe (originally Pavel Yazev's imbe_vocoder from OP25, built on
the ETSI/ITU fixed-point basic operators). Every arithmetic step keeps the C
code's 16/32-bit saturation and wrap-around.

IMBEEncoder below is the blip25 encoder (src/p25/blip25.py, bit-exact with
the DVSI reference vocoder) when it has been built with
python tools/build_native.py, and this one, PyIMBEEncoder, otherwise.

Input: 160 samples (20 ms) of signed 16-bit PCM at 8 kHz per call.
Output: 11 bytes - the 88-bit IMBE frame (u0..u7, unprotected), the form
P25 DFSI and the DVM FNE network carry.

GPL-2.0 (derived from dvmhost / OP25).
"""
import logging
import os
from typing import List, Optional, Sequence

import numpy as np

from src.p25 import tables as T

logger = logging.getLogger(__name__)

MAX_16 = 0x7FFF
MIN_16 = -0x8000
MAX_32 = 0x7FFFFFFF
MIN_32 = -0x80000000

FRAME = 160
NUM_HARMS_MAX = 56
NUM_HARMS_MIN = 9
NUM_BANDS_MAX = 12
MAX_BLOCK_LEN = 10
NUM_PRED_RES_BLKS = 6
PE_LPF_ORD = 21
PITCH_EST_FRAME = 301
B_NUM = NUM_HARMS_MAX - 1
FFTLENGTH = 256
PITCH_EST_BUF_SIZE = 621
EN_BIT_STREAM_LEN = 3 + 3 * 12 + 6 + 2 * 11 + 3
GAIN_QNT_TBL_SIZE = 64

CNST_0_9254_Q0_16 = 60647
CNST_0_33_Q0_16 = 0x5556
CNST_ONE_Q8_24 = 0x01000000
CNST_0_7_Q1_15 = 0x599A
CNST_0_4_Q1_15 = 0x3333
CNST_0_03_Q1_15 = 0x03D7
CNST_0_05_Q1_15 = 0x0666
CNST_1_125_Q8_8 = 0x0120
CNST_0_5_Q8_8 = 0x0080
CNST_0_125_Q8_8 = 0x0020
CNST_0_25_Q8_8 = 0x0040
CNST_0_8717_Q1_15 = 0x6F94
CNST_0_0031_Q1_15 = 0x0064
CNST_0_48_Q4_12 = 0x07AE
CNST_1_00_Q4_12 = 0x1000
CNST_0_85_Q4_12 = 0x0D9B
CNST_0_4_Q4_12 = 0x0666
CNST_0_05_Q4_12 = 0x00CD
CNST_0_5882_Q1_15 = 0x4B4B
CNST_0_2857_Q1_15 = 0x2492
CNST_0_5_Q1_15 = 0x4000
CNST_0_5_Q5_11 = 0x0400
CNST_1_0_Q1_15 = 0x7FFF
CNST_0_5_Q0_16 = 0x8000
X05_Q15 = 16384
ONE_Q15 = 32767


class VocoderError(ArithmeticError):
    """The C code abort()s here (div_s domain error). Callers substitute silence."""


# ---------------------------------------------------------------------------
#  Fixed-point basic operators (basic_op.cpp / math_sub.cpp)
# ---------------------------------------------------------------------------

def s16(x: int) -> int:
    return ((x + 0x8000) & 0xFFFF) - 0x8000


def s32(x: int) -> int:
    return ((x + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def sat16(x: int) -> int:
    return MAX_16 if x > MAX_16 else (MIN_16 if x < MIN_16 else x)


def sat32(x: int) -> int:
    return MAX_32 if x > MAX_32 else (MIN_32 if x < MIN_32 else x)


def add(a, b):
    return sat16(a + b)


def sub(a, b):
    return sat16(a - b)


def negate(a):
    return MAX_16 if a == MIN_16 else -a


def shr(a, n):
    if n < 0:
        return shl(a, -max(n, -16))
    if n >= 15:
        return -1 if a < 0 else 0
    return a >> n


def shl(a, n):
    if n < 0:
        return shr(a, -max(n, -16))
    if n > 15:
        if a != 0:
            return MAX_16 if a > 0 else MIN_16
        return 0
    r = a << n
    if r != s16(r):
        return MAX_16 if a > 0 else MIN_16
    return r


def mult(a, b):
    return sat16((a * b) >> 15)


def mult_r(a, b):
    return sat16((a * b + 0x4000) >> 15)


def L_mult(a, b):
    p = a * b
    return MAX_32 if p == 0x40000000 else p * 2


def L_add(a, b):
    return sat32(a + b)


def L_sub(a, b):
    return sat32(a - b)


def L_mac(acc, a, b):
    return sat32(acc + L_mult(a, b))


def L_msu(acc, a, b):
    return sat32(acc - L_mult(a, b))


def L_deposit_h(a):
    return a << 16


def extract_h(L):
    return s16(L >> 16)


def extract_l(L):
    return s16(L)


def L_round(L):
    return extract_h(sat32(L + 0x8000))


def L_shr(L, n):
    if n < 0:
        return L_shl(L, -max(n, -32))
    if n >= 31:
        return -1 if L < 0 else 0
    return L >> n


def L_shl(L, n):
    if n <= 0:
        return L_shr(L, -max(n, -32))
    while n > 0:
        if L > 0x3FFFFFFF:
            return MAX_32
        if L < -0x40000000:
            return MIN_32
        L *= 2
        n -= 1
    return L


def shr_r(a, n):
    if n > 15:
        return 0
    r = shr(a, n)
    if n > 0 and (a & (1 << (n - 1))):
        r += 1
    return s16(r)


def L_shr_r(L, n):
    if n > 31:
        return 0
    r = L_shr(L, n)
    if n > 0 and (L & (1 << (n - 1))):
        r += 1
    return s32(r)


def norm_s(a):
    if a == 0:
        return 0
    if a == -1:
        return 15
    if a < 0:
        a = ~a
    return 15 - a.bit_length()


def norm_l(L):
    if L == 0:
        return 0
    if L == -1:
        return 31
    if L < 0:
        L = ~L
    return 31 - L.bit_length()


def div_s(a, b):
    if a > b or a < 0 or b < 0:
        raise VocoderError(f'div_s domain error {a}/{b}')
    if b == 0:
        raise VocoderError('div_s by zero')
    if a == 0:
        return 0
    if a == b:
        return MAX_16
    return (a << 15) // b


def L_mpy_ls(L2, v1):
    sw = shr(extract_l(L2), 1) & 0x7FFF
    out = L_shr(L_mult(v1, sw), 15)
    return L_mac(out, v1, extract_h(L2))


def cos_fxp(x):
    sign = False
    tx = negate(x) if x < 0 else x
    if tx > X05_Q15:
        tx = sub(ONE_Q15, tx)
        sign = True
    i1 = shr(tx, 7)
    i2 = add(i1, 1)
    if i1 == 128:
        return 0
    m = shl(sub(tx, shl(i1, 7)), 8)
    ty = add(T.COS_TABLE[i1], mult(m, sub(T.COS_TABLE[i2], T.COS_TABLE[i1])))
    return negate(ty) if sign else ty


def sin_fxp(x):
    sign = False
    if x < 0:
        tx = negate(x)
        sign = True
    else:
        tx = x
    tx = sub(tx, X05_Q15) if tx > X05_Q15 else sub(X05_Q15, tx)
    ty = cos_fxp(tx)
    return negate(ty) if sign else ty


def sqrt_l_exp(L):
    """Returns (mantissa, exponent)."""
    if L <= 0:
        return 0, 0
    e = norm_l(L) & 0xFFFE
    L = L_shl(L, e)
    exp = e >> 1
    L = L_shr(L, 9)
    i = extract_h(L)
    L = L_shr(L, 1)
    a = extract_l(L) & 0x7FFF
    i = sub(i, 16)
    y = L_deposit_h(T.SQRT_TABLE[i])
    y = L_msu(y, sub(T.SQRT_TABLE[i], T.SQRT_TABLE[i + 1]), a)
    return y, exp


def Log2(x):
    if x <= 0:
        return 0
    exp = norm_s(x)
    x = shl(x, exp)
    i = shr(x, 9)
    a = shl(x & 0x1FF, 6)
    i = sub(i, 32)
    y = L_deposit_h(T.LOG_TABLE[i])
    y = L_msu(y, sub(T.LOG_TABLE[i], T.LOG_TABLE[i + 1]), a)
    y = L_shr(y, 9)
    exp = sub(12, exp)
    return L_add(y, L_deposit_h(shl(exp, 6)))


# ---------------------------------------------------------------------------
#  Vectorised helpers (exact: fall back to the sequential loop on saturation)
# ---------------------------------------------------------------------------

def _sat_sum(terms: np.ndarray, init: int = 0) -> int:
    """Sequential L_add(acc, term) over int32-range terms, exactly."""
    if terms.size == 0:
        return init
    cs = np.cumsum(terms, dtype=np.int64)
    if init:
        cs += init
    if cs.max() <= MAX_32 and cs.min() >= MIN_32:
        return int(cs[-1])
    acc = init
    for t in terms.tolist():
        acc = sat32(acc + t)
    return acc


def _L_mult_v(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    p = a.astype(np.int64) * b.astype(np.int64)
    out = p * 2
    out[p == 0x40000000] = MAX_32
    return out


def _L_shr_v(x: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return x
    if n >= 31:
        return np.where(x < 0, -1, 0).astype(np.int64)
    return x >> n


def _sat16_v(x: np.ndarray) -> np.ndarray:
    return np.clip(x, MIN_16, MAX_16)


def _sat32_v(x: np.ndarray) -> np.ndarray:
    return np.clip(x, MIN_32, MAX_32)


def _L_round_v(x: np.ndarray) -> np.ndarray:
    r = _sat32_v(x + 0x8000) >> 16
    return ((r + 0x8000) & 0xFFFF) - 0x8000


def _get_bit_allocation(num_harms: int, out: List[int]) -> None:
    if num_harms == NUM_HARMS_MIN:
        offset = 0
    else:
        index = num_harms - NUM_HARMS_MIN - 1
        offset = T.BIT_ALLOCATION_OFFSET_TBL[index >> 2] + (3 + (index >> 2)) * (index & 0x3)
    k = 0
    for _ in range(0, num_harms - 1, 4):
        tmp = T.BIT_ALLOCATION_TBL[offset]
        offset += 1
        out[k + 3] = tmp & 0xF
        out[k + 2] = (tmp >> 4) & 0xF
        out[k + 1] = (tmp >> 8) & 0xF
        out[k + 0] = (tmp >> 12) & 0xF
        k += 4


def _deqnt_by_step(qval, step_size, bit_num):
    if bit_num == 0:
        return 0
    res = s32(step_size * (qval - (1 << (bit_num - 1))))
    return L_add(res, (step_size * CNST_0_5_Q0_16) >> 16)


def _qnt_by_step(val, step_size, bit_num):
    step16 = s16(step_size)
    shift = norm_s(step16)
    tmp = div_s(0x4000, shl(step16, shift))
    q_index = shr_r(mult(val, tmp), sub(9, shift))
    max_val = s16(1 << (bit_num - 1)) if bit_num >= 1 else 0
    min_val = negate(max_val)
    if q_index < min_val:
        return 0
    if q_index >= max_val:
        return s16((1 << bit_num) - 1)
    return s16(max_val + q_index)


def _tbl_quant(val, q_tbl, size):
    lo, hi = 0, size - 1
    if val >= q_tbl[hi]:
        return hi
    if val <= q_tbl[lo]:
        return lo
    while hi - lo != 1:
        mid = lo + ((hi - lo) >> 1)
        if q_tbl[mid] > val:
            hi = mid
        else:
            lo = mid
    return hi if q_tbl[hi] - val <= val - q_tbl[lo] else lo


_b0_cache = {}


def _best_b0(freqs) -> Optional[int]:
    """Pitch index whose harmonics land closest to every tone in freqs.
    Among choices within 0.3% (or 2 Hz) of the best error, prefer the highest
    fundamental - fewer, wider-spaced harmonics give the cleanest tone."""
    if freqs in _b0_cache:
        return _b0_cache[freqs]
    scored = []
    for b0 in range(208):
        f0 = 16000.0 / (b0 + 39.5)
        if any(f < f0 * 0.75 for f in freqs):
            continue                            # tone below the first harmonic
        err = max(abs(f - round(f / f0) * f0) for f in freqs)
        scored.append((err, -f0, b0))
    if not scored:
        _b0_cache[freqs] = None
        return None
    exact = [s for s in scored if s[0] < 0.01]
    if exact:
        # Already on the grid (e.g. produced by codec_tone_frequencies): keep
        # the highest exact fundamental, which is the one that produced it.
        choice = min(exact, key=lambda s: s[1])[2]
    else:
        best = min(s[0] for s in scored)
        tol = max(best + 2.0, max(freqs) * 0.003)
        choice = min((s for s in scored if s[0] <= tol), key=lambda s: s[1])[2]
    _b0_cache[freqs] = choice
    return choice


def tone_hint(center):
    """(b0, harmonic numbers) for the known tone frequencies at the centre of
    the analysis window (NaN = no tone), or None."""
    freqs = tuple(round(float(f), 2) for f in np.asarray(center).ravel() if f == f and f > 0)
    if not freqs:
        return None
    b0 = _best_b0(freqs)
    if b0 is None:
        return None
    f0 = 16000.0 / (b0 + 39.5)
    return b0, [int(round(f / f0)) for f in freqs]      # 1-based harmonic numbers


def codec_tone_frequencies(freqs) -> tuple:
    """The frequencies IMBE will actually reproduce for these tones (nearest
    harmonics of the chosen pitch, within ~0.4%). Generating tones at exactly
    these frequencies means analysis and synthesis agree, so the tone level
    doesn't beat against the harmonic grid."""
    freqs = tuple(float(f) for f in freqs)
    b0 = _best_b0(tuple(round(f, 1) for f in freqs))
    if b0 is None:
        return freqs
    f0 = 16000.0 / (b0 + 39.5)
    return tuple(round(f / f0) * f0 for f in freqs)


# ---------------------------------------------------------------------------
#  Encoder
# ---------------------------------------------------------------------------

class PyIMBEEncoder:
    """Stateful encoder: feed consecutive 20 ms frames of one call."""

    native = False
    tone_hints = True           # uses the alert tones' tone track

    def __init__(self):
        self.prev_pitch = 158
        self.prev_prev_pitch = 158
        self.prev_e_p = 0
        self.prev_prev_e_p = 0
        self.pitch_est_buf = [0] * PITCH_EST_BUF_SIZE
        self.pitch_ref_buf = [0] * PITCH_EST_BUF_SIZE
        self.tone_buf = np.full((PITCH_EST_BUF_SIZE, 2), np.nan)
        self.pe_lpf_mem = [0] * PE_LPF_ORD
        self.dc_rmv_mem = 0
        self.num_harms_prev2 = 30
        self.sa_prev2 = [0] * (NUM_HARMS_MAX + 2)
        self.v_uv_dsn = [0] * NUM_BANDS_MAX
        self.th_max = 0
        self._fft_init()
        # IMBE_PARAM
        self.e_p_val = 0
        self.pitch = 0
        self.ref_pitch = 0
        self.fund_freq = 0
        self.num_harms = 0
        self.num_bands = 0
        self.p_v_uv_dsn = [0] * NUM_HARMS_MAX
        self.b_vec = [0] * (NUM_HARMS_MAX + 3)
        self.bit_alloc = [0] * (B_NUM + 4)
        self.sa = [0] * NUM_HARMS_MAX
        self.l_uv = 0

    # -- public ----------------------------------------------------------

    def encode(self, samples: Sequence[int], tones: Optional[Sequence] = None) -> bytes:
        """160 PCM samples -> 11-byte IMBE frame.

        tones: optional per-sample tone frequencies for these samples (an
        array of shape (160, 2) in Hz, NaN where there is no tone; one or two
        tones per sample). See _apply_tone_hint."""
        if len(samples) != FRAME:
            raise ValueError(f'IMBE frames are {FRAME} samples')
        fv = self.encode_frame_vector(samples, tones)
        return pack_frame_vector(fv)

    def encode_frame_vector(self, samples: Sequence[int], tones: Optional[Sequence] = None) -> List[int]:
        snd = [int(s) for s in samples]
        pe, pr = self.pitch_est_buf, self.pitch_ref_buf
        keep = PITCH_EST_BUF_SIZE - FRAME
        pe[:keep] = pe[FRAME:]
        pr[:keep] = pr[FRAME:]
        pr[keep:] = self._dc_rmv(snd)
        pe[keep:] = self._pe_lpf(pr[keep:])
        hb = self.tone_buf
        hb[:keep] = hb[FRAME:]
        hb[keep:] = np.asarray(tones, dtype=np.float64) if tones is not None else np.nan
        self._pitch_est(pe)
        # The analysis window below is pitch_ref_buf[40:261], centred on 150.
        hinted = self._apply_tone_hint(hb[150])

        fft_re = [0] * FFTLENGTH
        sig = pr[40:]
        wr = T.WR
        k = 0
        for i in range(146, 256):
            fft_re[i] = mult(sig[k], wr[k])
            k += 1
        fft_re[0] = sig[k]
        k += 1
        w = len(wr) - 1
        for i in range(1, 111):
            fft_re[i] = mult(sig[k], wr[w])
            k += 1
            w -= 1
        re, im = self._fft(fft_re, [0] * FFTLENGTH)
        if not hinted:
            self._pitch_ref(re, im)
        self._v_uv_det(re, im)
        if hinted:
            self._shape_tone_frame(hinted)
        self._sa_encode()
        return self._encode_frame_vector()

    # -- known-tone pitch ----------------------------------------------------

    def _apply_tone_hint(self, center) -> bool:
        """IMBE can only describe sound as harmonics of a voice pitch
        (b0: f0 = 16000 / (b0 + 39.5) Hz, about 65-405 Hz). The pitch tracker
        is built for speech: on a pure tone every multiple of the tone's period
        looks equally good, so it locks onto one that buries the tone among
        dozens of harmonics or flips between candidates - heard as warble.
        When the audio is a tone we generated, we know its frequency, so pick
        the pitch that puts each tone exactly on a harmonic (fewest harmonics
        among the accurate choices) and skip the search. Voicing and
        amplitudes are still measured from the audio."""
        hint = tone_hint(center)
        if hint is None:
            return False
        b0, harmonics = hint
        ref_pitch = (b0 << 7) + 0x13C0          # Q8.8 samples, centre of the b0 step
        shift = norm_s(ref_pitch)
        tmp = div_s(0x4000, shl(ref_pitch, shift))
        self.ref_pitch = ref_pitch
        self.fund_freq = L_shl(tmp, shift + 11)
        return harmonics

    def _shape_tone_frame(self, tone_harmonics) -> None:
        """Make a known-tone frame decode cleanly on any IMBE decoder:
        - mark every harmonic voiced: the standard adds random phase to voiced
          harmonics in proportion to how many are unvoiced, which makes a
          steady tone warble from frame to frame;
        - replace the near-silent non-tone harmonics with a fixed floor 30 dB
          down. IMBE's amplitude quantiser is predictive; a lone spike far
          above everything else drives it into a frame-to-frame limit cycle
          (the tone's level wobbles even though the input is constant). A
          -30 dB floor keeps it stable: measured wobble drops to under 0.5 dB
          with ~97% of the energy still on the tone."""
        n = self.num_harms
        tones = {h - 1 for h in tone_harmonics if 1 <= h <= n}
        if not tones:
            return
        peak = max(self.sa[j] for j in tones)
        floor = max(1, peak >> 5)
        for j in range(n):
            if j not in tones:
                self.sa[j] = floor
            self.p_v_uv_dsn[j] = 1
        for b in range(self.num_bands):
            self.v_uv_dsn[b] = 1
        self.b_vec[1] = (1 << self.num_bands) - 1
        self.l_uv = 0

    # -- front end -------------------------------------------------------

    def _dc_rmv(self, snd):
        out = []
        mem = self.dc_rmv_mem
        for x in snd:
            t = L_deposit_h(x)
            mem = L_add(mem, t)
            out.append(L_round(mem))
            mem = L_mpy_ls(mem, 0x7EB8)
            mem = L_sub(mem, t)
        self.dc_rmv_mem = mem
        return out

    _LPF_COEF = np.array(T.LPF_COEF, dtype=np.int64)

    # (i, i + lag, valid) index grids for the 130 autocorrelation lags in _e_p
    _ep_i = np.arange(PITCH_EST_FRAME - 21)[None, :]
    _ep_lag = np.arange(21, 151)[:, None]
    _EP_IDX = (np.broadcast_to(_ep_i, (130, PITCH_EST_FRAME - 21)),
               np.minimum(_ep_i + _ep_lag, PITCH_EST_FRAME - 1),
               (_ep_i + _ep_lag < PITCH_EST_FRAME).astype(np.int64))

    def _pe_lpf(self, sig):
        """21-tap FIR. Every output's tap products are formed at once; the
        running sum is exact unless an intermediate saturates, in which case
        that output is recomputed with the sequential saturating sum."""
        hist = np.array(self.pe_lpf_mem[1:] + list(sig), dtype=np.int64)
        win = np.lib.stride_tricks.sliding_window_view(hist, PE_LPF_ORD)    # (160, 21)
        terms = _L_mult_v(win, np.broadcast_to(self._LPF_COEF, win.shape))
        cs = np.cumsum(terms, axis=1)
        bad = (cs.max(axis=1) > MAX_32) | (cs.min(axis=1) < MIN_32)
        sums = cs[:, -1]
        out = _L_round_v(sums).tolist()
        for k in np.nonzero(bad)[0].tolist():
            out[k] = L_round(_sat_sum(terms[k]))
        self.pe_lpf_mem = hist[-PE_LPF_ORD:].tolist()
        return out

    # -- pitch estimation ------------------------------------------------

    def _e_p(self, sigin):
        wi = np.array(T.WI, dtype=np.int64)
        s = np.array(sigin[:PITCH_EST_FRAME], dtype=np.int64)
        w = _sat16_v((s * wi + 0x4000) >> 15)          # mult_r

        def mpy_ls_v(L2, v1):
            lo = (((L2 + 0x8000) & 0xFFFF) - 0x8000) >> 1 & 0x7FFF
            out = _L_mult_v(v1, lo) >> 15
            hi = ((L2 >> 16) + 0x8000 & 0xFFFF) - 0x8000
            return _sat32_v(out + _L_mult_v(v1, hi))

        sq = _L_mult_v(s, s)
        L_sum = _sat_sum(mpy_ls_v(sq, wi))
        if L_sum == MAX_32:
            L_sum = _sat_sum(mpy_ls_v(sq >> 5, wi))
            scale = 5
        else:
            scale = 0
        L_e0 = _sat_sum(_L_shr_v(_L_mult_v(w, w), scale))

        corr = [0] * 259
        # w comes from mult_r(x, wi) with 0 < wi < 32768, so it is never -32768 and
        # L_mult(a, b) is exactly 2ab. With no shift, if the absolute sum of all
        # terms fits in 32 bits no partial sum can saturate, so the sequential
        # saturating sum equals a plain correlation. Otherwise use the exact path.
        # All 130 lags at once: row r holds the terms for lag 21 + r; every
        # partial sum is checked, and only a row that would saturate is redone
        # with the sequential saturating sum.
        idx = self._EP_IDX
        terms = ((2 * w[idx[0]] * w[idx[1]]) >> scale) * idx[2]
        cs = np.cumsum(terms, axis=1)
        row_ok = (cs.max(axis=1) <= MAX_32) & (cs.min(axis=1) >= MIN_32)
        last = cs[:, -1].tolist()
        for r in range(130):
            if not row_ok[r]:
                lag = 21 + r
                last[r] = _sat_sum(_L_shr_v(_L_mult_v(w[:PITCH_EST_FRAME - lag], w[lag:]), scale))
        corr[0:259:2] = last
        for i in range(1, 258, 2):
            corr[i] = L_shr(L_add(corr[i - 1], corr[i + 1]), 1)

        res = [0] * 203
        den_part_acc = CNST_0_8717_Q1_15
        step = 42
        beg = 0
        L_e0 = L_shr(L_e0, 7)
        for i in range(203):
            L_tmp = 0
            j = beg
            while j <= 258:
                L_tmp = L_add(L_tmp, corr[j])
                j += step
            L_tmp = L_shr(L_tmp, 6)
            L_tmp = L_add(L_tmp, L_e0)
            L_tmp = s32(L_tmp * step)
            L_num = L_sub(L_sum, L_tmp)
            beg += 1
            step += 1
            L_den = L_mpy_ls(L_sum, den_part_acc)
            if L_num < L_den and L_den != 0:
                if L_num <= 0:
                    res[i] = 0
                else:
                    t = norm_l(L_den)
                    t = div_s(extract_h(L_shl(L_num, t)), extract_h(L_shl(L_den, t)))
                    res[i] = shr(t, 3)
            else:
                res[i] = CNST_1_00_Q4_12
            den_part_acc = sub(den_part_acc, CNST_0_0031_Q1_15)
        return res

    def _pitch_est(self, buf):
        mm = T.MIN_MAX_TBL
        arr0 = self._e_p(buf[0:])
        min_index = (mm[self.prev_pitch] >> 8) & 0xFF
        max_index = mm[self.prev_pitch] & 0xFF
        p = pb = min_index
        e_cur = arr0[min_index]
        p += 1
        while p <= max_index:
            if arr0[p] < e_cur:
                e_cur = arr0[p]
                pb = p
            p += 1
        ceb = add(e_cur, add(self.prev_e_p, self.prev_prev_e_p))
        if ceb <= CNST_0_48_Q4_12:
            self.prev_prev_pitch = self.prev_pitch
            self.prev_pitch = pb
            self.prev_prev_e_p = self.prev_e_p
            self.prev_e_p = arr0[pb]
            self.pitch = pb + 42
            self.e_p_val = self.prev_e_p
            return

        arr1 = self._e_p(buf[FRAME:])
        arr2 = self._e_p(buf[2 * FRAME:])
        p0_est = 0
        cef_est = s16(arr0[0] + arr1[0] + arr2[0])
        arr2_min = [0] * 203
        for p1 in range(203):
            p2 = (mm[p1] >> 8) & 0xFF
            p2_max = mm[p1] & 0xFF
            s_tmp = arr2[p1]
            while p2 <= p2_max:
                if arr2[p2] < s_tmp:
                    s_tmp = arr2[p2]
                p2 += 1
            arr2_min[p1] = s_tmp
        est_save = [0] * 203
        e1p1_e2p2_est = 0
        for p0 in range(203):
            e1p1_e2p2_est = s16(arr1[p0] + arr2_min[p0])
            p1 = (mm[p0] >> 8) & 0xFF
            p1_max = mm[p0] & 0xFF
            while p1 <= p1_max:
                v = add(arr1[p1], arr2_min[p1])
                if v < e1p1_e2p2_est:
                    e1p1_e2p2_est = v
                p1 += 1
            est_save[p0] = e1p1_e2p2_est
            cef = add(arr0[p0], e1p1_e2p2_est)
            if cef < cef_est:
                cef_est = cef
                p0_est = p0

        pf = p0_est
        if pf >= 42:
            if pf < 84:
                i = 1
            elif pf < 126:
                i = 2
            elif pf < 168:
                i = 3
            else:
                i = 4
            p_fp = ((pf + 42) << 8) & 0xFFFF
            tmp = 0
            while i:
                i -= 1
                if i == 0:
                    tmp = p_fp >> 1
                elif i == 1:
                    tmp = ((p_fp * 0x5555) & 0xFFFFFFFF) >> 16
                elif i == 2:
                    tmp = p_fp >> 2
                elif i == 3:
                    tmp = ((p_fp * 0x3333) & 0xFFFFFFFF) >> 16
                tmp &= 0xFFFF
                p_index = s16(((tmp + 0x0080) >> 8) - 42)
                cef = add(arr0[p_index], est_save[p_index])
                if cef <= CNST_0_85_Q4_12 and mult_r(cef, CNST_0_5882_Q1_15) <= cef_est:
                    pf = p_index
                    break
                if cef <= CNST_0_4_Q4_12 and mult_r(cef, CNST_0_2857_Q1_15) <= cef_est:
                    pf = p_index
                    break
                if cef <= CNST_0_05_Q4_12:
                    pf = p_index
                    break

        cef = add(arr0[pf], est_save[pf])
        p = pb if ceb <= cef else pf
        self.prev_prev_pitch = self.prev_pitch
        self.prev_pitch = p
        self.prev_prev_e_p = self.prev_e_p
        self.prev_e_p = arr0[p]
        self.pitch = p + 42
        self.e_p_val = self.prev_e_p

    # -- FFT -------------------------------------------------------------

    def _fft_init(self):
        half = FFTLENGTH >> 1
        step = shl(2, norm_s(half))
        theta = 0
        self.wr_array = [0] * (half + 1)
        self.wi_array = [0] * (half + 1)
        for i in range(half + 1):
            self.wr_array[i] = cos_fxp(theta)
            self.wi_array[i] = sin_fxp(theta)
            theta = ONE_Q15 if i >= half - 1 else add(theta, step)
        self._fft_plan()

    def _fft_plan(self):
        """Precompute fft.cpp's bit-reversal permutation and, per butterfly
        stage, every (i, j, twiddle) triple, so a stage is a few array ops.
        All butterflies within one stage touch disjoint elements, so doing
        them at once gives exactly the sequential result."""
        nn = FFTLENGTH
        n = 2 * nn
        perm = list(range(n + 1))
        j = 1
        for i in range(1, n, 2):
            if j > i:
                perm[j], perm[i] = perm[i], perm[j]
                perm[j + 1], perm[i + 1] = perm[i + 1], perm[j + 1]
            m = nn
            while m >= 2 and j > m:
                j -= m
                m >>= 1
            j += m
        stages = []
        mmax = 2
        index_step = nn
        while n > mmax:
            istep = mmax << 1
            index_step >>= 1
            ii_all, wr_all, wi_all = [], [], []
            index = 0
            wr, wi = ONE_Q15, 0
            for m in range(1, mmax, 2):
                for i in range(m, n + 1, istep):
                    ii_all.append(i)
                    wr_all.append(wr)
                    wi_all.append(wi)
                index += index_step
                wr = self.wr_array[index]
                wi = self.wi_array[index]
            ii = np.array(ii_all, dtype=np.int64)
            stages.append((ii, ii + mmax, np.array(wr_all, dtype=np.int64), np.array(wi_all, dtype=np.int64)))
            mmax = istep
        self._fft_perm = np.array(perm, dtype=np.int64)
        self._fft_stages = stages

    def _fft(self, re_in, im_in):
        """fft.cpp (forward, isign = 1) on interleaved 1-based data."""
        data = np.zeros(2 * FFTLENGTH + 1, dtype=np.int64)
        data[1::2] = re_in
        data[2::2] = im_in
        data = data[self._fft_perm]
        lo, hi = MIN_32, MAX_32
        for ii, jj, wr, wi in self._fft_stages:
            dr, di = data[jj], data[jj + 1]
            tr = np.clip((_L_mult_v(wr, dr) >> 1) - (_L_mult_v(wi, di) >> 1), lo, hi)
            ti = np.clip((_L_mult_v(wr, di) >> 1) + (_L_mult_v(wi, dr) >> 1), lo, hi)
            a = data[ii] << 15                              # L_shr(L_deposit_h(x), 1)
            b = data[ii + 1] << 15
            data[jj] = _L_round_v(np.clip(a - tr, lo, hi))
            data[ii] = _L_round_v(np.clip(a + tr, lo, hi))
            data[jj + 1] = _L_round_v(np.clip(b - ti, lo, hi))
            data[ii + 1] = _L_round_v(np.clip(b + ti, lo, hi))
        return data[1::2].tolist(), data[2::2].tolist()

    # -- pitch refinement -------------------------------------------------

    def _pitch_ref(self, re, im):
        wr_sp = T.WR_SP
        pitch_est = sub(shl(self.pitch, 7), CNST_1_125_Q8_8)
        L_diff_min = MAX_32
        pitch_cand = 0
        fund_freq_cand = 0
        sp_re = [0] * (2 * FFTLENGTH)
        sp_im = [0] * (2 * FFTLENGTH)
        re_np = np.array(re, dtype=np.int64)
        im_np = np.array(im, dtype=np.int64)
        for _ in range(19):
            shift = norm_s(pitch_est)
            tmp = div_s(0x4000, shl(pitch_est, shift))
            fund_freq = L_shl(tmp, shift + 11)
            acc = fund_freq
            half = L_shr(fund_freq, 1)
            acc_a = L_sub(fund_freq, half)
            acc_b = L_add(fund_freq, half)
            up_lim = extract_h(L_shr(s32((CNST_0_9254_Q0_16 * pitch_est) & 0xFFFFFFFF), 1))
            up_lim = sub(up_lim, CNST_0_5_Q8_8)
            up_lim = s16(up_lim & 0xFF00)
            up_lim = shr(mult(up_lim, extract_h(fund_freq)), 1)
            index_b = 0
            while index_b <= up_lim:
                ha = extract_h(acc_a)
                hb = extract_h(acc_b)
                index_a = (ha >> 8) + (1 if ha & 0xFF else 0)
                index_b = (hb >> 8) + (1 if hb & 0xFF else 0)
                if index_b >= 50:
                    L_tmp = L_shl(L_deposit_h(index_a), 8)
                    L_tmp = L_sub(L_tmp, acc)
                    L_tmp = L_add(L_tmp, 0x00020000)
                    L_tmp = L_shr(L_tmp, 2)
                    tbl = []
                    amp_re = amp_im = 0
                    ia = index_a
                    while ia < index_b:
                        iw = extract_h(L_tmp)
                        if iw < 0 and (L_tmp & 0xFFFF):
                            iw = add(iw, 1)
                        iw = add(iw, 160)
                        tbl.append(iw)
                        if 0 <= iw <= 320:
                            amp_re = L_mac(amp_re, re[ia], wr_sp[iw])
                            amp_im = L_mac(amp_im, im[ia], wr_sp[iw])
                        ia += 1
                        L_tmp = L_add(L_tmp, 0x400000)
                    hre = extract_h(amp_re)
                    him = extract_h(amp_im)
                    ia = index_a
                    for iw in tbl:
                        if iw < 0 or iw > 320:
                            sp_re[ia] = sp_im[ia] = 0
                        else:
                            sp_im[ia] = mult(mult(him, wr_sp[iw]), 0x6666)
                            sp_re[ia] = mult(mult(hre, wr_sp[iw]), 0x6666)
                        ia += 1
                acc_a = L_add(acc_a, fund_freq)
                acc_b = L_add(acc_b, fund_freq)
                acc = L_add(acc, fund_freq)
            if up_lim >= 50:
                rt = np.clip(re_np[50:up_lim + 1] - np.array(sp_re[50:up_lim + 1], dtype=np.int64), MIN_16, MAX_16)
                it = np.clip(im_np[50:up_lim + 1] - np.array(sp_im[50:up_lim + 1], dtype=np.int64), MIN_16, MAX_16)
                terms = np.empty(2 * rt.size, dtype=np.int64)
                terms[0::2] = _L_mult_v(rt, rt)
                terms[1::2] = _L_mult_v(it, it)
                L_sum = _sat_sum(terms)
            else:
                L_sum = 0
            if L_sum < L_diff_min:
                L_diff_min = L_sum
                pitch_cand = pitch_est
                fund_freq_cand = fund_freq
            pitch_est = add(pitch_est, CNST_0_125_Q8_8)
        self.ref_pitch = pitch_cand
        self.fund_freq = fund_freq_cand

    # -- voiced/unvoiced decision ----------------------------------------

    @staticmethod
    def _voiced_sa_calc(num, den):
        L, e = sqrt_l_exp(L_mpy_ls(num, den))
        return extract_h(L_shr(L, e - 3))

    @staticmethod
    def _unvoiced_sa_calc(num, den):
        shift = norm_s(den)
        tmp = div_s(0x4000, shl(den, shift))
        L = L_shl(L_mpy_ls(num, tmp), shift + 2)
        L, e = sqrt_l_exp(L)
        L = L_shr(L, e - 2 + 6)
        return extract_h(L_mpy_ls(L, 0x4A76))

    def _v_uv_det(self, re, im):
        wr_sp = T.WR_SP
        fund_freq = self.fund_freq
        tmp = shr(add(shr(self.ref_pitch, 1), CNST_0_25_Q8_8), 8)
        num_harms = extract_h(s32((CNST_0_9254_Q0_16 * tmp) & 0xFFFFFFFF))
        num_harms = max(NUM_HARMS_MIN, min(NUM_HARMS_MAX, num_harms))
        if num_harms <= 36:
            num_bands = extract_h(s32(((num_harms + 2) * CNST_0_33_Q0_16) & 0xFFFFFFFF))
        else:
            num_bands = NUM_BANDS_MAX
        self.num_harms = num_harms
        self.num_bands = num_bands

        th_lf = 0
        for j in range(64):
            th_lf = L_mac(th_lf, re[j], re[j])
            th_lf = L_mac(th_lf, im[j], im[j])
        th_hf = 0
        for j in range(64, 128):
            th_hf = L_mac(th_hf, re[j], re[j])
            th_hf = L_mac(th_hf, im[j], im[j])
        th0 = L_add(th_lf, th_hf)
        if th0 > self.th_max:
            self.th_max = L_shr(L_add(self.th_max, th0), 1)
        else:
            self.th_max = L_add(L_mpy_ls(self.th_max, 0x7EB8), L_mpy_ls(th0, 0x0148))
        M_fcn_num = L_add(th0, L_mpy_ls(self.th_max, 0x0051))
        M_fcn_den = L_add(th0, L_mpy_ls(self.th_max, 0x0148))
        if M_fcn_den == 0:
            M_fcn = 0x2000
        else:
            t = norm_l(M_fcn_den)
            M_fcn_den = L_shl(M_fcn_den, t)
            M_fcn_num = L_shl(M_fcn_num, t)
            M_fcn = div_s(extract_h(M_fcn_num), extract_h(M_fcn_den))
            L_tmp = L_add(L_shl(th_hf, 2), th_hf)
            if th_lf < L_tmp:
                t = norm_l(L_tmp)
                M_fcn_den = L_shl(L_tmp, t)
                th_lf = L_shl(th_lf, t)
                t = div_s(extract_h(th_lf), extract_h(M_fcn_den))
                L_tmp, t = sqrt_l_exp(L_deposit_h(t))
                if t:
                    L_tmp = L_shr(L_tmp, t)
                M_fcn = mult(M_fcn, extract_h(L_tmp))

        fund_fr_step = L_shl(L_mpy_ls(fund_freq, 0x6488), 2)
        uv_harms_cnt = 0
        b1_vec = 0
        band_cnt = 0
        num_harms_cnt = 0
        D_num = D_den = 0
        fund_fr_acc = 0
        acc = fund_freq
        half = L_shr(fund_freq, 1)
        acc_a = L_sub(fund_freq, half)
        acc_b = L_add(fund_freq, half)
        dsn_thr = 0
        M_num = [0] * NUM_HARMS_MAX
        M_den = [0] * NUM_HARMS_MAX
        index_a_save = index_b = 0
        v_uv = self.v_uv_dsn
        sa = self.sa
        pv = self.p_v_uv_dsn

        def decide(first, last):
            nonlocal b1_vec, uv_harms_cnt
            b1_vec = s16(b1_vec << 1)
            if D_den > D_num and D_den != 0:
                t = norm_l(D_den)
                Dk = div_s(extract_h(L_shl(D_num, t)), extract_h(L_shl(D_den, t)))
            else:
                Dk = MAX_16
            if Dk < dsn_thr:
                v_uv[band_cnt] = 1
                b1_vec |= 1
                for k in range(first, last):
                    sa[k] = self._voiced_sa_calc(M_num[k], M_den[k])
                    pv[k] = 1
            else:
                v_uv[band_cnt] = 0
                for k in range(first, last):
                    sa[k] = self._unvoiced_sa_calc(M_num[k], s16(index_b - index_a_save))
                    pv[k] = 0
                    uv_harms_cnt += 1

        for j in range(num_harms):
            ha = extract_h(acc_a)
            hb = extract_h(acc_b)
            index_a = (ha >> 8) + (1 if ha & 0xFF else 0)
            index_b = (hb >> 8) + (1 if hb & 0xFF else 0)
            L_tmp = L_shl(L_deposit_h(index_a), 8)
            L_tmp = L_sub(L_tmp, acc)
            L_tmp = L_add(L_tmp, 0x00020000)
            L_tmp = L_shr(L_tmp, 2)
            index_a_save = index_a
            if num_harms_cnt == 0:
                if self.e_p_val > 0x08CD and band_cnt >= 1:
                    dsn_thr = 0
                elif v_uv[band_cnt] == 1:
                    dsn_thr = mult(M_fcn, sub(0x4800, mult(0x164A, extract_h(fund_fr_acc))))
                else:
                    dsn_thr = mult(M_fcn, sub(0x3999, mult(0x11D5, extract_h(fund_fr_acc))))
                fund_fr_acc = L_add(fund_fr_acc, fund_fr_step)
            M_den_sum = 0
            amp_re = amp_im = 0
            tbl = []
            ia = index_a
            while ia < index_b:
                iw = extract_h(L_tmp)
                if iw < 0 and (L_tmp & 0xFFFF):
                    iw = add(iw, 1)
                iw = add(iw, 160)
                tbl.append(iw)
                if 0 <= iw <= 320:
                    w = wr_sp[iw]
                    amp_re = L_mac(amp_re, re[ia], w)
                    amp_im = L_mac(amp_im, im[ia], w)
                    M_den_sum = L_add(M_den_sum, mult(w, w))
                ia += 1
                L_tmp = L_add(L_tmp, 0x400000)
            sc_coef = div_s(0x4000, extract_l(L_shr(M_den_sum, 1)))
            im2 = mult(extract_h(amp_im), sc_coef)
            re2 = mult(extract_h(amp_re), sc_coef)
            M_num_sum = 0
            ia = index_a
            for iw in tbl:
                if iw < 0 or iw > 320:
                    srr = sri = 0
                else:
                    sri = mult(im2, wr_sp[iw])
                    srr = mult(re2, wr_sp[iw])
                rt = sub(re[ia], srr)
                it = sub(im[ia], sri)
                D_num = L_mac(D_num, rt, rt)
                D_num = L_mac(D_num, it, it)
                M_num_sum = L_mac(M_num_sum, re[ia], re[ia])
                M_num_sum = L_mac(M_num_sum, im[ia], im[ia])
                ia += 1
            M_den[j] = sc_coef
            M_num[j] = M_num_sum
            D_den = L_add(D_den, M_num_sum)
            num_harms_cnt += 1
            if num_harms_cnt == 3 and band_cnt < num_bands - 1:
                decide(j - 2, j + 1)
                D_num = D_den = 0
                num_harms_cnt = 0
                band_cnt += 1
            acc_a = L_add(acc_a, fund_freq)
            acc_b = L_add(acc_b, fund_freq)
            acc = L_add(acc, fund_freq)

        if num_harms_cnt:
            decide(num_harms - num_harms_cnt, num_harms)

        self.l_uv = uv_harms_cnt
        self.b_vec[1] = b1_vec
        self.b_vec[0] = shr(sub(self.ref_pitch, 0x1380), 7)

    # -- spectral amplitudes ---------------------------------------------

    @staticmethod
    def _dct(inp, m_lim, i_lim):
        if m_lim == 1:
            intl, intl2 = CNST_0_5_Q1_15, CNST_1_0_Q1_15
        else:
            intl = div_s(CNST_0_5_Q5_11, m_lim << 11)
            intl2 = shl(intl, 1)
        intl &= 0xFFFF
        intl2 &= 0xFFFF
        out = [0] * i_lim
        total = 0
        for m in range(m_lim):
            total = L_add(total, inp[m])
        out[0] = extract_l(L_mpy_ls(total, s16(intl2)))
        begin = intl
        step = intl2
        for i in range(1, i_lim):
            total = 0
            a = begin
            for m in range(m_lim):
                total = L_add(total, mult(inp[m], cos_fxp(s16(a))))
                a = (a + step) & 0xFFFF
            out[i] = extract_l(L_mpy_ls(total, s16(intl2)))
            step = (step + intl2) & 0xFFFF
            begin = (begin + intl) & 0xFFFF
        return out

    @staticmethod
    def _idct(inp, m_lim, i_lim):
        if m_lim == 1:
            intl, intl2 = CNST_0_5_Q1_15, CNST_1_0_Q1_15
        else:
            intl = div_s(CNST_0_5_Q5_11, m_lim << 11)
            intl2 = shl(intl, 1)
        intl &= 0xFFFF
        intl2 &= 0xFFFF
        out = [0] * i_lim
        step = intl
        for i in range(i_lim):
            total = 0
            a = step
            for m in range(1, m_lim):
                total = L_add(total, L_shr(L_mult(inp[m], cos_fxp(s16(a))), 7))
                a = (a + step) & 0xFFFF
            total = L_add(total, L_shr(L_deposit_h(inp[0]), 8))
            out[i] = extract_l(L_shr_r(total, 8))
            step = (step + intl2) & 0xFFFF
        return out

    def _predict(self, num_harms, k_coef, ro_coef, values, is_encode):
        """Shared interpolation of the previous frame's amplitudes."""
        prev = self.sa_prev2
        out = [0] * num_harms
        k_acc = k_coef
        total = 0
        for i in range(num_harms):
            index = (k_acc >> 24) & 0xFFFF
            si = s16(((k_acc - (index << 24)) & 0xFFFFFFFF) >> 9)
            if si == 0:
                t = L_mpy_ls(prev[index], ro_coef)
                if is_encode:
                    out[i] = L_sub(values[i], t)
                    total = L_add(total, prev[index])
                else:
                    out[i] = L_add(values[i], t)
            else:
                t = L_mpy_ls(prev[index], sub(0x7FFF, si))
                if is_encode:
                    total = L_add(total, t)
                    v = L_sub(values[i], L_mpy_ls(t, ro_coef))
                else:
                    v = L_add(values[i], L_mpy_ls(t, ro_coef))
                t = L_mpy_ls(prev[index + 1], si)
                if is_encode:
                    total = L_add(total, t)
                    v = L_sub(v, L_mpy_ls(t, ro_coef))
                else:
                    v = L_add(v, L_mpy_ls(t, ro_coef))
                out[i] = v
            k_acc = (k_acc + k_coef) & 0xFFFFFFFF
        return out, total

    def _sa_encode(self):
        num_harms = self.num_harms
        prev_n = self.num_harms_prev2
        if num_harms == prev_n:
            k_coef = CNST_ONE_Q8_24
        elif num_harms > prev_n:
            k_coef = div_s(prev_n << 9, num_harms << 9) << 9
        else:
            k_coef = 0
            t = prev_n
            while t > num_harms:
                t -= num_harms
                k_coef += CNST_ONE_Q8_24
            k_coef += div_s(t << 9, num_harms << 9) << 9
        k_coef &= 0xFFFFFFFF

        if num_harms <= 15:
            ro_coef = CNST_0_4_Q1_15
        elif num_harms <= 24:
            ro_coef = s16(num_harms * CNST_0_03_Q1_15 - CNST_0_05_Q1_15)
        else:
            ro_coef = CNST_0_7_Q1_15

        prev = self.sa_prev2
        for i in range(prev_n + 1, NUM_HARMS_MAX + 2):
            prev[i] = prev[prev_n]

        logs = [Log2(self.sa[i]) for i in range(num_harms)]
        vec32, total = self._predict(num_harms, k_coef, ro_coef, logs, True)

        sh = norm_s(num_harms)
        inv = div_s(0x4000, num_harms << sh)
        total = L_shr(L_mpy_ls(L_mpy_ls(total, ro_coef), inv), 14 - sh)
        t_vec = [extract_h(L_shl(L_add(vec32[i], total), 5)) for i in range(num_harms)]

        index = num_harms - NUM_HARMS_MIN
        _get_bit_allocation(num_harms, self.bit_alloc)
        ba = self.bit_alloc
        b_vec = self.b_vec
        item = T.LMPRBL_TBL[index]
        pos = 0
        bp = 8
        bap = 5
        gain_vec = [0] * 6
        for i in range(NUM_PRED_RES_BLKS):
            bl_len = (item >> 28) & 0xF
            item = (item << 4) & 0xFFFFFFFF
            c_vec = self._dct(t_vec[pos:pos + bl_len], bl_len, bl_len)
            gain_vec[i] = c_vec[0]
            for j in range(1, bl_len):
                nb = ba[bap]
                bap += 1
                if nb:
                    step_size = extract_h(s32(s32(T.HI_ORD_STD_TBL[j - 1] * T.HI_ORD_STEP_SIZE_TBL[nb - 1]) << 1))
                    b_vec[bp] = _qnt_by_step(c_vec[j], step_size & 0xFFFF, nb)
                else:
                    b_vec[bp] = 0
                bp += 1
            pos += bl_len

        gain_r = self._dct(gain_vec, NUM_PRED_RES_BLKS, NUM_PRED_RES_BLKS)
        gss = T.GAIN_STEP_SIZE_TBL[index * 5:index * 5 + 5]
        b_vec[2] = _tbl_quant(gain_r[0], T.GAIN_QNT_TBL, GAIN_QNT_TBL_SIZE)
        for j in range(1, 6):
            b_vec[2 + j] = _qnt_by_step(gain_r[j], gss[j - 1], ba[j - 1])

        gain_vec[0] = T.GAIN_QNT_TBL[b_vec[2]]
        for i in range(1, 6):
            gain_vec[i] = extract_l(L_shr(_deqnt_by_step(b_vec[2 + i], gss[i - 1], ba[i - 1]), 5))
        gain_r = self._idct(gain_vec, NUM_PRED_RES_BLKS, NUM_PRED_RES_BLKS)

        t_vec = [0] * NUM_HARMS_MAX
        item = T.LMPRBL_TBL[index]
        pos = 0
        bp = 8
        bap = 5
        for i in range(NUM_PRED_RES_BLKS):
            bl_len = (item >> 28) & 0xF
            item = (item << 4) & 0xFFFFFFFF
            c_vec = [0] * MAX_BLOCK_LEN
            c_vec[0] = gain_r[i]
            for j in range(1, bl_len):
                nb = ba[bap]
                bap += 1
                if nb:
                    step_size = extract_h(s32(s32(T.HI_ORD_STD_TBL[j - 1] * T.HI_ORD_STEP_SIZE_TBL[nb - 1]) << 1))
                    c_vec[j] = extract_l(L_shr(_deqnt_by_step(b_vec[bp], step_size & 0xFFFF, nb), 5))
                else:
                    c_vec[j] = 0
                bp += 1
            t_vec[pos:pos + bl_len] = self._idct(c_vec, bl_len, bl_len)
            pos += bl_len

        for i in range(prev_n + 1, NUM_HARMS_MAX + 2):
            prev[i] = prev[prev_n]
        recon = [L_shr(L_deposit_h(t_vec[i]), 5) for i in range(num_harms)]
        vec32, _ = self._predict(num_harms, k_coef, ro_coef, recon, False)
        for i in range(1, num_harms + 1):
            prev[i] = L_sub(vec32[i - 1], total)
        self.num_harms_prev2 = num_harms

    # -- bit packing -----------------------------------------------------

    def _encode_frame_vector(self):
        num_harms = self.num_harms
        num_bands = self.num_bands
        fv = [0] * 8
        _get_bit_allocation(num_harms, self.bit_alloc)
        ba = self.bit_alloc
        b_vec = self.b_vec
        bits = [0] * (EN_BIT_STREAM_LEN + 64)
        idx = 0
        bit_thr = 9 if num_harms == 0xB else ba[0]
        bit_mask = shl(1, bit_thr - 1)
        while idx < EN_BIT_STREAM_LEN - (num_bands - 3):
            for i in range(num_harms - 1):
                if bit_thr and bit_thr <= ba[i]:
                    bits[idx] = 1 if (b_vec[3 + i] & bit_mask) else 0
                    idx += 1
            bit_thr -= 1
            bit_mask = shr(bit_mask, 1)

        fv[0] = shl(b_vec[0] & 0xFC, 4) | (b_vec[2] & 0x38)
        idx = 0
        fv[0] |= 4 if bits[0] else 0
        fv[0] |= 2 if bits[1] else 0
        fv[0] |= 1 if bits[2] else 0
        idx = 3
        for v in range(1, 4):
            t = 0
            for _ in range(12):
                t = (t << 1) | bits[idx]
                idx += 1
            fv[v] = s16(t)
        idx -= num_bands + 2
        bit_mask = shl(1, num_bands - 1)
        for _ in range(num_bands):
            bits[idx] = 1 if (b_vec[1] & bit_mask) else 0
            idx += 1
            bit_mask >>= 1
        bits[idx] = 1 if (b_vec[2] & 0x04) else 0
        idx += 1
        bits[idx] = 1 if (b_vec[2] & 0x02) else 0
        idx += 1
        idx -= num_bands + 2
        for v in range(4, 7):
            t = 0
            for _ in range(11):
                t = (t << 1) | bits[idx]
                idx += 1
            fv[v] = s16(t)
        fv[7] = shl(b_vec[0] & 0x03, 1) | shl(b_vec[2] & 0x01, 3)
        fv[7] |= 0x40 if bits[idx] else 0
        fv[7] |= 0x20 if bits[idx + 1] else 0
        fv[7] |= 0x10 if bits[idx + 2] else 0
        fv[7] |= 0x01 if b_vec[num_harms + 2] else 0
        return fv


def pack_frame_vector(fv: Sequence[int]) -> bytes:
    """MBEEncoder::encode: u0-u3 12 bits, u4-u6 11 bits, u7 7 bits, MSB first."""
    widths = (12, 12, 12, 12, 11, 11, 11, 7)
    acc = 0
    for value, width in zip(fv, widths):
        acc = (acc << width) | (value & ((1 << width) - 1))
    return acc.to_bytes(11, 'big')


def _choose_encoder():
    """The blip25 encoder (reference-exact, much faster) if it has been
    built, else this pure Python one."""
    if os.environ.get('CCP_PURE_PYTHON_IMBE', '').strip() not in ('', '0'):
        return PyIMBEEncoder
    from src.p25 import blip25
    if blip25.available():
        return blip25.Blip25Encoder
    logger.info(f"Using the pure Python IMBE encoder ({blip25.load_error()})")
    return PyIMBEEncoder


IMBEEncoder = _choose_encoder()


# A silent frame, for padding and as a fallback if the encoder ever fails.
_silence_cache: List[bytes] = []


def silence_frame() -> bytes:
    if not _silence_cache:
        enc = IMBEEncoder()
        frame = b''
        for _ in range(4):
            frame = enc.encode([0] * FRAME)
        _silence_cache.append(frame)
    return _silence_cache[0]
