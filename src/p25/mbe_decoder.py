"""IMBE (P25 Phase 1) voice decoder - a port of mbelib's 4400 bps decoder.

dvmhost/dvmbridge decode P25 network audio with mbelib
(mbe_processImbe4400DataF). It was written independently of the OP25
encoder, so decoding our encoder's frames with it is a real check that the
bits mean what we think they mean. It also lets the dashboard preview how a
page will actually sound on a radio.

Float arithmetic (numpy float64 rather than C float), so output is close to,
not bit-identical with, mbelib - fine for both uses.

IMBEDecoder below is the blip25 decoder (src/p25/blip25.py, bit-exact with
the DVSI reference vocoder) when it has been built with
python tools/build_native.py, and PyIMBEDecoder otherwise. Decoders say how
to scale their output to int16 PCM in `gain`.
"""
import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.p25 import mbe_tables as M

N = 160
WS = np.array(M.WS, dtype=np.float64)


@dataclass
class MBEParms:
    w0: float = 0.09378
    L: int = 30
    K: int = 10
    Vl: List[int] = field(default_factory=lambda: [0] * 57)
    Ml: List[float] = field(default_factory=lambda: [0.0] * 57)
    log2Ml: List[float] = field(default_factory=lambda: [0.0] * 57)
    PHIl: List[float] = field(default_factory=lambda: [0.0] * 57)
    PSIl: List[float] = field(default_factory=lambda: [math.pi / 2] * 57)
    repeat: int = 0

    def copy_from(self, other: 'MBEParms') -> None:
        self.w0, self.L, self.K, self.repeat = other.w0, other.L, other.K, other.repeat
        self.Vl = list(other.Vl)
        self.Ml = list(other.Ml)
        self.Ml[0] = 0.0
        self.log2Ml = list(other.log2Ml)
        self.PHIl = list(other.PHIl)
        self.PSIl = list(other.PSIl)


def _bits(frame: bytes) -> List[int]:
    return [(frame[i >> 3] >> (7 - (i & 7))) & 1 for i in range(88)]


def _bin(bits) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v


class PyIMBEDecoder:
    native = False
    gain = 7.0                  # mbelib's own float-to-short scaling

    def __init__(self, uvquality: int = 3, seed: Optional[int] = None):
        self.uvquality = uvquality
        self.rng = random.Random(seed)
        self.cur = MBEParms()
        self.prev = MBEParms()
        self.prev_enh = MBEParms()
        self._init_parms()

    def _init_parms(self):
        self.prev = MBEParms()
        self.cur = MBEParms()
        self.cur.copy_from(self.prev)
        self.prev_enh = MBEParms()
        self.prev_enh.copy_from(self.prev)

    # -- parameter decoding (mbe_decodeImbe4400Parms) ---------------------

    def decode_params(self, frame: bytes) -> bool:
        """Fills self.cur from an 88-bit frame. False if the frame is invalid."""
        d = _bits(frame)
        cur, prev = self.cur, self.prev
        cur.repeat = prev.repeat
        b0 = _bin(d[0:6] + d[85:87])
        if b0 > 207:
            return False
        cur.w0 = (4 * math.pi) / (b0 + 39.5)
        L = int(0.9254 * int((math.pi / cur.w0) + 0.25))
        if L > 56 or L < 9:
            return False
        cur.L = L
        L9 = L - 9
        K = int((L + 2) / 3) if L < 37 else 12
        cur.K = K

        bb = [[0] * 12 for _ in range(58)]
        for i, (r, c) in zip(range(6, 85), M.BO[L9]):
            bb[r][c] = d[i]

        j, k = 1, K - 1
        for i in range(1, L + 1):
            cur.Vl[i] = bb[1][k]
            if j == 3:
                j = 1
                k = k - 1 if k > 0 else 0
            else:
                j += 1

        b2 = _bin([bb[2][5], bb[2][4], bb[2][3], bb[2][2], bb[2][1], bb[2][0]])
        Gm = [0.0] * 7
        Gm[1] = M.B2[b2]
        for i in range(2, 7):
            nbits, step = M.BA[L9][i - 2]
            nbits = int(nbits)
            bm = _bin([bb[i + 1][jj] for jj in range(nbits - 1, -1, -1)])
            Gm[i] = step * (bm - 2 ** (nbits - 1) + 0.5)

        Ri = [0.0] * 7
        for i in range(1, 7):
            s = 0.0
            for m in range(1, 7):
                am = 1 if m == 1 else 2
                s += am * Gm[m] * math.cos(math.pi * (m - 1) * (i - 0.5) / 6)
            Ri[i] = s

        Cik = [[0.0] * 11 for _ in range(7)]
        m = 8
        for i in range(1, 7):
            Cik[i][1] = Ri[i]
            for k in range(2, M.IMBEJI[L9][i - 1] + 1):
                Bm = M.HOBA[L9][m - 8]
                if Bm == 0:
                    Cik[i][k] = 0.0
                else:
                    bm = _bin([bb[m][Bm - b - 1] for b in range(Bm)])
                    Cik[i][k] = (M.QUANTSTEP[Bm - 1] * M.STANDDEV[k - 2]) * ((bm - 2 ** (Bm - 1)) + 0.5)
                m += 1

        Tl = [0.0] * 57
        l = 1
        for i in range(1, 7):
            ji = M.IMBEJI[L9][i - 1]
            for jj in range(1, ji + 1):
                s = 0.0
                for k in range(1, ji + 1):
                    ak = 1 if k == 1 else 2
                    s += ak * Cik[i][k] * math.cos(math.pi * (k - 1) * (jj - 0.5) / ji)
                Tl[l] = s
                l += 1

        if L <= 15:
            rho = 0.4
        elif L <= 24:
            rho = 0.03 * L - 0.05
        else:
            rho = 0.7
        if L > prev.L:
            for l in range(prev.L + 1, L + 1):
                prev.Ml[l] = prev.Ml[prev.L]
                prev.log2Ml[l] = prev.log2Ml[prev.L]
        intkl = [0] * 57
        deltal = [0.0] * 57
        # mbelib reads log2Ml[intkl + 1], one past the end when the previous
        # frame had 56 harmonics; its weight is then exactly 0.
        plog = prev.log2Ml + [0.0]
        s77 = 0.0
        for l in range(1, L + 1):
            fl = (prev.L / L) * l
            intkl[l] = int(fl)
            deltal[l] = fl - intkl[l]
            s77 += (1 - deltal[l]) * plog[intkl[l]] + deltal[l] * plog[intkl[l] + 1]
        s77 = (rho / L) * s77
        for l in range(1, L + 1):
            c1 = rho * (1 - deltal[l]) * plog[intkl[l]]
            c2 = rho * deltal[l] * plog[intkl[l] + 1]
            cur.log2Ml[l] = Tl[l] + c1 + c2 - s77
            cur.Ml[l] = 2.0 ** cur.log2Ml[l]
        return True

    # -- synthesis (mbe_processImbe4400DataF) -------------------------------

    def decode(self, frame: bytes) -> np.ndarray:
        """One 11-byte frame -> 160 float samples (mbelib scale)."""
        if not self.decode_params(frame):
            self.cur.copy_from(self.prev)
            self.cur.repeat += 1
        else:
            self.cur.repeat = 0
        if self.cur.repeat <= 3:
            self.prev.copy_from(self.cur)
            self._enhance(self.cur)
            out = self._synthesize(self.cur, self.prev_enh)
            self.prev_enh.copy_from(self.cur)
            return out
        self._init_parms()
        return np.zeros(N)

    @staticmethod
    def _enhance(mp: MBEParms) -> None:
        L = mp.L
        Ml = mp.Ml
        Rm0 = sum(Ml[l] * Ml[l] for l in range(1, L + 1))
        Rm1 = sum(Ml[l] * Ml[l] * math.cos(mp.w0 * l) for l in range(1, L + 1))
        R2m0, R2m1 = Rm0 * Rm0, Rm1 * Rm1
        for l in range(1, L + 1):
            if Ml[l] != 0:
                den = mp.w0 * Rm0 * (R2m0 - R2m1)
                num = 0.96 * math.pi * ((R2m0 + R2m1) - 2 * Rm0 * Rm1 * math.cos(mp.w0 * l))
                ratio = num / den if den else 0.0
                Wl = math.sqrt(Ml[l]) * (ratio ** 0.25 if ratio > 0 else 0.0)
                if 8 * l <= L:
                    pass
                elif Wl > 1.2:
                    Ml[l] *= 1.2
                elif Wl < 0.5:
                    Ml[l] *= 0.5
                else:
                    Ml[l] *= Wl
        total = sum(Ml[l] * Ml[l] for l in range(1, L + 1))
        gamma = 1.0 if total == 0 else math.sqrt(Rm0 / total)
        for l in range(1, L + 1):
            Ml[l] *= gamma

    def _rand(self, size=None):
        if size is None:
            return self.rng.random()
        return np.array([self.rng.random() for _ in range(size)])

    def _rand_phase(self):
        return self.rng.random() * (2 * math.pi) - math.pi

    def _synthesize(self, cur: MBEParms, prev: MBEParms) -> np.ndarray:
        q = self.uvquality
        uvthreshold = (2700.0 * math.pi) / 4000
        uvsine = 1.3591409 * math.e
        uvrand = 2.0
        qfactor = (1 / math.e) if q == 1 else math.log(q) / q
        uvstep = 1.0 / q
        uvoffset = (uvstep * (q - 1)) / 2
        num_uv = sum(1 for l in range(1, cur.L + 1) if cur.Vl[l] == 0)
        cw0, pw0 = cur.w0, prev.w0
        out = np.zeros(N)
        n = np.arange(N, dtype=np.float64)
        ws_cur, ws_prev = WS[0:N], WS[N:2 * N]

        if cur.L > prev.L:
            maxl = cur.L
            for l in range(prev.L + 1, maxl + 1):
                prev.Ml[l] = 0.0
                prev.Vl[l] = 1
        else:
            maxl = prev.L
            for l in range(cur.L + 1, maxl + 1):
                cur.Ml[l] = 0.0
                cur.Vl[l] = 1

        for l in range(1, 57):
            cur.PSIl[l] = prev.PSIl[l] + (pw0 + cw0) * (l * N / 2)
            if l <= int(cur.L / 4):
                cur.PHIl[l] = cur.PSIl[l]
            else:
                cur.PHIl[l] = cur.PSIl[l] + (num_uv * self._rand_phase()) / cur.L

        def unvoiced(w0, l, phases):
            c = np.zeros(N)
            w0l = w0 * l
            for i in range(q):
                c += np.cos(w0 * n * (l + i * uvstep - uvoffset) + phases[i])
                if w0l > uvthreshold:
                    c += (w0l - uvthreshold) * uvrand * self._rand(N)
            return c * uvsine * qfactor

        for l in range(1, maxl + 1):
            cw0l, pw0l = cw0 * l, pw0 * l
            cv, pv = cur.Vl[l], prev.Vl[l]
            if cv == 0 and pv == 1:
                ph = [self._rand_phase() for _ in range(q)]
                out += ws_prev * prev.Ml[l] * np.cos(pw0l * n + prev.PHIl[l])
                out += unvoiced(cw0, l, ph) * ws_cur * cur.Ml[l]
            elif cv == 1 and pv == 0:
                ph = [self._rand_phase() for _ in range(q)]
                out += ws_cur * cur.Ml[l] * np.cos(cw0l * (n - N) + cur.PHIl[l])
                out += unvoiced(pw0, l, ph) * ws_prev * prev.Ml[l]
            elif cv == 1 or pv == 1:
                out += ws_prev * prev.Ml[l] * np.cos(pw0l * n + prev.PHIl[l])
                out += ws_cur * cur.Ml[l] * np.cos(cw0l * (n - N) + cur.PHIl[l])
            else:
                ph = [self._rand_phase() for _ in range(q)]
                ph2 = [self._rand_phase() for _ in range(q)]
                out += unvoiced(pw0, l, ph) * ws_prev * prev.Ml[l]
                out += unvoiced(cw0, l, ph2) * ws_cur * cur.Ml[l]
        return out


def _choose_decoder():
    if os.environ.get('CCP_PURE_PYTHON_IMBE', '').strip() not in ('', '0'):
        return PyIMBEDecoder
    from src.p25 import blip25
    return blip25.Blip25Decoder if blip25.available() else PyIMBEDecoder


IMBEDecoder = _choose_decoder()


def decode_to_pcm(frames) -> np.ndarray:
    """Decode a sequence of IMBE frames to int16 PCM at 8 kHz."""
    dec = IMBEDecoder(seed=1)
    pcm = np.concatenate([dec.decode(f) for f in frames]) if frames else np.zeros(0)
    return np.clip(np.round(pcm * dec.gain), -32768, 32767).astype(np.int16)
