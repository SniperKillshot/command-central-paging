"""Transmit 8 kHz PCM as a P25 group voice call through the DVM FNE peer.

Encoding runs on its own thread while LDUs go out every 180 ms. The pure
Python vocoder may be slower than real time on small hosts, so transmission
starts only once the encoder is far enough ahead that it can't run dry -
on a fast host that is almost immediately, on a slow one it is a short
pre-buffer delay instead of a stuttering call.
"""
import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.dvm import protocol as dvm
from src.p25 import frames as F
from src.p25.crypto import ALGO_NAMES, CallEncryptor, build_ldu
from src.p25.imbe import FRAME, IMBEEncoder, VocoderError, silence_frame

logger = logging.getLogger(__name__)

LDU_SECONDS = 0.18
FRAMES_PER_LDU = 9
MIN_PREBUFFER_FRAMES = 18


@dataclass
class TxResult:
    ok: bool
    status: str            # 'sent', 'failed'
    detail: str
    seconds: float = 0.0
    ldus: int = 0


def pcm_to_frames(pcm: np.ndarray, tones: Optional[np.ndarray] = None) -> List[Tuple[List[int], Optional[np.ndarray]]]:
    """Split int16 PCM into 160-sample frames, zero-padded to whole superframes
    (LDU1 + LDU2 = 18 frames) so the call ends on an LDU2 boundary. Each frame
    comes with its slice of the tone track (or None)."""
    pcm = np.asarray(pcm, dtype=np.int16)
    n = -(-pcm.size // FRAME)
    n = -(-n // 18) * 18
    padded = np.zeros(n * FRAME, dtype=np.int16)
    padded[:pcm.size] = pcm
    track = None
    if tones is not None and np.any(np.isfinite(tones)):
        track = np.full((n * FRAME, 2), np.nan)
        track[:min(len(tones), n * FRAME)] = tones[:n * FRAME]
    return [(padded[i * FRAME:(i + 1) * FRAME].tolist(),
             None if track is None else track[i * FRAME:(i + 1) * FRAME]) for i in range(n)]


class _EncoderThread(threading.Thread):
    def __init__(self, pcm_frames):
        super().__init__(name='imbe-encoder', daemon=True)
        self.pcm_frames = pcm_frames
        self.out: List[bytes] = []
        self.cond = threading.Condition()
        self.started_at = time.monotonic()
        self.stop_flag = False
        self.errors = 0

    def run(self):
        enc = IMBEEncoder()
        for samples, tones in self.pcm_frames:
            if self.stop_flag:
                break
            try:
                frame = enc.encode(samples, tones)
            except VocoderError:
                self.errors += 1
                enc = IMBEEncoder()          # the C code would abort; restart cleanly
                frame = silence_frame()
            with self.cond:
                self.out.append(frame)
                self.cond.notify_all()

    def rate(self) -> Optional[float]:
        done = len(self.out)
        elapsed = time.monotonic() - self.started_at
        return done / elapsed if done >= 9 and elapsed > 0 else None

    def wait_for(self, count: int, timeout: float) -> bool:
        with self.cond:
            return self.cond.wait_for(lambda: len(self.out) >= count or not self.is_alive(), timeout)


def encode_pcm(pcm: np.ndarray, tones: Optional[np.ndarray] = None) -> List[bytes]:
    """Encode a whole clip synchronously (previews, tests)."""
    t = _EncoderThread(pcm_to_frames(pcm, tones))
    t.run()
    return t.out


class VoiceCall:
    def __init__(self, peer, params: F.CallParams, control_byte: int = 0,
                 max_seconds: float = 120.0, encryptor: Optional[CallEncryptor] = None):
        """encryptor: encrypt the call with that key (None = clear)."""
        self.peer = peer
        self.params = params
        self.encryptor = encryptor
        self.control_byte = control_byte
        self.max_seconds = max_seconds

    def transmit(self, pcm: np.ndarray, tones: Optional[np.ndarray] = None) -> TxResult:
        frames = pcm_to_frames(pcm, tones)
        total = len(frames)
        if total / 50.0 > self.max_seconds:
            keep = int(self.max_seconds * 50) // 18 * 18
            logger.warning(f"P25 call trimmed from {total / 50:.0f}s to the {self.max_seconds:.0f}s limit")
            frames = frames[:keep]
            total = keep
        if not self.peer.connected:
            return TxResult(False, 'failed', self._not_connected())

        enc = _EncoderThread(frames)
        enc.start()
        try:
            self._prebuffer(enc, total)
            return self._send(enc, total)
        finally:
            enc.stop_flag = True

    def _not_connected(self) -> str:
        reason = self.peer.status.get('last_error') or self.peer.state.value.replace('_', ' ')
        return f'not connected to the FNE ({reason})'

    def _prebuffer(self, enc: _EncoderThread, total: int) -> None:
        """Wait until the encoder will stay ahead of 50 frames/s playback."""
        enc.wait_for(min(total, MIN_PREBUFFER_FRAMES), timeout=30)
        while enc.is_alive():
            r = enc.rate()
            done = len(enc.out)
            if r is None:
                enc.wait_for(done + 1, 1)
                continue
            if r >= 50 * 1.15:
                return
            # time left for the encoder vs time left for playback, 15% margin
            if (total - done) / r <= (total / 50.0) * 0.85:
                return
            enc.wait_for(done + 9, 1)

    def _send(self, enc: _EncoderThread, total: int) -> TxResult:
        p = self.params
        stream_id = self.peer.new_stream_id()
        n_ldus = total // FRAMES_PER_LDU
        t0 = time.monotonic()
        underruns = 0
        secure = f', {ALGO_NAMES[self.encryptor.algo]} key 0x{self.encryptor.kid:X}' if self.encryptor else ''
        logger.info(f"P25 call start: TG {p.dst_id} from RID {p.src_id}, {total / 50:.1f}s, stream {stream_id}{secure}")
        try:
            for k in range(n_ldus):
                need = (k + 1) * FRAMES_PER_LDU
                if len(enc.out) < need:
                    underruns += 1
                    enc.wait_for(need, timeout=10)
                    if len(enc.out) < need:
                        return self._end(stream_id, False, 'failed', 'voice encoder stalled', t0, k)
                imbe = enc.out[k * FRAMES_PER_LDU:need]
                pkt = build_ldu(p, k, imbe, self.control_byte, self.encryptor)

                delay = t0 + k * LDU_SECONDS - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                if not self.peer.connected:
                    return self._end(stream_id, False, 'failed', 'FNE connection lost mid-call', t0, k, tdu=False)
                refusal = self.peer.call_refusal(stream_id)
                if refusal:
                    return self._end(stream_id, False, 'failed', f'FNE {refusal}', t0, k)
                if not self.peer.send(dvm.Func.PROTOCOL, dvm.PROTOCOL_P25, pkt, stream_id, k):
                    return self._end(stream_id, False, 'failed', 'network send failed', t0, k)

            # let the last LDU play out before the terminator
            delay = t0 + n_ldus * LDU_SECONDS - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            time.sleep(0.05)     # a reject for the final LDUs can still arrive
            refusal = self.peer.call_refusal(stream_id)
            if refusal:
                return self._end(stream_id, False, 'failed', f'FNE {refusal}', t0, n_ldus)
            note = f' ({underruns} encoder underrun(s))' if underruns else ''
            if enc.errors:
                note += f' ({enc.errors} frame(s) replaced with silence)'
            return self._end(stream_id, True, 'sent',
                             f'{n_ldus * LDU_SECONDS:.1f}s voice call to TG {p.dst_id}{note}', t0, n_ldus)
        finally:
            self.peer.forget_call(stream_id)

    def _end(self, stream_id, ok, status, detail, t0, ldus, tdu=True) -> TxResult:
        if tdu and self.peer.connected:
            self.peer.send(dvm.Func.PROTOCOL, dvm.PROTOCOL_P25, F.tdu(self.params, 0),
                           stream_id, dvm.END_OF_CALL_SEQ)
        secs = time.monotonic() - t0
        (logger.info if ok else logger.warning)(f"P25 call end: TG {self.params.dst_id}: {detail}")
        return TxResult(ok, status, detail, secs, ldus)
