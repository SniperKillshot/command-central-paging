"""Decode P25 calls heard from the FNE and fan the audio out to listeners.

The FNE peer hands every P25 network frame to `RxAudio.on_p25`. While at
least one dispatch console is listening, LDU1/LDU2 frames are queued to a
worker thread that IMBE-decodes them (one decoder per call) into 180 ms
chunks of 8 kHz audio. Each listener gets its own bounded queue, so a slow
browser drops audio instead of holding anything up.

Encrypted calls are decrypted when a configured key matches their algorithm
and key ID (src/p25/crypto.py); without one they are reported, not played.

Listener stream format (all little-endian), one message after another:
    u8 type   1 = audio, 2 = call ended, 3 = encrypted call (no key), 0 = keepalive
    u8 flags  audio: bit 0 = decrypted (the call is encrypted on air)
    u16 talkgroup
    u32 source radio ID
    u32 stream ID
    u16 sample count, then that many int16 samples at 8 kHz (audio only)
"""
import logging
import queue
import struct
import threading
from typing import Dict, Optional

import numpy as np

from src.p25 import frames as F
from src.p25.crypto import ALGO_NAMES, CallDecryptor, KeyStore, ldu1_header_sync, ldu2_sync
from src.p25.mbe_decoder import IMBEDecoder

logger = logging.getLogger(__name__)

MSG_KEEPALIVE, MSG_AUDIO, MSG_END, MSG_ENCRYPTED = 0, 1, 2, 3
HEADER = struct.Struct('<BBHIIH')
DUID_LDU1, DUID_LDU2 = 0x05, 0x0A
DUID_TDU, DUID_TDULC = 0x03, 0x0F
LISTENER_QUEUE = 100            # messages (~18 s of one call) before dropping
LDU_VOICE_END = F.LDU1_LAYOUT[-1][1] + F.LDU1_LAYOUT[-1][2]    # last voice frame's end
FLAG_DECRYPTED = 0x01


def message(kind: int, tgid: int, src: int, stream: int, pcm: Optional[np.ndarray] = None,
            flags: int = 0) -> bytes:
    n = 0 if pcm is None else pcm.size
    head = HEADER.pack(kind, flags, tgid & 0xFFFF, src & 0xFFFFFFFF, stream & 0xFFFFFFFF, n)
    return head if pcm is None else head + pcm.astype('<i2').tobytes()


def is_encrypted(payload: bytes) -> bool:
    """From an LDU1: the link control's service options (encrypted bit), or
    the algorithm ID dvmhost appends when the frame type says it is valid."""
    if len(payload) >= 182 and payload[180] == F.FRAME_TYPE_HDU_VALID and payload[181] not in (0, F.ALGO_UNENCRYPT):
        return True
    _, off, _ = F.LDU1_LAYOUT[2]
    return len(payload) > off + 3 and bool(payload[off + 3] & 0x40)


class _Call:
    def __init__(self, tgid, src):
        self.tgid = tgid
        self.src = src
        self.decoder = IMBEDecoder()
        self.encrypted = False          # encrypted and no key: not played
        self.decryptor: Optional[CallDecryptor] = None


def _clear(algo: int) -> bool:
    return algo in (0, F.ALGO_UNENCRYPT)


class RxAudio:
    def __init__(self, keys: Optional[KeyStore] = None):
        self.keys = keys if keys is not None else KeyStore()
        self._listeners: Dict[int, queue.Queue] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._work: queue.Queue = queue.Queue(maxsize=500)
        self._calls: Dict[int, _Call] = {}
        self._thread = threading.Thread(target=self._run, name='rx-audio', daemon=True)
        self._thread.start()

    # -- listeners ----------------------------------------------------------

    def listen(self):
        """Register a listener; returns (id, queue of message bytes)."""
        q: queue.Queue = queue.Queue(maxsize=LISTENER_QUEUE)
        with self._lock:
            self._next_id += 1
            self._listeners[self._next_id] = q
            return self._next_id, q

    def unlisten(self, listener_id: int) -> None:
        with self._lock:
            self._listeners.pop(listener_id, None)

    @property
    def listening(self) -> int:
        return len(self._listeners)

    def _publish(self, data: bytes) -> None:
        with self._lock:
            targets = list(self._listeners.values())
        for q in targets:
            try:
                q.put_nowait(data)
            except queue.Full:
                pass            # that listener is behind; it loses this chunk

    # -- frames from the FNE --------------------------------------------------

    def on_p25(self, stream_id: int, hdr, payload: bytes) -> None:
        """Called on the FNE peer's thread for every P25 frame; never blocks."""
        if not self._listeners and stream_id not in self._calls:
            return
        if hdr.duid in (DUID_LDU1, DUID_LDU2, DUID_TDU, DUID_TDULC):
            try:
                self._work.put_nowait((stream_id, hdr, bytes(payload)))
            except queue.Full:
                pass

    def _run(self) -> None:
        while True:
            stream_id, hdr, payload = self._work.get()
            try:
                self._handle(stream_id, hdr, payload)
            except Exception:
                logger.exception('RX audio decode failed')
                self._calls.pop(stream_id, None)

    def _handle(self, stream_id, hdr, payload) -> None:
        call = self._calls.get(stream_id)
        if hdr.duid in (DUID_TDU, DUID_TDULC):
            if call:
                del self._calls[stream_id]
                self._publish(message(MSG_END, call.tgid, call.src, stream_id))
            return
        if call is None:
            if not self._listeners:
                return
            call = self._calls[stream_id] = _Call(hdr.dst_id, hdr.src_id)
            if len(self._calls) > 32:                     # calls that never sent a TDU
                self._calls.pop(next(iter(self._calls)))
        if call.encrypted or len(payload) < LDU_VOICE_END:
            return
        ldu1 = hdr.duid == DUID_LDU1
        sync = ldu1_header_sync(payload) if ldu1 else ldu2_sync(payload)
        encrypted = is_encrypted(payload) if ldu1 else bool(sync and not _clear(sync[0]))
        if encrypted and call.decryptor is None:
            if not sync or _clear(sync[0]):
                return                  # joined late: algorithm and key ID come with the next LDU2
            algo, kid, _ = sync
            key = self.keys.key(algo, kid)
            if key is None:
                call.encrypted = True
                self._publish(message(MSG_ENCRYPTED, call.tgid, call.src, stream_id))
                return
            call.decryptor = CallDecryptor(algo, kid, key)
            logger.info(f"RX: TG {call.tgid} from {call.src} is {ALGO_NAMES.get(algo, hex(algo))} "
                        f"key 0x{kid:X} - decrypting")
        imbe = F.ldu_imbe(payload)
        if call.decryptor is not None:
            imbe = call.decryptor.ldu1(payload, imbe) if ldu1 else call.decryptor.ldu2(payload, imbe)
            if imbe is None:
                return                  # not in sync yet (waiting for an MI)
        pcm = np.concatenate([call.decoder.decode(f) for f in imbe])
        pcm = np.clip(np.round(pcm * call.decoder.gain), -32768, 32767)
        self._publish(message(MSG_AUDIO, call.tgid, call.src, stream_id, pcm,
                              FLAG_DECRYPTED if call.decryptor else 0))
