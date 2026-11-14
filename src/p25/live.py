"""Live talk: microphone audio streamed from the dispatch console, encoded and
sent as P25 voice calls while the talk button is held.

The browser posts 8 kHz 16-bit audio every ~100 ms. An encoder thread turns
it into IMBE frames as it arrives, and a sender thread puts out one LDU per
talkgroup every 180 ms, after a short jitter buffer. If audio arrives late
the gap is covered with silence (the call must keep its 180 ms cadence), and
if a backlog builds up the oldest audio is dropped so the delay doesn't keep
growing. Releasing the button (or the browser going quiet) ends the calls
with a terminator.

One live talk at a time: the console is one operator's microphone.
"""
import logging
import secrets
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.audio import dsp
from src.dvm import protocol as dvm
from src.p25 import frames as F
from src.p25.imbe import FRAME, IMBEEncoder, VocoderError, silence_frame
from src.p25.voice import FRAMES_PER_LDU, LDU_SECONDS, encode_pcm
from src.state import record_dispatch

logger = logging.getLogger(__name__)

PREBUFFER_FRAMES = 18       # 360 ms jitter buffer before the first LDU
MAX_BACKLOG_FRAMES = 40     # beyond this, drop old audio back to the jitter buffer
LATE_GRACE = 0.06           # wait this long past an LDU's time for audio before using silence
IDLE_TIMEOUT = 2.5          # no audio from the browser for this long: it's gone, end the call
LEAD_SILENCE_FRAMES = 9     # one LDU of silence first, so radios unmute before the speech


class LiveError(Exception):
    pass


class LiveSession:
    def __init__(self, peer, targets: List[Dict], max_seconds: float, lead_frames: List[bytes], sound: Optional[str]):
        """targets: [{'tgid', 'name', 'params': CallParams}]"""
        self.id = secrets.token_hex(8)
        self.peer = peer
        self.targets = [{**t, 'stream': peer.new_stream_id(), 'refused': None} for t in targets]
        self.max_seconds = max_seconds
        self.sound = sound
        self.started = datetime.now().isoformat(timespec='seconds')
        self.state = 'tone' if lead_frames else 'live'
        self.error: Optional[str] = None
        self.underruns = 0
        self.dropped = 0
        self.ldus = 0

        self._pcm: deque = deque()            # int16 chunks from the browser
        self._pcm_lock = threading.Lock()
        self._frames: deque = deque([silence_frame()] * LEAD_SILENCE_FRAMES + list(lead_frames))
        self._lead_left = len(self._frames)
        self._cond = threading.Condition()
        self._input_done = False               # browser said stop (or vanished)
        self._encoded_all = False
        self._last_feed = time.monotonic()
        self._t_start = time.monotonic()
        self.done = threading.Event()
        threading.Thread(target=self._encode_loop, name='live-encoder', daemon=True).start()
        threading.Thread(target=self._send_loop, name='live-sender', daemon=True).start()

    # -- from the web side ------------------------------------------------------

    def feed(self, pcm16: bytes) -> None:
        if self._input_done:
            return
        x = np.frombuffer(pcm16[:len(pcm16) // 2 * 2], dtype='<i2')
        with self._pcm_lock:
            self._pcm.append(x)
        self._last_feed = time.monotonic()
        with self._cond:
            self._cond.notify_all()

    def stop(self) -> None:
        self._input_done = True
        with self._cond:
            self._cond.notify_all()

    def status(self) -> Dict:
        return {'session': self.id, 'state': self.state, 'error': self.error,
                'talkgroups': [t['tgid'] for t in self.targets if not t['refused']],
                'refused': {t['tgid']: t['refused'] for t in self.targets if t['refused']},
                'seconds': round(time.monotonic() - self._t_start, 1), 'started': self.started}

    # -- encoding -----------------------------------------------------------------

    def _take_samples(self) -> Optional[np.ndarray]:
        """Next 160 samples, or None if not that many have arrived yet."""
        with self._pcm_lock:
            have = sum(c.size for c in self._pcm)
            if have < FRAME:
                return None
            joined = np.concatenate(list(self._pcm))
            self._pcm.clear()
            if joined.size > FRAME:
                self._pcm.append(joined[FRAME:])
            return joined[:FRAME]

    def _encode_loop(self) -> None:
        enc = IMBEEncoder()
        # mic audio gets the same treatment as recorded pages, block by block:
        # a fixed gain from the first speech heard, capped, then limited
        gain = None
        try:
            while True:
                samples = self._take_samples()
                if samples is None:
                    if self._input_done:
                        break
                    if time.monotonic() - self._last_feed > IDLE_TIMEOUT:
                        logger.warning('Live talk: no audio from the console - ending the call')
                        self._input_done = True
                        break
                    with self._cond:
                        self._cond.wait(0.05)
                    continue
                x = samples.astype(np.float64) / 32768
                rms = float(np.sqrt(np.mean(x * x)))
                if gain is None and rms > 0.01:
                    gain = min(10 ** (-18 / 20) / rms, 10 ** (12 / 20))
                y = np.clip(x * (gain or 1.0), -0.89, 0.89)
                try:
                    frame = enc.encode(dsp.to_pcm16(y).tolist())
                except VocoderError:
                    enc = IMBEEncoder()
                    frame = silence_frame()
                with self._cond:
                    self._frames.append(frame)
                    self._cond.notify_all()
        finally:
            self._encoded_all = True
            with self._cond:
                self._cond.notify_all()

    # -- sending ------------------------------------------------------------------

    def _take_ldu(self, deadline: float) -> Optional[List[bytes]]:
        """Nine frames for the next LDU: real audio if it's there by the
        deadline, padded with silence otherwise. None once everything is sent."""
        with self._cond:
            self._cond.wait_for(lambda: len(self._frames) >= FRAMES_PER_LDU or self._encoded_all,
                                max(0.0, deadline - time.monotonic()))
            if self._encoded_all and not self._frames:
                return None
            if len(self._frames) < FRAMES_PER_LDU and not self._encoded_all:
                self.underruns += 1
            if self._lead_left == 0 and len(self._frames) > MAX_BACKLOG_FRAMES:
                # behind (or speech piled up during the lead tone): drop the
                # oldest audio so the delay goes back to the jitter buffer
                while len(self._frames) > PREBUFFER_FRAMES:
                    self._frames.popleft()
                    self.dropped += 1
            out =[self._frames.popleft() for _ in range(min(FRAMES_PER_LDU, len(self._frames)))]
            self._lead_left = max(0, self._lead_left - len(out))
        return out + [silence_frame()] * (FRAMES_PER_LDU - len(out))

    def _send_loop(self) -> None:
        with self._cond:
            # the lead silence (and tone) count: speech keeps buffering while they play
            self._cond.wait_for(lambda: len(self._frames) >= PREBUFFER_FRAMES or self._encoded_all, 5.0)
        t0 = time.monotonic()
        k = 0
        detail = ''
        ok = True
        try:
            while True:
                if time.monotonic() - self._t_start > self.max_seconds:
                    detail = f'stopped at the {self.max_seconds:.0f}s limit'
                    self.stop()
                imbe = self._take_ldu(t0 + k * LDU_SECONDS + LATE_GRACE)
                if imbe is None:
                    if k % 2 == 1:            # finish the superframe on an LDU2
                        imbe = [silence_frame()] * FRAMES_PER_LDU
                    else:
                        break
                if self._lead_left == 0 and self.state == 'tone':
                    self.state = 'live'
                delay = t0 + k * LDU_SECONDS - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                if not self.peer.connected:
                    ok, detail = False, 'FNE connection lost mid-call'
                    break
                live = self._send_ldu(k, imbe)
                if not live:
                    ok, detail = False, 'FNE refused the call on every talkgroup'
                    break
                k += 1
                self.ldus = k
        except Exception as e:  # never leave a call keyed
            logger.exception('Live talk sender failed')
            ok, detail = False, f'internal error: {e}'
        finally:
            self.stop()
            self._finish(ok, detail, k)

    def _send_ldu(self, k: int, imbe: List[bytes]) -> int:
        live = 0
        for t in self.targets:
            if t['refused']:
                continue
            refusal = self.peer.call_refusal(t['stream'])
            if refusal:
                t['refused'] = refusal
                logger.warning(f"Live talk: TG {t['tgid']} {refusal}")
                continue
            pkt = F.ldu1(t['params'], imbe, 0, F.FRAME_TYPE_HDU_VALID) if k % 2 == 0 else F.ldu2(t['params'], imbe, 0)
            if self.peer.send(dvm.Func.PROTOCOL, dvm.PROTOCOL_P25, pkt, t['stream'], k):
                live += 1
        return live

    def _finish(self, ok: bool, detail: str, ldus: int) -> None:
        secs = ldus * LDU_SECONDS
        for t in self.targets:
            if self.peer.connected:
                self.peer.send(dvm.Func.PROTOCOL, dvm.PROTOCOL_P25, F.tdu(t['params'], 0),
                               t['stream'], dvm.END_OF_CALL_SEQ)
            self.peer.forget_call(t['stream'])
            note = detail
            if t['refused']:
                note = f"FNE {t['refused']}"
            elif self.underruns or self.dropped:
                note = (note + ' ' if note else '') + f'({self.underruns} gap(s) filled, {self.dropped} frame(s) dropped)'
            sent = ok and not t['refused'] and ldus > 0
            record_dispatch({
                'rule': 'Dispatch console', 'event_type': 'manual', 'event': '', 'sound': self.sound,
                'destination': t['name'], 'talkgroup': str(t['tgid']), 'backend': 'p25',
                'message': '(live voice)', 'success': sent, 'status': 'sent' if sent else 'failed',
                'detail': (f'{secs:.1f}s live call' + (f' - {note}' if note else '')),
                'timestamp': datetime.now().isoformat(),
            })
        self.error = None if ok else detail
        self.state = 'ended' if ok else 'failed'
        logger.info(f"Live talk ended: {', '.join(str(t['tgid']) for t in self.targets)}, {secs:.1f}s"
                    + (f' - {detail}' if detail else ''))
        self.done.set()


class LiveTalk:
    """Starts and tracks live talk sessions (one at a time)."""

    def __init__(self, pager, settings: Optional[Dict] = None):
        self.pager = pager
        self.settings = settings or {}
        self.session: Optional[LiveSession] = None
        self._lock = threading.Lock()
        self._tone_cache: Dict[str, List[bytes]] = {}

    def _lead_frames(self, sound: Optional[str]) -> List[bytes]:
        if not sound:
            return []
        if sound not in self._tone_cache:
            audio, tones = self.pager.sounds.render_with_tones(sound)
            audio = dsp.apply_gain_db(audio, float(self.pager.settings.get('sound_gain_db', 0)))
            gap = dsp.silence(float(self.pager.settings.get('sound_gap_ms', 400)) / 1000)
            track = np.concatenate([tones, np.full((gap.size, 2), np.nan)])
            frames = encode_pcm(dsp.to_pcm16(np.concatenate([audio, gap])), track)
            # encode_pcm pads to a whole superframe; trailing silence is fine here
            self._tone_cache[sound] = frames
        return self._tone_cache[sound]

    def start(self, targets: List[Dict], sound: Optional[str] = None) -> LiveSession:
        """targets: [{'tgid', 'name'}]. Raises LiveError if it can't start."""
        if not targets:
            raise LiveError('select at least one talkgroup')
        with self._lock:
            if self.session and not self.session.done.is_set():
                tgs = ', '.join(str(t) for t in self.session.status()['talkgroups'])
                raise LiveError(f'busy - a live call is already in progress (TG {tgs})')
            for t in targets:
                problem = self.pager.check_ready(t['tgid'])
                if problem:
                    raise LiveError(problem)
            if sound and not self.pager.sounds.exists(sound):
                raise LiveError(f"unknown sound '{sound}'")
            lead = self._lead_frames(sound)
            max_s = float(self.pager.settings.get('max_call_seconds', 120))
            self.session = LiveSession(self.pager.peer,
                                       [{**t, 'params': self.pager.call_params(t['tgid'])} for t in targets],
                                       max_s, lead, sound)
            logger.info(f"Live talk started: TG {', '.join(str(t['tgid']) for t in targets)}")
            return self.session

    def get(self, session_id: str) -> LiveSession:
        s = self.session
        if s is None or s.id != session_id:
            raise LiveError('that live call has ended')
        return s

    def active(self) -> Optional[Dict]:
        s = self.session
        return s.status() if s and not s.done.is_set() else None
