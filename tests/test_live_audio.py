"""Listening to calls heard from the FNE, and live talk from the console."""
import queue
import struct
import threading
import time
import unittest

import numpy as np

from src import state
from src.audio.tones import SoundLibrary
from src.dvm import protocol as dvm
from src.dvm.protocol import P25Header
from src.p25 import frames as F
from src.p25.live import LiveError, LiveTalk
from src.p25.mbe_decoder import decode_to_pcm
from src.p25.pager import P25Pager
from src.p25.rx_audio import HEADER, MSG_AUDIO, MSG_ENCRYPTED, MSG_END, RxAudio
from src.p25.voice import encode_pcm


def speechlike(seconds, rate=8000):
    t = np.arange(int(rate * seconds)) / rate
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    x = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 700, 1100), 1))
    return (x * env * 6000).astype(np.int16)


class FakePeer:
    connected = True

    def __init__(self):
        self.sent = []
        self.refuse = {}
        self._ids = iter(range(100, 10_000))
        self.lock = threading.Lock()

    def new_stream_id(self):
        return next(self._ids)

    def call_refusal(self, stream_id):
        return self.refuse.get(stream_id)

    def forget_call(self, stream_id):
        pass

    def active_talkgroup_ids(self):
        return set()

    def send(self, func, subfunc, payload, stream_id=None, seq=0):
        with self.lock:
            self.sent.append((stream_id, seq, payload))
        return True


def read_messages(q, until_type, timeout=10):
    out = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data = q.get(timeout=0.2)
        except queue.Empty:
            continue
        kind, _, tg, src, stream, n = HEADER.unpack_from(data)
        pcm = np.frombuffer(data[HEADER.size:], dtype='<i2')
        out.append((kind, tg, src, stream, pcm))
        if kind == until_type:
            break
    return out


class RxAudioTests(unittest.TestCase):
    def setUp(self):
        self.rx = RxAudio()
        self.frames = encode_pcm(speechlike(0.72))           # 36 frames = 4 LDUs

    def feed_call(self, stream, params, frames):
        for k in range(len(frames) // 9):
            imbe = frames[k * 9:(k + 1) * 9]
            pkt = F.ldu1(params, imbe) if k % 2 == 0 else F.ldu2(params, imbe)
            self.rx.on_p25(stream, P25Header(0, params.src_id, params.dst_id, pkt[22]), pkt)
        tdu = F.tdu(params)
        self.rx.on_p25(stream, P25Header(0, params.src_id, params.dst_id, tdu[22]), tdu)

    def test_nothing_decoded_without_listeners(self):
        self.feed_call(1, F.CallParams(src_id=5, dst_id=1001), self.frames)
        time.sleep(0.2)
        self.assertEqual(self.rx._calls, {})

    def test_heard_call_is_decoded_and_ended(self):
        lid, q = self.rx.listen()
        self.feed_call(42, F.CallParams(src_id=123456, dst_id=1001), self.frames)
        msgs = read_messages(q, MSG_END)
        self.assertEqual([m[0] for m in msgs], [MSG_AUDIO] * 4 + [MSG_END])
        self.assertTrue(all(m[1:4] == (1001, 123456, 42) for m in msgs))
        audio = np.concatenate([m[4] for m in msgs[:4]])
        self.assertEqual(audio.size, 36 * 160)
        # same level as the preview decoder (unvoiced phases are random, so not bit-equal)
        ref = decode_to_pcm(self.frames)
        rms = lambda a: np.sqrt(np.mean(a.astype(float) ** 2))
        self.assertAlmostEqual(rms(audio) / rms(ref), 1.0, delta=0.1)
        self.rx.unlisten(lid)

    def test_encrypted_call_is_not_decoded(self):
        _, q = self.rx.listen()
        self.feed_call(43, F.CallParams(src_id=7, dst_id=2002, alg_id=0x84, key_id=1), self.frames)
        msgs = read_messages(q, MSG_END)
        self.assertEqual([m[0] for m in msgs], [MSG_ENCRYPTED, MSG_END])


class LiveTalkTests(unittest.TestCase):
    def setUp(self):
        with state._lock:
            state.app_state['dispatches'] = []
        self.peer = FakePeer()
        self.pager = P25Pager({'source_id': 9999}, None, SoundLibrary(''), lambda: {}, peer=self.peer)
        self.live = LiveTalk(self.pager)

    def talk(self, session, pcm, chunk=800, realtime=True):
        for i in range(0, pcm.size, chunk):
            session.feed(pcm[i:i + chunk].astype('<i2').tobytes())
            if realtime:
                time.sleep(chunk / 8000)
        session.stop()
        self.assertTrue(session.done.wait(15))

    def packets(self, stream):
        return [(seq, p) for s, seq, p in self.peer.sent if s == stream]

    def test_live_call_to_two_talkgroups(self):
        s = self.live.start([{'tgid': 1001, 'name': 'Ops'}, {'tgid': 2002, 'name': 'Fire'}])
        self.assertEqual(s.status()['talkgroups'], [1001, 2002])
        with self.assertRaises(LiveError):                    # one at a time
            self.live.start([{'tgid': 3003, 'name': 'X'}])
        speech = speechlike(1.5)
        self.talk(s, speech)
        self.assertEqual(s.state, 'ended')
        self.assertIsNone(self.live.active())
        for t in s.targets:
            pk = self.packets(t['stream'])
            ldus = [p for seq, p in pk if seq != dvm.END_OF_CALL_SEQ]
            self.assertEqual(pk[-1][1][22], F.DUID_TDU)
            self.assertEqual(len(ldus) % 2, 0)                 # whole superframes
            self.assertEqual([p[22] for p in ldus[:2]], [F.DUID_LDU1, F.DUID_LDU2])
            self.assertEqual(int.from_bytes(ldus[0][8:11], 'big'), t['tgid'])
            audio = decode_to_pcm([f for p in ldus for f in F.ldu_imbe(p)])
            self.assertLess(np.abs(audio[:8 * 160]).max(), 400)    # the lead-in silence LDU (IMBE silence is faint noise)
            self.assertGreater(np.sqrt(np.mean(audio[9 * 160:].astype(float) ** 2)), 500)
            # all the speech made it, with nothing dropped
            self.assertGreaterEqual(len(ldus) * 9, 9 + speech.size // 160)
        hist = state.snapshot()['dispatches']
        self.assertEqual({h['talkgroup'] for h in hist}, {'1001', '2002'})
        self.assertTrue(all(h['success'] and h['message'] == '(live voice)' for h in hist))
        self.assertEqual(s.dropped, 0)

    def test_lead_tone_goes_first(self):
        s = self.live.start([{'tgid': 1001, 'name': 'Ops'}], sound='triple_beep')
        self.assertEqual(s.state, 'tone')
        self.talk(s, speechlike(0.5))
        ldus = [p for seq, p in self.packets(s.targets[0]['stream']) if seq != dvm.END_OF_CALL_SEQ]
        tone_frames = s.lead_count
        self.assertGreater(tone_frames, 0)
        self.assertGreaterEqual(len(ldus) * 9, 9 + tone_frames)
        self.assertEqual(s.status()['state'], 'ended')

    def test_browser_going_quiet_ends_the_call(self):
        s = self.live.start([{'tgid': 1001, 'name': 'Ops'}])
        s.feed(speechlike(0.3).tobytes())
        self.assertTrue(s.done.wait(10))                       # no stop() - idle timeout
        self.assertEqual(self.packets(s.targets[0]['stream'])[-1][1][22], F.DUID_TDU)

    def test_refused_talkgroup_is_dropped(self):
        s = self.live.start([{'tgid': 1001, 'name': 'Ops'}, {'tgid': 2002, 'name': 'Fire'}])
        self.peer.refuse[s.targets[1]['stream']] = 'rejected by the FNE'
        self.talk(s, speechlike(0.8))
        self.assertEqual(s.status()['refused'], {2002: 'rejected by the FNE'})
        hist = {h['talkgroup']: h for h in state.snapshot()['dispatches']}
        self.assertTrue(hist['1001']['success'])
        self.assertFalse(hist['2002']['success'])

    def test_not_ready(self):
        self.peer.connected = False
        self.peer.status = {'last_error': 'down'}
        self.peer.state = type('S', (), {'value': 'connecting'})()
        with self.assertRaises(LiveError):
            self.live.start([{'tgid': 1001, 'name': 'Ops'}])


class LiveApiTests(unittest.TestCase):
    def setUp(self):
        from src.console import DispatchConsole
        from src.web_ui import Services, create_app
        state.update_dvm_state({'enabled': True, 'connected': True, 'active_talkgroups': [],
                                'calls': [], 'recent_calls': []})
        self.peer = FakePeer()
        pager = P25Pager({'source_id': 9999}, None, SoundLibrary(''), lambda: {}, peer=self.peer)
        self.live = LiveTalk(pager)
        self.rx = RxAudio()
        console = DispatchConsole({'talkgroups': [{'id': 1001, 'name': 'Ops'}]}, [], None, None, self.live)
        app = create_app(services=Services(console=console, rx_audio=self.rx))
        self.client = app.test_client()

    def tearDown(self):
        state.update_dvm_state({'enabled': False})

    def test_live_talk_over_http(self):
        r = self.client.post('/api/dispatch/live', json={'talkgroups': [1001]})
        self.assertEqual(r.status_code, 200, r.get_json())
        sid = r.get_json()['session']
        tiles = self.client.get('/api/dispatch/state').get_json()['talkgroups']
        self.assertTrue(tiles[0]['live'])
        self.assertEqual(self.client.post('/api/dispatch/live', json={'talkgroups': [1001]}).status_code, 409)
        pcm = speechlike(1.0)
        for i in range(0, pcm.size, 800):
            r = self.client.post(f'/api/dispatch/live/{sid}/audio', data=pcm[i:i + 800].astype('<i2').tobytes())
            self.assertEqual(r.status_code, 200)
            time.sleep(0.1)
        self.assertEqual(self.client.post(f'/api/dispatch/live/{sid}/stop').status_code, 200)
        self.assertTrue(self.live.session.done.wait(10))
        self.assertEqual(self.client.post(f'/api/dispatch/live/nope/audio', data=b'').status_code, 409)
        sent = [p for _, seq, p in self.peer.sent]
        self.assertEqual(sent[-1][22], F.DUID_TDU)
        self.assertGreaterEqual(len(sent) - 1, 6)

    def test_listen_stream(self):
        resp = self.client.get('/api/dispatch/listen', buffered=False)
        chunks = iter(resp.response)
        self.assertEqual(HEADER.unpack(next(chunks))[0], 0)       # keepalive first
        frames = encode_pcm(speechlike(0.36))
        params = F.CallParams(src_id=55, dst_id=1001)
        for k, pkt in enumerate((F.ldu1(params, frames[:9]), F.ldu2(params, frames[9:18]))):
            self.rx.on_p25(9, P25Header(0, 55, 1001, pkt[22]), pkt)
        got = next(chunks)
        kind, _, tg, src, stream, n = HEADER.unpack_from(got)
        self.assertEqual((kind, tg, src, stream, n), (MSG_AUDIO, 1001, 55, 9, 1440))
        self.assertEqual(len(got), HEADER.size + 1440 * 2)
        self.assertEqual(self.rx.listening, 1)
        resp.close()
        self.assertEqual(self.rx.listening, 0)


if __name__ == '__main__':
    unittest.main()
