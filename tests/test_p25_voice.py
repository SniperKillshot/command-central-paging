"""Tests for P25 voice paging: IMBE vocoder, P25 network frames, the voice
call transmitter (against a fake FNE), alert sounds, voice selection and
the paging API."""
import math
import struct
import time
import unittest

import numpy as np

from src.audio import dsp
from src.audio.tones import BUILTIN, SoundLibrary
from src.dispatcher import Dispatcher
from src.dvm import protocol as dvm
from src.p25 import frames as F
from src.p25.imbe import IMBEEncoder, PyIMBEEncoder, pack_frame_vector, silence_frame
from src.p25.mbe_decoder import IMBEDecoder, PyIMBEDecoder, decode_to_pcm
from src.p25.pager import P25Pager
from src.p25.voice import encode_pcm, pcm_to_frames
from src.paging import MockBackend, P25Backend
from src.settings_store import SettingsStore
from src.web_ui import Services, create_app
from tests.test_dvm_peer import FakeFNE, make_peer, wait_for

TG = 1001
SRC_RID = 1234567


def tone(freq=440.0, seconds=0.5, amp=0.4):
    t = np.arange(int(8000 * seconds)) / 8000
    return dsp.to_pcm16(amp * np.sin(2 * np.pi * freq * t))


def speechlike(seconds=1.0, f0=140.0):
    """Harmonic signal with a moving formant - enough to exercise voicing."""
    n = int(8000 * seconds)
    t = np.arange(n) / 8000
    x = np.zeros(n)
    for k in range(1, 25):
        if f0 * k * 1.05 > 3800:
            break
        f = f0 * k * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))
        env = np.exp(-((f - 700 - 300 * np.sin(2 * np.pi * 2 * t)) / 500) ** 2)
        x += env * np.sin(2 * np.pi * np.cumsum(f) / 8000) / k
    return dsp.to_pcm16(0.5 * x / np.max(np.abs(x)))


class FakeTTS:
    """Stands in for the real engines: a tone whose length follows the text."""
    def __init__(self):
        self.calls = []
        self.engines = {'fake': self}
        self.label = 'Fake'

    def available(self):
        return True

    def voices(self):
        return [{'id': 'v1', 'name': 'Voice One'}, {'id': 'v2', 'name': 'Voice Two'}]

    def describe(self):
        return [{'name': 'fake', 'label': 'Fake', 'available': True, 'voices': self.voices()}]

    def first_available(self):
        return ('fake', 'v1')

    def synthesize(self, text, engine, voice, rate=100):
        self.calls.append((text, engine, voice, rate))
        secs = min(1.0, 0.05 * len(text.split()) + 0.2)
        t = np.arange(int(16000 * secs)) / 16000
        return 0.3 * np.sin(2 * np.pi * 300 * t), 16000


class VoiceFNE(FakeFNE):
    """FakeFNE that also collects P25 traffic and can refuse it the way
    dvmfne does (in-call control REJECT_TRAFFIC)."""
    def __init__(self, reject_p25=False, **kw):
        self.reject_p25 = reject_p25
        self.p25 = []
        self.p25_times = []
        super().__init__(**kw)

    def _handle(self, frame):
        if frame.func == dvm.Func.PROTOCOL and frame.subfunc == dvm.PROTOCOL_P25:
            self.p25.append(frame)
            self.p25_times.append(time.monotonic())
            if self.reject_p25 and frame.payload[22] == F.DUID_LDU1:
                body = b'\x00' * 6 + struct.pack('>I', frame.peer_id) + bytes([dvm.ICC_REJECT_TRAFFIC]) \
                    + TG.to_bytes(3, 'big') + b'\x00'
                self.send(dvm.Func.INCALL_CTRL, dvm.PROTOCOL_P25, body, stream_id=frame.stream_id)
            return
        super()._handle(frame)


# ---------------------------------------------------------------------------

class VocoderTests(unittest.TestCase):
    def test_frame_packing_widths(self):
        fv = [0xFFF, 0, 0xFFF, 0, 0x7FF, 0, 0x7FF, 0x7F]
        bits = '1' * 12 + '0' * 12 + '1' * 12 + '0' * 12 + '1' * 11 + '0' * 11 + '1' * 11 + '1' * 7
        self.assertEqual(pack_frame_vector(fv), int(bits, 2).to_bytes(11, 'big'))

    def test_independent_decoder_reads_what_the_encoder_meant(self):
        """mbelib (independent code) must recover the harmonic count, every
        voicing decision and the quantised amplitudes the encoder chose."""
        enc, dec = PyIMBEEncoder(), PyIMBEDecoder(seed=1)   # reads its internal state
        pcm = np.concatenate([speechlike(0.6), tone(1000, 0.3), np.zeros(800, dtype=np.int16)])
        frames = 0
        for i in range(0, len(pcm) - 159, 160):
            frame = enc.encode(pcm[i:i + 160].tolist())
            self.assertTrue(dec.decode_params(frame))
            L = dec.cur.L
            self.assertEqual(L, enc.num_harms)
            self.assertEqual([dec.cur.Vl[l] for l in range(1, L + 1)], enc.p_v_uv_dsn[:L])
            for l in range(1, L + 1):
                self.assertAlmostEqual(dec.cur.log2Ml[l], enc.sa_prev2[l] / 2 ** 22, delta=0.12)
            dec.prev.copy_from(dec.cur)
            frames += 1
        self.assertEqual(frames, 50)         # 0.6 s + 0.3 s + 0.1 s = 8000 samples

    def test_round_trip_keeps_pitch_and_loudness(self):
        pcm = tone(250, 1.0, 0.3)
        out = decode_to_pcm(encode_pcm(pcm)).astype(float)
        seg = out[2400:4400]                          # steady state, past the codec delay
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
        peak_hz = np.argmax(spec) * 8000 / seg.size
        self.assertAlmostEqual(peak_hz, 250, delta=12)
        rms_db = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) / 32768)
        self.assertGreater(rms_db, -30)

    def test_known_tones_stay_steady_through_the_codec(self):
        """Generated alert tones must not warble: steady level frame to frame
        and nearly all energy on the intended frequencies."""
        lib = SoundLibrary('')

        def wobble_and_purity(sig, targets):
            levels = {t: [] for t in targets}
            for i in range(0, len(sig) - 320, 160):
                X = np.abs(np.fft.rfft(sig[i:i + 320] * np.hanning(320), 4096)) ** 2
                f = np.fft.rfftfreq(4096, 1 / 8000)
                for t in targets:
                    levels[t].append(10 * np.log10(X[np.abs(f - t) < 30].sum() + 1e-9))
            P = np.abs(np.fft.rfft(sig * np.hanning(len(sig)), 8 * len(sig))) ** 2
            f = np.fft.rfftfreq(8 * len(sig), 1 / 8000)
            on = sum(P[np.abs(f - t) < 15].sum() for t in targets) / P.sum()
            return max(np.std(v) for v in levels.values()), on

        for name, targets in (('noaa_wat', [1050]), ('eas_attention', [853, 960]), ('alert_tone', [1000])):
            x, track = lib.render_with_tones(name)
            x, track = x[:12000], track[:12000]
            out = decode_to_pcm(encode_pcm(dsp.to_pcm16(x), track)).astype(float)[3000:10000]
            wobble, on = wobble_and_purity(out, targets)
            self.assertLess(wobble, 0.8, name)
            self.assertGreater(on, 0.9, name)

    def test_silence_frame_decodes_quiet(self):
        out = decode_to_pcm([silence_frame()] * 10).astype(float)
        self.assertLess(np.max(np.abs(out)), 500)


class FrameTests(unittest.TestCase):
    P = F.CallParams(src_id=SRC_RID, dst_id=TG)

    def test_rs_codewords_have_zero_syndromes(self):
        exp, log, v = [0] * 126, [0] * 64, 1
        for i in range(63):
            exp[i], log[v] = v, i
            v <<= 1
            v = v ^ 0x43 if v & 0x40 else v
        for i in range(63, 126):
            exp[i] = exp[i - 63]

        def syndromes(cw, n):
            hb = [(int.from_bytes(cw, 'big') >> (138 - 6 * i)) & 0x3F for i in range(24)]
            out = []
            for r in range(1, n + 1):
                acc = 0
                for s in hb:
                    acc = (0 if acc == 0 else exp[log[acc] + r]) ^ s
                out.append(acc)
            return out
        self.assertEqual(syndromes(F.rs_241213(bytes(range(1, 10))), 12), [0] * 12)
        self.assertEqual(syndromes(F.rs_24169(bytes(range(20, 32))), 8), [0] * 8)

    def test_ldu1_layout(self):
        imbe = [bytes([i] * 11) for i in range(1, 10)]
        pkt = F.ldu1(self.P, imbe)
        self.assertEqual(len(pkt), 201)
        self.assertEqual(pkt[:4], b'P25D')
        self.assertEqual(int.from_bytes(pkt[5:8], 'big'), SRC_RID)
        self.assertEqual(int.from_bytes(pkt[8:11], 'big'), TG)
        self.assertEqual((pkt[11:13], pkt[16:19]), (b'\x00\x01', bytes.fromhex('0bb800')))
        self.assertEqual((pkt[22], pkt[23], pkt[180], pkt[181]), (0x05, 178, 0x01, 0x80))
        self.assertEqual([pkt[o] for _, o, _ in F.LDU1_LAYOUT], list(range(0x62, 0x6B)))
        self.assertEqual(F.ldu_imbe(pkt), imbe)
        # link control (LCO, MFId, service options, TG, RID) rides in voice frames 3-5
        lc = pkt[61:64] + pkt[78:81] + pkt[95:98]
        self.assertEqual(lc, bytes([0x00, 0x00, 0x04]) + TG.to_bytes(3, 'big') + SRC_RID.to_bytes(3, 'big'))

    def test_ldu2_and_tdu(self):
        imbe = [bytes([0x10 + i] * 11) for i in range(9)]
        pkt = F.ldu2(self.P, imbe)
        self.assertEqual((len(pkt), pkt[22], pkt[23]), (189, 0x0A, 178))
        self.assertEqual([pkt[o] for _, o, _ in F.LDU2_LAYOUT], list(range(0x6B, 0x74)))
        self.assertEqual(F.ldu_imbe(pkt), imbe)
        self.assertEqual(pkt[111 + 1], 0x80)              # voice 15 carries ALGID (unencrypted)
        t = F.tdu(self.P)
        self.assertEqual((len(t), t[22], t[23]), (32, 0x03, 24))

    def test_pcm_padded_to_superframes(self):
        self.assertEqual(len(pcm_to_frames(np.zeros(161, dtype=np.int16))), 18)
        self.assertEqual(len(pcm_to_frames(np.zeros(160 * 18, dtype=np.int16))), 18)


class SoundTests(unittest.TestCase):
    def test_builtin_sounds_render(self):
        lib = SoundLibrary(sounds_dir='')
        for name in BUILTIN:
            x = lib.render(name)
            self.assertGreater(x.size, 2000, name)
            self.assertLessEqual(np.max(np.abs(x)), 1.0, name)
        with self.assertRaises(KeyError):
            lib.render('nope')

    def test_custom_wav_overrides_builtin(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'alert_tone.wav'), 'wb') as f:
                f.write(dsp.wav_bytes(tone(600, 0.25), 16000))
            with open(os.path.join(d, 'bad name!.wav'), 'wb') as f:
                f.write(b'')
            lib = SoundLibrary(d)
            entry = [s for s in lib.list() if s['id'] == 'alert_tone']
            self.assertEqual(entry[0]['source'], 'file')
            self.assertFalse(lib.exists('bad name!'))
            self.assertEqual(lib.render('alert_tone').size, 1000)   # 0.25 s at 16 kHz -> 8 kHz

    def test_resample_preserves_frequency(self):
        t = np.arange(22050) / 22050
        y = dsp.resample(np.sin(2 * np.pi * 1000 * t), 22050, 8000)
        self.assertAlmostEqual(y.size, 8000, delta=2)
        spec = np.abs(np.fft.rfft(y))
        self.assertAlmostEqual(np.argmax(spec) * 8000 / y.size, 1000, delta=2)


class VoiceCallTests(unittest.TestCase):
    P25 = {'source_id': SRC_RID, 'lead_in_ms': 0, 'tail_ms': 0, 'sound_gap_ms': 100}

    def setUp(self):
        self.fne = None
        self.peer = None

    def tearDown(self):
        if self.peer:
            self.peer.stop()
        if self.fne:
            self.fne.close()

    def connect(self, **fne_kw):
        self.fne = VoiceFNE(**fne_kw)
        self.peer = make_peer(self.fne)
        self.peer.start()
        self.assertTrue(wait_for(lambda: self.peer.connected))

    def pager(self, **over):
        settings = dict(self.P25, **over)
        return P25Pager(settings, FakeTTS(), SoundLibrary(''), lambda: {'engine': 'fake', 'voice': 'v1'}, self.peer)

    def test_page_goes_out_as_a_p25_call(self):
        self.connect()
        result = self.pager().page(TG, 'Test page', sound='triple_beep')
        self.assertTrue(result.ok, result.detail)
        self.assertTrue(wait_for(lambda: self.fne.p25 and self.fne.p25[-1].payload[22] == F.DUID_TDU))
        pkts = self.fne.p25
        duids = [f.payload[22] for f in pkts]
        self.assertEqual(duids[-1], F.DUID_TDU)
        voice = pkts[:-1]
        self.assertGreaterEqual(len(voice), 4)
        self.assertEqual(len(voice) % 2, 0)
        self.assertEqual([f.payload[22] for f in voice], [F.DUID_LDU1, F.DUID_LDU2] * (len(voice) // 2))
        self.assertEqual({f.stream_id for f in pkts}.__len__(), 1)
        self.assertEqual([f.seq for f in voice], list(range(len(voice))))
        self.assertEqual(pkts[-1].seq, dvm.END_OF_CALL_SEQ)
        self.assertTrue(all(int.from_bytes(f.payload[8:11], 'big') == TG for f in pkts))
        self.assertTrue(all(int.from_bytes(f.payload[5:8], 'big') == SRC_RID for f in pkts))
        # paced like a radio: one LDU every 180 ms
        gaps = np.diff(self.fne.p25_times[:len(voice)])
        self.assertAlmostEqual(float(np.median(gaps)), 0.18, delta=0.03)
        # and the audio inside decodes to something with energy
        imbe = [fr for f in voice for fr in F.ldu_imbe(f.payload)]
        self.assertGreater(np.max(np.abs(decode_to_pcm(imbe).astype(float))), 1000)

    def test_fne_rejection_stops_the_call(self):
        self.connect(reject_p25=True)
        result = self.pager().page(TG, 'Test page')
        self.assertFalse(result.ok)
        self.assertIn('rejected by the FNE', result.detail)
        self.assertTrue(wait_for(lambda: self.fne.p25[-1].payload[22] == F.DUID_TDU))
        self.assertLessEqual(len(self.fne.p25), 4)

    def test_readiness_checks(self):
        self.connect()
        self.assertIn('source_id', self.pager(source_id=0).check_ready(TG))
        self.assertIn('not a number', self.pager().check_ready('ops'))
        self.fne.send_active_tgs([(2002, 0x01)])
        self.assertTrue(wait_for(lambda: self.peer.active_talkgroup_ids() == {2002}))
        self.assertIn('not active on the FNE', self.pager().check_ready(TG))
        self.assertIsNone(self.pager(require_active_talkgroup=False).check_ready(TG))
        self.assertIsNone(self.pager().check_ready(2002))

    def test_backend_reports_disconnected_reason(self):
        self.fne = VoiceFNE(password='other')
        self.peer = make_peer(self.fne)
        self.peer.start()
        self.assertTrue(wait_for(lambda: self.peer.status['last_error']))
        backend = P25Backend({}, p25_pager=self.pager())
        r = backend.send({'id': 'x', 'talkgroup': str(TG)}, 'hello')
        self.assertEqual(r.status, 'failed')
        self.assertIn('unauthorized', r.detail)

    def test_backend_without_fne(self):
        r = P25Backend({}).send({'id': 'x', 'talkgroup': str(TG)}, 'hello')
        self.assertEqual((r.success, r.status), (False, 'failed'))
        self.assertIn('not enabled', r.detail)


class VoiceSelectionTests(unittest.TestCase):
    def test_runtime_voice_is_used_and_persisted(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'settings.json')
            store = SettingsStore(path, {'tts': {'engine': 'fake', 'voice': 'v1', 'rate': 100}})
            tts = FakeTTS()
            pager = P25Pager({}, tts, SoundLibrary(''), lambda: store.get('tts'))
            pager.render('one two')
            store.update('tts', {'voice': 'v2', 'rate': 130})
            pager.render('three')
            pager.render('four', voice={'voice': 'v1'})          # per-destination override
            self.assertEqual([(c[2], c[3]) for c in tts.calls], [('v1', 100), ('v2', 130), ('v1', 130)])
            self.assertEqual(SettingsStore(path, {}).get('tts')['voice'], 'v2')


class PagingApiTests(unittest.TestCase):
    DESTS = [{'id': 'p25-ops', 'name': 'P25 Ops', 'backend': 'mock', 'talkgroup': '1001'}]

    def make(self, token=''):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        sounds = SoundLibrary('')
        self.dispatcher = Dispatcher([], self.DESTS, {'mock': MockBackend({})}, sound_exists=sounds.exists)
        tts = FakeTTS()
        store = SettingsStore(self.tmp.name + '/s.json', {'tts': {'engine': 'fake', 'voice': 'v1', 'rate': 100}})
        pager = P25Pager({}, tts, sounds, lambda: store.get('tts'))
        svc = Services(dispatcher=self.dispatcher, destinations=self.DESTS, tts=tts, sounds=sounds,
                       settings=store, pager=pager)
        return create_app(services=svc, api_token=token).test_client()

    def tearDown(self):
        self.dispatcher.shutdown()
        self.tmp.cleanup()

    def test_manual_page_and_sound_on_command(self):
        from src import state
        c = self.make()
        r = c.post('/api/page', json={'destination': 'p25-ops', 'sound': 'hi_lo'})
        self.assertEqual(r.status_code, 202)
        self.dispatcher.shutdown()
        last = state.snapshot()['dispatches'][0]
        self.assertEqual((last['sound'], last['event_type'], last['status']), ('hi_lo', 'manual', 'sent'))

    def test_page_validation(self):
        c = self.make()
        self.assertEqual(c.post('/api/page', json={'destination': 'nowhere', 'message': 'x'}).status_code, 400)
        self.assertEqual(c.post('/api/page', json={'destination': 'p25-ops'}).status_code, 400)
        self.assertEqual(c.post('/api/page', json={'destination': 'p25-ops', 'sound': 'kazoo'}).status_code, 400)

    def test_api_token(self):
        c = self.make(token='s3cret')
        self.assertEqual(c.post('/api/page', json={'destination': 'p25-ops', 'message': 'x'}).status_code, 401)
        r = c.post('/api/page', json={'destination': 'p25-ops', 'message': 'x'}, headers={'X-Api-Token': 's3cret'})
        self.assertEqual(r.status_code, 202)
        self.assertTrue(c.get('/api/state').get_json()['api_token_required'])

    def test_voice_selection_api_and_previews(self):
        c = self.make()
        data = c.get('/api/tts').get_json()
        self.assertEqual(data['current']['voice'], 'v1')
        self.assertEqual(c.post('/api/tts', json={'engine': 'fake', 'voice': 'zzz'}).status_code, 400)
        r = c.post('/api/tts', json={'engine': 'fake', 'voice': 'v2', 'rate': 120})
        self.assertEqual(r.get_json()['current'], {'engine': 'fake', 'voice': 'v2', 'rate': 120})
        wav = c.post('/api/tts/preview', json={'text': 'hello there', 'radio': True})
        self.assertEqual((wav.status_code, wav.mimetype), (200, 'audio/wav'))
        audio, rate = dsp.read_wav(wav.data)
        self.assertEqual(rate, 8000)
        self.assertGreater(np.max(np.abs(audio)), 0.02)
        snd = c.get('/api/sounds/chime.wav?radio=0')
        self.assertEqual(snd.status_code, 200)
        self.assertEqual(c.get('/api/sounds/nope.wav').status_code, 404)
        self.assertIn('wail', [s['id'] for s in c.get('/api/sounds').get_json()])


if __name__ == '__main__':
    unittest.main()
