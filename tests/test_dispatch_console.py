"""Dispatch console: tile list, live state and paging from the console."""
import struct
import unittest

import numpy as np

from src import state
from src.console import DispatchConsole
from src.dispatcher import Dispatcher
from src.paging.base import PageResult
from src.web_ui import Services, create_app

DESTINATIONS = [
    {'id': 'ops', 'name': 'Ops Page', 'backend': 'p25', 'talkgroup': '1001'},
    {'id': 'fire', 'name': 'Fire', 'backend': 'p25', 'talkgroup': '2002'},
    {'id': 'pa', 'name': 'PA', 'backend': 'mock', 'extension': '8500'},
]
SETTINGS = {'talkgroups': [{'id': 3003, 'name': 'Command', 'color': 'red'}, {'id': 1001, 'name': 'Ops'}]}


class FakeBackend:
    def __init__(self):
        self.sent = []

    def send(self, destination, message, sound=None, voice=None, audio=None):
        self.sent.append({'dest': destination, 'message': message, 'sound': sound, 'audio': audio})
        return PageResult(True, 'sent', 'ok')


class FakeSounds:
    def list(self):
        return [{'id': 'alert_tone', 'label': 'Alert tone (1 kHz)'}, {'id': 'hi_lo', 'label': 'Hi-lo'}]

    def exists(self, name):
        return name in ('alert_tone', 'hi_lo')


def set_dvm(**kw):
    state.update_dvm_state({'enabled': True, 'connected': True, 'state': 'running',
                            'active_talkgroups': [], 'calls': [], 'recent_calls': [], **kw})


class ConsoleTests(unittest.TestCase):
    def setUp(self):
        with state._lock:
            state.app_state['dispatches'] = []
        set_dvm(active_talkgroups=[{'tgid': 1001, 'affiliated': True}, {'tgid': 4004, 'affiliated': False}])
        self.backend = FakeBackend()
        self.dispatcher = Dispatcher([], DESTINATIONS, {'p25': self.backend, 'mock': FakeBackend()},
                                     sound_exists=FakeSounds().exists)
        self.console = DispatchConsole(SETTINGS, DESTINATIONS, self.dispatcher, FakeSounds())

    def tearDown(self):
        self.dispatcher.shutdown()
        state.update_dvm_state({'enabled': False})

    def test_tile_order_and_names(self):
        tiles = self.console.talkgroups(state.snapshot()['dvm'])
        self.assertEqual([t['tgid'] for t in tiles], [3003, 1001, 2002, 4004])
        by = {t['tgid']: t for t in tiles}
        self.assertEqual(by[1001]['name'], 'Ops')                 # config name wins
        self.assertEqual(by[1001]['destination'], 'ops')          # but pages through the destination
        self.assertEqual(by[2002]['name'], 'Fire')
        self.assertEqual(by[4004]['name'], 'TG 4004')
        self.assertEqual(by[3003]['color'], 'red')
        self.assertTrue(by[1001]['on_fne'] and by[1001]['affiliated'])
        self.assertFalse(by[3003]['on_fne'])

    def test_fne_talkgroups_can_be_hidden(self):
        c = DispatchConsole({**SETTINGS, 'show_fne_talkgroups': False}, DESTINATIONS)
        self.assertNotIn(4004, [t['tgid'] for t in c.talkgroups(state.snapshot()['dvm'])])

    def test_state_shows_calls_heard(self):
        set_dvm(calls=[{'src_id': 555, 'dst_id': 2002, 'started': '2026-01-01T10:00:00', 'active': True}],
                recent_calls=[{'src_id': 777, 'dst_id': 1001, 'started': '2026-01-01T09:00:00',
                               'ended': '2026-01-01T09:00:05', 'active': False}])
        s = self.console.state()
        by = {t['tgid']: t for t in s['talkgroups']}
        self.assertEqual(by[2002]['rx']['src_id'], 555)
        self.assertIsNone(by[1001]['rx'])
        self.assertEqual(by[1001]['last_rx']['src_id'], 777)
        self.assertEqual([a['kind'] for a in s['activity']], ['rx', 'rx'])
        self.assertEqual([t['id'] for t in s['tones']], ['alert_tone', 'hi_lo'])

    def test_page_any_talkgroup(self):
        self.assertEqual(self.console.page([1001, 4004], 'Test message', 'hi_lo'), 2)
        self.dispatcher.shutdown()
        dests = [p['dest'] for p in self.backend.sent]
        self.assertEqual(dests[0]['id'], 'ops')                       # configured destination
        self.assertEqual((dests[1]['backend'], dests[1]['talkgroup']), ('p25', '4004'))
        self.assertEqual(self.backend.sent[1]['sound'], 'hi_lo')
        hist = state.snapshot()['dispatches']
        self.assertEqual({h['talkgroup'] for h in hist}, {'1001', '4004'})
        self.assertTrue(all(h['rule'] == 'Dispatch console' for h in hist))

    def test_tx_state_while_queued(self):
        import threading
        gate = threading.Event()
        orig = self.backend.send

        def slow(*a, **kw):
            gate.wait(5)
            return orig(*a, **kw)
        self.backend.send = slow
        self.console.page([1001, 2002], 'hello')
        tiles = {t['tgid']: t for t in self.console.state()['talkgroups']}
        self.assertEqual(tiles[2002]['tx'], 'queued')
        self.assertIn(tiles[1001]['tx'], ('queued', 'sending'))
        gate.set()
        self.dispatcher.shutdown()
        tiles = {t['tgid']: t for t in self.console.state()['talkgroups']}
        self.assertIsNone(tiles[1001]['tx'])
        self.assertTrue(tiles[1001]['last_page']['success'])

    def test_bad_requests(self):
        with self.assertRaises(ValueError):
            self.console.page([], 'x')
        with self.assertRaises(ValueError):
            self.console.page([70000], 'x')
        with self.assertRaises(ValueError):
            self.console.page([1001], '')
        with self.assertRaises(KeyError):
            self.console.page([1001], '', 'nope')

    def test_talk_resamples_recording(self):
        rate = 48000
        t = np.arange(rate * 2) / rate
        pcm = (np.sin(2 * np.pi * 440 * t) * 8000).astype('<i2').tobytes()
        self.assertEqual(self.console.talk([2002], pcm, rate, 'alert_tone'), 1)
        self.dispatcher.shutdown()
        sent = self.backend.sent[0]
        self.assertEqual(sent['sound'], 'alert_tone')
        self.assertAlmostEqual(len(sent['audio']) / 8000, 2.0, places=2)
        self.assertEqual(state.snapshot()['dispatches'][0]['message'], '(recorded voice)')
        with self.assertRaises(ValueError):
            self.console.talk([2002], b'\x00\x00' * 100, rate)        # too short


class LayoutTests(unittest.TestCase):
    def setUp(self):
        import tempfile, os
        from src.settings_store import SettingsStore
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'settings.json')
        self.store = SettingsStore(self.path, {})
        set_dvm(active_talkgroups=[{'tgid': 1001}, {'tgid': 4004}, {'tgid': 5005}])
        self.console = DispatchConsole({**SETTINGS, 'tones': ['hi_lo']}, DESTINATIONS, None, FakeSounds(),
                                       store=self.store)

    def tearDown(self):
        state.update_dvm_state({'enabled': False})
        self.tmp.cleanup()

    def ids(self):
        return [t['tgid'] for t in self.console.talkgroups(state.snapshot()['dvm'])]

    def test_editor_starts_from_config(self):
        e = self.console.editor()
        self.assertEqual([t['id'] for t in e['talkgroups']], [3003, 1001, 2002])   # config + destination
        self.assertEqual(e['fne_talkgroups'], [{'id': 4004, 'name': ''}, {'id': 5005, 'name': ''}])
        self.assertEqual(e['tones'], ['hi_lo'])
        self.assertFalse(e['saved_in_browser'])

    def test_save_replaces_config_and_persists(self):
        self.console.save_layout({'talkgroups': [{'id': 2002, 'name': '  Fire   Ground ', 'color': 'red'},
                                                 {'id': '4004', 'name': '', 'color': 'pink'}],
                                  'show_fne_talkgroups': False, 'tones': []})
        self.assertEqual(self.ids(), [2002, 4004])          # destination 1001 no longer added
        tiles = self.console.talkgroups(state.snapshot()['dvm'])
        self.assertEqual((tiles[0]['name'], tiles[0]['color'], tiles[0]['destination']), ('Fire Ground', 'red', 'fire'))
        self.assertEqual((tiles[1]['name'], tiles[1]['color']), ('TG 4004', None))
        self.assertEqual(self.console.tones(), [])           # none chosen means none
        # a restart reads it back
        from src.settings_store import SettingsStore
        again = DispatchConsole(SETTINGS, DESTINATIONS, store=SettingsStore(self.path, {}))
        self.assertEqual([t['tgid'] for t in again.talkgroups({})], [2002, 4004])
        self.assertTrue(again.editor()['saved_in_browser'])

    def test_bad_layouts_rejected(self):
        for bad in ({'talkgroups': 'x'},
                    {'talkgroups': [{'id': 0}]},
                    {'talkgroups': [{'id': 70000}]},
                    {'talkgroups': [{'id': 5}, {'id': '5'}]},
                    {'talkgroups': [], 'tones': ['nope']}):
            with self.assertRaises(ValueError, msg=bad):
                self.console.save_layout(bad)
        self.assertFalse(self.console.editor()['saved_in_browser'])

    def test_reset_goes_back_to_config(self):
        self.console.save_layout({'talkgroups': [{'id': 7}], 'tones': []})
        self.console.reset_layout()
        self.assertEqual(self.ids(), [3003, 1001, 2002, 4004, 5005])
        self.assertEqual([t['id'] for t in self.console.tones()], ['hi_lo'])

    def test_layout_api(self):
        app = create_app(services=Services(console=self.console), api_token='tok')
        c = app.test_client()
        self.assertEqual([t['id'] for t in c.get('/api/dispatch/layout').get_json()['fne_talkgroups']], [4004, 5005])
        body = {'talkgroups': [{'id': 1001, 'name': 'Ops'}], 'tones': ['alert_tone']}
        self.assertEqual(c.post('/api/dispatch/layout', json=body).status_code, 401)
        r = c.post('/api/dispatch/layout', json=body, headers={'X-Api-Token': 'tok'})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()['talkgroups'],
                         [{'id': 1001, 'name': 'Ops', 'default_name': 'Ops Page', 'color': None}])
        r = c.post('/api/dispatch/layout', json={'talkgroups': [{'id': 'abc'}]}, headers={'X-Api-Token': 'tok'})
        self.assertEqual(r.status_code, 400)
        self.assertIn('not a talkgroup', r.get_json()['error'])
        r = c.post('/api/dispatch/layout/reset', headers={'X-Api-Token': 'tok'})
        self.assertFalse(r.get_json()['saved_in_browser'])


class ConsoleApiTests(unittest.TestCase):
    def setUp(self):
        set_dvm()
        self.backend = FakeBackend()
        self.dispatcher = Dispatcher([], DESTINATIONS, {'p25': self.backend}, sound_exists=FakeSounds().exists)
        console = DispatchConsole(SETTINGS, DESTINATIONS, self.dispatcher, FakeSounds())
        app = create_app(services=Services(dispatcher=self.dispatcher, console=console), api_token='tok')
        self.client = app.test_client()

    def tearDown(self):
        self.dispatcher.shutdown()
        state.update_dvm_state({'enabled': False})

    def test_page_and_talk_endpoints(self):
        self.assertEqual(self.client.get('/dispatch').status_code, 200)
        s = self.client.get('/api/dispatch/state').get_json()
        self.assertTrue(s['api_token_required'])
        self.assertEqual(s['talkgroups'][0]['tgid'], 3003)

        r = self.client.post('/api/dispatch/page', json={'talkgroups': [3003], 'sound': 'hi_lo'})
        self.assertEqual(r.status_code, 401)
        r = self.client.post('/api/dispatch/page', json={'talkgroups': [3003], 'sound': 'hi_lo'},
                             headers={'X-Api-Token': 'tok'})
        self.assertEqual(r.status_code, 202)

        pcm = struct.pack('<4000h', *([1000, -1000] * 2000))
        r = self.client.post('/api/dispatch/talk?talkgroups=1001,2002&rate=8000', data=pcm,
                             headers={'X-Api-Token': 'tok', 'Content-Type': 'application/octet-stream'})
        self.assertEqual(r.status_code, 202, r.get_json())
        self.assertEqual(r.get_json()['queued'], 2)
        r = self.client.post('/api/dispatch/talk?talkgroups=1001&rate=5', data=pcm,
                             headers={'X-Api-Token': 'tok'})
        self.assertEqual(r.status_code, 400)
        self.dispatcher.shutdown()
        self.assertEqual(len(self.backend.sent), 3)


if __name__ == '__main__':
    unittest.main()
