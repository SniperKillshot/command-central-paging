"""Talkgroup names from the FNE's REST API (src/dvm/rest.py) and how the
dispatch console uses them."""
import hashlib
import unittest

from src import state
from src.console import DispatchConsole
from src.dvm.rest import AUTH_HEADER, FNERest, FNERestError, TalkgroupDirectory, password_hash

TGS = {'status': 200, 'tgs': [
    {'name': 'Ops 1', 'alias': 'OPS1', 'invalid': False, 'source': {'tgid': 1001, 'slot': 1},
     'config': {'active': True, 'affiliated': False}},
    {'name': '', 'alias': 'FIRE', 'invalid': False, 'source': {'tgid': 2002, 'slot': 1}, 'config': {}},
    {'name': 'Old', 'alias': '', 'invalid': True, 'source': {'tgid': 3003, 'slot': 1}, 'config': {}},
    {'name': 'Statewide', 'alias': '', 'invalid': False, 'source': {'tgid': 10037, 'slot': 1}, 'config': {}},
]}


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class FakeFNE:
    """Stands in for requests.Session against dvmfne's REST API."""

    def __init__(self, password='secret'):
        self.hash = hashlib.sha256(password.encode()).hexdigest()
        self.tokens = 0
        self.valid = None
        self.calls = []

    def put(self, url, json=None, timeout=None, verify=None):
        self.calls.append(('PUT', url, json))
        if url.endswith('/auth') and json and json.get('auth') == self.hash:
            self.tokens += 1
            self.valid = str(1000 + self.tokens)
            return FakeResponse(200, {'status': 200, 'token': self.valid})
        return FakeResponse(400, {'status': 400, 'message': 'invalid password'})

    def get(self, url, headers=None, timeout=None, verify=None):
        self.calls.append(('GET', url, headers))
        if (headers or {}).get(AUTH_HEADER) != self.valid:
            return FakeResponse(401, {'status': 401, 'message': 'invalid authentication token'})
        return FakeResponse(200, TGS)


class RestTests(unittest.TestCase):
    def test_password_hash_is_hex_sha256(self):
        self.assertEqual(password_hash('PASSWORD'), hashlib.sha256(b'PASSWORD').hexdigest())
        self.assertEqual(len(password_hash('x')), 64)

    def test_logs_in_and_reads_talkgroups(self):
        fne = FakeFNE()
        client = FNERest('http://fne:9990/', 'secret', session=fne)
        tgs = client.talkgroups()
        self.assertEqual(fne.calls[0][:2], ('PUT', 'http://fne:9990/auth'))
        self.assertEqual(fne.calls[1][2], {AUTH_HEADER: '1001'})
        self.assertEqual(tgs[0], {'tgid': 1001, 'slot': 1, 'name': 'Ops 1', 'alias': 'OPS1',
                                  'active': True, 'affiliated': False, 'invalid': False})
        self.assertEqual(len(tgs), 4)

    def test_logs_in_again_when_the_token_is_dropped(self):
        fne = FakeFNE()
        client = FNERest('http://fne:9990', 'secret', session=fne)
        client.talkgroups()
        fne.valid = 'something else'          # FNE restarted / token replaced
        client.talkgroups()
        self.assertEqual(fne.tokens, 2)

    def test_wrong_password(self):
        client = FNERest('http://fne:9990', 'wrong', session=FakeFNE())
        with self.assertRaises(FNERestError) as e:
            client.talkgroups()
        self.assertIn('invalid password', str(e.exception))


class DirectoryTests(unittest.TestCase):
    def directory(self, fne=None):
        return TalkgroupDirectory({}, client=FNERest('http://fne:9990', 'secret', session=fne or FakeFNE()))

    def test_names_prefer_name_then_alias_and_skip_invalid(self):
        d = self.directory()
        self.assertTrue(d.refresh())
        self.assertEqual(d.names(), {1001: 'Ops 1', 2002: 'FIRE', 10037: 'Statewide'})
        self.assertEqual(d.status['count'], 3)
        self.assertTrue(d.status['ok'])

    def test_keeps_last_names_when_the_fne_stops_answering(self):
        fne = FakeFNE()
        d = self.directory(fne)
        d.refresh()
        fne.hash = 'changed'                  # password changed on the FNE
        fne.valid = None
        self.assertFalse(d.refresh())
        self.assertFalse(d.status['ok'])
        self.assertIn('invalid password', d.status['error'])
        self.assertEqual(d.name(1001), 'Ops 1')

    def test_settings_build_the_url(self):
        d = TalkgroupDirectory({'address': 'fne.example', 'rest_port': 9991, 'rest_https': True,
                                'rest_password': 'p'})
        self.assertEqual(d.client.base_url, 'https://fne.example:9991')
        d = TalkgroupDirectory({'address': 'fne.example', 'rest_address': '10.0.0.5'})
        self.assertEqual(d.client.base_url, 'http://10.0.0.5:9990')


class ConsoleNameTests(unittest.TestCase):
    DESTS = [{'id': 'fire', 'name': 'Fire Page', 'backend': 'p25', 'talkgroup': '2002'}]

    def setUp(self):
        self.names = TalkgroupDirectory({}, client=FNERest('http://fne:9990', 'secret', session=FakeFNE()))
        self.names.refresh()
        state.update_dvm_state({'enabled': True, 'connected': True,
                                'active_talkgroups': [{'tgid': 1001}, {'tgid': 2002}, {'tgid': 10037},
                                                      {'tgid': 4444}]})

    def tearDown(self):
        state.update_dvm_state({'enabled': False})

    def tiles(self, settings):
        c = DispatchConsole(settings, self.DESTS, names=self.names)
        return c, {t['tgid']: t for t in c.talkgroups(state.snapshot()['dvm'])}

    def test_order_own_name_then_destination_then_fne(self):
        _, t = self.tiles({'talkgroups': [{'id': 1001, 'name': 'Dispatch'}, {'id': 2002}, {'id': 10037}]})
        self.assertEqual(t[1001]['name'], 'Dispatch')       # its own name wins
        self.assertEqual(t[2002]['name'], 'Fire Page')      # destination name over the FNE's
        self.assertEqual(t[10037]['name'], 'Statewide')     # FNE name
        self.assertEqual(t[4444]['name'], 'TG 4444')        # nothing anywhere

    def test_tg_placeholder_is_not_a_name(self):
        _, t = self.tiles({'talkgroups': [{'id': 10037, 'name': 'TG 10037'}, {'id': 1001, 'name': 'tg1001'}]})
        self.assertEqual(t[10037]['name'], 'Statewide')
        self.assertEqual(t[1001]['name'], 'Ops 1')

    def test_editor_shows_fne_names(self):
        c, _ = self.tiles({'talkgroups': [{'id': 1001}], 'include_destinations': False})
        e = c.editor()
        self.assertEqual(e['talkgroups'], [{'id': 1001, 'name': '', 'default_name': 'Ops 1', 'color': None}])
        fne = {t['id']: t['name'] for t in e['fne_talkgroups']}
        self.assertEqual(fne[10037], 'Statewide')
        self.assertEqual(fne[4444], '')
        self.assertTrue(e['fne_names']['ok'])

    def test_activity_names_calls_on_untiled_talkgroups(self):
        state.update_dvm_state({'enabled': True, 'connected': True, 'active_talkgroups': [],
                                'calls': [{'dst_id': 10037, 'src_id': 251933, 'started': '2026-11-15T12:00:00',
                                           'active': True}]})
        c = DispatchConsole({'talkgroups': [], 'show_fne_talkgroups': False}, [], names=self.names)
        heard = [i for i in c.state()['activity'] if i['kind'] == 'rx']   # other tests leave page history
        self.assertEqual(heard[0]['name'], 'Statewide')

    def test_works_without_names(self):
        c = DispatchConsole({'talkgroups': [{'id': 10037}]}, [])
        self.assertEqual(c.talkgroups(state.snapshot()['dvm'])[0]['name'], 'TG 10037')
        self.assertIsNone(c.state()['fne_names'])


if __name__ == '__main__':
    unittest.main()
