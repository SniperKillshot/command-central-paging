"""Tests for Forewarned -> event -> rule -> page routing.

Run from the project root:  python -m unittest discover -s tests
"""
import copy
import unittest

from src import state
from src.dispatcher import Dispatcher, render_message, rule_matches
from src.events import (EOC_STATE_CHANGED, LEVEL_CHANGED, WEATHER_ALERT_CLEARED,
                        WEATHER_ALERT_ISSUED, WEATHER_ALERT_UPDATED, EventDetector)
from src.paging import MockBackend, P25Backend
from src.web_ui import create_app

STORM = {
    'event': 'Severe Thunderstorm Warning for Townsville and Herbert and Lower Burdekin',
    'headline': 'Severe Thunderstorm Warning for damaging winds and large hail',
    'description': 'Very long BOM text...',
    'severity': 'Severe',
    'urgency': 'Immediate',
    'areas': 'Townsville, Herbert and Lower Burdekin',
    'onset': '2026-11-11T15:00:00+10:00',
    'expires': '2026-11-11T18:00:00+10:00',
    'source': 'BOM',
    'filename': 'IDQ21035.xml',
}

FLOOD = {
    'event': 'Flood Watch - Ross River',
    'headline': 'Flood Watch for Ross River',
    'severity': 'Moderate',
    'areas': 'Ross River',
    'filename': 'IDQ20885.xml',
}


def status(level='none', alerts=(), eoc=None, reason=''):
    return {
        'local_alert_state': {'active': level != 'none', 'level': level, 'reason': reason, 'triggered_by': []},
        'weather_alerts': [copy.deepcopy(a) for a in alerts],
        'eoc_states': eoc or {},
    }


DESTINATIONS = [
    {'id': 'site-wide', 'name': 'All Call', 'backend': 'mock', 'extension': '8500'},
    {'id': 'p25-ops', 'name': 'P25 Ops', 'backend': 'p25', 'talkgroup': '1001'},
]

RULES = [
    {'name': 'Storm to P25', 'events': ['weather_alert_issued', 'weather_alert_updated'],
     'match': {'event_contains': ['severe thunderstorm']},
     'destinations': ['p25-ops'],
     'message': 'Attention all units. {headline}. Areas affected: {areas}.'},
    {'name': 'Escalation', 'events': 'level_changed',
     'match': {'min_level': 'warning', 'direction': 'escalation'},
     'destinations': ['site-wide'],
     'message': 'Alert level raised to {level_upper}. {reason}.'},
]


def make_dispatcher(rules=RULES):
    backends = {'mock': MockBackend({}), 'p25': P25Backend({})}
    return Dispatcher(rules, DESTINATIONS, backends)


def reset_state():
    with state._lock:
        state.app_state['events'] = []
        state.app_state['dispatches'] = []
        state.app_state['paging']['last_result'] = None


class EventDetectorTests(unittest.TestCase):
    def test_first_status_is_baseline_only(self):
        d = EventDetector()
        self.assertEqual(d.process(status('warning', [STORM])), [])

    def test_fire_on_startup_pages_existing_warnings(self):
        d = EventDetector(fire_on_startup=True)
        types = [e.type for e in d.process(status('warning', [STORM]))]
        self.assertIn(WEATHER_ALERT_ISSUED, types)
        self.assertIn(LEVEL_CHANGED, types)

    def test_new_warning_carries_warning_data(self):
        d = EventDetector()
        d.process(status())
        events = d.process(status('warning', [STORM], reason='Severe Thunderstorm Warning'))
        issued = [e for e in events if e.type == WEATHER_ALERT_ISSUED]
        self.assertEqual(len(issued), 1)
        data = issued[0].data
        self.assertEqual(data['event_short'], 'Severe Thunderstorm Warning')
        self.assertEqual(data['areas'], STORM['areas'])
        self.assertEqual(data['level'], 'warning')
        level = [e for e in events if e.type == LEVEL_CHANGED][0]
        self.assertEqual((level.data['previous_level'], level.data['level'], level.data['direction']),
                         ('none', 'warning', 'escalation'))

    def test_same_status_twice_only_fires_once(self):
        d = EventDetector()
        d.process(status())
        self.assertTrue(d.process(status('warning', [STORM])))
        self.assertEqual(d.process(status('warning', [STORM])), [])

    def test_reissued_warning_is_an_update(self):
        d = EventDetector()
        d.process(status('warning', [STORM]))
        changed = dict(STORM, areas='Townsville, Charters Towers', expires='2026-11-11T20:00:00+10:00')
        types = [e.type for e in d.process(status('warning', [changed]))]
        self.assertEqual(types, [WEATHER_ALERT_UPDATED])

    def test_cancelled_warning_is_cleared(self):
        d = EventDetector()
        d.process(status('warning', [STORM, FLOOD]))
        events = d.process(status('watch', [FLOOD]))
        cleared = [e for e in events if e.type == WEATHER_ALERT_CLEARED]
        self.assertEqual([e.data['id'] for e in cleared], ['IDQ21035.xml'])
        level = [e for e in events if e.type == LEVEL_CHANGED][0]
        self.assertEqual(level.data['direction'], 'deescalation')

    def test_eoc_state_change(self):
        d = EventDetector()
        url = 'https://disaster.townsville.qld.gov.au/'
        d.process(status(eoc={url: {'state': 'alert'}}))
        events = d.process(status(eoc={url: {'state': 'stand up'}}))
        self.assertEqual([(e.type, e.data['previous_state'], e.data['state']) for e in events],
                         [(EOC_STATE_CHANGED, 'alert', 'stand up')])


class RuleTests(unittest.TestCase):
    def test_storm_routes_to_p25_with_rendered_message(self):
        d = EventDetector()
        d.process(status())
        events = d.process(status('warning', [STORM]))
        pages = [p for e in events for p in make_dispatcher().plan(e)]
        storm = [p for p in pages if p['rule'] == 'Storm to P25']
        self.assertEqual(len(storm), 1)
        self.assertEqual(storm[0]['destination']['id'], 'p25-ops')
        self.assertEqual(storm[0]['message'],
                         'Attention all units. Severe Thunderstorm Warning for damaging winds and large hail. '
                         'Areas affected: Townsville, Herbert and Lower Burdekin.')

    def test_non_matching_warning_does_not_page(self):
        d = EventDetector()
        d.process(status())
        events = [e for e in d.process(status('watch', [FLOOD])) if e.type == WEATHER_ALERT_ISSUED]
        self.assertEqual(make_dispatcher().plan(events[0]), [])

    def test_deescalation_does_not_match_escalation_rule(self):
        d = EventDetector()
        d.process(status('emergency'))
        events = d.process(status('warning'))
        self.assertEqual(make_dispatcher().plan(events[0]), [])

    def test_min_level(self):
        self.assertTrue(rule_matches({'min_level': 'warning'}, {'level': 'emergency'}))
        self.assertFalse(rule_matches({'min_level': 'warning'}, {'level': 'watch'}))

    def test_missing_placeholders_render_blank(self):
        self.assertEqual(render_message('{headline}. Expires {expires}. {nonsense}.', {'headline': 'Storm'}),
                         'Storm. Expires.')

    def test_invalid_rules_are_skipped(self):
        dispatcher = make_dispatcher([
            {'name': 'bad event', 'events': 'thunder', 'destinations': ['site-wide']},
            {'name': 'bad dest', 'events': 'level_changed', 'destinations': ['nowhere']},
            {'name': 'disabled', 'events': 'level_changed', 'destinations': ['site-wide'], 'enabled': False},
            {'name': 'good', 'events': 'level_changed', 'destinations': ['site-wide']},
        ])
        self.assertEqual([r['name'] for r in dispatcher.rules], ['good'])


class ConfigFileTests(unittest.TestCase):
    def test_example_config_rules_all_load(self):
        """Rules must survive a real YAML load (a bare `on:` key becomes True)."""
        import os
        from src.config import Config
        cfg = Config(os.path.join(os.path.dirname(__file__), '..', 'config.yaml.example'))
        routing = cfg.get('alert_routing')
        dispatcher = Dispatcher(routing['rules'], cfg.get('paging_destinations'),
                                {'mock': MockBackend({}), 'p25': P25Backend({})})
        enabled = [r for r in routing['rules'] if r.get('enabled', True)]
        self.assertEqual(len(dispatcher.rules), len(enabled))
        self.assertGreater(len(enabled), 0)

    def test_legacy_on_key_still_works(self):
        import yaml
        rules = yaml.safe_load("- name: x\n  on: level_changed\n  destinations: [site-wide]\n")
        self.assertEqual(len(make_dispatcher(rules).rules), 1)


class PushEndToEndTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.detector = EventDetector()
        self.dispatcher = make_dispatcher()

        def on_status(s):
            events = self.detector.process(s)
            state.record_events(events)
            return self.dispatcher.handle(events)

        self.on_status = on_status
        self.client = create_app(on_status=on_status, push_token='secret').test_client()

    def push(self, payload, token='secret'):
        return self.client.post('/api/forewarned/push', json=payload, headers={'X-Forewarned-Token': token})

    def test_push_then_poll_pages_once(self):
        self.on_status(status())                       # baseline from a poll
        r = self.push(status('warning', [STORM], reason='Severe Thunderstorm Warning'))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()['pages_queued'], 2)   # P25 storm page + escalation All Call
        self.assertEqual(self.on_status(status('warning', [STORM], reason='Severe Thunderstorm Warning')), 0)
        self.dispatcher.shutdown()

        dispatches = state.snapshot()['dispatches']
        by_dest = {d['destination']: d for d in dispatches}
        self.assertEqual(len(dispatches), 2)
        self.assertEqual(by_dest['P25 Ops']['status'], 'failed')   # no FNE connection in tests
        self.assertFalse(by_dest['P25 Ops']['success'])
        self.assertEqual(by_dest['All Call']['status'], 'sent')
        self.assertEqual(by_dest['All Call']['message'], 'Alert level raised to WARNING. Severe Thunderstorm Warning.')

    def test_push_rejects_bad_token(self):
        self.assertEqual(self.push(status('warning'), token='wrong').status_code, 401)
        self.dispatcher.shutdown()

    def test_push_rejects_non_status_body(self):
        self.assertEqual(self.push({'hello': 'world'}).status_code, 400)
        self.dispatcher.shutdown()


if __name__ == '__main__':
    unittest.main()
