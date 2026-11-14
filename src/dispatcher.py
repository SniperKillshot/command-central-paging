"""Routes alert events to paging destinations according to configured rules.

A rule says: for these event types, if the event matches these conditions,
send this message (a template filled from the event's warning data) to these
destinations. Pages run one at a time on a single worker thread, because a
paging channel (a radio talkgroup, a PA zone) can only carry one at a time
and the poller must never block waiting for a transmission to finish.
"""
import itertools
import logging
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from src.events import EVENT_TYPES, LEVEL_RANK, AlertEvent
from src.paging.base import PageResult
from src.state import record_dispatch

logger = logging.getLogger(__name__)

DEFAULT_MESSAGES = {
    'level_changed': 'Alert level is now {level_upper}. {reason}.',
    'weather_alert_issued': '{headline}. Areas: {areas}.',
    'weather_alert_updated': 'Update. {headline}. Areas: {areas}.',
    'weather_alert_cleared': '{event_short} has been cancelled.',
    'eoc_state_changed': 'Local Disaster Management Group status is now {state_upper}.',
}


class _BlankMissing(dict):
    """Template values: unknown placeholders render as empty, never raise."""
    def __missing__(self, key):
        return ''


def render_message(template: str, data: Dict[str, Any]) -> str:
    values = _BlankMissing({k: ('' if v is None else v) for k, v in data.items()})
    text = template.format_map(values)
    # Tidy what empty fields leave behind ("Expires . ." -> "Expires.") so
    # TTS doesn't read out odd pauses.
    text = ' '.join(text.split())
    text = re.sub(r'\s+([.,])', r'\1', text)
    text = re.sub(r'([.,])[.,]+', r'\1', text)
    return text.strip()


def rule_matches(match: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """Every condition must hold. Supported condition forms:
      min_level: warning            -> event's alert level is at least this
      <field>_contains: [a, b]      -> field contains any of these (case-insensitive)
      <field>: [a, b]               -> field equals one of these (case-insensitive)
    """
    for key, expected in (match or {}).items():
        if key == 'min_level':
            if LEVEL_RANK.get(data.get('level', 'none'), 0) < LEVEL_RANK.get(str(expected).lower(), 0):
                return False
            continue
        options = expected if isinstance(expected, list) else [expected]
        options = [str(o).lower() for o in options]
        if key.endswith('_contains'):
            value = str(data.get(key[:-len('_contains')], '')).lower()
            if not any(o in value for o in options):
                return False
        else:
            if str(data.get(key, '')).lower() not in options:
                return False
    return True


class Dispatcher:
    def __init__(self, rules: List[Dict[str, Any]], destinations: List[Dict[str, Any]], backends: Dict[str, Any],
                 sound_exists: Optional[Callable[[str], bool]] = None):
        self.destinations = {d['id']: d for d in destinations}
        self.backends = backends
        self.sound_exists = sound_exists
        self.rules = self._validate(rules or [])
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pager')
        # pages waiting for or using the channel, for the dispatch console
        self._jobs: Dict[int, Dict[str, Any]] = {}
        self._job_ids = itertools.count(1)
        self._jobs_lock = threading.Lock()

    def _validate(self, rules):
        valid = []
        for i, rule in enumerate(rules):
            name = rule.get('name', f'rule {i + 1}')
            # 'events' is the documented key. YAML 1.1 parses a bare `on:` key
            # as boolean True, so accept that (and a literal 'on') too.
            on = rule.get('events', rule.get('on', rule.get(True, [])))
            on = on if isinstance(on, list) else [on]
            bad = [t for t in on if t not in EVENT_TYPES]
            missing = [d for d in rule.get('destinations', []) if d not in self.destinations]
            if not on or bad:
                logger.error(f"Routing rule '{name}' skipped: bad 'events' {bad or on} (valid: {', '.join(EVENT_TYPES)})")
                continue
            if missing:
                logger.error(f"Routing rule '{name}' skipped: unknown destination(s) {missing}")
                continue
            sound = rule.get('sound')
            if sound and self.sound_exists and not self.sound_exists(sound):
                logger.error(f"Routing rule '{name}' skipped: unknown sound '{sound}'")
                continue
            if rule.get('message') == '' and not sound:
                logger.error(f"Routing rule '{name}' skipped: empty message and no sound")
                continue
            if rule.get('enabled', True) is False:
                logger.info(f"Routing rule '{name}' is disabled")
                continue
            valid.append({**rule, 'name': name, 'events': on})
        logger.info(f"Loaded {len(valid)} routing rule(s)")
        return valid

    def plan(self, event: AlertEvent) -> List[Dict[str, Any]]:
        """Work out which pages an event produces, without sending anything."""
        pages = []
        for rule in self.rules:
            if event.type not in rule['events'] or not rule_matches(rule.get('match'), event.data):
                continue
            if rule.get('message') == '':
                message = ''        # sound-only rule
            else:
                message = render_message(rule.get('message') or DEFAULT_MESSAGES[event.type], event.data)
            for dest_id in rule.get('destinations', []):
                pages.append({'rule': rule['name'], 'destination': self.destinations[dest_id],
                              'message': message, 'sound': rule.get('sound'), 'event': event})
        return pages

    def handle(self, events: List[AlertEvent]) -> int:
        count = 0
        for event in events:
            for page in self.plan(event):
                self._submit(page)
                count += 1
        return count

    def _submit(self, page) -> Future:
        dest = page['destination']
        job_id = next(self._job_ids)
        with self._jobs_lock:
            self._jobs[job_id] = {'id': job_id, 'destination': dest['id'], 'talkgroup': dest.get('talkgroup'),
                                  'state': 'queued', 'rule': page['rule']}
        return self._executor.submit(self._run_job, job_id, page)

    def _run_job(self, job_id, page):
        with self._jobs_lock:
            self._jobs[job_id]['state'] = 'sending'
        try:
            return self._send(page)
        finally:
            with self._jobs_lock:
                self._jobs.pop(job_id, None)

    def jobs(self) -> List[Dict[str, Any]]:
        """Pages queued or being sent, in order."""
        with self._jobs_lock:
            return [dict(j) for j in self._jobs.values()]

    def manual(self, destination_id: str, message: str = '', sound: Optional[str] = None,
               voice: Optional[Dict[str, Any]] = None, source: str = 'Manual page') -> Future:
        """Queue a page on demand (dashboard / API). Runs on the same single
        worker as automatic pages, so it waits its turn on the channel."""
        dest = self.destinations.get(destination_id)
        if dest is None:
            raise KeyError(f"unknown destination '{destination_id}'")
        return self.manual_to(dest, message, sound, voice, source)

    def manual_to(self, dest: Dict[str, Any], message: str = '', sound: Optional[str] = None,
                  voice: Optional[Dict[str, Any]] = None, source: str = 'Manual page',
                  audio: Any = None) -> Future:
        """Like manual(), to a destination that need not be in the config (the
        dispatch console pages any talkgroup). audio: recorded speech (8 kHz
        float) sent in place of a spoken message."""
        if not (message or '').strip() and not sound and audio is None:
            raise ValueError('give a message and/or a sound')
        if sound and self.sound_exists and not self.sound_exists(sound):
            raise KeyError(f"unknown sound '{sound}'")
        page = {'rule': source, 'destination': dest, 'message': (message or '').strip(),
                'sound': sound or None, 'voice': voice, 'audio': audio, 'event': None}
        return self._submit(page)

    def _send(self, page):
        dest = page['destination']
        backend = self.backends.get(dest.get('backend', 'mock'))
        try:
            if backend is None:
                result = PageResult(False, 'failed', f"no backend '{dest.get('backend')}' loaded")
            else:
                kwargs = {'sound': page.get('sound'), 'voice': page.get('voice')}
                if page.get('audio') is not None:
                    kwargs['audio'] = page['audio']
                result = backend.send(dest, page['message'], **kwargs)
        except Exception as e:  # a failed page must never kill the pager thread
            logger.exception(f"Page to {dest['id']} failed")
            result = PageResult(False, 'failed', str(e))
        event = page.get('event')
        record_dispatch({
            'rule': page['rule'],
            'event_type': event.type if event else 'manual',
            'event': (event.data.get('event_short') or event.data.get('level', '')) if event else '',
            'sound': page.get('sound'),
            'destination': dest.get('name', dest['id']),
            'talkgroup': dest.get('talkgroup'),
            'backend': dest.get('backend', 'mock'),
            'message': page['message'] or ('(recorded voice)' if page.get('audio') is not None else ''),
            'success': result.success,
            'status': result.status,
            'detail': result.detail,
            'timestamp': result.timestamp,
        })
        return result

    def shutdown(self):
        self._executor.shutdown(wait=True)
