"""Turns successive Forewarned status payloads into discrete alert events.

Both the background poller and the push webhook feed status payloads in here.
Because events are derived by diffing against the last-seen payload, a change
that arrives by push and then again on the next poll only produces events once.
"""
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LEVEL_RANK = {'none': 0, 'advisory': 1, 'watch': 2, 'warning': 3, 'emergency': 4}

LEVEL_CHANGED = 'level_changed'
WEATHER_ALERT_ISSUED = 'weather_alert_issued'
WEATHER_ALERT_UPDATED = 'weather_alert_updated'
WEATHER_ALERT_CLEARED = 'weather_alert_cleared'
EOC_STATE_CHANGED = 'eoc_state_changed'

EVENT_TYPES = (LEVEL_CHANGED, WEATHER_ALERT_ISSUED, WEATHER_ALERT_UPDATED,
               WEATHER_ALERT_CLEARED, EOC_STATE_CHANGED)

# Fields that, when changed on an alert with the same ID, count as an update.
_ALERT_FINGERPRINT_FIELDS = ('event', 'headline', 'severity', 'areas', 'onset', 'expires')


@dataclass
class AlertEvent:
    type: str
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {'type': self.type, 'data': self.data, 'timestamp': self.timestamp}


def short_event_name(event: str) -> str:
    """'Severe Thunderstorm Warning for Townsville...' -> 'Severe Thunderstorm Warning'.
    Mirrors how Forewarned shortens event names for its own reason text."""
    short = event.split(' for ')[0]
    return short.split(' - ')[0].strip()


def _alert_id(alert: Dict[str, Any]) -> str:
    # Forewarned keys alerts by the BOM product file name (e.g. IDQ21035.xml).
    return alert.get('filename') or alert.get('id') or alert.get('headline') or alert.get('event', '')


def _alert_payload(alert: Dict[str, Any]) -> Dict[str, Any]:
    event = alert.get('event', '')
    return {
        'id': _alert_id(alert),
        'event': event,
        'event_short': short_event_name(event),
        'headline': alert.get('headline') or event,
        'description': alert.get('description', ''),
        'severity': alert.get('severity', 'Unknown'),
        'urgency': alert.get('urgency', 'Unknown'),
        'areas': alert.get('areas', ''),
        'onset': alert.get('onset', ''),
        'expires': alert.get('expires', ''),
        'source': alert.get('source', ''),
    }


class EventDetector:
    """Diffs each new Forewarned status against the previous one.

    The first payload after startup only sets a baseline unless
    fire_on_startup is True - otherwise every restart during an active
    warning would re-page everyone.
    """

    def __init__(self, fire_on_startup: bool = False):
        self.fire_on_startup = fire_on_startup
        self._lock = threading.Lock()
        self._level: Optional[str] = None
        self._alerts: Dict[str, Dict[str, Any]] = {}
        self._eoc: Dict[str, str] = {}

    def process(self, status: Dict[str, Any]) -> List[AlertEvent]:
        local = status.get('local_alert_state') or {}
        level = local.get('level') or 'none'
        alerts = {_alert_id(a): _alert_payload(a) for a in (status.get('weather_alerts') or [])}
        eoc = {url: (info or {}).get('state', 'inactive')
               for url, info in (status.get('eoc_states') or {}).items()}

        # Diff and swap baseline under one lock so a push and a poll racing
        # each other can't both see the same change.
        with self._lock:
            first = self._level is None
            prev_level = 'none' if first else self._level
            prev_alerts = {} if first else self._alerts
            prev_eoc = {} if first else self._eoc
            self._level, self._alerts, self._eoc = level, alerts, eoc

        if first and not self.fire_on_startup:
            logger.info(f"Event baseline set: level={level}, {len(alerts)} active warning(s)")
            return []

        events: List[AlertEvent] = []

        if level != prev_level:
            escalation = LEVEL_RANK.get(level, 0) > LEVEL_RANK.get(prev_level, 0)
            events.append(AlertEvent(LEVEL_CHANGED, {
                'level': level,
                'level_upper': level.upper(),
                'previous_level': prev_level,
                'direction': 'escalation' if escalation else 'deescalation',
                'reason': local.get('reason', ''),
                'triggered_by': local.get('triggered_by', []),
            }))

        for alert_id, alert in alerts.items():
            old = prev_alerts.get(alert_id)
            if old is None:
                events.append(AlertEvent(WEATHER_ALERT_ISSUED, {**alert, 'level': level}))
            elif any(old.get(f) != alert.get(f) for f in _ALERT_FINGERPRINT_FIELDS):
                events.append(AlertEvent(WEATHER_ALERT_UPDATED, {**alert, 'level': level}))

        for alert_id, alert in prev_alerts.items():
            if alert_id not in alerts:
                events.append(AlertEvent(WEATHER_ALERT_CLEARED, {**alert, 'level': level}))

        for url in set(eoc) | set(prev_eoc):
            state, old_state = eoc.get(url, 'inactive'), prev_eoc.get(url, 'inactive')
            if state != old_state:
                events.append(AlertEvent(EOC_STATE_CHANGED, {
                    'site': url,
                    'state': state,
                    'state_upper': state.upper(),
                    'previous_state': old_state,
                    'level': level,
                }))

        for e in events:
            logger.info(f"Event: {e.type} {e.data.get('event_short') or e.data.get('state') or e.data.get('level', '')}")
        return events
