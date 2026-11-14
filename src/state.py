"""Shared application state, updated by background pollers and read by the Flask routes."""
import threading
from datetime import datetime
from typing import Any, Dict, Optional

_lock = threading.Lock()

app_state: Dict[str, Any] = {
    'forewarned': {
        'reachable': False,
        'last_seen': None,
        'local_alert_state': {
            'active': False,
            'level': 'none',
            'reason': '',
            'timestamp': None,
            'triggered_by': [],
        },
        'weather_alerts': [],
        'eoc_states': {},
    },
    'manual_override': {
        'level': None,
    },
    'paging': {
        'destinations': [],
        'last_result': None,
    },
    # DVM FNE peer connection (see src/dvm/peer.py for the full field list).
    'dvm': {
        'enabled': False,
    },
    # Most recent first, capped at HISTORY_LIMIT.
    'events': [],
    'dispatches': [],
}

HISTORY_LIMIT = 50


def update_forewarned_state(*, reachable: bool, status: Optional[Dict[str, Any]] = None) -> None:
    """Update the cached Forewarned status. On unreachable, keep the last-known
    alert data but flip reachable=False so the UI can show a distinct
    "unreachable" indicator rather than silently reading as "all clear"."""
    with _lock:
        app_state['forewarned']['reachable'] = reachable
        if reachable and status is not None:
            app_state['forewarned']['last_seen'] = datetime.now().isoformat()
            app_state['forewarned']['local_alert_state'] = status.get(
                'local_alert_state', app_state['forewarned']['local_alert_state']
            )
            app_state['forewarned']['weather_alerts'] = status.get('weather_alerts', [])
            app_state['forewarned']['eoc_states'] = status.get('eoc_states', {})


def update_manual_override(level: Optional[str]) -> None:
    with _lock:
        app_state['manual_override']['level'] = level


def update_paging_destinations(destinations: list) -> None:
    with _lock:
        app_state['paging']['destinations'] = destinations


def update_paging_result(result: Dict[str, Any]) -> None:
    with _lock:
        app_state['paging']['last_result'] = result


def record_events(events: list) -> None:
    if not events:
        return
    with _lock:
        app_state['events'] = ([e.to_dict() for e in reversed(events)] + app_state['events'])[:HISTORY_LIMIT]


def record_dispatch(entry: Dict[str, Any]) -> None:
    with _lock:
        app_state['dispatches'] = ([entry] + app_state['dispatches'])[:HISTORY_LIMIT]
        app_state['paging']['last_result'] = entry


def update_dvm_state(status: Dict[str, Any]) -> None:
    with _lock:
        app_state['dvm'] = status


def snapshot() -> Dict[str, Any]:
    """Return a shallow-copied snapshot of app_state safe to serialize."""
    with _lock:
        return {
            'forewarned': dict(app_state['forewarned']),
            'manual_override': dict(app_state['manual_override']),
            'paging': dict(app_state['paging']),
            'dvm': dict(app_state['dvm']),
            'events': list(app_state['events']),
            'dispatches': list(app_state['dispatches']),
        }
