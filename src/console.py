"""Dispatch console: talkgroup tiles with live activity, and paging them.

The tiles come from, in order:
  1. dispatch.talkgroups (name, colour and order chosen there)
  2. paging destinations with backend p25 not already listed
     (until the tiles are edited in the browser, which lists them explicitly)
  3. any other talkgroup the FNE reports active (dispatch.show_fne_talkgroups)

The dispatch settings start from config.yaml; once the tiles are edited on
the console they are saved in the settings store and override it.

A tile that matches a p25 destination pages through that destination (so
its own voice and name apply); any other talkgroup gets a destination made
on the fly. Everything goes through the dispatcher, so console pages queue
with automatic ones and show in the page history.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from src.audio import dsp
from src.state import snapshot

logger = logging.getLogger(__name__)

MAX_TALK_SECONDS = 60
MAX_TILES = 100
MAX_NAME = 40
COLORS = ('blue', 'green', 'amber', 'red', 'purple', 'teal', 'grey')


def _tgid(v) -> Optional[int]:
    try:
        tg = int(str(v).strip())
    except (TypeError, ValueError):
        return None
    return tg if 1 <= tg <= 0xFFFF else None


class DispatchConsole:
    def __init__(self, settings: Dict[str, Any], destinations: List[Dict[str, Any]], dispatcher=None,
                 sounds=None, live=None, store=None):
        self.config = settings or {}
        self.store = store         # SettingsStore: tile setup saved from the browser
        self.destinations = destinations or []
        self.dispatcher = dispatcher
        self.sounds = sounds
        self.live = live           # LiveTalk, if there is an FNE connection

    @property
    def settings(self) -> Dict[str, Any]:
        saved = self.store.get('dispatch') if self.store else {}
        return {**self.config, **saved}

    def tones(self) -> List[Dict[str, str]]:
        """Alert sound buttons: dispatch.tones in that order. An empty list in
        config.yaml means every sound; one saved from the browser means none."""
        if self.sounds is None:
            return []
        available = {s['id']: s for s in self.sounds.list()}
        s = self.settings
        wanted = s.get('tones') or ([] if s.get('tones_explicit') else list(available))
        return [{'id': i, 'label': available[i]['label']} for i in wanted if i in available]

    # -- editing the tiles from the browser ---------------------------------------

    def editor(self) -> Dict[str, Any]:
        """Everything the tile editor needs."""
        dvm = snapshot().get('dvm') or {}
        tiles = [t for t in self.talkgroups(dvm) if t['source'] != 'fne']
        listed = {t['tgid'] for t in tiles}
        s = self.settings
        return {
            'talkgroups': [{'id': t['tgid'], 'name': t['name'], 'color': t['color']} for t in tiles],
            'show_fne_talkgroups': bool(s.get('show_fne_talkgroups', True)),
            'tones': [t['id'] for t in self.tones()],
            'available_tones': self.sounds.list() if self.sounds else [],
            'colors': list(COLORS),
            'fne_talkgroups': [t['tgid'] for t in dvm.get('active_talkgroups') or [] if t['tgid'] not in listed],
            'saved_in_browser': bool(self.store and self.store.get('dispatch')),
        }

    def save_layout(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and save a tile setup from the editor. Raises ValueError."""
        if self.store is None:
            raise ValueError('settings storage is not available')
        rows = data.get('talkgroups')
        if not isinstance(rows, list):
            raise ValueError('talkgroups must be a list')
        if len(rows) > MAX_TILES:
            raise ValueError(f'at most {MAX_TILES} tiles')
        tiles, seen = [], set()
        for n, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise ValueError(f'tile {n}: bad entry')
            tg = _tgid(row.get('id'))
            if tg is None:
                raise ValueError(f"tile {n}: '{row.get('id')}' is not a talkgroup number (1-65535)")
            if tg in seen:
                raise ValueError(f'talkgroup {tg} is on more than one tile')
            seen.add(tg)
            name = ' '.join(str(row.get('name') or '').split())[:MAX_NAME] or f'TG {tg}'
            color = row.get('color') if row.get('color') in COLORS else None
            tiles.append({'id': tg, 'name': name, 'color': color})
        available = [x['id'] for x in (self.sounds.list() if self.sounds else [])]
        tones = data.get('tones', [])
        if not isinstance(tones, list) or any(t not in available for t in tones):
            raise ValueError('unknown alert sound in tones')
        values = {
            'talkgroups': tiles,
            'show_fne_talkgroups': bool(data.get('show_fne_talkgroups', True)),
            'tones': list(dict.fromkeys(tones)),
            'tones_explicit': True,
            'include_destinations': False,     # the list now names every tile itself
        }
        self.store.update('dispatch', values)
        logger.info(f"Dispatch tiles saved from the console: {len(tiles)} tile(s)")
        return self.editor()

    def reset_layout(self) -> Dict[str, Any]:
        """Forget the browser-saved setup and go back to config.yaml."""
        if self.store is not None:
            self.store.clear('dispatch')
            logger.info('Dispatch tiles reset to config.yaml')
        return self.editor()

    # -- talkgroups -------------------------------------------------------------

    def talkgroups(self, dvm: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        settings = self.settings
        tiles: Dict[int, Dict[str, Any]] = {}
        by_tg = {}
        for d in self.destinations:
            tg = _tgid(d.get('talkgroup'))
            if d.get('backend') == 'p25' and tg and tg not in by_tg:
                by_tg[tg] = d

        for t in settings.get('talkgroups') or []:
            tg = _tgid(t.get('id', t.get('tgid')) if isinstance(t, dict) else t)
            if tg is None:
                logger.warning(f"dispatch.talkgroups: skipping bad entry {t!r}")
                continue
            t = t if isinstance(t, dict) else {}
            dest = by_tg.get(tg)
            tiles.setdefault(tg, {
                'tgid': tg, 'name': str(t.get('name') or (dest or {}).get('name') or f'TG {tg}'),
                'color': t.get('color') if t.get('color') in COLORS else None,
                'destination': (dest or {}).get('id'), 'source': 'config',
            })
        if settings.get('include_destinations', True):
            for tg, dest in by_tg.items():
                tiles.setdefault(tg, {'tgid': tg, 'name': str(dest.get('name') or f'TG {tg}'), 'color': None,
                                      'destination': dest['id'], 'source': 'destination'})
        fne = {t['tgid']: t for t in (dvm or {}).get('active_talkgroups') or []}
        if settings.get('show_fne_talkgroups', True):
            for tg in sorted(fne):
                tiles.setdefault(tg, {'tgid': tg, 'name': f'TG {tg}', 'color': None,
                                      'destination': None, 'source': 'fne'})
        for tile in tiles.values():
            info = fne.get(tile['tgid'])
            tile['on_fne'] = info is not None
            tile['affiliated'] = bool(info and info.get('affiliated'))
        return list(tiles.values())

    def _destination(self, tg: int, name: str, dest_id: Optional[str]) -> Dict[str, Any]:
        for d in self.destinations:
            if d['id'] == dest_id:
                return d
        return {'id': f'tg-{tg}', 'name': name, 'backend': 'p25', 'talkgroup': str(tg)}

    # -- state ------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        snap = snapshot()
        dvm = snap.get('dvm') or {}
        tiles = self.talkgroups(dvm)
        by_tg = {t['tgid']: t for t in tiles}
        for t in tiles:
            t.update(rx=None, tx=None, live=False, last_rx=None, last_page=None)

        for call in reversed(dvm.get('recent_calls') or []):
            if call.get('dst_id') in by_tg:
                by_tg[call['dst_id']]['last_rx'] = call
        for call in dvm.get('calls') or []:
            if call.get('dst_id') in by_tg:
                by_tg[call['dst_id']]['rx'] = call
        for job in (self.dispatcher.jobs() if self.dispatcher else []):
            t = by_tg.get(_tgid(job.get('talkgroup')))
            if t and t['tx'] != 'sending':
                t['tx'] = job['state']
        live = self.live.active() if self.live else None
        for tg in (live or {}).get('talkgroups', []):
            if tg in by_tg:
                by_tg[tg]['tx'] = 'sending'
                by_tg[tg]['live'] = True
        for d in reversed(snap.get('dispatches') or []):
            t = by_tg.get(_tgid(d.get('talkgroup')))
            if t:
                t['last_page'] = {k: d.get(k) for k in ('timestamp', 'success', 'status', 'detail', 'message', 'sound')}

        fw = snap.get('forewarned') or {}
        return {
            'now': datetime.now().isoformat(timespec='milliseconds'),
            'talkgroups': tiles,
            'tones': self.tones(),
            'live': live,
            'live_available': self.live is not None,
            'fne': {k: dvm.get(k) for k in ('enabled', 'connected', 'state', 'last_error', 'master', 'rtt_ms')},
            'alert': {'level': (fw.get('local_alert_state') or {}).get('level', 'none'),
                      'reason': (fw.get('local_alert_state') or {}).get('reason', ''),
                      'reachable': fw.get('reachable', False)},
            'activity': self._activity(dvm, snap.get('dispatches') or [], by_tg),
        }

    @staticmethod
    def _activity(dvm, dispatches, by_tg) -> List[Dict[str, Any]]:
        """Recent calls heard and pages sent, newest first."""
        items = []
        for c in (dvm.get('calls') or []) + (dvm.get('recent_calls') or []):
            tile = by_tg.get(c.get('dst_id'))
            items.append({'kind': 'rx', 'time': c.get('started'), 'tgid': c.get('dst_id'),
                          'name': tile['name'] if tile else f"TG {c.get('dst_id')}",
                          'text': f"RID {c.get('src_id')}", 'active': c.get('active', False),
                          'ended': c.get('ended')})
        for d in dispatches[:20]:
            items.append({'kind': 'tx', 'time': d.get('timestamp'), 'tgid': _tgid(d.get('talkgroup')),
                          'name': d.get('destination'), 'ok': d.get('success'),
                          'text': (f"[{d['sound']}] " if d.get('sound') else '') + (d.get('message') or ''),
                          'detail': d.get('detail')})
        items.sort(key=lambda i: i.get('time') or '', reverse=True)
        return items[:30]

    # -- paging -----------------------------------------------------------------

    def _targets(self, talkgroups) -> List[Dict[str, Any]]:
        if not isinstance(talkgroups, list) or not talkgroups:
            raise ValueError('select at least one talkgroup')
        tiles = {t['tgid']: t for t in self.talkgroups(snapshot().get('dvm'))}
        out = []
        for v in talkgroups:
            tg = _tgid(v)
            if tg is None:
                raise ValueError(f"'{v}' is not a talkgroup number (1-65535)")
            tile = tiles.get(tg, {'name': f'TG {tg}', 'destination': None})
            out.append(self._destination(tg, tile['name'], tile.get('destination')))
        return out

    def live_start(self, talkgroups, sound: Optional[str] = None):
        """Start a live call to these talkgroups; returns the LiveSession."""
        from src.p25.live import LiveError
        if self.live is None:
            raise LiveError('live talk needs the DVM FNE connection (dvm.enabled)')
        targets = [{'tgid': _tgid(d['talkgroup']), 'name': d.get('name', d['id'])}
                   for d in self._targets(talkgroups)]
        return self.live.start(targets, sound)

    def page(self, talkgroups, message: str = '', sound: Optional[str] = None) -> int:
        """Queue a page (alert sound and/or spoken message) to each talkgroup."""
        if self.dispatcher is None:
            raise RuntimeError('paging not configured')
        targets = self._targets(talkgroups)
        for dest in targets:
            self.dispatcher.manual_to(dest, message, sound, source='Dispatch console')
        return len(targets)

    def talk(self, talkgroups, pcm16: bytes, rate: int, sound: Optional[str] = None) -> int:
        """Queue recorded speech (little-endian 16-bit mono at `rate`) to each
        talkgroup, optionally after an alert sound."""
        if self.dispatcher is None:
            raise RuntimeError('paging not configured')
        if not 8000 <= rate <= 192000:
            raise ValueError(f'unsupported sample rate {rate}')
        x = np.frombuffer(pcm16[:len(pcm16) // 2 * 2], dtype='<i2').astype(np.float64) / 32768
        if x.size < rate * 0.3:
            raise ValueError('recording too short - hold the talk button while speaking')
        if x.size > rate * MAX_TALK_SECONDS:
            raise ValueError(f'recording longer than {MAX_TALK_SECONDS}s')
        audio = dsp.resample(x, rate, dsp.RATE)
        targets = self._targets(talkgroups)
        for dest in targets:
            self.dispatcher.manual_to(dest, '', sound, source='Dispatch console', audio=audio)
        return len(targets)
