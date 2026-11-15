"""Talkgroup names from the DVM FNE's REST API.

The peer connection only carries talkgroup numbers; the names live in the
FNE's talkgroup rules file, which dvmfne serves over its REST API (off by
default - the FNE needs restEnable: true, a restPassword, and restAddress
reachable from here; see the README).

    PUT /auth       {"auth": hex(SHA-256(password))} -> {"token": "<decimal>"}
    GET /tg/query   header X-DVM-Auth-Token: <token>
                    -> {"tgs": [{"name", "alias", "invalid",
                                 "source": {"tgid", "slot"},
                                 "config": {"active", "affiliated", ...}}]}

TalkgroupDirectory fetches the list at start and every few minutes after,
keeping the last good copy if the FNE stops answering.
"""
import hashlib
import logging
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

AUTH_HEADER = 'X-DVM-Auth-Token'
RETRY_SECONDS = 60          # after a failed fetch
TIMEOUT = 8


class FNERestError(Exception):
    pass


def password_hash(password: str) -> str:
    """What PUT /auth wants: hex SHA-256 of the password (the FNE uses at
    most its first 64 characters)."""
    return hashlib.sha256(password[:64].encode('utf-8')).hexdigest()


class FNERest:
    """Minimal client for the dvmfne REST API."""

    def __init__(self, base_url: str, password: str, verify_tls: bool = True,
                 session: Optional[requests.Session] = None, timeout: float = TIMEOUT):
        self.base_url = base_url.rstrip('/')
        self.password = password
        self.verify = verify_tls
        self.session = session or requests.Session()
        self.timeout = timeout
        self.token: Optional[str] = None

    def _error(self, r: requests.Response, what: str) -> FNERestError:
        try:
            msg = r.json().get('message') or ''
        except ValueError:
            msg = ''
        return FNERestError(f'{what}: HTTP {r.status_code}{" - " + msg if msg else ""}')

    def authenticate(self) -> None:
        try:
            r = self.session.put(f'{self.base_url}/auth', json={'auth': password_hash(self.password)},
                                 timeout=self.timeout, verify=self.verify)
        except requests.RequestException as e:
            raise FNERestError(f'FNE REST API unreachable at {self.base_url}: {e}') from e
        if r.status_code != 200:
            raise self._error(r, 'FNE REST login refused')
        token = (r.json() or {}).get('token')
        if not token:
            raise FNERestError('FNE REST login returned no token')
        self.token = str(token)

    def get(self, path: str) -> Dict[str, Any]:
        """GET with the auth token, logging in (again) when needed."""
        for attempt in range(2):
            if self.token is None:
                self.authenticate()
            try:
                r = self.session.get(f'{self.base_url}{path}', headers={AUTH_HEADER: self.token},
                                     timeout=self.timeout, verify=self.verify)
            except requests.RequestException as e:
                raise FNERestError(f'FNE REST API unreachable at {self.base_url}: {e}') from e
            if r.status_code in (401, 403) and attempt == 0:
                self.token = None            # expired or replaced: log in again once
                continue
            if r.status_code != 200:
                raise self._error(r, f'FNE REST {path}')
            return r.json()
        raise FNERestError(f'FNE REST {path}: not authorised')

    def talkgroups(self) -> List[Dict[str, Any]]:
        """The FNE's talkgroup rules: [{'tgid', 'slot', 'name', 'alias',
        'active', 'affiliated', 'invalid'}]."""
        out = []
        for tg in self.get('/tg/query').get('tgs') or []:
            source, config = tg.get('source') or {}, tg.get('config') or {}
            try:
                tgid = int(source.get('tgid'))
            except (TypeError, ValueError):
                continue
            out.append({
                'tgid': tgid, 'slot': int(source.get('slot') or 0),
                'name': str(tg.get('name') or '').strip(), 'alias': str(tg.get('alias') or '').strip(),
                'active': bool(config.get('active')), 'affiliated': bool(config.get('affiliated')),
                'invalid': bool(tg.get('invalid')),
            })
        return out


class TalkgroupDirectory:
    """Talkgroup names from the FNE, refreshed in the background."""

    def __init__(self, settings: Dict[str, Any], client: Optional[FNERest] = None,
                 on_status: Optional[Callable[[Dict[str, Any]], None]] = None):
        s = settings or {}
        self.refresh_seconds = max(60.0, float(s.get('names_refresh_minutes', 10)) * 60)
        if client is None:
            host = s.get('rest_address') or s.get('address') or ''
            scheme = 'https' if s.get('rest_https') else 'http'
            client = FNERest(f"{scheme}://{host}:{int(s.get('rest_port') or 9990)}",
                             str(s.get('rest_password') or ''), bool(s.get('rest_verify_tls', True)))
        self.client = client
        self.on_status = on_status
        self._names: Dict[int, str] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.status: Dict[str, Any] = {'enabled': True, 'ok': None, 'count': 0, 'fetched': None,
                                       'error': None, 'url': client.base_url}

    # -- lookups ----------------------------------------------------------------

    def name(self, tgid: int) -> Optional[str]:
        with self._lock:
            return self._names.get(int(tgid))

    def names(self) -> Dict[int, str]:
        with self._lock:
            return dict(self._names)

    # -- fetching ---------------------------------------------------------------

    def refresh(self) -> bool:
        """Fetch the list now. Keeps the previous names if it fails."""
        try:
            tgs = self.client.talkgroups()
        except (FNERestError, ValueError) as e:
            first = self.status['ok'] is not False or self.status['error'] != str(e)
            self.status.update(ok=False, error=str(e))
            if first:
                logger.warning(f'Talkgroup names: {e}')
            self._publish()
            return False
        names = {}
        for tg in tgs:
            label = tg['name'] or tg['alias']
            if label and not tg['invalid']:
                names.setdefault(tg['tgid'], label)
        with self._lock:
            changed = names != self._names
            self._names = names
        if changed or self.status['ok'] is not True:
            logger.info(f'Talkgroup names: {len(names)} from the FNE')
        self.status.update(ok=True, error=None, count=len(names),
                           fetched=datetime.now().isoformat(timespec='seconds'))
        self._publish()
        return True

    def _publish(self) -> None:
        if self.on_status:
            try:
                self.on_status(dict(self.status))
            except Exception:
                logger.exception('talkgroup name status callback failed')

    def _run(self) -> None:
        while not self._stop.is_set():
            ok = self.refresh()
            self._stop.wait(self.refresh_seconds if ok else RETRY_SECONDS)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name='fne-tg-names', daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
