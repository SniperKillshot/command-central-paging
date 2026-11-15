"""DVM FNE peer client.

Logs in to a dvmfne master as a network peer and keeps the connection up,
following dvmhost's Network::clock() state machine:

    RPTL (login, peer ID)       -> ACK carrying a 4-byte salt
    RPTK (SHA-256 of salt+pw)   -> ACK
    RPTC (JSON peer config)     -> ACK  => running
    PING every ping_interval    -> PONG (with the FNE's clock)

Failed or silent connections are torn down and retried. A NAK says why the
FNE refused us (wrong password, peer ID not in its ACL, duplicate login...),
and that reason is shown on the dashboard.

While connected the FNE also sends this peer its active talkgroup list and
repeats network voice traffic; both are recorded for the dashboard so the
link can be seen working in both directions.
"""
import logging
import random
import socket
import threading
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

from src.dvm import protocol as p
from src.dvm.protocol import Func

logger = logging.getLogger(__name__)

SOFTWARE_ID = 'CCPAGING_R01'
RECV_TIMEOUT = 0.25            # seconds; also the timer resolution
CALL_IDLE_TIMEOUT = 3.0        # a heard call with no frames for this long has ended
RECENT_CALLS = 20
LONG_BACKOFF = 60.0            # after ACL / duplicate-connection rejections
ALIAS_FIRST_DELAY = 3.0        # after login, before asking for the radio alias table
ALIAS_RETRY = 60.0             # no (complete) answer: ask again this much later


class PeerState(str, Enum):
    STOPPED = 'stopped'
    CONNECTING = 'connecting'            # waiting to (re)open the socket and send RPTL
    WAITING_LOGIN = 'waiting_login'      # RPTL sent, waiting for the salt
    WAITING_AUTH = 'waiting_auth'        # RPTK sent
    WAITING_CONFIG = 'waiting_config'    # RPTC sent
    RUNNING = 'running'


def _now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


class FNEPeer:
    def __init__(self, settings: Dict[str, Any],
                 on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
                 clock: Callable[[], float] = time.monotonic):
        try:
            self.peer_id = int(settings.get('peer_id'))
        except (TypeError, ValueError):
            raise ValueError('dvm.peer_id must be a number') from None
        if not 1 <= self.peer_id <= p.MAX_PEER_ID:
            raise ValueError(f'dvm.peer_id must be between 1 and {p.MAX_PEER_ID}')
        self.address = str(settings.get('address') or '').strip()
        if not self.address:
            raise ValueError('dvm.address (the FNE host) is required')
        self.port = int(settings.get('port') or p.DEFAULT_PORT)
        self.password = str(settings.get('password') or '')
        key = p.parse_preshared_key(settings.get('preshared_key'))
        self._cipher = p.PacketCipher(key) if key else None
        self.ping_interval = float(settings.get('ping_interval', 10))
        self.retry_interval = float(settings.get('retry_interval', 10))
        self.timeout = float(settings.get('timeout', 60))
        self.peer_config = p.build_peer_config(settings, SOFTWARE_ID)
        self.identity = self.peer_config['identity']
        # radio aliases, asked of the FNE after login and every so often
        # (NET_FUNC 0x85 to its metadata port; 0 minutes = never)
        self.alias_interval = float(settings.get('alias_sync_minutes', 10) or 0) * 60

        self.on_status = on_status
        # on_voice(stream_id, P25Header, payload): every P25 frame heard, e.g. for RX audio
        self.on_voice: Optional[Callable[[int, Any, bytes], None]] = None
        self._clock = clock
        self._rng = random.SystemRandom()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._send_lock = threading.Lock()
        self._sock: Optional[socket.socket] = None
        self._master_addr = None
        self._metadata_addr = None
        self._aliases: Dict[int, str] = {}
        self._alias_lock = threading.Lock()
        self._alias_stream_id = 0
        self._alias_frags = p.FragmentAssembler()
        self._next_alias_sync = 0.0

        self.state = PeerState.STOPPED
        self._retry_at = 0.0
        self._step_started = 0.0
        self._last_tx = 0.0
        self._last_rx = 0.0
        self._last_ping = 0.0
        self._ping_sent_at: Optional[float] = None
        self._login_stream_id = 0
        self._salt = b''
        self._pkt_seq = 0
        # stream ID -> call heard from the network (several talkgroups can be busy at once)
        self._calls: Dict[int, Dict[str, Any]] = {}
        self._call_last_frame: Dict[int, float] = {}

        self.status: Dict[str, Any] = {
            'enabled': True,
            'state': self.state.value,
            'connected': False,
            'master': f'{self.address}:{self.port}',
            'peer_id': self.peer_id,
            'identity': self.identity,
            'encrypted': self._cipher is not None,
            'remote_peer_id': None,
            'connected_since': None,
            'last_pong': None,
            'rtt_ms': None,
            'last_error': None,
            'reconnects': 0,
            'metadata_port': None,
            'radio_aliases': 0,
            'aliases_synced': None,
            'active_talkgroups': [],
            'rx_frames': {},
            'last_call': None,
            'calls': [],             # calls in progress, oldest first
            'recent_calls': [],      # ended calls, newest first
        }
        self._talkgroups: Dict[int, Dict[str, Any]] = {}
        # stream ID -> (ICC command, destination ID) for calls the FNE refused
        self._icc: Dict[int, tuple] = {}

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self.state == PeerState.RUNNING

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._set_state(PeerState.CONNECTING)
        self._retry_at = self._clock()
        self._thread = threading.Thread(target=self._run, name='dvm-peer', daemon=True)
        self._thread.start()
        logger.info(f"DVM peer {self.peer_id} ({self.identity}) starting, FNE {self.address}:{self.port}"
                    f"{' (encrypted)' if self._cipher else ''}")

    def stop(self, timeout: float = 2.0) -> None:
        """Disconnect cleanly (RPT_DISC) and stop the thread."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    def snapshot(self) -> Dict[str, Any]:
        return dict(self.status)

    def send(self, func: int, subfunc: int, payload: bytes, stream_id: Optional[int] = None,
             seq: int = p.END_OF_CALL_SEQ, metadata: bool = False) -> bool:
        """Send one message to the FNE (its metadata port, traffic port + 1, if
        metadata). Safe to call from other threads (this is what a voice page
        transmitter will use)."""
        addr = self._metadata_addr if metadata else self._master_addr
        if self._sock is None or addr is None:
            return False
        datagram = p.encode_frame(payload, func, subfunc,
                                  stream_id if stream_id is not None else self.new_stream_id(),
                                  self.peer_id, self.peer_id, seq, int(self._clock() * 8000))
        if self._cipher:
            datagram = self._cipher.wrap(datagram)
        try:
            with self._send_lock:
                self._sock.sendto(datagram, addr)
        except OSError as e:
            logger.warning(f"DVM peer send failed: {e}")
            return False
        self._last_tx = self._clock()
        return True

    def alias(self, rid) -> Optional[str]:
        """The radio's alias from the FNE's alias table, or None."""
        try:
            rid = int(rid)
        except (TypeError, ValueError):
            return None
        with self._alias_lock:
            return self._aliases.get(rid)

    def aliases(self) -> Dict[int, str]:
        with self._alias_lock:
            return dict(self._aliases)

    def new_stream_id(self) -> int:
        return self._rng.randint(0x00000001, 0xFFFFFFFE)

    def active_talkgroup_ids(self) -> set:
        return set(self._talkgroups)

    def call_refusal(self, stream_id: int) -> Optional[str]:
        """If the FNE sent an in-call control (reject/busy) for this stream,
        a human-readable reason; otherwise None."""
        icc = self._icc.get(stream_id)
        return p.ICC_NAMES.get(icc[0], f'in-call control 0x{icc[0]:02X}') if icc else None

    def forget_call(self, stream_id: int) -> None:
        self._icc.pop(stream_id, None)

    # ------------------------------------------------------------------
    #  Thread loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    if self.state == PeerState.CONNECTING:
                        if self._clock() >= self._retry_at:
                            self._connect()
                        else:
                            self._stop.wait(RECV_TIMEOUT)
                        continue
                    self._receive_one()
                    self._tick()
                except Exception as e:  # never let the peer thread die
                    logger.exception("DVM peer error")
                    self._reconnect(f'internal error: {e}', self.retry_interval, send_disc=False)
        finally:
            if self.state == PeerState.RUNNING:
                self.send(Func.RPT_DISC, p.SUBFUNC_NOP, p.SINGLE_BYTE)
                logger.info(f"DVM peer {self.peer_id} disconnected from FNE")
            self._close_socket()
            self._set_state(PeerState.STOPPED)

    def _connect(self) -> None:
        self._close_socket()
        try:
            infos = socket.getaddrinfo(self.address, self.port, type=socket.SOCK_DGRAM)
        except socket.gaierror as e:
            self._reconnect(f'cannot resolve {self.address}: {e}', self.retry_interval, send_disc=False)
            return
        family, socktype, proto, _, sockaddr = infos[0]
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(RECV_TIMEOUT)
        if hasattr(socket, 'SIO_UDP_CONNRESET'):
            # Windows: stop ICMP "port unreachable" surfacing as recv errors.
            try:
                sock.ioctl(socket.SIO_UDP_CONNRESET, False)
            except OSError:
                pass
        self._sock = sock
        self._master_addr = sockaddr
        self._metadata_addr = (sockaddr[0], sockaddr[1] + 1) + tuple(sockaddr[2:])
        self._talkgroups.clear()
        self.status['active_talkgroups'] = []
        self._send_login()

    def _receive_one(self) -> None:
        try:
            data, addr = self._sock.recvfrom(p.MAX_DATAGRAM)
        except socket.timeout:
            return
        except ConnectionResetError:
            return
        except OSError as e:
            if not self._stop.is_set():
                self._reconnect(f'socket error: {e}', self.retry_interval, send_disc=False)
            return
        if addr[:2] != self._master_addr[:2]:
            logger.debug(f"DVM peer ignoring packet from unexpected source {addr}")
            return
        self._handle_datagram(data)

    def _tick(self) -> None:
        now = self._clock()
        if self.state in (PeerState.WAITING_LOGIN, PeerState.WAITING_AUTH, PeerState.WAITING_CONFIG):
            if now - self._step_started >= self.timeout:
                self._reconnect('login timed out - no answer from FNE', self.retry_interval, send_disc=False)
            elif now - self._last_tx >= self.retry_interval:
                logger.info(f"DVM peer {self.peer_id} retrying {self.state.value}")
                self._resend_step()
        elif self.state == PeerState.RUNNING:
            if now - self._last_rx >= self.timeout:
                self._reconnect(f'connection timed out - nothing from FNE for {int(self.timeout)}s',
                                self.retry_interval, send_disc=False)
            elif now - self._last_ping >= self.ping_interval:
                self._last_ping = now
                self._ping_sent_at = now
                self.send(Func.PING, p.SUBFUNC_NOP, p.SINGLE_BYTE)
            if self.alias_interval and now >= self._next_alias_sync:
                self._request_aliases(now)

        for stream_id, last in list(self._call_last_frame.items()):
            if now - last > CALL_IDLE_TIMEOUT:
                self._end_call(stream_id)

    # ------------------------------------------------------------------
    #  Login sequence
    # ------------------------------------------------------------------

    def _next_seq(self, reset: bool = False) -> int:
        if reset:
            self._pkt_seq = 0
        seq = self._pkt_seq
        self._pkt_seq = (self._pkt_seq + 1) % p.END_OF_CALL_SEQ
        return seq

    def _send_login(self) -> None:
        self._login_stream_id = self.new_stream_id()
        self.status['remote_peer_id'] = None
        self._set_state(PeerState.WAITING_LOGIN)
        self._step_started = self._clock()
        self.send(Func.RPTL, p.SUBFUNC_NOP, p.login_payload(self.peer_id),
                  self._login_stream_id, self._next_seq(reset=True))

    def _send_auth(self) -> None:
        self.send(Func.RPTK, p.SUBFUNC_NOP, p.auth_payload(self.peer_id, self._salt, self.password),
                  self._login_stream_id, self._next_seq())

    def _send_config(self) -> None:
        self.send(Func.RPTC, p.SUBFUNC_NOP, p.config_payload(self.peer_config),
                  self._login_stream_id, p.END_OF_CALL_SEQ)

    def _resend_step(self) -> None:
        {PeerState.WAITING_LOGIN: lambda: self.send(
            Func.RPTL, p.SUBFUNC_NOP, p.login_payload(self.peer_id), self._login_stream_id,
            self._next_seq(reset=True)),
         PeerState.WAITING_AUTH: self._send_auth,
         PeerState.WAITING_CONFIG: self._send_config}[self.state]()

    # ------------------------------------------------------------------
    #  Incoming messages
    # ------------------------------------------------------------------

    def _handle_datagram(self, data: bytes) -> None:
        if self._cipher:
            data = self._cipher.unwrap(data)
            if data is None:
                logger.debug("DVM peer dropped an unencrypted packet (pre-shared key is set)")
                return
        try:
            frame = p.decode_frame(data)
        except p.FrameError as e:
            logger.debug(f"DVM peer dropped malformed packet: {e}")
            return
        if frame.peer_id != self.peer_id:
            logger.debug(f"DVM peer ignoring packet for peer {frame.peer_id}")
            return
        self._last_rx = self._clock()

        handler = {
            Func.ACK: self._on_ack,
            Func.NAK: self._on_nak,
            Func.MST_DISC: self._on_master_disconnect,
            Func.PONG: self._on_pong,
            Func.MASTER: self._on_master,
            Func.PROTOCOL: self._on_protocol,
            Func.INCALL_CTRL: self._on_incall_ctrl,
            Func.RADIO_ALIAS_SYNC: self._on_alias_sync,
        }.get(frame.func)
        if handler:
            handler(frame)
        else:
            logger.debug(f"DVM peer ignoring function 0x{frame.func:02X}/0x{frame.subfunc:02X}")

    def _on_ack(self, frame: p.Frame) -> None:
        if self.state == PeerState.WAITING_LOGIN:
            try:
                self._salt = p.ack_salt(frame.payload)
            except p.FrameError as e:
                self._reconnect(str(e), self.retry_interval, send_disc=False)
                return
            self._set_state(PeerState.WAITING_AUTH)
            self._step_started = self._clock()
            self._send_auth()
        elif self.state == PeerState.WAITING_AUTH:
            self._set_state(PeerState.WAITING_CONFIG)
            self._step_started = self._clock()
            self._send_config()
        elif self.state == PeerState.WAITING_CONFIG:
            now = self._clock()
            self._login_stream_id = 0
            self._next_seq(reset=True)
            self._last_ping = now
            self.status.update({
                'remote_peer_id': frame.ssrc,
                'connected_since': _now_iso(),
                'last_error': None,
                'metadata_port': bool(p.ack_flags(frame.payload) & 0x80),
            })
            self._set_state(PeerState.RUNNING)
            self._next_alias_sync = now + ALIAS_FIRST_DELAY
            logger.info(f"DVM peer {self.peer_id} logged in to FNE {self.address}:{self.port} "
                        f"(FNE peer ID {frame.ssrc})")

    # ------------------------------------------------------------------
    #  Radio aliases (dvmhost MetadataNetwork radioAliasSync)
    # ------------------------------------------------------------------

    def _request_aliases(self, now: float) -> None:
        """Ask the FNE's metadata port for its radio alias table. The answer
        comes back on this socket as fragments with the same stream ID."""
        self._alias_stream_id = self.new_stream_id()
        self._alias_frags.clear()
        self._next_alias_sync = now + ALIAS_RETRY       # pushed out once an answer completes
        self.send(Func.RADIO_ALIAS_SYNC, p.SUBFUNC_NOP, p.SINGLE_BYTE, self._alias_stream_id,
                  metadata=True)

    def _on_alias_sync(self, frame: p.Frame) -> None:
        if frame.subfunc != p.SUBFUNC_NOP or not self._alias_stream_id or frame.stream_id != self._alias_stream_id:
            return
        try:
            data = self._alias_frags.add(frame.payload)
        except p.FrameError as e:
            logger.warning(f"DVM peer {self.peer_id}: radio alias sync failed: {e}")
            self._alias_stream_id = 0
            return
        if data is None:
            return                                      # more fragments to come
        self._alias_stream_id = 0
        aliases = p.parse_radio_aliases(data)
        with self._alias_lock:
            changed = aliases != self._aliases
            self._aliases = aliases
        self._next_alias_sync = self._clock() + self.alias_interval
        self.status.update(radio_aliases=len(aliases), aliases_synced=_now_iso())
        if changed:
            logger.info(f"DVM peer {self.peer_id}: {len(aliases)} radio alias(es) from the FNE")
        self._publish()

    def _on_nak(self, frame: p.Frame) -> None:
        reason = p.nak_reason(frame.payload)
        text = p.NAK_REASONS.get(reason, 'general failure') if reason is not None else 'refused'
        logger.warning(f"DVM peer {self.peer_id} NAK from FNE during {self.state.value}: {text}")
        if reason in (p.NAK_PEER_ACL, p.NAK_DUPLICATE_CONN):
            backoff = LONG_BACKOFF
        else:
            backoff = self.retry_interval
        self._reconnect(f'FNE refused: {text}', backoff, send_disc=False)

    def _on_master_disconnect(self, frame: p.Frame) -> None:
        logger.warning(f"DVM peer {self.peer_id}: FNE closed the connection")
        self._reconnect('FNE closed the connection', self.retry_interval, send_disc=False)

    def _on_pong(self, frame: p.Frame) -> None:
        if self._ping_sent_at is not None:
            self.status['rtt_ms'] = round((self._clock() - self._ping_sent_at) * 1000, 1)
            self._ping_sent_at = None
        self.status['last_pong'] = _now_iso()
        self._publish()

    def _on_master(self, frame: p.Frame) -> None:
        if frame.subfunc == p.MASTER_ACTIVE_TGS:
            for tgid, info in p.parse_talkgroup_list(frame.payload, active=True):
                self._talkgroups[tgid] = {'tgid': tgid, **info}
        elif frame.subfunc == p.MASTER_DEACTIVE_TGS:
            for tgid, _ in p.parse_talkgroup_list(frame.payload, active=False):
                self._talkgroups.pop(tgid, None)
        else:
            return
        self.status['active_talkgroups'] = [self._talkgroups[k] for k in sorted(self._talkgroups)]
        logger.info(f"DVM peer {self.peer_id}: FNE reports {len(self._talkgroups)} active talkgroup(s)")
        self._publish()

    def _on_incall_ctrl(self, frame: p.Frame) -> None:
        if len(frame.payload) < 14:
            return
        command = frame.payload[10]
        dst = int.from_bytes(frame.payload[11:14], 'big')
        self._icc[frame.stream_id] = (command, dst)
        if len(self._icc) > 64:
            self._icc.pop(next(iter(self._icc)))
        logger.warning(f"DVM peer {self.peer_id}: FNE in-call control for TG {dst}: "
                       f"{p.ICC_NAMES.get(command, hex(command))}")

    def _on_protocol(self, frame: p.Frame) -> None:
        name = p.PROTOCOL_NAMES.get(frame.subfunc, f'0x{frame.subfunc:02X}')
        counts = dict(self.status['rx_frames'])
        counts[name] = counts.get(name, 0) + 1
        self.status['rx_frames'] = counts
        if frame.subfunc != p.PROTOCOL_P25:
            return
        hdr = p.parse_p25_header(frame.payload)
        if hdr is None:
            return
        ending = hdr.duid in (p.P25_DUID_TDU, p.P25_DUID_TDULC) or frame.seq == p.END_OF_CALL_SEQ
        if frame.stream_id not in self._calls:
            if ending:
                return          # stray terminator for a call we never heard
            call = {'protocol': 'p25', 'src_id': hdr.src_id, 'dst_id': hdr.dst_id,
                    'stream_id': frame.stream_id, 'started': _now_iso(), 'ended': None, 'active': True}
            self._calls[frame.stream_id] = call
            self.status['last_call'] = call
            self._publish_calls()
            logger.info(f"DVM peer heard P25 call from {hdr.src_id} to TG {hdr.dst_id}")
            self._publish()
        self._call_last_frame[frame.stream_id] = self._clock()
        if self.on_voice:
            try:
                self.on_voice(frame.stream_id, hdr, frame.payload)
            except Exception:
                logger.exception("DVM peer voice callback failed")
        if ending:
            self._end_call(frame.stream_id)

    def _end_call(self, stream_id: int) -> None:
        call = self._calls.pop(stream_id, None)
        self._call_last_frame.pop(stream_id, None)
        if call is None:
            return
        ended = {**call, 'active': False, 'ended': _now_iso()}
        if (self.status.get('last_call') or {}).get('stream_id') == stream_id:
            self.status['last_call'] = ended
        self.status['recent_calls'] = ([ended] + self.status['recent_calls'])[:RECENT_CALLS]
        self._publish_calls()
        self._publish()

    def _end_all_calls(self) -> None:
        for stream_id in list(self._calls):
            self._end_call(stream_id)

    def _publish_calls(self) -> None:
        self.status['calls'] = [dict(c) for c in self._calls.values()]

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _reconnect(self, reason: str, delay: float, send_disc: bool = True) -> None:
        if send_disc and self.state == PeerState.RUNNING:
            self.send(Func.RPT_DISC, p.SUBFUNC_NOP, p.SINGLE_BYTE)
        was_running = self.state == PeerState.RUNNING
        self._end_all_calls()
        self._close_socket()
        self.status['last_error'] = reason
        self.status['connected_since'] = None
        self.status['rtt_ms'] = None
        if was_running:
            self.status['reconnects'] += 1
        self._retry_at = self._clock() + delay
        logger.warning(f"DVM peer {self.peer_id}: {reason}; retrying in {delay:g}s")
        self._set_state(PeerState.CONNECTING)

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None

    def _set_state(self, state: PeerState) -> None:
        self.state = state
        self.status['state'] = state.value
        self.status['connected'] = state == PeerState.RUNNING
        self._publish()

    def _publish(self) -> None:
        if self.on_status:
            try:
                self.on_status(self.snapshot())
            except Exception:
                logger.exception("DVM peer status callback failed")
