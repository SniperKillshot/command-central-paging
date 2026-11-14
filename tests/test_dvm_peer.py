"""Tests for the DVM FNE peer connection.

FakeFNE is a minimal master that follows dvmhost's src/fne/network/
TrafficNetwork.cpp for the parts a peer sees: salt challenge, SHA-256 check,
config ACK, PING/PONG, NAK reasons, the active talkgroup list and repeated
P25 traffic.
"""
import hashlib
import json
import socket
import struct
import threading
import time
import unittest

from src.dvm import FNEPeer, PeerState
from src.dvm import protocol as p
from src.dvm.protocol import Func

FNE_PEER_ID = 1
PEER_ID = 9000001
PASSWORD = 'RPT1234'
PSK_HEX = '000102030405060708090A0B0C0D0E0F000102030405060708090A0B0C0D0E0F'


class FakeFNE:
    def __init__(self, password=PASSWORD, preshared_key=None, acl_reject=False, answer_pings=True):
        self.password = password
        self.cipher = p.PacketCipher(p.parse_preshared_key(preshared_key)) if preshared_key else None
        self.acl_reject = acl_reject
        self.answer_pings = answer_pings
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', 0))
        self.sock.settimeout(0.1)
        self.port = self.sock.getsockname()[1]
        self.received = []          # decoded frames, in order
        self.peer_config = None
        self.salt = b''
        self.peer_addr = None
        self.state = {}             # peer_id -> state name
        self.logins = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join(1)
        self.sock.close()

    # -- transport --------------------------------------------------------
    def send(self, func, subfunc, payload, stream_id=1, peer_id=PEER_ID, seq=p.END_OF_CALL_SEQ):
        data = p.encode_frame(payload, func, subfunc, stream_id, peer_id, FNE_PEER_ID, seq)
        if self.cipher:
            data = self.cipher.wrap(data)
        self.sock.sendto(data, self.peer_addr)

    def _run(self):
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(4096)
            except (socket.timeout, OSError):
                continue
            if self.cipher:
                data = self.cipher.unwrap(data)
                if data is None:
                    continue
            frame = p.decode_frame(data)
            self.received.append(frame)
            self.peer_addr = addr
            self._handle(frame)

    # -- master logic (TrafficNetwork.cpp) ---------------------------------
    def _writePeerACK(self, frame, data=b''):
        buf = struct.pack('>I', frame.peer_id) + b'\x00\x00' + data
        self.send(Func.ACK, p.SUBFUNC_NOP, buf + b'\x00' * 4, frame.stream_id)

    def _writePeerNAK(self, frame, reason):
        buf = b'\x00' * 6 + struct.pack('>IH', frame.peer_id, reason)
        self.send(Func.NAK, p.SUBFUNC_NOP, buf, frame.stream_id)

    def _handle(self, frame):
        f = frame.func
        if f == Func.RPTL:
            self.logins += 1
            if self.acl_reject:
                self._writePeerNAK(frame, p.NAK_PEER_ACL)
                return
            self.salt = struct.pack('>I', 0x1234ABCD)
            self.state[frame.peer_id] = 'auth'
            self._writePeerACK(frame, self.salt)
        elif f == Func.RPTK:
            expected = hashlib.sha256(self.salt + self.password.encode()).digest()
            if self.state.get(frame.peer_id) == 'auth' and frame.payload[8:] == expected:
                self.state[frame.peer_id] = 'config'
                self._writePeerACK(frame)
            else:
                self._writePeerNAK(frame, 3)
        elif f == Func.RPTC:
            if self.state.get(frame.peer_id) != 'config':
                self._writePeerNAK(frame, 4)
                return
            self.peer_config = json.loads(frame.payload[8:].decode())
            self.state[frame.peer_id] = 'running'
            self._writePeerACK(frame, b'\x80')
        elif f == Func.PING:
            if self.answer_pings and self.state.get(frame.peer_id) == 'running':
                now = int(time.time() * 1000)
                self.send(Func.PONG, p.SUBFUNC_NOP, b'\x00' * 6 + struct.pack('>Q', now), frame.stream_id)
        elif f == Func.RPT_DISC:
            self.state[frame.peer_id] = 'disconnected'

    # -- unsolicited master -> peer traffic ---------------------------------
    def send_active_tgs(self, tgs):
        body = struct.pack('>I', len(tgs)) + b''.join(struct.pack('>IB', tg, flags) for tg, flags in tgs)
        self.send(Func.MASTER, p.MASTER_ACTIVE_TGS, b'\x00' * 6 + body, stream_id=7)

    def send_p25(self, src, dst, duid, stream_id, seq):
        payload = bytearray(181)
        payload[0:4] = b'P25D'
        payload[5:8] = src.to_bytes(3, 'big')
        payload[8:11] = dst.to_bytes(3, 'big')
        payload[22] = duid
        self.send(Func.PROTOCOL, p.PROTOCOL_P25, bytes(payload), stream_id=stream_id, seq=seq)

    def send_master_disconnect(self):
        self.send(Func.MST_DISC, p.SUBFUNC_NOP, b'\x00')


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def make_peer(fne, **overrides):
    settings = {'address': '127.0.0.1', 'port': fne.port, 'peer_id': PEER_ID, 'password': PASSWORD,
                'identity': 'CCPAGING', 'ping_interval': 0.3, 'retry_interval': 0.5, 'timeout': 2}
    settings.update(overrides)
    return FNEPeer(settings)


class ProtocolTests(unittest.TestCase):
    def test_crc_is_ccitt_false(self):
        self.assertEqual(p.crc16(b'123456789'), 0x29B1)

    def test_login_packet_bytes_match_dvmhost_layout(self):
        data = p.encode_frame(p.login_payload(PEER_ID), Func.RPTL, p.SUBFUNC_NOP,
                              stream_id=0x01020304, peer_id=PEER_ID, ssrc=PEER_ID, seq=0, timestamp=0x0A0B0C0D)
        payload = b'RPTL' + PEER_ID.to_bytes(4, 'big')
        crc = p.crc16(payload)
        expected = (bytes([0x90, 0x56, 0x00, 0x00, 0x0A, 0x0B, 0x0C, 0x0D]) + PEER_ID.to_bytes(4, 'big')
                    + bytes([0x00, 0xFE, 0x00, 0x04])
                    + crc.to_bytes(2, 'big') + bytes([0x60, 0xFF]) + bytes([1, 2, 3, 4])
                    + PEER_ID.to_bytes(4, 'big') + (8).to_bytes(4, 'big') + payload)
        self.assertEqual(data, expected)
        self.assertEqual(p.decode_frame(data).payload, payload)

    def test_corrupt_payload_rejected(self):
        data = bytearray(p.encode_frame(b'hello', Func.PING, p.SUBFUNC_NOP, 1, 2, 2, 0))
        data[-1] ^= 0xFF
        with self.assertRaises(p.FrameError):
            p.decode_frame(bytes(data))

    def test_preshared_key_round_trip_and_half_key(self):
        self.assertEqual(p.parse_preshared_key(PSK_HEX[:32]), bytes.fromhex(PSK_HEX[:32] * 2))
        self.assertIsNone(p.parse_preshared_key(''))
        with self.assertRaises(ValueError):
            p.parse_preshared_key('xyz')
        c = p.PacketCipher(p.parse_preshared_key(PSK_HEX))
        frame = p.encode_frame(b'RPTL1234', Func.RPTL, p.SUBFUNC_NOP, 5, 6, 6, 0)
        wrapped = c.wrap(frame)
        self.assertEqual(wrapped[:2], b'\xC0\xFE')
        self.assertEqual(p.decode_frame(c.unwrap(wrapped)).payload, b'RPTL1234')
        self.assertIsNone(c.unwrap(frame))   # plaintext dropped, as the FNE does

    def test_settings_validation(self):
        with self.assertRaises(ValueError):
            FNEPeer({'address': 'x', 'peer_id': 0})
        with self.assertRaises(ValueError):
            FNEPeer({'address': '', 'peer_id': 5})


class PeerConnectionTests(unittest.TestCase):
    def setUp(self):
        self.fne = None
        self.peer = None

    def tearDown(self):
        if self.peer:
            self.peer.stop()
        if self.fne:
            self.fne.close()

    def connect(self, fne_kwargs=None, **peer_kwargs):
        self.fne = FakeFNE(**(fne_kwargs or {}))
        self.peer = make_peer(self.fne, **peer_kwargs)
        self.peer.start()

    def test_logs_in_and_sends_config(self):
        self.connect()
        self.assertTrue(wait_for(lambda: self.peer.connected), self.peer.status)
        st = self.peer.snapshot()
        self.assertEqual(st['remote_peer_id'], FNE_PEER_ID)
        self.assertTrue(st['metadata_port'])
        self.assertEqual(self.fne.peer_config['identity'], 'CCPAGING')
        self.assertEqual(self.fne.peer_config['software'], 'CCPAGING_R01')
        funcs = [f.func for f in self.fne.received[:3]]
        self.assertEqual(funcs, [Func.RPTL, Func.RPTK, Func.RPTC])
        # RPTL/RPTK/RPTC share the login stream ID; SSRC is our peer ID
        self.assertEqual(len({f.stream_id for f in self.fne.received[:3]}), 1)
        self.assertTrue(all(f.ssrc == PEER_ID for f in self.fne.received[:3]))

    def test_keepalive_ping_pong(self):
        self.connect()
        self.assertTrue(wait_for(lambda: self.peer.status['rtt_ms'] is not None))
        self.assertIsNotNone(self.peer.status['last_pong'])

    def test_wrong_password_is_reported(self):
        self.connect(password='wrong')
        self.assertTrue(wait_for(lambda: self.peer.status['last_error']))
        self.assertFalse(self.peer.connected)
        self.assertIn('unauthorized', self.peer.status['last_error'])

    def test_acl_rejection_backs_off(self):
        self.connect(fne_kwargs={'acl_reject': True})
        self.assertTrue(wait_for(lambda: self.peer.status['last_error']))
        self.assertIn('access control list', self.peer.status['last_error'])
        time.sleep(1.2)   # > retry_interval, < LONG_BACKOFF
        self.assertEqual(self.fne.logins, 1)

    def test_encrypted_link(self):
        self.connect(fne_kwargs={'preshared_key': PSK_HEX}, preshared_key=PSK_HEX)
        self.assertTrue(wait_for(lambda: self.peer.connected), self.peer.status)
        self.assertTrue(self.peer.status['encrypted'])

    def test_unencrypted_peer_cannot_reach_encrypted_fne(self):
        self.connect(fne_kwargs={'preshared_key': PSK_HEX})
        time.sleep(0.8)
        self.assertFalse(self.peer.connected)
        self.assertEqual(self.fne.received, [])

    def test_talkgroups_and_heard_call(self):
        self.connect()
        self.assertTrue(wait_for(lambda: self.peer.connected))
        self.fne.send_active_tgs([(1001, 0x01), (2002, 0x41)])
        self.assertTrue(wait_for(lambda: len(self.peer.status['active_talkgroups']) == 2))
        tgs = {t['tgid']: t for t in self.peer.status['active_talkgroups']}
        self.assertTrue(tgs[2002]['affiliated'])
        self.assertFalse(tgs[1001]['affiliated'])

        self.fne.send_p25(src=123456, dst=1001, duid=p.P25_DUID_LDU1, stream_id=99, seq=0)
        self.assertTrue(wait_for(lambda: (self.peer.status['last_call'] or {}).get('active')))
        call = self.peer.status['last_call']
        self.assertEqual((call['src_id'], call['dst_id']), (123456, 1001))
        self.fne.send_p25(src=123456, dst=1001, duid=p.P25_DUID_TDU, stream_id=99, seq=p.END_OF_CALL_SEQ)
        self.assertTrue(wait_for(lambda: not self.peer.status['last_call']['active']))
        self.assertEqual(self.peer.status['rx_frames']['p25'], 2)

    def test_heard_frames_go_to_voice_callback(self):
        heard = []
        self.connect()
        self.peer.on_voice = lambda stream, hdr, payload: heard.append((stream, hdr.dst_id, hdr.duid))
        self.assertTrue(wait_for(lambda: self.peer.connected))
        self.fne.send_p25(src=1, dst=1001, duid=p.P25_DUID_LDU1, stream_id=5, seq=0)
        self.fne.send_p25(src=1, dst=1001, duid=p.P25_DUID_TDU, stream_id=5, seq=p.END_OF_CALL_SEQ)
        self.assertTrue(wait_for(lambda: len(heard) == 2))
        self.assertEqual(heard, [(5, 1001, p.P25_DUID_LDU1), (5, 1001, p.P25_DUID_TDU)])

    def test_simultaneous_calls_on_different_talkgroups(self):
        self.connect()
        self.assertTrue(wait_for(lambda: self.peer.connected))
        self.fne.send_p25(src=111, dst=1001, duid=p.P25_DUID_LDU1, stream_id=7, seq=0)
        self.fne.send_p25(src=222, dst=2002, duid=p.P25_DUID_LDU1, stream_id=8, seq=0)
        self.assertTrue(wait_for(lambda: len(self.peer.status['calls']) == 2))
        self.assertEqual({c['dst_id'] for c in self.peer.status['calls']}, {1001, 2002})
        self.fne.send_p25(src=111, dst=1001, duid=p.P25_DUID_TDU, stream_id=7, seq=p.END_OF_CALL_SEQ)
        self.assertTrue(wait_for(lambda: len(self.peer.status['calls']) == 1))
        self.assertEqual(self.peer.status['calls'][0]['dst_id'], 2002)
        self.assertEqual(self.peer.status['recent_calls'][0]['src_id'], 111)
        # the other call ends by going quiet
        self.assertTrue(wait_for(lambda: not self.peer.status['calls'], timeout=6))
        self.assertEqual(len(self.peer.status['recent_calls']), 2)

    def test_master_disconnect_triggers_relogin(self):
        self.connect()
        self.assertTrue(wait_for(lambda: self.peer.connected))
        self.fne.send_master_disconnect()
        self.assertTrue(wait_for(lambda: not self.peer.connected))
        self.assertTrue(wait_for(lambda: self.peer.connected and self.fne.logins == 2), self.peer.status)
        self.assertEqual(self.peer.status['reconnects'], 1)

    def test_silent_fne_times_out(self):
        self.connect(fne_kwargs={'answer_pings': False}, timeout=1)
        self.assertTrue(wait_for(lambda: self.peer.connected))
        self.assertTrue(wait_for(lambda: 'timed out' in (self.peer.status['last_error'] or ''), 4))

    def test_stop_sends_disconnect(self):
        self.connect()
        self.assertTrue(wait_for(lambda: self.peer.connected))
        self.peer.stop()
        self.assertTrue(wait_for(lambda: self.fne.state.get(PEER_ID) == 'disconnected'))
        self.assertEqual(self.peer.state, PeerState.STOPPED)
        self.peer = None


if __name__ == '__main__':
    unittest.main()
