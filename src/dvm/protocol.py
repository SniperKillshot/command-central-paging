"""DVM FNE network protocol: packet framing and message payloads.

Ported from dvmhost's src/common/network (RTPHeader, RTPFNEHeader,
FrameQueue, Network.cpp) and the master side in src/fne/network/
TrafficNetwork.cpp. Every datagram is:

    12 bytes  RTP header       V=2, X=1, PT=0x56, seq, timestamp, SSRC (sender's peer ID)
     4 bytes  RTP extension    type 0x00FE ("DVM frame start"), length 4 (words)
    16 bytes  FNE header       CRC-16 of payload, function, sub-function,
                               stream ID, peer ID, payload length
     N bytes  payload

All integers are big-endian. The CRC is CRC-16/CCITT-FALSE (poly 0x1021,
init 0xFFFF, MSB first), which is exactly binascii.crc_hqx(data, 0xFFFF).

If the FNE has `encrypted: true`, every datagram is additionally wrapped as
0xC0FE + AES-256-ECB(datagram zero-padded to 16 bytes) with the pre-shared key.
"""
import binascii
import hashlib
import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

DEFAULT_PORT = 62031

RTP_VERSION = 2
RTP_PAYLOAD_TYPE = 0x56          # DVM_RTP_PAYLOAD_TYPE (0x57 is also accepted on receive)
DVM_FRAME_START = 0xFE           # RTP extension payload type for an FNE header
RTP_HEADER_LEN = 12
RTP_EXT_HEADER_LEN = 4
FNE_HEADER_LEN = 16
HEADER_LEN = RTP_HEADER_LEN + RTP_EXT_HEADER_LEN + FNE_HEADER_LEN   # 32
FNE_EXT_LEN_WORDS = 4
END_OF_CALL_SEQ = 0xFFFF         # RTP_END_OF_CALL_SEQ; also used for control messages
MAX_PEER_ID = 999999999
MAX_DATAGRAM = 4096

AES_WRAPPED_MAGIC = b'\xC0\xFE'
AES_KEY_LEN = 32
AES_BLOCK = 16


class Func(IntEnum):
    PROTOCOL = 0x00
    MASTER = 0x01
    RPTL = 0x60          # login
    RPTK = 0x61          # authorisation
    RPTC = 0x62          # configuration
    RPT_DISC = 0x70      # peer disconnect
    MST_DISC = 0x71      # master disconnect
    PING = 0x74
    PONG = 0x75
    GRANT_REQ = 0x7A
    INCALL_CTRL = 0x7B
    KEY_REQ = 0x7C
    KEY_RSP = 0x7D
    ACK = 0x7E
    NAK = 0x7F
    TRANSFER = 0x90
    ANNOUNCE = 0x91
    REPL = 0x92
    NET_TREE = 0x93


SUBFUNC_NOP = 0xFF

# NET_FUNC::PROTOCOL sub-functions
PROTOCOL_DMR = 0x00
PROTOCOL_P25 = 0x01
PROTOCOL_NXDN = 0x02
PROTOCOL_P25_P2 = 0x03
PROTOCOL_ANALOG = 0x0F
PROTOCOL_NAMES = {PROTOCOL_DMR: 'dmr', PROTOCOL_P25: 'p25', PROTOCOL_NXDN: 'nxdn',
                  PROTOCOL_P25_P2: 'p25_phase2', PROTOCOL_ANALOG: 'analog'}

# NET_FUNC::MASTER sub-functions
MASTER_WL_RID = 0x00
MASTER_BL_RID = 0x01
MASTER_ACTIVE_TGS = 0x02
MASTER_DEACTIVE_TGS = 0x03
MASTER_HA_PARAMS = 0xA3

# NET_CONN_NAK_REASON
NAK_REASONS = {
    0: 'general failure',
    1: 'digital mode not enabled on FNE',
    2: 'illegal/unknown packet',
    3: 'unauthorized (wrong password, or address mismatch)',
    4: 'bad connection state',
    5: 'invalid configuration data',
    6: 'FNE demanded connection reset',
    7: 'peer ID rejected by FNE access control list',
    8: 'FNE has reached maximum permitted connections',
    9: 'duplicate connection (this peer ID is already connected)',
}
NAK_PEER_ACL = 7
NAK_FNE_MAX_CONN = 8
NAK_DUPLICATE_CONN = 9

# NET_FUNC::INCALL_CTRL commands (payload byte 10; destination ID at 11-13)
ICC_BUSY_DENY = 0x00
ICC_REJECT_TRAFFIC = 0x01
ICC_NAMES = {ICC_BUSY_DENY: 'busy - another call is active on that talkgroup',
             ICC_REJECT_TRAFFIC: 'rejected by the FNE (radio ID not allowed, or talkgroup not active)'}

# P25 data unit IDs (byte 22 of a P25D message)
P25_DUID_HDU = 0x00
P25_DUID_TDU = 0x03
P25_DUID_LDU1 = 0x05
P25_DUID_TSDU = 0x07
P25_DUID_LDU2 = 0x0A
P25_DUID_PDU = 0x0C
P25_DUID_TDULC = 0x0F


class FrameError(ValueError):
    pass


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE, as dvmhost's edac::CRC::createCRC16."""
    return binascii.crc_hqx(data, 0xFFFF)


@dataclass
class Frame:
    func: int
    subfunc: int
    stream_id: int
    peer_id: int
    payload: bytes
    seq: int = 0
    ssrc: int = 0
    timestamp: int = 0
    marker: bool = False


def encode_frame(payload: bytes, func: int, subfunc: int, stream_id: int, peer_id: int,
                 ssrc: int, seq: int, timestamp: int = 0) -> bytes:
    if not payload:
        raise FrameError('payload must not be empty (the FNE rejects zero-length messages)')
    rtp = struct.pack('>BBHII', (RTP_VERSION << 6) | 0x10, RTP_PAYLOAD_TYPE & 0x7F,
                      seq & 0xFFFF, timestamp & 0xFFFFFFFF, ssrc & 0xFFFFFFFF)
    ext = struct.pack('>HH', DVM_FRAME_START, FNE_EXT_LEN_WORDS)
    fne = struct.pack('>HBBIII', crc16(payload), func & 0xFF, subfunc & 0xFF,
                      stream_id & 0xFFFFFFFF, peer_id & 0xFFFFFFFF, len(payload))
    return rtp + ext + fne + payload


def decode_frame(data: bytes) -> Frame:
    if len(data) < HEADER_LEN:
        raise FrameError(f'packet too short ({len(data)} bytes)')
    b0, b1, seq, timestamp, ssrc = struct.unpack_from('>BBHII', data, 0)
    if (b0 >> 6) != RTP_VERSION:
        raise FrameError('not an RTP v2 packet')
    if not b0 & 0x10:
        raise FrameError('RTP extension header missing')
    payload_type = b1 & 0x7F
    if payload_type not in (RTP_PAYLOAD_TYPE, RTP_PAYLOAD_TYPE + 1):
        raise FrameError(f'unexpected RTP payload type 0x{payload_type:02X}')
    ext_type, ext_len = struct.unpack_from('>HH', data, RTP_HEADER_LEN)
    if ext_type != DVM_FRAME_START or ext_len != FNE_EXT_LEN_WORDS:
        raise FrameError('not a DVM FNE frame')
    crc, func, subfunc, stream_id, peer_id, length = struct.unpack_from(
        '>HBBIII', data, RTP_HEADER_LEN + RTP_EXT_HEADER_LEN)
    if length == 0 or HEADER_LEN + length > len(data):
        raise FrameError(f'bad message length {length} for {len(data)}-byte packet')
    payload = bytes(data[HEADER_LEN:HEADER_LEN + length])
    if crc16(payload) != crc:
        raise FrameError('CRC mismatch')
    return Frame(func, subfunc, stream_id, peer_id, payload, seq, ssrc, timestamp, bool(b1 & 0x80))


# ---------------------------------------------------------------------------
#  Pre-shared key packet encryption
# ---------------------------------------------------------------------------

def parse_preshared_key(text: Optional[str]) -> Optional[bytes]:
    """dvmhost's presharedKey: 64 hex chars (32 bytes). A 32-char key is doubled
    on itself, as dvmhost does. Blank means no encryption."""
    key = (text or '').strip()
    if not key:
        return None
    if len(key) == 32:
        key = key + key
    if len(key) != 64:
        raise ValueError('preshared_key must be 64 hex characters (32 bytes)')
    try:
        return bytes.fromhex(key)
    except ValueError:
        raise ValueError('preshared_key contains non-hex characters') from None


class PacketCipher:
    """AES-256-ECB datagram wrapping, matching dvmhost's udp::Socket."""

    def __init__(self, key: bytes):
        if len(key) != AES_KEY_LEN:
            raise ValueError('pre-shared key must be 32 bytes')
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ImportError as e:
            raise RuntimeError("DVM pre-shared key encryption needs the 'cryptography' "
                               "package (pip install cryptography)") from e
        # ECB is what the FNE uses; it is not our choice to make.
        self._cipher = Cipher(algorithms.AES(key), modes.ECB())

    @staticmethod
    def _pad(data: bytes) -> bytes:
        return data + b'\x00' * (-len(data) % AES_BLOCK)

    def wrap(self, datagram: bytes) -> bytes:
        enc = self._cipher.encryptor()
        return AES_WRAPPED_MAGIC + enc.update(self._pad(datagram)) + enc.finalize()

    def unwrap(self, datagram: bytes) -> Optional[bytes]:
        """Returns None for packets without the magic (the FNE discards those too)."""
        if len(datagram) < 2 + AES_BLOCK or datagram[:2] != AES_WRAPPED_MAGIC:
            return None
        body = datagram[2:]
        dec = self._cipher.decryptor()
        return (dec.update(self._pad(body)) + dec.finalize())[:len(body)]


# ---------------------------------------------------------------------------
#  Login sequence payloads (peer -> master)
# ---------------------------------------------------------------------------

def login_payload(peer_id: int) -> bytes:
    return b'RPTL' + struct.pack('>I', peer_id)


def auth_payload(peer_id: int, salt: bytes, password: str) -> bytes:
    digest = hashlib.sha256(salt + password.encode('utf-8')).digest()
    return b'RPTK' + struct.pack('>I', peer_id) + digest


def config_payload(config: Dict) -> bytes:
    # Bytes 4-7 are left zero, exactly as dvmhost's writeConfig does.
    return b'RPTC' + b'\x00' * 4 + json.dumps(config, separators=(',', ':')).encode('utf-8')


def build_peer_config(settings: Dict, software: str) -> Dict:
    """The RPTC JSON. The FNE only acts on identity, software, conventionalPeer,
    sysView and externalPeer; the rest is shown in its peer list / REST API.
    Field names and nesting follow dvmhost's Network::writeConfig."""
    return {
        'identity': str(settings.get('identity') or 'CCPAGING'),
        'rxFrequency': int(settings.get('rx_frequency', 0) or 0),
        'txFrequency': int(settings.get('tx_frequency', 0) or 0),
        'info': {
            'latitude': float(settings.get('latitude', 0.0) or 0.0),
            'longitude': float(settings.get('longitude', 0.0) or 0.0),
            'height': int(settings.get('height', 0) or 0),
            'location': str(settings.get('location') or ''),
        },
        'channel': {
            'txPower': int(settings.get('tx_power', 0) or 0),
            'txOffsetMhz': 0.0,
            'chBandwidthKhz': 12.5,
            'channelId': 0,
            'channelNo': 0,
        },
        'rcon': {'password': '', 'port': 0},
        'conventionalPeer': bool(settings.get('conventional_peer', False)),
        'software': software,
    }


SINGLE_BYTE = b'\x00'   # payload for PING and RPT_DISC


# ---------------------------------------------------------------------------
#  Master -> peer payload parsing
# ---------------------------------------------------------------------------

def ack_salt(payload: bytes) -> bytes:
    """RPTL ACK: the master's ACK payload carries the 4-byte salt at offset 6."""
    if len(payload) < 10:
        raise FrameError('RPTL ACK too short to carry a salt')
    return payload[6:10]


def ack_flags(payload: bytes) -> int:
    """RPTC ACK: byte 6 bit 0x80 means the FNE runs its metadata port (port + 1)."""
    return payload[6] if len(payload) > 6 else 0


def nak_reason(payload: bytes) -> Optional[int]:
    return struct.unpack_from('>H', payload, 10)[0] if len(payload) >= 12 else None


def pong_server_time_ms(payload: bytes) -> Optional[int]:
    return struct.unpack_from('>Q', payload, 6)[0] if len(payload) >= 14 else None


def parse_talkgroup_list(payload: bytes, active: bool) -> List[Tuple[int, Dict]]:
    """MASTER_SUBFUNC_ACTIVE_TGS / DEACTIVE_TGS: count at 6, then 5-byte
    entries from 10 (32-bit TGID, then slot/flags byte)."""
    if len(payload) < 10:
        return []
    count = struct.unpack_from('>I', payload, 6)[0]
    out = []
    offs = 10
    for _ in range(count):
        if offs + 5 > len(payload):
            break
        tgid = struct.unpack_from('>I', payload, offs)[0] & 0xFFFFFF
        flags = payload[offs + 4]
        info = {'slot': flags & 0x03}
        if active:
            info['affiliated'] = bool(flags & 0x40)
            info['non_preferred'] = bool(flags & 0x80)
        out.append((tgid, info))
        offs += 5
    return out


@dataclass
class P25Header:
    lco: int
    src_id: int
    dst_id: int
    duid: int


def parse_p25_header(payload: bytes) -> Optional[P25Header]:
    if len(payload) < 23 or payload[:4] != b'P25D':
        return None
    src = int.from_bytes(payload[5:8], 'big')
    dst = int.from_bytes(payload[8:11], 'big')
    return P25Header(payload[4], src, dst, payload[22])
