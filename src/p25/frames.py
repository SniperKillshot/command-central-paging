"""P25 network voice frames for the DVM FNE ("P25D" messages).

Byte layouts follow dvmhost:
  - BaseNetwork::createP25_MessageHdr (24-byte header)
  - bridge/network/PeerNetwork::createP25_LDU1Message_Raw / _LDU2Message_Raw
  - p25/dfsi/LC::encodeLDU1 / encodeLDU2 (DFSI voice frames 1-18)
  - BaseNetwork::createP25_TDUMessage
  - edac/RS634717 encode241213 / encode24169 (link control FEC)

An LDU carries nine 20 ms IMBE frames, so one LDU goes out every 180 ms.
"""
from dataclasses import dataclass, field
from typing import Sequence

from src.p25 import tables as T

TAG_P25_DATA = b'P25D'
MSG_HDR_SIZE = 24
PACKET_PAD = 8
LDU1_PACKET_LEN = 193 + PACKET_PAD
LDU2_PACKET_LEN = 181 + PACKET_PAD
TDU_PACKET_LEN = MSG_HDR_SIZE + PACKET_PAD
IMBE_LEN = 11

DUID_TDU = 0x03
DUID_LDU1 = 0x05
DUID_LDU2 = 0x0A

FRAME_TYPE_DATA_UNIT = 0x00
FRAME_TYPE_HDU_VALID = 0x01

LCO_GROUP = 0x00
MFG_STANDARD = 0x00
ALGO_UNENCRYPT = 0x80
WACN_STD_DEFAULT = 0xBB800
SID_STD_DEFAULT = 0x001

NET_CTRL_GRANT_DEMAND = 0x80
NET_CTRL_SWITCH_OVER = 0x20

# (frame type, offset in packet, frame length) for DFSI voice frames 1-9 / 10-18
LDU1_LAYOUT = ((0x62, 24, 22), (0x63, 46, 14), (0x64, 60, 17), (0x65, 77, 17), (0x66, 94, 17),
               (0x67, 111, 17), (0x68, 128, 17), (0x69, 145, 17), (0x6A, 162, 16))
LDU2_LAYOUT = ((0x6B, 24, 22), (0x6C, 46, 14), (0x6D, 60, 17), (0x6E, 77, 17), (0x6F, 94, 17),
               (0x70, 111, 17), (0x71, 128, 17), (0x72, 145, 17), (0x73, 162, 16))


# ---------------------------------------------------------------------------
#  Reed-Solomon over GF(2^6) (P25 link control FEC)
# ---------------------------------------------------------------------------

def _gf6_mult(a: int, b: int) -> int:
    p = 0
    for _ in range(6):
        if b & 1:
            p ^= a
        a <<= 1
        if a & 0x40:
            a ^= 0x43
        b >>= 1
    return p


def _hexbits(data: bytes, count: int):
    v = int.from_bytes(data, 'big')
    total = len(data) * 8
    return [(v >> (total - 6 * (i + 1))) & 0x3F for i in range(count)]


def _rs_encode(data: bytes, matrix, k: int) -> bytes:
    """Systematic encode of k 6-bit symbols into 24 symbols (18 bytes)."""
    hb = _hexbits(data.ljust(18, b'\x00'), k)
    out = 0
    for i in range(24):
        c = 0
        for j in range(k):
            c ^= _gf6_mult(hb[j], matrix[j][i])
        out = (out << 6) | c
    return out.to_bytes(18, 'big')


def rs_241213(lc9: bytes) -> bytes:
    """RS(24,12,13): 9 bytes of LDU1 link control -> 18 bytes."""
    return _rs_encode(lc9, T.ENCODE_MATRIX, 12)


def rs_24169(es12: bytes) -> bytes:
    """RS(24,16,9): 12 bytes of LDU2 encryption sync -> 18 bytes."""
    return _rs_encode(es12, T.ENCODE_MATRIX_24169, 16)


# ---------------------------------------------------------------------------
#  Messages
# ---------------------------------------------------------------------------

@dataclass
class CallParams:
    src_id: int
    dst_id: int
    lco: int = LCO_GROUP
    mfid: int = MFG_STANDARD
    priority: int = 4
    emergency: bool = False
    sys_id: int = SID_STD_DEFAULT
    net_id: int = WACN_STD_DEFAULT
    alg_id: int = ALGO_UNENCRYPT
    key_id: int = 0
    mi: bytes = field(default_factory=lambda: bytes(9))
    lsd1: int = 0
    lsd2: int = 0

    @property
    def service_options(self) -> int:
        enc = self.alg_id != ALGO_UNENCRYPT
        return (0x80 if self.emergency else 0) | (0x40 if enc else 0) | (self.priority & 0x07)


def _header(buf: bytearray, duid: int, p: CallParams, control_byte: int) -> None:
    buf[0:4] = TAG_P25_DATA
    buf[4] = p.lco
    buf[5:8] = p.src_id.to_bytes(3, 'big')
    buf[8:11] = p.dst_id.to_bytes(3, 'big')
    buf[11:13] = p.sys_id.to_bytes(2, 'big')
    buf[14] = control_byte
    buf[15] = p.mfid
    buf[16:19] = p.net_id.to_bytes(3, 'big')
    buf[20] = p.lsd1
    buf[21] = p.lsd2
    buf[22] = duid


def _check(imbe: Sequence[bytes]) -> None:
    if len(imbe) != 9 or any(len(f) != IMBE_LEN for f in imbe):
        raise ValueError('an LDU needs exactly nine 11-byte IMBE frames')


def ldu1(p: CallParams, imbe: Sequence[bytes], control_byte: int = 0,
         frame_type: int = FRAME_TYPE_HDU_VALID) -> bytes:
    _check(imbe)
    buf = bytearray(LDU1_PACKET_LEN)
    _header(buf, DUID_LDU1, p, control_byte)
    lc = bytes([p.lco, p.mfid, p.service_options]) + p.dst_id.to_bytes(3, 'big') + p.src_id.to_bytes(3, 'big')
    rs = rs_241213(lc)
    for n, (ftype, off, length) in enumerate(LDU1_LAYOUT):
        f = bytearray(length)
        f[0] = ftype
        if n == 0:
            f[10:21] = imbe[0]                      # byte 6 = RSSI (0)
        elif n == 1:
            f[1:12] = imbe[1]
        elif n == 8:
            f[1], f[2] = p.lsd1, p.lsd2
            f[4:15] = imbe[8]
        else:
            k = (n - 2) * 3
            f[1:4] = rs[k:k + 3]
            f[5:16] = imbe[n]
        buf[off:off + length] = f
    buf[23] = MSG_HDR_SIZE + sum(length for _, _, length in LDU1_LAYOUT)
    buf[180] = frame_type
    if frame_type == FRAME_TYPE_HDU_VALID:
        buf[181] = p.alg_id
        buf[182:184] = p.key_id.to_bytes(2, 'big')
        buf[184:193] = p.mi
    return bytes(buf)


def ldu2(p: CallParams, imbe: Sequence[bytes], control_byte: int = 0) -> bytes:
    _check(imbe)
    buf = bytearray(LDU2_PACKET_LEN)
    _header(buf, DUID_LDU2, p, control_byte)
    es = bytes(p.mi) + bytes([p.alg_id]) + p.key_id.to_bytes(2, 'big')
    rs = rs_24169(es)
    for n, (ftype, off, length) in enumerate(LDU2_LAYOUT):
        f = bytearray(length)
        f[0] = ftype
        if n == 0:
            f[10:21] = imbe[0]
        elif n == 1:
            f[1:12] = imbe[1]
        elif n == 8:
            f[1], f[2] = p.lsd1, p.lsd2
            f[4:15] = imbe[8]
        else:
            k = (n - 2) * 3
            f[1:4] = rs[k:k + 3]
            f[5:16] = imbe[n]
        buf[off:off + length] = f
    buf[23] = MSG_HDR_SIZE + sum(length for _, _, length in LDU2_LAYOUT)
    buf[180] = FRAME_TYPE_DATA_UNIT
    return bytes(buf)


def tdu(p: CallParams, control_byte: int = 0) -> bytes:
    buf = bytearray(TDU_PACKET_LEN)
    _header(buf, DUID_TDU, p, control_byte)
    buf[23] = MSG_HDR_SIZE
    return bytes(buf)


def ldu_imbe(packet: bytes):
    """Extract the nine IMBE frames from an LDU1/LDU2 packet (for tests and
    for decoding received calls)."""
    layout = LDU1_LAYOUT if packet[22] == DUID_LDU1 else LDU2_LAYOUT
    out = []
    for n, (_, off, _) in enumerate(layout):
        start = off + (10 if n == 0 else 1 if n == 1 else 4 if n == 8 else 5)
        out.append(bytes(packet[start:start + IMBE_LEN]))
    return out
