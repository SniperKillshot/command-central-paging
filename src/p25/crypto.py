"""P25 Phase 1 voice encryption: AES-256, ARC4 (ADP) and DES-OFB.

A port of dvmhost's p25::crypto::P25Crypto (src/common/p25/Crypto.cpp) and
how its bridge uses it (src/bridge/HostBridge.P25.cpp), so radios and other
DVM consoles decode what we send and we decode theirs.

  - Each superframe (LDU1 + LDU2, 18 IMBE frames) has a 72-bit message
    indicator (MI). Only its first 8 bytes feed the cipher.
  - The keystream for a superframe comes from the key and its MI:
      AES-256  output feedback from IV = MI[0:8] || LFSR(MI) (240 bytes)
      ARC4     RC4 keyed with key[5] || MI[0:8]                (469 bytes)
      DES      output feedback from MI[0:8] (DES-OFB)          (224 bytes)
  - Each 11-byte IMBE frame is XORed with the keystream at a fixed offset
    for its position (LDU1 or LDU2, frame 0-8). Encrypt = decrypt.
  - The MI steps once per superframe (a 64-bit LFSR). LDU1 carries the
    superframe's own MI (in dvmhost's network header); LDU2's encryption
    sync carries the NEXT superframe's MI, which is how a receiver keeps up.

Keys (traffic encryption keys) come from config: see KeyStore.
"""
import logging
import secrets
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.p25 import frames as F

logger = logging.getLogger(__name__)

ALGO_UNENCRYPT = F.ALGO_UNENCRYPT   # 0x80
ALGO_DES = 0x81
ALGO_AES = 0x84
ALGO_ARC4 = 0xAA
ALGO_BY_NAME = {'aes': ALGO_AES, 'aes256': ALGO_AES, 'aes-256': ALGO_AES,
                'arc4': ALGO_ARC4, 'adp': ALGO_ARC4, 'rc4': ALGO_ARC4,
                'des': ALGO_DES, 'des-ofb': ALGO_DES}
ALGO_NAMES = {ALGO_AES: 'AES-256', ALGO_ARC4: 'ARC4', ALGO_DES: 'DES-OFB'}
KEY_BYTES = {ALGO_AES: 32, ALGO_ARC4: 5, ALGO_DES: 8}

MI_BYTES = 9
IMBE_BYTES = 11
_MASK64 = (1 << 64) - 1


class CryptoError(Exception):
    pass


# ---------------------------------------------------------------------------
#  MI
# ---------------------------------------------------------------------------

def _feedback(v: int) -> int:
    """Taps of x^64 + x^62 + x^46 + x^38 + x^27 + x^15 + 1."""
    return ((v >> 63) ^ (v >> 61) ^ (v >> 45) ^ (v >> 37) ^ (v >> 26) ^ (v >> 14)) & 1


def new_mi() -> bytes:
    return secrets.token_bytes(MI_BYTES)


def next_mi(mi: bytes) -> bytes:
    """The following superframe's MI: 64 LFSR steps over MI[0:8]; MI[8]
    is carried along unchanged (dvmhost generateNextMI)."""
    v = int.from_bytes(mi[:8], 'big')
    for _ in range(64):
        v = ((v << 1) & _MASK64) | _feedback(v)
    return v.to_bytes(8, 'big') + bytes(mi[8:9])


def mi_to_iv(mi: bytes) -> bytes:
    """AES IV: the 64 bits the LFSR shifts out (MI[0:8] itself), then its
    state after 64 steps (dvmhost expandMIToIV)."""
    return bytes(mi[:8]) + next_mi(mi)[:8]


# ---------------------------------------------------------------------------
#  Keystreams
# ---------------------------------------------------------------------------

def _rc4(key: bytes, n: int) -> bytes:
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + key[i % len(key)]) & 0xFF
        s[i], s[j] = s[j], s[i]
    out = bytearray(n)
    i = j = 0
    for k in range(n):
        i = (i + 1) & 0xFF
        j = (j + s[i]) & 0xFF
        s[i], s[j] = s[j], s[i]
        out[k] = s[(s[i] + s[j]) & 0xFF]
    return bytes(out)


def _ecb_chain(cipher, block: bytes, blocks: int) -> bytes:
    """Output feedback: encrypt the previous output block, `blocks` times."""
    enc = cipher.encryptor()
    out = bytearray()
    for _ in range(blocks):
        block = enc.update(block)
        out += block
    return bytes(out)


def _ciphers():
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except ImportError as e:
        raise CryptoError("AES and DES need the 'cryptography' package") from e
    return Cipher, algorithms, modes


def keystream(algo: int, key: bytes, mi: bytes) -> bytes:
    if algo == ALGO_AES:
        if len(key) != 32:
            raise CryptoError('AES-256 keys are 32 bytes')
        Cipher, algorithms, modes = _ciphers()
        return _ecb_chain(Cipher(algorithms.AES(key), modes.ECB()), mi_to_iv(mi), 15)
    if algo == ALGO_ARC4:
        k5 = bytes(max(5 - len(key), 0)) + bytes(key[:5])
        return _rc4(k5 + bytes(mi[:8]), 469)
    if algo == ALGO_DES:
        Cipher, algorithms, modes = _ciphers()
        try:
            from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
        except ImportError:
            TripleDES = algorithms.TripleDES
        k8 = bytes(max(8 - len(key), 0)) + bytes(key[:8])
        # the same key three times is single DES
        return _ecb_chain(Cipher(TripleDES(k8 * 3), modes.ECB()), bytes(mi[:8]), 28)
    raise CryptoError(f'unsupported algorithm 0x{algo:02X}')


def frame_offset(algo: int, ldu2: bool, n: int) -> int:
    """Where IMBE frame n (0-8) of an LDU starts in the superframe keystream."""
    base = {ALGO_AES: 16, ALGO_DES: 8, ALGO_ARC4: 0}[algo]
    skip = 267 if algo == ALGO_ARC4 else IMBE_BYTES
    return base + (101 if ldu2 else 0) + n * IMBE_BYTES + skip + (2 if n >= 8 else 0)


def crypt_ldu(algo: int, ks: bytes, imbe: Sequence[bytes], ldu2: bool) -> List[bytes]:
    """XOR an LDU's nine IMBE frames with their slices of the keystream."""
    out = []
    for n, f in enumerate(imbe):
        o = frame_offset(algo, ldu2, n)
        out.append(bytes(a ^ b for a, b in zip(f, ks[o:o + IMBE_BYTES])))
    return out


# ---------------------------------------------------------------------------
#  Keys
# ---------------------------------------------------------------------------

def _int(v, what: str) -> int:
    """YAML ints as they are; strings as hex (as dvmhost reads key IDs)."""
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    s = str(v or '').strip().lower()
    try:
        return int(s.removeprefix('0x'), 16)
    except ValueError:
        raise CryptoError(f"{what} '{v}' is not a number") from None


class KeyStore:
    """Traffic encryption keys, from config:

        encryption:
          keys:
            - {kid: 0x561, algo: aes, key: "<64 hex digits>", talkgroups: [500]}
            - {kid: 0x201, algo: arc4, key: "<10 hex digits>", talkgroups: [503]}

    Received calls are decrypted with whichever key matches their algorithm
    and key ID. Calls this console sends to a listed talkgroup are encrypted
    with that key; other talkgroups are sent in the clear. A talkgroup listed
    under a key that is missing or invalid is refused rather than sent clear.
    """

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self._keys: Dict[Tuple[int, int], bytes] = {}
        self._by_tg: Dict[int, Tuple[int, int]] = {}
        self._missing: Dict[int, Tuple[int, int]] = {}
        for n, k in enumerate((settings or {}).get('keys') or [], 1):
            try:
                self._add(k)
            except CryptoError as e:
                logger.error(f'encryption.keys entry {n}: {e} - skipped')

    def _add(self, k: Dict[str, Any]) -> None:
        if not isinstance(k, dict):
            raise CryptoError('not a mapping')
        algo = ALGO_BY_NAME.get(str(k.get('algo', '')).strip().lower())
        if algo is None:
            raise CryptoError(f"algo '{k.get('algo')}' is not aes, arc4 or des")
        kid = _int(k.get('kid', k.get('key_id')), 'kid')
        if not 0 <= kid <= 0xFFFF:
            raise CryptoError(f'kid {kid} is not 16-bit')
        tgs = [_int(tg, 'talkgroup') if isinstance(tg, str) and tg.lower().startswith('0x') else int(tg)
               for tg in k.get('talkgroups') or []]
        try:
            key = self._parse_key(algo, kid, k.get('key'))
        except CryptoError:
            for tg in tgs:                      # meant to be encrypted: never send these in the clear
                self._missing[tg] = (algo, kid)
            raise
        self._keys[(algo, kid)] = key
        for tg in tgs:
            self._by_tg[tg] = (algo, kid)

    @staticmethod
    def _parse_key(algo: int, kid: int, value) -> bytes:
        hexkey = ''.join(str(value or '').split()).lower().removeprefix('0x')
        if not hexkey:
            raise CryptoError(f'kid 0x{kid:X} has no key')
        try:
            key = bytes.fromhex(hexkey)
        except ValueError:
            raise CryptoError(f'kid 0x{kid:X}: key is not hex') from None
        want = KEY_BYTES[algo]
        if len(key) > want or (algo == ALGO_AES and len(key) != want):
            raise CryptoError(f'kid 0x{kid:X}: {ALGO_NAMES[algo]} keys are {want} bytes '
                              f'({want * 2} hex digits), got {len(key)}')
        return key

    def __len__(self) -> int:
        return len(self._keys)

    def key(self, algo: int, kid: int) -> Optional[bytes]:
        return self._keys.get((algo, kid))

    def for_talkgroup(self, tg: int) -> Optional[Tuple[int, int, bytes]]:
        """(algo, kid, key) to send to this talkgroup with, or None (clear)."""
        hit = self._by_tg.get(int(tg))
        return (hit[0], hit[1], self._keys[hit]) if hit else None

    def talkgroups(self) -> Dict[int, Tuple[int, int]]:
        return dict(self._by_tg)

    def missing_key(self, tg: int) -> Optional[Tuple[int, int]]:
        """(algo, kid) if this talkgroup is listed for a key that failed to
        load: it must not be sent to in the clear."""
        tg = int(tg)
        return None if tg in self._by_tg else self._missing.get(tg)


# ---------------------------------------------------------------------------
#  Calls
# ---------------------------------------------------------------------------

class CallEncryptor:
    """Builds the LDUs of one outgoing encrypted call. ldu(k) must be called
    for k = 0, 1, 2... in order (LDU1, LDU2, LDU1...)."""

    def __init__(self, algo: int, kid: int, key: bytes, mi: Optional[bytes] = None):
        self.algo, self.kid, self.key = algo, kid, key
        self.mi = bytes(mi) if mi else new_mi()
        self._ks = keystream(algo, key, self.mi)

    def params(self, p: F.CallParams) -> F.CallParams:
        return replace(p, alg_id=self.algo, key_id=self.kid, mi=self.mi)

    def ldu(self, p: F.CallParams, k: int, imbe: Sequence[bytes], control_byte: int = 0) -> bytes:
        if k % 2 == 0:
            return F.ldu1(self.params(p), crypt_ldu(self.algo, self._ks, imbe, False), control_byte,
                          F.FRAME_TYPE_HDU_VALID)
        frames = crypt_ldu(self.algo, self._ks, imbe, True)
        self.mi = next_mi(self.mi)          # LDU2 announces the next superframe's MI
        self._ks = keystream(self.algo, self.key, self.mi)
        return F.ldu2(self.params(p), frames, control_byte)


def build_ldu(p: F.CallParams, k: int, imbe: Sequence[bytes], control_byte: int = 0,
              encryptor: Optional[CallEncryptor] = None) -> bytes:
    """LDU k of a call (even = LDU1, odd = LDU2), encrypted if given an encryptor."""
    if encryptor is not None:
        return encryptor.ldu(p, k, imbe, control_byte)
    if k % 2 == 0:
        return F.ldu1(p, imbe, control_byte, F.FRAME_TYPE_HDU_VALID)
    return F.ldu2(p, imbe, control_byte)


def ldu1_header_sync(payload: bytes) -> Optional[Tuple[int, int, bytes]]:
    """(algo, kid, MI) from dvmhost's header block on an LDU1, if present."""
    if len(payload) >= 193 and payload[180] == F.FRAME_TYPE_HDU_VALID:
        return payload[181], int.from_bytes(payload[182:184], 'big'), bytes(payload[184:193])
    return None


def ldu2_sync(payload: bytes) -> Optional[Tuple[int, int, bytes]]:
    """(algo, kid, next MI) from an LDU2's encryption sync. The RS(24,16,9)
    code is systematic, so its first 12 bytes are MI + algo + kid."""
    rs = bytearray()
    for n, (_, off, length) in enumerate(F.LDU2_LAYOUT):
        if 2 <= n <= 7:
            if len(payload) < off + 4:
                return None
            rs += payload[off + 1:off + 4]
    es = bytes(rs[:12])
    return es[9], int.from_bytes(es[10:12], 'big'), es[:9]


class CallDecryptor:
    """Decrypts one received call. A call joined late is silent until its
    first LDU2 gives the MI for the next superframe."""

    def __init__(self, algo: int, kid: int, key: bytes):
        self.algo, self.kid, self.key = algo, kid, key
        self._ks: Optional[bytes] = None

    def set_mi(self, mi: bytes) -> None:
        self._ks = keystream(self.algo, self.key, mi)

    @property
    def synced(self) -> bool:
        return self._ks is not None

    def ldu1(self, payload: bytes, imbe: Sequence[bytes]) -> Optional[List[bytes]]:
        if self._ks is None:
            sync = ldu1_header_sync(payload)
            if sync and sync[0] == self.algo and sync[1] == self.kid and any(sync[2]):
                self.set_mi(sync[2])
        return None if self._ks is None else crypt_ldu(self.algo, self._ks, imbe, False)

    def ldu2(self, payload: bytes, imbe: Sequence[bytes]) -> Optional[List[bytes]]:
        out = None if self._ks is None else crypt_ldu(self.algo, self._ks, imbe, True)
        sync = ldu2_sync(payload)
        if sync and sync[0] == self.algo and sync[1] == self.kid:
            self.set_mi(sync[2])
        else:
            self._ks = None                 # lost sync: wait for the next LDU2
        return out
