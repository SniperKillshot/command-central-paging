"""P25 voice encryption (src/p25/crypto.py): the primitives against
independent references, dvmhost's frame layout, and whole calls through the
send and receive paths."""
import time
import unittest

import numpy as np

from src.p25 import crypto as C
from src.p25 import frames as F
from src.p25.pager import P25Pager
from src.p25.rx_audio import FLAG_DECRYPTED, HEADER, MSG_AUDIO, MSG_ENCRYPTED, MSG_END, RxAudio
from src.p25.voice import encode_pcm
from src.dvm.protocol import P25Header
from src.audio.tones import SoundLibrary
from tests.test_live_audio import FakePeer, read_messages, speechlike

AES_KEY = bytes(range(32))
ARC4_KEY = bytes.fromhex('0123456789')
DES_KEY = bytes.fromhex('0123456789ABCDEF')
MI = bytes.fromhex('112233445566778899')


def dvmhost_next_mi(mi: bytes) -> bytes:
    """Literal transcription of dvmhost P25Crypto::generateNextMI."""
    n = bytearray(mi)
    for _ in range(64):
        carry = ((n[0] >> 7) ^ (n[0] >> 5) ^ (n[2] >> 5) ^ (n[3] >> 5) ^ (n[4] >> 2) ^ (n[6] >> 6)) & 0x01
        for i in range(7):
            n[i] = ((n[i] & 0x7F) << 1) | (n[i + 1] >> 7)
        n[7] = ((n[7] & 0x7F) << 1) | carry
    return bytes(n)


def dvmhost_expand_mi_to_iv(mi: bytes) -> bytes:
    """Literal transcription of dvmhost expandMIToIV / stepLFSR."""
    lfsr = int.from_bytes(mi[:8], 'big')
    overflow = 0
    for _ in range(64):
        ov = (lfsr >> 63) & 1
        fb = ((lfsr >> 63) ^ (lfsr >> 61) ^ (lfsr >> 45) ^ (lfsr >> 37) ^ (lfsr >> 26) ^ (lfsr >> 14)) & 1
        lfsr = ((lfsr << 1) | fb) & ((1 << 64) - 1)
        overflow = (overflow << 1) | ov
    return overflow.to_bytes(8, 'big') + lfsr.to_bytes(8, 'big')


def ofb_cipher(algorithm, iv):
    from cryptography.hazmat.primitives.ciphers import Cipher
    try:
        from cryptography.hazmat.decrepit.ciphers.modes import OFB
    except ImportError:
        from cryptography.hazmat.primitives.ciphers.modes import OFB
    return Cipher(algorithm, OFB(iv)).encryptor()


class PrimitiveTests(unittest.TestCase):
    def test_rc4_known_answer(self):
        # the classic vector: RC4("Key") XOR "Plaintext"
        ks = C._rc4(b'Key', 9)
        self.assertEqual(bytes(a ^ b for a, b in zip(ks, b'Plaintext')).hex().upper(), 'BBF316E8D940AF0AD3')

    def test_mi_stepping_matches_dvmhost(self):
        mi = MI
        for _ in range(5):
            self.assertEqual(C.next_mi(mi), dvmhost_next_mi(mi))
            mi = C.next_mi(mi)
        self.assertEqual(mi[8], MI[8])                     # MI[8] rides along unchanged

    def test_iv_matches_dvmhost(self):
        for mi in (MI, bytes(9), bytes([0xFF] * 9)):
            self.assertEqual(C.mi_to_iv(mi), dvmhost_expand_mi_to_iv(mi))

    def test_aes_keystream_is_ofb(self):
        from cryptography.hazmat.primitives.ciphers import algorithms
        ofb = ofb_cipher(algorithms.AES(AES_KEY), C.mi_to_iv(MI)).update(bytes(240))
        self.assertEqual(C.keystream(C.ALGO_AES, AES_KEY, MI), ofb)

    def test_arc4_keystream_uses_key_then_mi(self):
        self.assertEqual(C.keystream(C.ALGO_ARC4, ARC4_KEY, MI), C._rc4(ARC4_KEY + MI[:8], 469))
        # short keys are left-padded with zeros to 5 bytes, as dvmhost does
        self.assertEqual(C.keystream(C.ALGO_ARC4, b'\x01\x02', MI), C._rc4(bytes(3) + b'\x01\x02' + MI[:8], 469))

    def test_des_keystream_is_ofb(self):
        try:
            from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
        except ImportError:
            from cryptography.hazmat.primitives.ciphers.algorithms import TripleDES
        ofb = ofb_cipher(TripleDES(DES_KEY * 3), MI[:8]).update(bytes(224))
        self.assertEqual(C.keystream(C.ALGO_DES, DES_KEY, MI), ofb)

    def test_frame_offsets_match_dvmhost(self):
        want = {C.ALGO_AES: (27, 117, 128, 218), C.ALGO_ARC4: (267, 357, 368, 458), C.ALGO_DES: (19, 109, 120, 210)}
        for algo, (l1_0, l1_8, l2_0, l2_8) in want.items():
            self.assertEqual((C.frame_offset(algo, False, 0), C.frame_offset(algo, False, 8),
                              C.frame_offset(algo, True, 0), C.frame_offset(algo, True, 8)),
                             (l1_0, l1_8, l2_0, l2_8), C.ALGO_NAMES[algo])
            ks = C.keystream(algo, {C.ALGO_AES: AES_KEY, C.ALGO_ARC4: ARC4_KEY, C.ALGO_DES: DES_KEY}[algo], MI)
            self.assertGreaterEqual(len(ks), l2_8 + 11)


class KeyStoreTests(unittest.TestCase):
    def test_parses_keys_and_talkgroups(self):
        ks = C.KeyStore({'keys': [
            {'kid': 0x561, 'algo': 'aes', 'key': AES_KEY.hex(), 'talkgroups': [500]},
            {'kid': '563', 'algo': 'AES', 'key': AES_KEY.hex().upper(), 'talkgroups': [501, 502]},
            {'kid': '0x201', 'algo': 'arc4', 'key': '01 23 45 67 89', 'talkgroups': [503]},
            {'kid': 0x273F, 'algo': 'adp', 'key': '0123456789', 'talkgroups': ['10047']},
        ]})
        self.assertEqual(len(ks), 4)
        self.assertEqual(ks.key(C.ALGO_AES, 0x563), AES_KEY)
        self.assertEqual(ks.for_talkgroup(503), (C.ALGO_ARC4, 0x201, ARC4_KEY))
        self.assertEqual(ks.for_talkgroup(10047)[:2], (C.ALGO_ARC4, 0x273F))
        self.assertIsNone(ks.for_talkgroup(504))

    def test_bad_entries_are_skipped(self):
        with self.assertLogs('src.p25.crypto', 'ERROR') as logs:
            ks = C.KeyStore({'keys': [
                {'kid': 1, 'algo': 'aes', 'key': '00' * 16},       # AES-128 is not P25 AES-256
                {'kid': 2, 'algo': 'arc4', 'key': '00' * 6},       # too long
                {'kid': 3, 'algo': 'blowfish', 'key': '00'},
                {'kid': 4, 'algo': 'arc4', 'key': ''},
                {'kid': 5, 'algo': 'arc4', 'key': 'zz'},
                {'kid': 6, 'algo': 'arc4', 'key': '0011223344'},
            ]})
        self.assertEqual(len(logs.records), 5)
        self.assertEqual(len(ks), 1)


def frames_for(seconds):
    return encode_pcm(speechlike(seconds))


class CallTests(unittest.TestCase):
    P = F.CallParams(src_id=252007, dst_id=500)

    def call(self, algo, key, n_ldus=6):
        frames = frames_for(n_ldus * 0.18)
        enc = C.CallEncryptor(algo, 0x561, key, mi=MI)
        packets = [C.build_ldu(self.P, k, frames[k * 9:(k + 1) * 9], 0, enc) for k in range(n_ldus)]
        return frames, packets

    def test_encrypted_ldus_carry_algo_kid_and_mi_chain(self):
        _, pkts = self.call(C.ALGO_AES, AES_KEY)
        mi = MI
        for k, pkt in enumerate(pkts):
            if k % 2 == 0:
                self.assertEqual(C.ldu1_header_sync(pkt), (C.ALGO_AES, 0x561, mi))
                lc_svc = pkt[F.LDU1_LAYOUT[2][1] + 3]
                self.assertTrue(lc_svc & 0x40, 'LDU1 link control says encrypted')
            else:
                mi = C.next_mi(mi)                                # LDU2 announces the next MI
                self.assertEqual(C.ldu2_sync(pkt), (C.ALGO_AES, 0x561, mi))

    def test_round_trip_every_algorithm(self):
        for algo, key in ((C.ALGO_AES, AES_KEY), (C.ALGO_ARC4, ARC4_KEY), (C.ALGO_DES, DES_KEY)):
            frames, pkts = self.call(algo, key)
            self.assertNotEqual(F.ldu_imbe(pkts[0]), frames[:9], 'frames are encrypted on the wire')
            dec = C.CallDecryptor(algo, 0x561, key)
            out = []
            for k, pkt in enumerate(pkts):
                imbe = F.ldu_imbe(pkt)
                out += dec.ldu1(pkt, imbe) if k % 2 == 0 else dec.ldu2(pkt, imbe)
            self.assertEqual(out, frames, C.ALGO_NAMES[algo])

    def test_late_join_syncs_from_the_first_ldu2(self):
        frames, pkts = self.call(C.ALGO_ARC4, ARC4_KEY)
        dec = C.CallDecryptor(C.ALGO_ARC4, 0x561, ARC4_KEY)
        self.assertIsNone(dec.ldu2(pkts[1], F.ldu_imbe(pkts[1])))     # joined at an LDU2: no MI yet
        out = []
        for k in range(2, len(pkts)):
            imbe = F.ldu_imbe(pkts[k])
            out += dec.ldu1(pkts[k], imbe) if k % 2 == 0 else dec.ldu2(pkts[k], imbe)
        self.assertEqual(out, frames[18:])

    def test_wrong_key_gives_different_frames(self):
        frames, pkts = self.call(C.ALGO_AES, AES_KEY)
        dec = C.CallDecryptor(C.ALGO_AES, 0x561, bytes(32))
        self.assertNotEqual(dec.ldu1(pkts[0], F.ldu_imbe(pkts[0])), frames[:9])

    def test_new_calls_get_fresh_mis(self):
        a = C.CallEncryptor(C.ALGO_AES, 1, AES_KEY)
        b = C.CallEncryptor(C.ALGO_AES, 1, AES_KEY)
        self.assertNotEqual(a.mi, b.mi)

    def test_clear_call_unchanged(self):
        frames = frames_for(0.36)
        self.assertEqual(C.build_ldu(self.P, 0, frames[:9]), F.ldu1(self.P, frames[:9], 0, F.FRAME_TYPE_HDU_VALID))
        self.assertEqual(C.build_ldu(self.P, 1, frames[9:18]), F.ldu2(self.P, frames[9:18], 0))


class PathTests(unittest.TestCase):
    KEYS = C.KeyStore({'keys': [{'kid': 0x561, 'algo': 'aes', 'key': AES_KEY.hex(), 'talkgroups': [500]}]})

    def test_pager_encrypts_listed_talkgroups_only(self):
        pager = P25Pager({'source_id': 252007}, None, SoundLibrary(''), lambda: {}, keys=self.KEYS)
        self.assertEqual((pager.encryptor(500).algo, pager.encryptor(500).kid), (C.ALGO_AES, 0x561))
        self.assertIsNone(pager.encryptor(504))

    def test_talkgroup_with_a_broken_key_is_refused_not_sent_clear(self):
        with self.assertLogs('src.p25.crypto', 'ERROR'):
            keys = C.KeyStore({'keys': [{'kid': 0x563, 'algo': 'aes', 'key': '', 'talkgroups': [501]}]})
        self.assertEqual(keys.missing_key(501), (C.ALGO_AES, 0x563))
        peer = FakePeer()
        pager = P25Pager({'source_id': 252007}, None, SoundLibrary(''), lambda: {}, peer=peer, keys=keys)
        problem = pager.check_ready(501)
        self.assertIn('not sending in the clear', problem)
        self.assertIsNone(pager.check_ready(504))
        result = pager.page(501, sound='alert_tone')
        self.assertFalse(result.ok)
        self.assertEqual(peer.sent, [])

    def test_rx_decrypts_with_a_key_and_reports_without(self):
        frames = frames_for(0.72)
        enc = C.CallEncryptor(C.ALGO_AES, 0x561, AES_KEY)
        p = F.CallParams(src_id=12007, dst_id=500)
        pkts = [C.build_ldu(p, k, frames[k * 9:(k + 1) * 9], 0, enc) for k in range(4)] + [F.tdu(p)]

        def play(rx, stream):
            _, q = rx.listen()
            for pkt in pkts:
                rx.on_p25(stream, P25Header(0, p.src_id, p.dst_id, pkt[22]), pkt)
            return read_messages(q, MSG_END)

        msgs = play(RxAudio(self.KEYS), 77)
        self.assertEqual([m[0] for m in msgs], [MSG_AUDIO] * 4 + [MSG_END])
        clear_rx = RxAudio()
        _, q = clear_rx.listen()
        for k in range(4):
            pkt = F.ldu1(p, frames[k * 9:(k + 1) * 9]) if k % 2 == 0 else F.ldu2(p, frames[k * 9:(k + 1) * 9])
            clear_rx.on_p25(78, P25Header(0, p.src_id, p.dst_id, pkt[22]), pkt)
        clear_rx.on_p25(78, P25Header(0, p.src_id, p.dst_id, pkts[-1][22]), pkts[-1])
        ref = np.concatenate([m[4] for m in read_messages(q, MSG_END)[:4]])
        got = np.concatenate([m[4] for m in msgs[:4]])
        np.testing.assert_array_equal(got, ref)                  # decrypted audio = the clear call's
        self.assertEqual(play(RxAudio(), 79)[0][0], MSG_ENCRYPTED)

    def test_decrypted_audio_is_flagged(self):
        frames = frames_for(0.36)
        enc = C.CallEncryptor(C.ALGO_ARC4, 0x201, ARC4_KEY)
        p = F.CallParams(src_id=1, dst_id=503)
        keys = C.KeyStore({'keys': [{'kid': 0x201, 'algo': 'arc4', 'key': ARC4_KEY.hex()}]})
        rx = RxAudio(keys)
        _, q = rx.listen()
        for k in range(2):
            pkt = C.build_ldu(p, k, frames[k * 9:(k + 1) * 9], 0, enc)
            rx.on_p25(5, P25Header(0, 1, 503, pkt[22]), pkt)
        rx.on_p25(5, P25Header(0, 1, 503, F.DUID_TDU), F.tdu(p))
        deadline = time.time() + 5
        flags = []
        while time.time() < deadline and len(flags) < 2:
            try:
                data = q.get(timeout=0.2)
            except Exception:
                continue
            kind, fl = HEADER.unpack_from(data)[:2]
            if kind == MSG_AUDIO:
                flags.append(fl)
        self.assertEqual(flags, [FLAG_DECRYPTED, FLAG_DECRYPTED])


if __name__ == '__main__':
    unittest.main()
