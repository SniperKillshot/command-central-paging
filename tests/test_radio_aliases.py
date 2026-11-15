"""Radio aliases over the FNE peer link (NET_FUNC 0x85, dvmhost r05a06_dev):
the fragment format, the alias file, and a peer asking a fake FNE."""
import socket
import struct
import threading
import unittest
import zlib

from src.console import DispatchConsole
from src.dvm import protocol as p
from src.dvm.protocol import Func
from tests.test_dvm_peer import FakeFNE, make_peer, wait_for

ALIAS_FILE = b"""# Radio alias table
251933,VK4ABC Mobile
12007,Base 7
241315,CCP Console,extra
999,
bad line
notanumber,Nope
"""


def dvmhost_encode(message: bytes):
    """Transcription of dvmhost PacketBuffer::encode (compression on)."""
    comp = zlib.compress(message)
    extended = len(comp) > p.MAX_256_COMPRESSED_SIZE
    hdr = p.EXT_FRAG_HDR_SIZE if extended else p.FRAG_HDR_SIZE
    bsize = p.EXT_FRAG_BLOCK_SIZE if extended else p.FRAG_BLOCK_SIZE
    count = -(-len(comp) // bsize)
    frags = []
    for i in range(count):
        f = bytearray(p.FRAG_SIZE)
        if i == 0 or extended:
            struct.pack_into('>II', f, 0, len(message), len(comp))
        if extended:
            struct.pack_into('>HH', f, 8, i, count - 1)
        else:
            f[8], f[9] = i, count - 1
        chunk = comp[i * bsize:(i + 1) * bsize]
        f[hdr:hdr + len(chunk)] = chunk
        frags.append(bytes(f))
    return frags


class FragmentTests(unittest.TestCase):
    def test_single_fragment(self):
        frags = dvmhost_encode(ALIAS_FILE)
        self.assertEqual(len(frags), 1)
        self.assertEqual(p.FragmentAssembler().add(frags[0]), ALIAS_FILE)

    def test_many_fragments_any_order(self):
        import random
        msg = random.Random(1).randbytes(5000)                          # incompressible
        frags = dvmhost_encode(msg)
        self.assertGreater(len(frags), 5)
        asm = p.FragmentAssembler()
        order = frags[1:] + frags[:1]                    # block 0 last
        results = [asm.add(f) for f in order]
        self.assertEqual(results[:-1], [None] * (len(frags) - 1))
        self.assertEqual(results[-1], msg)

    def test_extended_layout(self):
        import random
        msg = random.Random(2).randbytes(p.MAX_256_COMPRESSED_SIZE + 2000)
        frags = dvmhost_encode(msg)
        self.assertGreater(len(frags), 256)
        asm = p.FragmentAssembler()
        out = None
        for f in frags:
            out = asm.add(f)
        self.assertEqual(out, msg)

    def test_bad_input(self):
        asm = p.FragmentAssembler()
        with self.assertRaises(p.FrameError):
            asm.add(b'\x00' * 100)
        bad = bytearray(dvmhost_encode(ALIAS_FILE)[0])
        bad[12:20] = b'\xff' * 8                         # corrupt the zlib stream
        with self.assertRaises(p.FrameError):
            asm.add(bytes(bad))

    def test_alias_file(self):
        self.assertEqual(p.parse_radio_aliases(ALIAS_FILE),
                         {251933: 'VK4ABC Mobile', 12007: 'Base 7', 241315: 'CCP Console'})


class FakeMetadataPort:
    """The FNE's metadata port (traffic port + 1): answers RADIO_ALIAS_SYNC on
    the peer's traffic address, as dvmhost's MetadataNetwork does."""

    def __init__(self, fne: FakeFNE, alias_file: bytes):
        self.fne = fne
        self.alias_file = alias_file
        self.requests = []
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', fne.port + 1))
        self.sock.settimeout(0.1)
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def close(self):
        self._stop.set()
        self.sock.close()

    def _run(self):
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(4096)
            except (socket.timeout, OSError):
                continue
            frame = p.decode_frame(data)
            self.requests.append((frame, addr))
            if frame.func == Func.RADIO_ALIAS_SYNC and addr[0] == self.fne.peer_addr[0]:
                for frag in dvmhost_encode(self.alias_file):
                    self.fne.send(Func.RADIO_ALIAS_SYNC, p.SUBFUNC_NOP, frag, frame.stream_id, seq=0)


class PeerSyncTests(unittest.TestCase):
    def setUp(self):
        self.fne = FakeFNE()
        try:
            self.md = FakeMetadataPort(self.fne, ALIAS_FILE)
        except OSError:
            self.fne.close()
            self.skipTest('the port after the fake FNE is in use')

    def tearDown(self):
        self.md.close()
        self.fne.close()

    def test_peer_asks_the_metadata_port_and_gets_aliases(self):
        peer = make_peer(self.fne)
        peer.start()
        try:
            self.assertTrue(wait_for(lambda: peer.alias(251933) is not None, 10))
            self.assertEqual(peer.aliases(), {251933: 'VK4ABC Mobile', 12007: 'Base 7', 241315: 'CCP Console'})
            self.assertEqual(peer.status['radio_aliases'], 3)
            req, addr = self.md.requests[0]
            self.assertEqual((req.func, req.subfunc, req.payload, req.seq),
                             (Func.RADIO_ALIAS_SYNC, p.SUBFUNC_NOP, b'\x00', p.END_OF_CALL_SEQ))
            self.assertEqual(addr, self.fne.peer_addr)     # same socket as the traffic link
            # and the console shows them
            c = DispatchConsole({'talkgroups': []}, [], radio_aliases=peer.alias)
            self.assertEqual(c.with_alias({'src_id': 12007})['src_alias'], 'Base 7')
        finally:
            peer.stop()

    def test_replies_for_another_request_are_ignored(self):
        peer = make_peer(self.fne, alias_sync_minutes=0)       # never asks
        peer.start()
        try:
            self.assertTrue(wait_for(lambda: peer.connected, 5))
            for frag in dvmhost_encode(ALIAS_FILE):
                self.fne.send(Func.RADIO_ALIAS_SYNC, p.SUBFUNC_NOP, frag, 12345, seq=0)
            self.assertFalse(wait_for(lambda: peer.aliases(), 1))
            self.assertEqual(self.md.requests, [])
        finally:
            peer.stop()


if __name__ == '__main__':
    unittest.main()
