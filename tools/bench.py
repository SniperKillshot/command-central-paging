"""How fast is this machine at the P25 voice work?

    python tools/bench.py

Run from the project folder. P25 audio is 50 frames a second, so a frame
must be encoded in under 20 ms for live talk to keep up; pages are encoded
ahead of time, so on a slower machine they just take longer to start.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.p25.imbe import IMBEEncoder  # noqa: E402
from src.p25.mbe_decoder import IMBEDecoder  # noqa: E402

FRAMES = 100        # 2 seconds of audio
REALTIME_MS = 20.0


def speechlike(n):
    t = np.arange(n) / 8000
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    x = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 700, 1100), 1))
    return (x * env * 6000).astype(np.int16)


def main():
    pcm = speechlike(FRAMES * 160)
    enc = IMBEEncoder()
    enc.encode(pcm[:160].tolist())                     # warm up
    t = time.perf_counter()
    frames = [enc.encode(pcm[i * 160:(i + 1) * 160].tolist()) for i in range(FRAMES)]
    enc_ms = (time.perf_counter() - t) * 1000 / FRAMES

    dec = IMBEDecoder()
    t = time.perf_counter()
    for f in frames:
        dec.decode(f)
    dec_ms = (time.perf_counter() - t) * 1000 / FRAMES

    print(f"Python {sys.version.split()[0]}, numpy {np.__version__}")
    for what, cls in (('Encoder', IMBEEncoder), ('Decoder', IMBEDecoder)):
        if getattr(cls, 'native', False):
            print(f"{what}: compiled (src/p25/native)")
        else:
            print(f"{what}: pure Python - build the compiled one with: python tools/build_native.py")
    print(f"Encode: {enc_ms:6.1f} ms/frame  ({REALTIME_MS / enc_ms:4.1f}x real time)")
    print(f"Decode: {dec_ms:6.1f} ms/frame  ({REALTIME_MS / dec_ms:4.1f}x real time)")
    print()
    if enc_ms < REALTIME_MS * 0.8:
        print("Live talk: OK")
    elif enc_ms < REALTIME_MS:
        print("Live talk: borderline - expect gaps when anything else is busy")
    else:
        print("Live talk: too slow - the encoder can't keep up with speech")
    page = 15 * 50 * enc_ms / 1000
    print(f"A 15 s page takes about {max(page - 15, 0):.0f} s of encoding before it starts transmitting"
          if page > 15 else "Pages: start almost immediately")
    calls = int(REALTIME_MS * 0.7 // dec_ms)
    print(f"Listening: about {calls} simultaneous call(s) can be decoded" if calls else
          "Listening: too slow to decode even one call in real time")


if __name__ == '__main__':
    main()
