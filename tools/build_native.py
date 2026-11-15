"""Build the blip25 IMBE vocoder (src/p25/native/blip25).

    python tools/build_native.py

blip25-vocoder (https://github.com/openBLIP25/blip25-vocoder) is a Rust
implementation of the P25 vocoder, bit-exact with the DVSI reference codec.
This builds it, behind a small C interface, as one shared library that the
app loads automatically when it's there. Without it the app falls back to the
pure Python encoder (src/p25/imbe.py) and decoder (src/p25/mbe_decoder.py),
which are slow and not reference-exact.

Needs Rust 1.85 or later (cargo). The crate is vendored (with two patches,
src/p25/native/blip25/PATCHES.md), so no internet is needed:
  Alpine           apk add cargo        (Alpine 3.22 or later for Rust 1.85+)
  Debian / Ubuntu  rustup (https://rustup.rs) - the distro cargo is often too old
  Windows / macOS  rustup (https://rustup.rs); on Windows also Visual Studio
                   or its Build Tools ("Desktop development with C++")

The build targets the plain x86-64 / arm64 baseline (no CPU-specific
instructions), so a library built on one machine also runs on older CPUs of
the same OS and architecture.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIVE = os.path.join(ROOT, 'src', 'p25', 'native')
CRATE = os.path.join(NATIVE, 'blip25')
sys.path.insert(0, ROOT)


def lib_names():
    """(what cargo builds, what the app loads)"""
    if sys.platform == 'win32':
        return 'ccp_blip25.dll', '_blip25.dll'
    if sys.platform == 'darwin':
        return 'libccp_blip25.dylib', '_blip25.dylib'
    return 'libccp_blip25.so', '_blip25.so'


def find_cargo() -> str:
    cargo = shutil.which('cargo') or os.path.join(os.path.expanduser('~'), '.cargo', 'bin',
                                                   'cargo.exe' if sys.platform == 'win32' else 'cargo')
    if not os.path.exists(cargo):
        raise SystemExit('cargo not found. Install Rust 1.85+ (Alpine: apk add cargo; elsewhere: https://rustup.rs).')
    return cargo


def build() -> str:
    env = dict(os.environ)
    env.pop('RUSTFLAGS', None)
    if sys.platform.startswith('linux'):
        # musl (Alpine) Rust can default to static linking, which can't make a
        # shared library; this is already the default on glibc.
        env['RUSTFLAGS'] = '-C target-feature=-crt-static'
    result = subprocess.run([find_cargo(), 'build', '--release', '--locked'], cwd=CRATE, env=env)
    if result.returncode != 0:
        raise SystemExit('Build failed.')
    built, name = lib_names()
    src = os.path.join(CRATE, 'target', 'release', built)
    out = os.path.join(NATIVE, name)
    # A running service keeps the library open: on Windows it can't be
    # overwritten, but it can be renamed out of the way (the service keeps
    # the old one until it restarts).
    stale = out + '.old'
    if os.path.exists(stale):
        try:
            os.remove(stale)
        except OSError:
            pass                        # still loaded by a service from an earlier build
    if os.path.exists(out):
        try:
            os.remove(out)
        except OSError:
            os.replace(out, stale)
    shutil.copy2(src, out)
    return out


def main():
    out = build()
    print(f'Built:  {out}')
    from src.p25 import blip25
    blip25.load(force=True)
    if not blip25.available():
        raise SystemExit(f'Built, but it would not load: {blip25.load_error()}')
    print('The blip25 encoder and decoder load. Check them with:\n'
          '    python -m unittest tests.test_blip25\n'
          '    python tools/bench.py')


if __name__ == '__main__':
    main()
