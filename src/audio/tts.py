"""Text-to-speech engines.

Each engine wraps a command-line synthesiser and returns (float audio, rate).
Voices are selected per engine; speaking rate is a percentage (100 = the
engine's normal speed) mapped onto each engine's own scale.

  sapi      Windows built-in voices (System.Speech via Windows PowerShell)
  cepstral  Cepstral Swift ("swift" CLI) - the classic weather-radio voices
  espeak    eSpeak NG ("espeak-ng" CLI) - small, robotic, on every Linux
  piper     Piper neural TTS ("piper" CLI + .onnx voice models) - best free quality
"""
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.audio.dsp import read_wav

logger = logging.getLogger(__name__)

TIMEOUT = 60
VOICE_CACHE_SECONDS = 300


class TTSError(RuntimeError):
    pass


def _run(cmd: List[str], stdin: Optional[bytes] = None) -> subprocess.CompletedProcess:
    kwargs = {}
    if sys.platform == 'win32':
        kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        return subprocess.run(cmd, input=stdin, capture_output=True, timeout=TIMEOUT, check=False, **kwargs)
    except FileNotFoundError as e:
        raise TTSError(f'{cmd[0]} not found') from e
    except subprocess.TimeoutExpired as e:
        raise TTSError(f'{os.path.basename(cmd[0])} timed out') from e


class Engine:
    name = 'base'
    label = 'base'

    def __init__(self, settings: Dict):
        self.settings = settings or {}
        self._voices_cache: Tuple[float, List[Dict]] = (0.0, [])
        self._lock = threading.Lock()

    def available(self) -> bool:
        raise NotImplementedError

    def _list_voices(self) -> List[Dict]:
        raise NotImplementedError

    def voices(self) -> List[Dict]:
        with self._lock:
            ts, cached = self._voices_cache
            if cached and time.monotonic() - ts < VOICE_CACHE_SECONDS:
                return cached
            try:
                voices = self._list_voices() if self.available() else []
            except TTSError as e:
                logger.warning(f"TTS {self.name}: cannot list voices: {e}")
                voices = []
            self._voices_cache = (time.monotonic(), voices)
            return voices

    def synthesize(self, text: str, voice: str, rate: int = 100) -> Tuple[np.ndarray, int]:
        with tempfile.TemporaryDirectory(prefix='ccp-tts-') as tmp:
            out = os.path.join(tmp, 'out.wav')
            txt = os.path.join(tmp, 'text.txt')
            with open(txt, 'w', encoding='utf-8') as f:
                f.write(text)
            self._synth_to_file(text, txt, out, voice, rate)
            if not os.path.exists(out) or os.path.getsize(out) < 64:
                raise TTSError(f'{self.label} produced no audio')
            return read_wav(out)

    def _synth_to_file(self, text, text_path, out_path, voice, rate):
        raise NotImplementedError


class SapiEngine(Engine):
    name = 'sapi'
    label = 'Windows voices (SAPI)'
    _PS = 'powershell.exe'

    _SCRIPT = r'''
param([string]$TextPath, [string]$OutPath, [string]$Voice, [int]$Rate, [switch]$List)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($List) {
  foreach ($v in $s.GetInstalledVoices()) { if ($v.Enabled) { "{0}`t{1}`t{2}" -f $v.VoiceInfo.Name, $v.VoiceInfo.Culture, $v.VoiceInfo.Gender } }
  exit 0
}
if ($Voice) { $s.SelectVoice($Voice) }
$s.Rate = $Rate
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo 16000, ([System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen), ([System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile($OutPath, $fmt)
$s.Speak([System.IO.File]::ReadAllText($TextPath, [System.Text.Encoding]::UTF8))
$s.Dispose()
'''

    def available(self):
        return sys.platform == 'win32' and shutil.which(self._PS) is not None

    def _script_path(self, tmp):
        p = os.path.join(tmp, 'sapi.ps1')
        with open(p, 'w', encoding='utf-8-sig') as f:
            f.write(self._SCRIPT)
        return p

    def _ps(self, args: List[str]) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory(prefix='ccp-sapi-') as tmp:
            cmd = [self._PS, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                   '-File', self._script_path(tmp)] + args
            r = _run(cmd)
        if r.returncode != 0:
            raise TTSError(r.stderr.decode(errors='replace').strip()[:300] or 'PowerShell failed')
        return r

    def _list_voices(self):
        out = []
        for line in self._ps(['-List']).stdout.decode(errors='replace').splitlines():
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[0]:
                out.append({'id': parts[0], 'name': parts[0], 'language': parts[1],
                            'gender': parts[2].lower() if len(parts) > 2 else ''})
        return out

    def _synth_to_file(self, text, text_path, out_path, voice, rate):
        sapi_rate = int(max(-10, min(10, round((rate - 100) / 10))))
        self._ps(['-TextPath', text_path, '-OutPath', out_path, '-Voice', voice or '', '-Rate', str(sapi_rate)])


class CepstralEngine(Engine):
    name = 'cepstral'
    label = 'Cepstral Swift'

    def _exe(self):
        return self.settings.get('swift_path') or shutil.which('swift') or ''

    def available(self):
        exe = self._exe()
        return bool(exe) and os.path.exists(exe)

    def _list_voices(self):
        r = _run([self._exe(), '--voices'])
        out = []
        for line in r.stdout.decode(errors='replace').splitlines():
            cols = [c.strip() for c in line.split('|')]
            if len(cols) >= 7 and cols[0] and cols[0] != 'Voice' and not cols[0].startswith('-'):
                out.append({'id': cols[0], 'name': cols[0], 'language': cols[5], 'gender': cols[3],
                            'licensed': cols[2].lower() == 'yes'})
        return out

    def _synth_to_file(self, text, text_path, out_path, voice, rate):
        wpm = int(170 * rate / 100)
        cmd = [self._exe(), '-f', text_path, '-o', out_path, '-p', f'speech/rate={wpm}']
        if voice:
            cmd[1:1] = ['-n', voice]
        r = _run(cmd)
        if r.returncode != 0:
            raise TTSError(r.stderr.decode(errors='replace').strip()[:300] or 'swift failed')


class EspeakEngine(Engine):
    name = 'espeak'
    label = 'eSpeak NG'

    def _exe(self):
        return self.settings.get('espeak_path') or shutil.which('espeak-ng') or shutil.which('espeak') or ''

    def available(self):
        return bool(self._exe())

    def _list_voices(self):
        r = _run([self._exe(), '--voices'])
        out = []
        for line in r.stdout.decode(errors='replace').splitlines()[1:]:
            cols = line.split()
            if len(cols) >= 4:
                out.append({'id': cols[1], 'name': f'{cols[3]} ({cols[1]})', 'language': cols[1],
                            'gender': {'M': 'male', 'F': 'female'}.get(cols[2].split('/')[-1], '')})
        return out

    def _synth_to_file(self, text, text_path, out_path, voice, rate):
        cmd = [self._exe(), '-f', text_path, '-w', out_path, '-s', str(int(175 * rate / 100))]
        if voice:
            cmd += ['-v', voice]
        r = _run(cmd)
        if r.returncode != 0:
            raise TTSError(r.stderr.decode(errors='replace').strip()[:300] or 'espeak failed')


class PiperEngine(Engine):
    name = 'piper'
    label = 'Piper (neural)'

    def _exe(self):
        return self.settings.get('piper_path') or shutil.which('piper') or ''

    def _voices_dir(self):
        return self.settings.get('piper_voices_dir') or 'voices'

    def available(self):
        return bool(self._exe()) and os.path.isdir(self._voices_dir())

    def _list_voices(self):
        out = []
        for fn in sorted(os.listdir(self._voices_dir())):
            if fn.endswith('.onnx'):
                stem = fn[:-5]
                m = re.match(r'([a-z]{2}_[A-Z]{2})-(.+?)-(x_low|low|medium|high)$', stem)
                name = f'{m.group(2)} ({m.group(1)}, {m.group(3)})' if m else stem
                out.append({'id': stem, 'name': name, 'language': m.group(1) if m else ''})
        return out

    def _synth_to_file(self, text, text_path, out_path, voice, rate):
        model = os.path.join(self._voices_dir(), f'{voice}.onnx')
        if not voice or not os.path.exists(model):
            raise TTSError(f"piper voice model '{voice}' not found in {self._voices_dir()}")
        cmd = [self._exe(), '--model', model, '--output_file', out_path,
               '--length_scale', f'{100 / max(rate, 10):.3f}']
        r = _run(cmd, stdin=text.encode('utf-8'))
        if r.returncode != 0:
            raise TTSError(r.stderr.decode(errors='replace').strip()[-300:] or 'piper failed')


ENGINE_CLASSES = {cls.name: cls for cls in (SapiEngine, CepstralEngine, EspeakEngine, PiperEngine)}


class TTS:
    """All configured engines, plus the current default voice."""

    def __init__(self, settings: Dict):
        self.settings = settings or {}
        self.engines = {name: cls(self.settings) for name, cls in ENGINE_CLASSES.items()}

    def describe(self) -> List[Dict]:
        return [{'name': e.name, 'label': e.label, 'available': e.available(),
                 'voices': e.voices() if e.available() else []}
                for e in self.engines.values()]

    def first_available(self) -> Optional[Tuple[str, str]]:
        for e in self.engines.values():
            if e.available():
                voices = e.voices()
                if voices:
                    return e.name, voices[0]['id']
        return None

    def synthesize(self, text: str, engine: str, voice: str, rate: int = 100) -> Tuple[np.ndarray, int]:
        if not text.strip():
            raise TTSError('nothing to say')
        eng = self.engines.get(engine)
        if eng is None:
            raise TTSError(f"unknown TTS engine '{engine}' (known: {', '.join(self.engines)})")
        if not eng.available():
            raise TTSError(f'{eng.label} is not installed on this machine')
        t = time.monotonic()
        audio, sr = eng.synthesize(text, voice, int(rate or 100))
        logger.info(f"TTS {engine}/{voice or 'default'}: {audio.size / sr:.1f}s of speech in {time.monotonic() - t:.1f}s")
        return audio, sr
