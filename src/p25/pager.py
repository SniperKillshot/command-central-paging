"""P25 voice paging: alert sound + spoken message -> P25 group call.

A page is rendered as: [lead-in silence] [alert sound] [gap] [speech] [tail],
at 8 kHz, then IMBE-encoded and sent as one call to the destination's
talkgroup through the FNE connection.
"""
import logging
from typing import Dict, Optional

import numpy as np

from src.audio import dsp
from src.audio.tones import SoundLibrary
from src.audio.tts import TTS, TTSError
from src.p25 import frames as F
from src.p25.mbe_decoder import decode_to_pcm
from src.p25.voice import TxResult, VoiceCall, encode_pcm

logger = logging.getLogger(__name__)


def _hex_id(v, default):
    """System/network IDs are hex in dvmhost configs ("001", "BB800"). YAML
    turns unquoted all-digit values into ints, so ints are read back as the
    hex digits they were written as."""
    if v is None or v == '':
        return default
    return int(str(v).strip().lower().removeprefix('0x'), 16)


class P25Pager:
    def __init__(self, settings: Dict, tts: TTS, sounds: SoundLibrary, voice_settings, peer=None):
        """voice_settings: callable returning the current default
        {'engine', 'voice', 'rate'} (the dashboard can change it)."""
        self.settings = settings or {}
        self.tts = tts
        self.sounds = sounds
        self.voice_settings = voice_settings
        self.peer = peer

    # -- audio ---------------------------------------------------------------

    def resolve_voice(self, override: Optional[Dict] = None) -> Dict:
        v = dict(self.voice_settings() or {})
        for k, val in (override or {}).items():
            if val not in (None, ''):
                v[k] = val
        if not v.get('engine'):
            first = self.tts.first_available()
            if first:
                v['engine'] = first[0]
                v['voice'] = v.get('voice') or first[1]
        v.setdefault('rate', 100)
        return v

    def render(self, message: str = '', sound: Optional[str] = None, voice: Optional[Dict] = None) -> np.ndarray:
        """Float audio at 8 kHz for the whole page."""
        return self.render_with_tones(message, sound, voice)[0]

    def render_with_tones(self, message: str = '', sound: Optional[str] = None, voice: Optional[Dict] = None,
                          audio: Optional[np.ndarray] = None):
        """(float audio at 8 kHz, tone track). The tone track carries the known
        frequencies of generated alert sounds so the vocoder codes them cleanly;
        it is NaN for silence and speech. audio: recorded speech (8 kHz float)
        used in place of the spoken message."""
        lead = float(self.settings.get('lead_in_ms', 400)) / 1000
        gap = float(self.settings.get('sound_gap_ms', 400)) / 1000
        tail = float(self.settings.get('tail_ms', 300)) / 1000

        def blank(x):
            return np.full((len(x), 2), np.nan)

        parts = [dsp.silence(lead)]
        tracks = [blank(parts[0])]
        if sound:
            audio, track = self.sounds.render_with_tones(sound)
            parts.append(dsp.apply_gain_db(audio, float(self.settings.get('sound_gain_db', 0))))
            tracks.append(track)
        speech = None
        if audio is not None and len(audio):
            speech = dsp.condition_recording(np.asarray(audio, dtype=np.float64))
        elif message and message.strip():
            v = self.resolve_voice(voice)
            if not v.get('engine'):
                raise TTSError('no text-to-speech engine is installed')
            tts_audio, rate = self.tts.synthesize(message, v['engine'], v.get('voice', ''), int(v.get('rate', 100)))
            speech = dsp.condition_voice(dsp.resample(tts_audio, rate, dsp.RATE))
        if speech is not None:
            speech = dsp.apply_gain_db(speech, float(self.settings.get('voice_gain_db', 0)))
            if sound:
                parts.append(dsp.silence(gap))
                tracks.append(blank(parts[-1]))
            parts.append(speech)
            tracks.append(blank(speech))
        if len(parts) == 1:
            raise ValueError('nothing to send - give a message and/or a sound')
        parts.append(dsp.silence(tail))
        tracks.append(blank(parts[-1]))
        return np.concatenate(parts), np.concatenate(tracks)

    def preview(self, message: str = '', sound: Optional[str] = None, voice: Optional[Dict] = None,
                as_radio: bool = True) -> bytes:
        """WAV bytes of the page. as_radio=True runs it through the P25 vocoder
        and back, so it sounds like it will on a radio."""
        audio, tones = self.render_with_tones(message, sound, voice)
        pcm = dsp.to_pcm16(audio)
        if as_radio:
            pcm = decode_to_pcm(encode_pcm(pcm, tones))
        return dsp.wav_bytes(pcm)

    # -- transmission ----------------------------------------------------------

    def call_params(self, talkgroup: int) -> F.CallParams:
        s = self.settings
        return F.CallParams(
            src_id=int(s.get('source_id') or 0),
            dst_id=int(talkgroup),
            priority=int(s.get('priority', 4)),
            sys_id=_hex_id(s.get('sys_id'), F.SID_STD_DEFAULT),
            net_id=_hex_id(s.get('net_id'), F.WACN_STD_DEFAULT),
        )

    def check_ready(self, talkgroup) -> Optional[str]:
        """Why a page to this talkgroup can't go out right now, or None."""
        if self.peer is None:
            return 'DVM FNE connection not enabled (dvm.enabled)'
        if not self.settings.get('source_id'):
            return 'p25.source_id (the radio ID pages are sent from) is not set'
        try:
            tg = int(talkgroup)
        except (TypeError, ValueError):
            return f"destination talkgroup '{talkgroup}' is not a number"
        if not 1 <= tg <= 0xFFFF:
            return f'talkgroup {tg} is outside the P25 range 1-65535'
        if not self.peer.connected:
            return VoiceCall(self.peer, self.call_params(tg))._not_connected()
        active = self.peer.active_talkgroup_ids()
        if active and tg not in active and self.settings.get('require_active_talkgroup', True):
            return f'talkgroup {tg} is not active on the FNE'
        return None

    def page(self, talkgroup, message: str = '', sound: Optional[str] = None,
             voice: Optional[Dict] = None, audio: Optional[np.ndarray] = None) -> TxResult:
        problem = self.check_ready(talkgroup)
        if problem:
            return TxResult(False, 'failed', problem)
        try:
            audio, tones = self.render_with_tones(message, sound, voice, audio)
        except (TTSError, KeyError, ValueError) as e:
            return TxResult(False, 'failed', f'could not produce audio: {e}')
        call = VoiceCall(self.peer, self.call_params(int(talkgroup)),
                         max_seconds=float(self.settings.get('max_call_seconds', 120)))
        return call.transmit(dsp.to_pcm16(audio), tones)
