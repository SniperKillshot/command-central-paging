"""Common interface for paging backends (CME SIP, P25, two-tone, POCSAG...)."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class PageResult:
    success: bool
    status: str            # 'sent', 'failed', 'not_implemented'
    detail: str = ''
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class PagingBackend:
    """A backend delivers a text message to one destination. Voice backends
    render the text with TTS themselves."""

    name = 'base'

    def __init__(self, settings: Dict[str, Any], **context: Any):
        """`context` carries shared services a backend may need (for example
        `dvm_peer`, the FNE connection). Backends ignore what they don't use."""
        self.settings = settings
        self.context = context

    def send(self, destination: Dict[str, Any], message: str, sound: Optional[str] = None,
             voice: Optional[Dict[str, Any]] = None, audio: Any = None) -> PageResult:
        """message: text to speak (may be empty if only a sound is wanted).
        sound: optional alert sound name played first. voice: optional TTS
        voice override {engine, voice, rate}. audio: recorded speech (8 kHz
        float) used in place of the spoken message, for voice backends."""
        raise NotImplementedError
