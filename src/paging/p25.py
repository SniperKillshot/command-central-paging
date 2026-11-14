"""P25 voice paging backend.

This service is a peer on the DVM FNE (src/dvm). A page is spoken with the
configured TTS voice (optionally after an alert sound), encoded with the
IMBE vocoder, and sent as a P25 group voice call to the destination's
talkgroup. See src/p25/pager.py.
"""
import logging

from src.paging.base import PageResult, PagingBackend

logger = logging.getLogger(__name__)


class P25Backend(PagingBackend):
    name = 'p25'

    def send(self, destination, message, sound=None, voice=None, audio=None):
        target = destination.get('talkgroup') or '?'
        pager = self.context.get('p25_pager')
        if pager is None:
            # Without a pager there is no FNE connection to send through.
            logger.warning(f"[P25 NOT SENT] no DVM FNE connection configured - talkgroup {target}: {message}")
            return PageResult(success=False, status='failed',
                              detail=f'DVM FNE connection not enabled (dvm.enabled) - talkgroup {target} was not paged')
        voice = voice or destination.get('voice')
        result = pager.page(target, message, sound=sound, voice=voice, audio=audio)
        if not result.ok:
            logger.warning(f"[P25 NOT SENT] talkgroup {target}: {result.detail}")
            return PageResult(success=False, status='failed', detail=f'{result.detail} - talkgroup {target} was not paged')
        return PageResult(success=True, status='sent', detail=result.detail)
