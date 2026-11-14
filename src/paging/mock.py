"""Mock backend: logs the page and reports success. Lets the whole
event -> rule -> page pipeline be exercised with no paging hardware."""
import logging

from src.paging.base import PageResult, PagingBackend

logger = logging.getLogger(__name__)


class MockBackend(PagingBackend):
    name = 'mock'

    def send(self, destination, message, sound=None, voice=None, audio=None):
        prefix = f"[{sound}] " if sound else ''
        if audio is not None:
            message = f"{message} [recorded voice, {len(audio) / 8000:.1f}s]".strip()
        logger.info(f"[MOCK PAGE] -> {destination.get('name', destination.get('id'))}: {prefix}{message}")
        return PageResult(success=True, status='sent', detail='mock backend - nothing transmitted')
