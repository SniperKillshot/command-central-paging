"""Background poller for Forewarned's REST API."""
import logging
import time
from threading import Thread
from typing import Optional

import requests

from src.state import update_forewarned_state

logger = logging.getLogger(__name__)


class ForewarnedClient:
    """Polls Forewarned's GET /api/status on an interval and writes results into
    the shared app_state. Connection failures mark the state 'unreachable'
    rather than silently clearing the alert - an unreachable Forewarned should
    never read as "all clear" on an alerting console."""

    def __init__(self, base_url: str, poll_interval: int = 5):
        self.base_url = base_url.rstrip('/')
        self.poll_interval = poll_interval
        self._stop = False

    def poll_once(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/status", timeout=5)
            if response.status_code == 200:
                update_forewarned_state(reachable=True, status=response.json())
                return True
            logger.warning(f"Forewarned returned status {response.status_code}")
            update_forewarned_state(reachable=False)
            return False
        except requests.RequestException as e:
            logger.warning(f"Forewarned unreachable at {self.base_url}: {e}")
            update_forewarned_state(reachable=False)
            return False

    def _loop(self):
        logger.info(f"Forewarned poller started - {self.base_url}, every {self.poll_interval}s")
        while not self._stop:
            self.poll_once()
            time.sleep(self.poll_interval)

    def start(self) -> Thread:
        thread = Thread(target=self._loop, daemon=True)
        thread.start()
        return thread

    def stop(self):
        self._stop = True
