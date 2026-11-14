"""Background poller for Forewarned's REST API."""
import logging
import time
from threading import Thread
from typing import Any, Callable, Dict, Optional

import requests

from src.state import update_forewarned_state

logger = logging.getLogger(__name__)


class ForewarnedClient:
    """Polls Forewarned's GET /api/status on an interval and writes results into
    the shared app_state. Connection failures mark the state 'unreachable'
    rather than silently clearing the alert - an unreachable Forewarned should
    never read as "all clear" on an alerting console."""

    def __init__(self, base_url: str, poll_interval: int = 5,
                 on_status: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.base_url = base_url.rstrip('/')
        self.poll_interval = poll_interval
        self.on_status = on_status
        self._stop = False

    def poll_once(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                update_forewarned_state(reachable=True, status=status)
                if self.on_status:
                    try:
                        self.on_status(status)
                    except Exception:
                        logger.exception("Error handling Forewarned status")
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
