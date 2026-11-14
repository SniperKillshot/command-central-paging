"""Settings changed from the dashboard at runtime (the TTS voice, the dispatch tiles),
persisted to a small JSON file so they survive restarts. config.yaml holds
the defaults; this file only holds what was changed."""
import copy
import json
import logging
import os
import threading
from typing import Any, Dict

logger = logging.getLogger(__name__)


class SettingsStore:
    def __init__(self, path: str, defaults: Dict[str, Any]):
        self.path = path
        self.defaults = copy.deepcopy(defaults)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as f:
                    self._data = json.load(f) or {}
            except (OSError, ValueError) as e:
                logger.warning(f"Ignoring unreadable settings file {path}: {e}")

    def get(self, section: str) -> Dict[str, Any]:
        with self._lock:
            merged = dict(self.defaults.get(section) or {})
            merged.update(self._data.get(section) or {})
            return merged

    def update(self, section: str, values: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._data.setdefault(section, {}).update(values)
            self._write()
        return self.get(section)

    def clear(self, section: str) -> None:
        """Drop everything changed in a section (back to the defaults)."""
        with self._lock:
            if self._data.pop(section, None) is not None:
                self._write()

    def _write(self) -> None:
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp, self.path)
