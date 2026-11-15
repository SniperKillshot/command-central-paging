"""Configuration loading: config.yaml with environment variable overrides."""
import os
import logging
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    'forewarned': {
        'base_url': 'http://forewarned.local:5000',
        'poll_interval': 5,
        'push_token': '',
    },
    'home_assistant': {
        'base_url': 'http://homeassistant.local:8123',
        'token': '',
        'entities': {
            'advisory': 'input_boolean.forewarned_manual_advisory',
            'watch': 'input_boolean.forewarned_manual_watch',
            'warning': 'input_boolean.forewarned_manual_warning',
            'emergency': 'input_boolean.forewarned_manual_emergency',
        },
    },
    'cme': {
        'sip_server': '',
        'sip_domain': '',
        'sip_port': 5060,
        'transport': 'udp',
        'sip_user': '',
        'sip_password': '',
        'tts_voice': 'en-us',
        'tts_speed': 160,
    },
    'p25': {
        'source_id': 0,
        'sys_id': '001',
        'net_id': 'BB800',
        'priority': 4,
        'lead_in_ms': 500,
        'sound_gap_ms': 400,
        'tail_ms': 300,
        'voice_gain_db': 0,
        'sound_gain_db': 0,
        'max_call_seconds': 120,
        'require_active_talkgroup': True,
    },
    'tts': {
        'default': {'engine': '', 'voice': '', 'rate': 100},
        'swift_path': '',
        'espeak_path': '',
        'piper_path': '',
        'piper_voices_dir': 'voices',
    },
    'sounds_dir': 'sounds',
    'data_dir': 'data',
    'dispatch': {
        'talkgroups': [],
        'show_fne_talkgroups': True,
        'tones': [],
    },
    'dvm': {
        'enabled': False,
        'address': '',
        'port': 62031,
        'peer_id': 0,
        'password': '',
        'preshared_key': '',
        'identity': 'CCPAGING',
        'location': '',
        'conventional_peer': False,
        'ping_interval': 10,
        'retry_interval': 10,
        'timeout': 60,
    },
    'paging_destinations': [],
    'alert_routing': {
        'page_existing_on_startup': False,
        'rules': [],
    },
    'web': {
        'host': '0.0.0.0',
        'port': 5000,
        'api_token': '',
    },
}

# Env vars that override a specific dotted config path. Anything secret-shaped
# lives here so it never has to be committed in config.yaml.
ENV_OVERRIDES = {
    'FOREWARNED_BASE_URL': ('forewarned', 'base_url'),
    'FOREWARNED_PUSH_TOKEN': ('forewarned', 'push_token'),
    'HA_BASE_URL': ('home_assistant', 'base_url'),
    'HA_TOKEN': ('home_assistant', 'token'),
    'CME_SIP_SERVER': ('cme', 'sip_server'),
    'CME_SIP_DOMAIN': ('cme', 'sip_domain'),
    'CME_SIP_USER': ('cme', 'sip_user'),
    'CME_SIP_PASSWORD': ('cme', 'sip_password'),
    'DVM_FNE_ADDRESS': ('dvm', 'address'),
    'DVM_FNE_PASSWORD': ('dvm', 'password'),
    'DVM_FNE_PRESHARED_KEY': ('dvm', 'preshared_key'),
    'WEB_API_TOKEN': ('web', 'api_token'),
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base, returning base."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class Config:
    """Loads config.yaml (falling back to defaults), then applies env var overrides."""

    def __init__(self, path: str = 'config.yaml'):
        self.path = path
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        import copy
        data = copy.deepcopy(DEFAULT_CONFIG)

        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                user_config = yaml.safe_load(f) or {}
            _deep_merge(data, user_config)
            logger.info(f"Loaded configuration from {self.path}")
        else:
            logger.warning(f"{self.path} not found - using defaults (see config.yaml.example)")

        for env_var, path_tuple in ENV_OVERRIDES.items():
            value = os.getenv(env_var)
            if value is not None:
                node = data
                for key in path_tuple[:-1]:
                    node = node[key]
                node[path_tuple[-1]] = value
                logger.info(f"Config override from env: {env_var} -> {'.'.join(path_tuple)}")

        return data

    def get(self, *path: str, default: Any = None) -> Any:
        """Get a nested config value, e.g. config.get('forewarned', 'base_url')."""
        node: Any = self.data
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node
