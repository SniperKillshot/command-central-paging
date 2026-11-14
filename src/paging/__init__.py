"""Pluggable paging backends. Each destination in config names a backend."""
from src.paging.base import PageResult, PagingBackend
from src.paging.mock import MockBackend
from src.paging.p25 import P25Backend

BACKENDS = {
    'mock': MockBackend,
    'p25': P25Backend,
}


def create_backends(config, always=(), **context) -> dict:
    """Instantiate one backend object per backend type referenced by a destination.
    Backend-specific settings come from the top-level config section of the same
    name (e.g. a `p25:` block). `context` (e.g. dvm_peer=...) is passed to every
    backend. `always` names backends to load even if no destination uses them."""
    used = {d.get('backend', 'mock') for d in (config.get('paging_destinations') or [])} | set(always)
    backends = {}
    for name in used:
        cls = BACKENDS.get(name)
        if cls is None:
            raise ValueError(f"Unknown paging backend '{name}' (known: {', '.join(sorted(BACKENDS))})")
        backends[name] = cls(config.get(name, default={}) or {}, **context)
    return backends
