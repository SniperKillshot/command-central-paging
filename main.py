"""Command Central Paging - entry point."""
import atexit
import logging
import os
import signal
import sys

from src.audio.tones import SoundLibrary
from src.audio.tts import TTS
from src.config import Config
from src.console import DispatchConsole
from src.dispatcher import Dispatcher
from src.dvm import FNEPeer
from src.dvm.rest import TalkgroupDirectory
from src.events import EventDetector
from src.forewarned_client import ForewarnedClient
from src.p25.crypto import KeyStore
from src.p25.live import LiveTalk
from src.p25.pager import P25Pager
from src.p25.rx_audio import RxAudio
from src.paging import create_backends
from src.settings_store import SettingsStore
from src.state import record_events, update_dvm_state, update_paging_destinations
from src.web_ui import Services, create_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# One line per HTTP request is a lot: every open console polls once a second.
# CCP_ACCESS_LOG=1 brings it back.
if os.environ.get('CCP_ACCESS_LOG', '').strip() in ('', '0'):
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
# A service manager stops us with SIGTERM: exit normally so the atexit
# handlers run (the FNE is told we're leaving).
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


def main():
    logger.info("=" * 60)
    logger.info("Starting Command Central Paging")
    logger.info("=" * 60)

    config = Config()

    destinations = config.get('paging_destinations', default=[]) or []
    update_paging_destinations(destinations)

    tts_settings = config.get('tts', default={}) or {}
    tts = TTS(tts_settings)
    sounds = SoundLibrary(config.get('sounds_dir', default='sounds'))
    settings = SettingsStore(os.path.join(config.get('data_dir', default='data'), 'settings.json'),
                             {'tts': tts_settings.get('default') or {}})

    dvm_peer = None
    dvm_settings = config.get('dvm', default={}) or {}
    if dvm_settings.get('enabled'):
        dvm_peer = FNEPeer(dvm_settings, on_status=update_dvm_state)
        dvm_peer.start()
        atexit.register(dvm_peer.stop)   # tell the FNE we're leaving (RPT_DISC)
    else:
        logger.info("DVM FNE connection disabled (dvm.enabled: false)")

    tg_names = None
    if dvm_settings.get('enabled') and dvm_settings.get('rest_enabled'):
        if dvm_settings.get('rest_password'):
            tg_names = TalkgroupDirectory(dvm_settings)
            tg_names.start()
            atexit.register(tg_names.stop)
        else:
            logger.warning("dvm.rest_enabled is on but dvm.rest_password is empty - no talkgroup names")

    keys = KeyStore(config.get('encryption', default={}) or {})
    if len(keys):
        logger.info(f"Encryption: {len(keys)} key(s); sending encrypted to TG "
                    f"{', '.join(str(tg) for tg in sorted(keys.talkgroups())) or 'none'}")
    pager = P25Pager(config.get('p25', default={}) or {}, tts, sounds,
                     voice_settings=lambda: settings.get('tts'), peer=dvm_peer, keys=keys)

    routing = config.get('alert_routing', default={}) or {}
    detector = EventDetector(fire_on_startup=bool(routing.get('page_existing_on_startup', False)))
    # p25 is always loaded: the dispatch console can page any talkgroup
    backends = create_backends(config, always=('p25',), dvm_peer=dvm_peer, p25_pager=pager if dvm_peer else None)
    dispatcher = Dispatcher(routing.get('rules', []), destinations, backends, sound_exists=sounds.exists)
    rx_audio = live = None
    if dvm_peer:
        rx_audio = RxAudio(keys)
        dvm_peer.on_voice = rx_audio.on_p25
        live = LiveTalk(pager)
    console = DispatchConsole(config.get('dispatch', default={}) or {}, destinations, dispatcher, sounds, live,
                              store=settings, names=tg_names, keys=keys)

    def on_status(status) -> int:
        """Shared by the poller and the push webhook."""
        events = detector.process(status)
        record_events(events)
        return dispatcher.handle(events)

    forewarned_client = ForewarnedClient(
        base_url=config.get('forewarned', 'base_url'),
        poll_interval=config.get('forewarned', 'poll_interval', default=5),
        on_status=on_status,
    )
    forewarned_client.start()

    services = Services(dispatcher=dispatcher, destinations=destinations, tts=tts,
                        sounds=sounds, settings=settings, pager=pager, console=console,
                        rx_audio=rx_audio)
    app = create_app(on_status=on_status, push_token=config.get('forewarned', 'push_token', default=''),
                     services=services, api_token=config.get('web', 'api_token', default=''))
    host = config.get('web', 'host', default='0.0.0.0')
    port = config.get('web', 'port', default=5000)
    logger.info(f"Dashboard on http://{host}:{port}/ (dispatch console: /dispatch)")
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True,
    )


if __name__ == '__main__':
    main()
