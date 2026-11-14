"""Flask web application: dashboard + REST API."""
import hmac
import logging
import queue
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

from flask import Flask, Response, jsonify, render_template, request

from src.p25.live import LiveError
from src.p25.rx_audio import MSG_KEEPALIVE, message as rx_message
from src.state import snapshot, update_forewarned_state


def rx_keepalive() -> bytes:
    return rx_message(MSG_KEEPALIVE, 0, 0, 0)

logger = logging.getLogger(__name__)

MAX_TEXT = 1000


@dataclass
class Services:
    """What the API needs to page and to manage voices/sounds. Every field is
    optional so the dashboard still runs with only some of it configured."""
    dispatcher: Any = None
    destinations: Any = None
    tts: Any = None
    sounds: Any = None
    settings: Any = None
    pager: Any = None          # P25Pager - renders/previews audio even without an FNE link
    console: Any = None        # DispatchConsole - the /dispatch screen
    rx_audio: Any = None       # RxAudio - calls heard, for listening on the console


def create_app(on_status: Optional[Callable[[Dict[str, Any]], int]] = None,
               push_token: str = '', services: Optional[Services] = None,
               api_token: str = '') -> Flask:
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024    # a minute of 96 kHz mic audio
    svc = services or Services()

    def require_token(fn):
        """Gate actions that key a radio or change settings, if web.api_token is set."""
        @wraps(fn)
        def wrapper(*a, **kw):
            if api_token and not hmac.compare_digest(request.headers.get('X-Api-Token', ''), api_token):
                return jsonify({'error': 'invalid or missing X-Api-Token'}), 401
            return fn(*a, **kw)
        return wrapper

    def body() -> Dict[str, Any]:
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else {}

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

    @app.route('/api/state')
    def api_state():
        state = snapshot()
        state['api_token_required'] = bool(api_token)
        return jsonify(state)

    @app.route('/api/forewarned/push', methods=['POST'])
    def forewarned_push():
        """Forewarned pushes its full status payload (same shape as its own
        GET /api/status) here whenever its state changes. This goes through
        the same change detector as polling, so a change seen by both the
        push and the next poll only pages once."""
        if push_token:
            supplied = request.headers.get('X-Forewarned-Token', '')
            if not hmac.compare_digest(supplied, push_token):
                return jsonify({'error': 'invalid token'}), 401
        status = request.get_json(silent=True)
        if not isinstance(status, dict) or 'local_alert_state' not in status:
            return jsonify({'error': 'expected Forewarned status JSON with local_alert_state'}), 400
        update_forewarned_state(reachable=True, status=status)
        pages = on_status(status) if on_status else 0
        return jsonify({'success': True, 'pages_queued': pages})

    # -- destinations & manual paging -----------------------------------------

    @app.route('/api/destinations')
    def api_destinations():
        return jsonify([{k: d.get(k) for k in ('id', 'name', 'backend', 'talkgroup', 'extension')}
                        for d in (svc.destinations or [])])

    @app.route('/api/page', methods=['POST'])
    @require_token
    def api_page():
        """Page on command: {"destination": id, "message": "...", "sound": "hi_lo",
        "voice": {"engine", "voice", "rate"}}. Message and sound are each optional,
        but at least one is needed. Returns 202; the result shows in /api/state."""
        if svc.dispatcher is None:
            return jsonify({'error': 'paging not configured'}), 503
        b = body()
        message = str(b.get('message') or '')[:MAX_TEXT]
        try:
            svc.dispatcher.manual(str(b.get('destination') or ''), message,
                                  sound=b.get('sound') or None, voice=b.get('voice') or None,
                                  source=str(b.get('source') or 'Manual page')[:60])
        except (KeyError, ValueError) as e:
            return jsonify({'error': str(e).strip("'\"")}), 400
        return jsonify({'queued': True}), 202

    # -- text-to-speech -------------------------------------------------------

    @app.route('/api/tts')
    def api_tts():
        if svc.tts is None:
            return jsonify({'engines': [], 'current': {}})
        return jsonify({'engines': svc.tts.describe(), 'current': svc.settings.get('tts') if svc.settings else {}})

    @app.route('/api/tts', methods=['POST'])
    @require_token
    def api_tts_set():
        if svc.tts is None or svc.settings is None:
            return jsonify({'error': 'text-to-speech not configured'}), 503
        b = body()
        engine = str(b.get('engine') or '')
        eng = svc.tts.engines.get(engine)
        if eng is None or not eng.available():
            return jsonify({'error': f"engine '{engine}' is not available"}), 400
        voice = str(b.get('voice') or '')
        known = [v['id'] for v in eng.voices()]
        if voice and known and voice not in known:
            return jsonify({'error': f"voice '{voice}' not found for {eng.label}"}), 400
        try:
            rate = int(b.get('rate', 100))
        except (TypeError, ValueError):
            return jsonify({'error': 'rate must be a number'}), 400
        rate = max(50, min(200, rate))
        current = svc.settings.update('tts', {'engine': engine, 'voice': voice, 'rate': rate})
        logger.info(f"Default TTS voice set to {engine}/{voice or 'default'} at {rate}%")
        return jsonify({'current': current})

    def _wav(data: bytes) -> Response:
        return Response(data, mimetype='audio/wav', headers={'Cache-Control': 'no-store'})

    @app.route('/api/tts/preview', methods=['POST'])
    def api_tts_preview():
        """WAV of a message (and optional sound) as it would be sent. radio=true
        (default) runs it through the P25 vocoder so it sounds like a radio."""
        if svc.pager is None:
            return jsonify({'error': 'voice paging not configured'}), 503
        b = body()
        voice = {k: b.get(k) for k in ('engine', 'voice', 'rate') if b.get(k) not in (None, '')}
        try:
            data = svc.pager.preview(str(b.get('text') or '')[:MAX_TEXT], b.get('sound') or None,
                                     voice or None, as_radio=bool(b.get('radio', True)))
        except Exception as e:
            logger.warning(f"TTS preview failed: {e}")
            return jsonify({'error': str(e)}), 400
        return _wav(data)

    # -- alert sounds ---------------------------------------------------------

    @app.route('/api/sounds')
    def api_sounds():
        return jsonify(svc.sounds.list() if svc.sounds else [])

    @app.route('/api/sounds/<name>.wav')
    def api_sound_preview(name):
        if svc.pager is None or svc.sounds is None or not svc.sounds.exists(name):
            return jsonify({'error': 'unknown sound'}), 404
        radio = request.args.get('radio', '1') not in ('0', 'false', 'no')
        return _wav(svc.pager.preview('', name, None, as_radio=radio))

    # -- dispatch console -----------------------------------------------------

    @app.route('/dispatch')
    def dispatch_console():
        return render_template('dispatch.html')

    @app.route('/api/dispatch/state')
    def api_dispatch_state():
        if svc.console is None:
            return jsonify({'error': 'dispatch console not configured'}), 503
        state = svc.console.state()
        state['api_token_required'] = bool(api_token)
        state['listen_available'] = svc.rx_audio is not None
        return jsonify(state)

    @app.route('/api/dispatch/page', methods=['POST'])
    @require_token
    def api_dispatch_page():
        """{"talkgroups": [1001, 2002], "message": "...", "sound": "hi_lo"}"""
        if svc.console is None:
            return jsonify({'error': 'dispatch console not configured'}), 503
        b = body()
        try:
            n = svc.console.page(b.get('talkgroups'), str(b.get('message') or '')[:MAX_TEXT],
                                 b.get('sound') or None)
        except (KeyError, ValueError, RuntimeError) as e:
            return jsonify({'error': str(e).strip("'\"")}), 400
        return jsonify({'queued': n}), 202

    @app.route('/api/dispatch/talk', methods=['POST'])
    @require_token
    def api_dispatch_talk():
        """Recorded speech: body is raw little-endian 16-bit mono PCM;
        ?talkgroups=1001,2002&rate=48000[&sound=alert_tone]"""
        if svc.console is None:
            return jsonify({'error': 'dispatch console not configured'}), 503
        tgs = [t for t in request.args.get('talkgroups', '').split(',') if t.strip()]
        try:
            rate = int(request.args.get('rate', '0'))
            n = svc.console.talk(tgs, request.get_data(cache=False), rate, request.args.get('sound') or None)
        except (KeyError, ValueError, RuntimeError) as e:
            return jsonify({'error': str(e).strip("'\"")}), 400
        return jsonify({'queued': n}), 202

    # -- tile setup -------------------------------------------------------------

    @app.route('/api/dispatch/layout')
    def api_layout():
        if svc.console is None:
            return jsonify({'error': 'dispatch console not configured'}), 503
        return jsonify(svc.console.editor())

    @app.route('/api/dispatch/layout', methods=['POST'])
    @require_token
    def api_layout_save():
        """{"talkgroups": [{"id": 1001, "name": "Ops", "color": "blue"}, ...],
        "show_fne_talkgroups": true, "tones": ["alert_tone", ...]}"""
        if svc.console is None:
            return jsonify({'error': 'dispatch console not configured'}), 503
        try:
            return jsonify(svc.console.save_layout(body()))
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

    @app.route('/api/dispatch/layout/reset', methods=['POST'])
    @require_token
    def api_layout_reset():
        if svc.console is None:
            return jsonify({'error': 'dispatch console not configured'}), 503
        return jsonify(svc.console.reset_layout())

    # -- listen & live talk -----------------------------------------------------

    @app.route('/api/dispatch/listen')
    def api_dispatch_listen():
        """Audio of calls heard from the FNE, streamed as binary messages for as
        long as the client stays connected (format: src/p25/rx_audio.py)."""
        if svc.rx_audio is None:
            return jsonify({'error': 'no FNE connection to listen to'}), 503
        rx = svc.rx_audio

        def stream():
            listener, q = rx.listen()
            try:
                yield rx_keepalive()
                while True:
                    try:
                        yield q.get(timeout=2)
                    except queue.Empty:
                        yield rx_keepalive()     # also how a closed browser is noticed
            finally:
                rx.unlisten(listener)

        return Response(stream(), mimetype='application/octet-stream',
                        headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    def live_error(e):
        return jsonify({'error': str(e).strip("'\"")}), 409

    @app.route('/api/dispatch/live', methods=['POST'])
    @require_token
    def api_live_start():
        """{"talkgroups": [1001], "sound": optional lead tone} -> {"session": ...}"""
        if svc.console is None:
            return jsonify({'error': 'dispatch console not configured'}), 503
        b = body()
        try:
            session = svc.console.live_start(b.get('talkgroups'), b.get('sound') or None)
        except (KeyError, ValueError, LiveError) as e:
            return live_error(e)
        return jsonify(session.status())

    @app.route('/api/dispatch/live/<session_id>/audio', methods=['POST'])
    @require_token
    def api_live_audio(session_id):
        """Body: little-endian 16-bit mono PCM at 8 kHz."""
        try:
            session = svc.console.live.get(session_id)
        except (AttributeError, LiveError) as e:
            return live_error(e)
        session.feed(request.get_data(cache=False))
        return jsonify(session.status())

    @app.route('/api/dispatch/live/<session_id>/stop', methods=['POST'])
    @require_token
    def api_live_stop(session_id):
        try:
            session = svc.console.live.get(session_id)
        except (AttributeError, LiveError) as e:
            return live_error(e)
        session.stop()
        return jsonify(session.status())

    return app
