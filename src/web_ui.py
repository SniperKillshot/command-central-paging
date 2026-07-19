"""Flask web application: dashboard + REST API."""
import logging
from datetime import datetime

from flask import Flask, jsonify, render_template

from src.state import snapshot

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__, template_folder='../templates', static_folder='../static')

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

    @app.route('/api/state')
    def api_state():
        return jsonify(snapshot())

    return app
