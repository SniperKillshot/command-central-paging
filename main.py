"""Command Central Paging - entry point."""
import logging

from src.config import Config
from src.forewarned_client import ForewarnedClient
from src.web_ui import create_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Starting Command Central Paging")
    logger.info("=" * 60)

    config = Config()

    forewarned_client = ForewarnedClient(
        base_url=config.get('forewarned', 'base_url'),
        poll_interval=config.get('forewarned', 'poll_interval', default=5),
    )
    forewarned_client.start()

    app = create_app()
    app.run(
        host=config.get('web', 'host', default='0.0.0.0'),
        port=config.get('web', 'port', default=5000),
        debug=False,
    )


if __name__ == '__main__':
    main()
