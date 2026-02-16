"""Desktop GUI wrapper using pywebview."""

import logging
import threading
import time

from .config import WEB_HOST, WEB_PORT

logger = logging.getLogger(__name__)


def _start_server():
    """Start the FastAPI server in a background thread."""
    from .app import run_server
    run_server()


def launch():
    """Launch the desktop application."""
    import webview

    from . import database
    database.init_db()

    # Start FastAPI server in background thread
    server_thread = threading.Thread(target=_start_server, daemon=True)
    server_thread.start()

    # Give the server a moment to start
    time.sleep(1.5)

    url = f"http://{WEB_HOST}:{WEB_PORT}"
    logger.info(f"Opening desktop window at {url}")

    # Create and run the desktop window
    window = webview.create_window(
        title="Epstein Files Analyzer",
        url=url,
        width=1400,
        height=900,
        min_size=(800, 600),
        text_select=True,
    )
    webview.start(debug=False)
