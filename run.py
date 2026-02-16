#!/usr/bin/env python3
"""Entry point for the Epstein Files Analyzer.

Usage:
    python run.py              # Launch desktop GUI
    python run.py --web        # Launch web server only (open in browser)
    python run.py --headless   # For environments without a display
"""

import argparse
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Epstein Files Analyzer")
    parser.add_argument(
        "--web", action="store_true",
        help="Run as web server only (no desktop window)"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port for the web server"
    )
    args = parser.parse_args()

    from epstein_analyzer import database
    database.init_db()

    if args.port:
        from epstein_analyzer.config import WEB_PORT
        import epstein_analyzer.config as cfg
        cfg.WEB_PORT = args.port

    if args.web:
        import webbrowser
        from epstein_analyzer.config import WEB_HOST, WEB_PORT
        port = args.port or WEB_PORT
        url = f"http://{WEB_HOST}:{port}"
        print(f"\n  Epstein Files Analyzer")
        print(f"  Server: {url}")
        print(f"  Press Ctrl+C to stop\n")
        webbrowser.open(url)
        from epstein_analyzer.app import app
        import uvicorn
        uvicorn.run(app, host=WEB_HOST, port=port, log_level="info")
    else:
        from epstein_analyzer.desktop import launch
        launch()


if __name__ == "__main__":
    main()
