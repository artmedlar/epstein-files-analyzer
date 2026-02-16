#!/usr/bin/env python3
"""Entry point for the Epstein Files Analyzer.

Usage:
    python run.py              # Launch web server, open browser
    python run.py --port 9000  # Use a custom port
"""

import argparse
import logging
import sys
import os
import webbrowser

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
        "--port", type=int, default=None,
        help="Port for the web server (default: 8742)"
    )
    args = parser.parse_args()

    from epstein_analyzer import database
    database.init_db()

    if args.port:
        import epstein_analyzer.config as cfg
        cfg.WEB_PORT = args.port

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


if __name__ == "__main__":
    main()
