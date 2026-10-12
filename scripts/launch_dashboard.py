#!/usr/bin/env python3
"""
Dashboard launcher for FL Healthcare Benchmark.
Starts a local web server and opens the interactive comparison dashboard.

Usage:
    python scripts/launch_dashboard.py
    python scripts/launch_dashboard.py --port 8501 --no-browser
"""

import argparse
import http.server
import socket
import socketserver
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_free_port(start_port: int = 8501, max_tries: int = 50) -> int:
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    return start_port


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def log_message(self, format, *args):
        # Keep terminal clean, only log errors
        if args and str(args[1]).startswith("4"):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Launch Healthcare FL Comparison Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve on (default: 8501)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    port = find_free_port(args.port)
    dashboard_url = f"http://localhost:{port}/dashboard/index.html"

    print("=" * 70)
    print("  HEALTHCARE FL BENCHMARK — INTERACTIVE COMPARISON DASHBOARD")
    print("=" * 70)
    print(f"  Serving Repository: {REPO_ROOT}")
    print(f"  Dashboard URL:      {dashboard_url}")
    print("  Press Ctrl+C to stop the dashboard server.")
    print("=" * 70)

    if not args.no_browser:
        webbrowser.open(dashboard_url)

    with socketserver.TCPServer(("", port), QuietHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard server stopped.")


if __name__ == "__main__":
    main()
