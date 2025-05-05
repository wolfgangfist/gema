#!/usr/bin/env python3
"""
Simple HTTP server to serve the CSM interface and audio test pages.
This makes it easy to access the HTML files via HTTP rather than file:// protocol.
"""

import http.server
import socketserver
import os
import webbrowser
import argparse
from urllib.parse import urlparse

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(SCRIPT_DIR, "static")

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler that serves from the static directory"""
    
    def translate_path(self, path):
        """Translate URL path to file system path"""
        parsed_path = urlparse(path).path
        if parsed_path == "/":
            # Default to serving the CSM interface
            return os.path.join(STATIC_DIR, "csm-interface.html")
        elif parsed_path == "/audio-test":
            # Special path for the audio test page
            return os.path.join(STATIC_DIR, "audio-test.html")
        else:
            # Handle other files
            return os.path.join(STATIC_DIR, parsed_path.lstrip("/"))
    
    def log_message(self, format, *args):
        """Override to provide cleaner logging"""
        print(f"[HTTP] {self.address_string()} - {format % args}")

def serve_static(host="localhost", port=2999):
    """Start a simple HTTP server to serve static files"""
    
    handler = CustomHTTPRequestHandler
    
    with socketserver.TCPServer((host, port), handler) as httpd:
        print(f"Serving static files from: {STATIC_DIR}")
        print(f"Server started at http://{host}:{port}")
        print(f"Audio test page: http://{host}:{port}/audio-test")
        print("\nPress Ctrl+C to stop the server")
        
        # Open the browser
        webbrowser.open(f"http://{host}:{port}")
        
        # Serve until interrupted
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve CSM interface files")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=2999, help="Port to listen on")
    args = parser.parse_args()
    
    serve_static(args.host, args.port)