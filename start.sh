#!/bin/bash
# -------------------------------------------------------------
# Launch Script for ReviewIQ Product Intelligence & Sentiment Hub
# -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================="
echo "🚀 Starting ReviewIQ Product Intelligence & Sentiment Hub..."
echo "📂 Project Directory: $SCRIPT_DIR"
echo "🌐 URL: http://localhost:8000/executive_dashboard.html"
echo "=============================================================="

# Cross-platform browser launcher
open_browser() {
  sleep 1
  if command -v open > /dev/null; then
    open "http://localhost:8000/executive_dashboard.html"
  elif command -v xdg-open > /dev/null; then
    xdg-open "http://localhost:8000/executive_dashboard.html"
  elif command -v explorer.exe > /dev/null; then
    explorer.exe "http://localhost:8000/executive_dashboard.html"
  fi
}

open_browser &

# Start python server (zero external web dependencies required)
python3 server.py
