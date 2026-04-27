#!/bin/bash
# Start script for Render/Railway deployment
# PORT is injected automatically by Render
PORT=${PORT:-10000}
echo "Starting server on port $PORT"
exec gunicorn app:app --bind "0.0.0.0:${PORT}" --workers 1 --timeout 120 --log-level info
