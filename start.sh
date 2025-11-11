#!/bin/bash
# Start script for Railway deployment
# Read PORT from environment variable, default to 5000 if not set
PORT=${PORT:-5000}
echo "Starting server on port $PORT"
exec gunicorn app:app --bind "0.0.0.0:${PORT}" --workers 2 --timeout 120

