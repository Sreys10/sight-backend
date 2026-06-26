#!/bin/bash
# Start script for Render/Railway deployment
# PORT is injected automatically by Render
PORT=${PORT:-10000}
echo "Starting server on port $PORT"
exec uvicorn main:app --host "0.0.0.0" --port "${PORT}" --workers 1
