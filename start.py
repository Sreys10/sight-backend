#!/usr/bin/env python3
"""
Alternative start script using Python
This ensures PORT environment variable is properly read
"""
import os
import subprocess
import sys

# Get PORT from environment, default to 5000
port = os.getenv('PORT', '5000')

# Build gunicorn command
cmd = [
    'gunicorn',
    'app:app',
    '--bind', f'0.0.0.0:{port}',
    '--workers', '2',
    '--timeout', '120'
]

# Execute gunicorn
sys.exit(subprocess.call(cmd))





