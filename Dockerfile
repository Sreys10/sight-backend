# Use Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY image_detector.py .

# Copy start script
COPY start.sh .
RUN chmod +x start.sh

# Expose port (default 5000, Railway will override via PORT env var)
EXPOSE 5000

# Use start script to read PORT from environment
CMD ["./start.sh"]

