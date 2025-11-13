# Use Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY image_detector.py .
COPY face_matcher.py .

# Copy database folder for face matching
COPY database/ ./database/

# Copy start script
COPY start.sh .
RUN chmod +x start.sh

# Expose port (default 5000, Railway will override via PORT env var)
EXPOSE 5000

# Use start script to read PORT from environment
CMD ["./start.sh"]

