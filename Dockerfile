# Use Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
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

# Pre-download and extract InsightFace buffalo_s model during Docker build to optimize container startup time
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip \
    && mkdir -p /app/.insightface/models \
    && curl -L https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip -o /app/.insightface/models/buffalo_s.zip \
    && unzip -q /app/.insightface/models/buffalo_s.zip -d /app/.insightface/models/buffalo_s \
    && rm /app/.insightface/models/buffalo_s.zip \
    && apt-get purge -y --auto-remove curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .
RUN chmod +x start.sh

# Expose port (default 5000, Railway will override via PORT env var)
EXPOSE 5000

# Use start script to read PORT from environment
CMD ["./start.sh"]

