FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for compiling some python packages if wheel is missing)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY main.py .

# Copy pre-trained model (so the container starts ready to predict)
COPY models/ ./models/

# Create data directory
RUN mkdir -p data

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run in HTTP server mode (sidecar for Go bot)
CMD ["python", "main.py", "--serve", "--port", "8000"]
