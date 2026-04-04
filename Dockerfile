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

# Copy source code and pre-trained models
COPY src/ ./src/
COPY main.py .
COPY .env .

# We don't copy the models or data directory directly because we want 
# the container to maintain its own state or use volumes, but we need the dirs
RUN mkdir -p models data

# The default command runs the live monitor
CMD ["python", "main.py"]
