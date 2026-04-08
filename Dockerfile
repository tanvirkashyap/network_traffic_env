# Use slim for faster deployment
FROM python:3.10-slim

# Install system dependencies if needed (pandas/sklearn sometimes need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements first (to cache layers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Ensure the app can find your modules
ENV PYTHONPATH=/app

# HF Spaces run on port 7860
EXPOSE 7860

# DOUBLE CHECK THIS: 
# If app.py is in root, use "app:app". 
# If app.py is in /server, use "server.app:app"
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]