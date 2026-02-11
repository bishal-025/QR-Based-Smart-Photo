FROM python:3.11-slim

WORKDIR /app

# System deps (for opencv, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Make sure models directory exists (model file should already be there if trained)
RUN mkdir -p models data/raw

EXPOSE 8000 8501

# Default command is backend; docker-compose will override for frontend
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


