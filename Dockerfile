# Start from a Python 3.11 base image
FROM python:3.11-slim

# Install system dependency needed by implicit (OpenMP runtime)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (Docker caches this layer — if requirements don't change, it skips reinstalling)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY src/ src/
COPY models/ models/
COPY data/processed/ data/processed/

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
