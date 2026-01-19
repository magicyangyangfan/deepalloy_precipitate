FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for scientific packages
# - gcc: C compiler for building Python extensions
# - gfortran: Fortran compiler for scipy/numpy
# - libopenblas-dev: Linear algebra library
RUN apt-get update && apt-get install -y \
    gcc \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install kawin package with API dependencies
RUN pip install -e ".[api]"

# Set PYTHONPATH to include app directory
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose port
EXPOSE 8000

# Run the application
# Use gunicorn with uvicorn workers for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
