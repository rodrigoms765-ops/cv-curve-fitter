FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    JAX_PLATFORMS=cpu

# Create a non-root user with UID 1000 (required for Hugging Face Spaces Docker SDK)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY --chown=user:user . /app

# Switch to non-root user
USER user

# Expose Hugging Face Space default port
EXPOSE 7860

# Run FastAPI / Gradio application
CMD ["python", "app.py"]
