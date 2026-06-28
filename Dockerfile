FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user with UID 1000 for Hugging Face compatibility
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and set ownership to user 1000
COPY --chown=user:user . .

USER user

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Start app using uvicorn (binding to the dynamic PORT environment variable with fallback to 7860)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
