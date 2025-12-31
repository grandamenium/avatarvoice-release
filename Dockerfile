# Dockerfile for AvatarVoice Railway Deployment
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Create output directory
RUN mkdir -p ./output

# Set Python path
ENV PYTHONPATH=/app/src

# Set default environment variables
ENV DATA_DIR=./data/crema_d
ENV DATABASE_PATH=./data/voice_database.sqlite
ENV OUTPUT_DIR=./output

# Expose default Gradio port (Railway will override with PORT env var)
EXPOSE 7861

# Run the app
CMD ["python", "-m", "src.avatarvoice_ui.app"]
