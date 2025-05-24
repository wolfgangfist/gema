# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set environment variables
ENV VIRTUAL_ENV=/opt/venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NO_TORCH_COMPILE=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git ffmpeg libportaudio2 portaudio19-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory and clone the repo
WORKDIR /app

# Cache-buster: change this to force rebuild
ARG BUILD_VERSION
RUN echo "Build version: $BUILD_VERSION"

# Clone repo
RUN git clone https://github.com/nobandegani/csm-streaming-api.git

# Set working directory to the cloned repo
WORKDIR /app/csm-streaming-api

# Create and activate virtual environment, install dependencies
RUN python3 -m venv $VIRTUAL_ENV && \
    . $VIRTUAL_ENV/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Use the virtual environment path for PATH
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Optional speedup
#RUN pip install flash-attn

ARG HUGGINGFACE_TOKEN
RUN huggingface-cli login --token $HUGGINGFACE_TOKEN


# start
CMD bash -c "python setup.py && python main.py && uvicorn run_api:app --host 0.0.0.0 --port $API_INTERNAL_PORT"