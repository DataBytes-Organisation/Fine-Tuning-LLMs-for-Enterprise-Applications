#!/bin/bash

echo "Stopping any existing containers..."
docker stop llm-api-container 2>/dev/null || true
docker rm llm-api-container 2>/dev/null || true

echo "Building Docker image..."
docker build -t llm-api-image .

docker volume create model-weights

if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected - launching with GPU support"
    docker run --name llm-api-container \
        --gpus all \
        -d \
        -p 8000:8000 \
        -v $(pwd)/mistral-combined-finetuned-weights:/app/mistral-combined-finetuned-weights \
        --shm-size=2g \
        llm-api-image
else
    echo "Launching CPU version with limited resources"
    docker run --name llm-api-container \
        -d \
        -p 8000:8000 \
        -v $(pwd)/mistral-combined-finetuned-weights:/app/mistral-combined-finetuned-weights \
        --memory=6g \
        --cpus=2 \
        llm-api-image
fi

echo "Waiting for container..."
sleep 5
docker logs llm-api-container
