FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# System packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY app/requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copy source code and model
COPY app ./app
COPY mistral-combined-finetuned-weights ./mistral-combined-finetuned-weights

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

