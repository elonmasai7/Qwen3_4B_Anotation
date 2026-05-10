#!/bin/bash

set -e

echo "Setting up Annotation Platform..."

cd "$(dirname "$0")/.."

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -e .

echo "Creating .env file..."
cat > .env << EOF
PLATFORM_ENVIRONMENT=development
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=annotation_db
DATABASE_USER=postgres
DATABASE_PASSWORD=postgres

REDIS_HOST=localhost
REDIS_PORT=6379

KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_CONSUMER_GROUP=annotation-platform

MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=annotation-data

MODEL_NAME=Qwen/Qwen3-4B
MODEL_MAX_TOKENS=8192
MODEL_TEMPERATURE=0.1
MODEL_TOP_P=0.95

VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_GPU_MEMORY_UTILIZATION=0.85

ANNOTATION_NUM_BRANCHES=3
ANNOTATION_CONFIDENCE_THRESHOLD=0.8
ANNOTATION_MAX_RETRIES=3

RETRIEVAL_TOP_K=5
RETRIEVAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RETRIEVAL_SIMILARITY_THRESHOLD=0.7

LOGGING_LEVEL=INFO
LOGGING_FORMAT=json

MONITORING_ENABLED=true
MONITORING_METRICS_PORT=9090
EOF

echo "Running tests..."
pytest tests/ -v --tb=short || true

echo "Setup complete!"
echo "Run 'source venv/bin/activate' to activate the environment"
echo "Run 'python -m src.main' to start the server"