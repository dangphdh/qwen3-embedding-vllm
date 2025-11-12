# Qwen3 Embedding Service with vLLM

A high-performance embedding service for the **Qwen3-0.6B-Embedding-finetuned-sts-vi** model using vLLM and Docker.

## Features

- 🚀 Fast inference with vLLM optimization
- 🐳 Easy deployment with Docker and Docker Compose
- 🔌 OpenAI-compatible API endpoints
- �� Health check and monitoring endpoints
- ⚡ GPU acceleration support
- 🔄 Automatic restart on failure

## Model Information

- **Model**: [zzeennoo/Qwen3-0.6B-Embedding-finetuned-sts-vi](https://huggingface.co/zzeennoo/Qwen3-0.6B-Embedding-finetuned-sts-vi)
- **Type**: Embedding model fine-tuned for Vietnamese semantic text similarity
- **Base**: Qwen3-0.6B
- **Use Case**: Generate embeddings for Vietnamese text

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/dangphdh/qwen3-embedding-vllm.git
cd qwen3-embedding-vllm
```

### 2. Build and Run with Docker Compose

```bash
# Build and start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### 3. Alternative: Build and Run with Docker

```bash
# Build the image
docker build -t qwen3-embedding .

# Run the container
docker run -d \
  --name qwen3-embedding \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/cache:/root/.cache/huggingface \
  qwen3-embedding
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Generate Embeddings

#### Single Text

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Xin chào, đây là một ví dụ về văn bản tiếng Việt.",
    "model": "zzeennoo/Qwen3-0.6B-Embedding-finetuned-sts-vi"
  }'
```

#### Multiple Texts

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "Câu đầu tiên để tạo embedding.",
      "Câu thứ hai để tạo embedding."
    ]
  }'
```

### Python Client Example

```python
import requests

url = "http://localhost:8000/v1/embeddings"

# Single text
response = requests.post(
    url,
    json={
        "input": "Xin chào, đây là một ví dụ về văn bản tiếng Việt.",
        "model": "zzeennoo/Qwen3-0.6B-Embedding-finetuned-sts-vi"
    }
)

embeddings = response.json()
print(f"Embedding dimension: {len(embeddings['data'][0]['embedding'])}")
print(f"Embedding vector: {embeddings['data'][0]['embedding'][:10]}...")  # First 10 values

# Multiple texts
response = requests.post(
    url,
    json={
        "input": [
            "Câu đầu tiên.",
            "Câu thứ hai."
        ]
    }
)

embeddings = response.json()
for i, item in enumerate(embeddings['data']):
    print(f"Text {i+1} embedding dimension: {len(item['embedding'])}")
```

### Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

response = client.embeddings.create(
    input="Xin chào, đây là một ví dụ về văn bản tiếng Việt.",
    model="zzeennoo/Qwen3-0.6B-Embedding-finetuned-sts-vi"
)

print(response.data[0].embedding)
```

## Configuration

Environment variables can be configured in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `zzeennoo/Qwen3-0.6B-Embedding-finetuned-sts-vi` | HuggingFace model name |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `MAX_MODEL_LEN` | `8192` | Maximum sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory utilization ratio |

## API Endpoints

- `GET /` - Service information
- `GET /health` - Health check
- `POST /v1/embeddings` - Generate embeddings (OpenAI-compatible)
- `POST /embeddings` - Generate embeddings (simplified)

## Response Format

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, ...],
      "index": 0
    }
  ],
  "model": "zzeennoo/Qwen3-0.6B-Embedding-finetuned-sts-vi",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

## Troubleshooting

### GPU Not Detected

Ensure NVIDIA Container Toolkit is installed:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory

Reduce `GPU_MEMORY_UTILIZATION` in `docker-compose.yml`:

```yaml
environment:
  - GPU_MEMORY_UTILIZATION=0.7
```

### Port Already in Use

Change the port mapping in `docker-compose.yml`:

```yaml
ports:
  - "8001:8000"
```

## Development

### Local Development without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) for the high-performance inference engine
- [Qwen3 Embedding Model](https://huggingface.co/zzeennoo/Qwen3-0.6B-Embedding-finetuned-sts-vi) by zzeennoo
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue on GitHub.
