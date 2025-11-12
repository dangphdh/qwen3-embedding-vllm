FROM vllm/vllm-openai:latest

# Set working directory
WORKDIR /app

# Copy application files
COPY app.py /app/
COPY requirements.txt /app/

# Install additional dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Set environment variables
ENV EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B-GGUF"
ENV RERANKER_MODEL=""
ENV HOST="0.0.0.0"
ENV PORT=8000
ENV DEVICE="cpu"

# Run the application
CMD ["python", "app.py"]
