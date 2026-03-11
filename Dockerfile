FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
WORKDIR /app

# CPU-only PyTorch first — prevents pip pulling the full CUDA build
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

RUN mkdir -p data/docs data/uploads data/faiss_index \
    && chown -R appuser:appuser /app

USER appuser

ENV HF_HOME=/home/appuser/.cache/huggingface

# Cache embedding model into image layer
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Pre-build FAISS index — baked into image for zero cold-start latency
RUN python ingest.py

EXPOSE 7860

ENV GROQ_API_KEY=""
ENV HF_HOME=/home/appuser/.cache/huggingface

CMD ["python", "app.py"]