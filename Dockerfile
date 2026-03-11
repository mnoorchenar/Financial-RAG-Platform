# ── Base Image ─────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── System Dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root User (required by HuggingFace Spaces) ────────────────────────────
RUN useradd -m -u 1000 appuser
WORKDIR /app

# ── Install CPU-only PyTorch FIRST (before requirements.txt) ──────────────────
# Installing torch from the CPU wheel index prevents pip from pulling the
# full 2-3 GB CUDA build, which causes OOM / timeout on HuggingFace Spaces.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.2.2 \
        --index-url https://download.pytorch.org/whl/cpu

# ── Install Remaining Python Dependencies ─────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Application Source ────────────────────────────────────────────────────
# config.py is gitignored — bootstrapped from config.example.py below.
# GROQ_API_KEY is injected at runtime via HuggingFace Space Secret.
COPY --chown=appuser:appuser . .

# ── Prepare Data Directories ───────────────────────────────────────────────────
RUN mkdir -p data/docs data/uploads data/faiss_index \
    && chown -R appuser:appuser data/

# ── Switch to Non-root User ────────────────────────────────────────────────────
USER appuser

# ── Cache the Sentence-Transformer Model ──────────────────────────────────────
# Pre-downloading the model into the image layer means ingest.py and app.py
# never need an outbound network call to HuggingFace Hub at runtime.
# The model (~90 MB) is stored at the path HuggingFace Transformers expects.
ENV HF_HOME=/home/appuser/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ── Bootstrap config.py from the committed example template ───────────────────
# config.py is gitignored (holds the private Groq key for local dev).
# ingest.py only needs paths and embedding settings — not the Groq key —
# so the blank example template is perfectly sufficient for the build step.
RUN cp config.example.py config.py

# ── Pre-build FAISS Index (baked into image — zero cold-start latency) ─────────
# Downloads public PDFs from FDIC / NAIC / SEC.
# Falls back to the built-in synthetic corpus automatically if any download fails.
RUN python ingest.py

# ── Expose Port ────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Runtime Environment ────────────────────────────────────────────────────────
# GROQ_API_KEY must be set as a HuggingFace Space Secret.
# config.py reads it via os.environ.get("GROQ_API_KEY") with priority 1.
ENV GROQ_API_KEY=""
ENV HF_HOME=/home/appuser/.cache/huggingface

# ── Launch ─────────────────────────────────────────────────────────────────────
CMD ["python", "app.py"]