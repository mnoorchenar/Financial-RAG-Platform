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

# ── Install Python Dependencies ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy Application Source ────────────────────────────────────────────────────
# config.py is excluded via .dockerignore; GROQ_API_KEY is injected at runtime
COPY --chown=appuser:appuser . .

# ── Prepare Data Directories ───────────────────────────────────────────────────
RUN mkdir -p data/docs data/uploads data/faiss_index \
    && chown -R appuser:appuser data/

# ── Switch to Non-root User ────────────────────────────────────────────────────
USER appuser

# ── Pre-build FAISS Index (baked into image — zero cold-start latency) ─────────
# Downloads public PDFs from FDIC/NAIC/SEC; falls back to synthetic corpus
# if network is unavailable during the HuggingFace build.
RUN python ingest.py

# ── Expose Port ────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Environment ────────────────────────────────────────────────────────────────
# Set GROQ_API_KEY as a Space Secret in the HuggingFace UI.
# It will be injected as this environment variable at runtime.
ENV GROQ_API_KEY=""

# ── Launch ─────────────────────────────────────────────────────────────────────
CMD ["python", "app.py"]
