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

# ── Bootstrap config.py from the committed example template ───────────────────
# config.py is gitignored (it holds the private Groq key for local dev).
# ingest.py only needs paths and embedding settings from config — not the key —
# so copying the blank example is sufficient for the build step.
# At runtime, GROQ_API_KEY is injected via the HuggingFace Space Secret and
# config.py reads it through os.environ.get("GROQ_API_KEY") with priority 1.
RUN cp config.example.py config.py

# ── Pre-build FAISS Index (baked into image — zero cold-start latency) ─────────
# Downloads public PDFs from FDIC/NAIC/SEC; falls back to synthetic corpus
# automatically if any download fails during the HuggingFace build.
RUN python ingest.py

# ── Expose Port ────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Environment ────────────────────────────────────────────────────────────────
# Set GROQ_API_KEY as a Space Secret in the HuggingFace UI.
# It will be injected as this environment variable at runtime.
ENV GROQ_API_KEY=""

# ── Launch ─────────────────────────────────────────────────────────────────────
CMD ["python", "app.py"]