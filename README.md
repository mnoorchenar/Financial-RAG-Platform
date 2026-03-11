---
title: Financial Document Intelligence Platform using Retrieval-Augmented Generation RAG LangChain FAISS and LLaMA 3.3 70B
colorFrom: blue
colorTo: indigo
sdk: docker
---

<div align="center">

<h1>🏦 Financial Document Intelligence Platform</h1>
<h3>using Retrieval-Augmented Generation (RAG), LangChain, FAISS &amp; LLaMA 3.3 70B</h3>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=3000&pause=1000&color=3b82f6&center=true&vCenter=true&width=820&lines=RAG+Pipeline+%7C+LangChain+%7C+FAISS+%7C+LLaMA+3.3+70B;Banking+%26+Insurance+Document+Intelligence;Multi-turn+Memory+%7C+Confidence+Scoring+%7C+Guardrails;Deployed+on+HuggingFace+Spaces+via+Docker+SDK" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-06b6d4?style=for-the-badge&logo=chainlink&logoColor=white)](https://python.langchain.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-4f46e5?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-f59e0b?style=for-the-badge)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-3b82f6?style=for-the-badge)](https://faiss.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mnoorchenar/Financial-RAG-Platform)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

**🏦 Financial Document Intelligence Platform using Retrieval-Augmented Generation (RAG), LangChain, FAISS, and LLaMA 3.3 70B** — A production-ready RAG system targeting the banking and insurance domain, built to demonstrate real-world LLM engineering skills aligned with current MLOps, LLM Engineer, and AI/NLP Developer job market demands. Ingests real publicly available financial PDFs from FDIC, NAIC, and SEC; embeds them locally using `sentence-transformers/all-MiniLM-L6-v2` with no external embedding API; stores vectors in a FAISS index pre-baked at Docker build time for zero cold-start latency; and uses LangChain's `ConversationalRetrievalChain` wired to the Groq API (LLaMA 3.3 70B) for sub-second multi-turn answers with source citations, confidence scoring, and domain guardrails.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [HuggingFace Spaces Deployment](#-huggingface-spaces-deployment)
- [Dashboard Modules](#-dashboard-modules)
- [ML Models](#-ml-models)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr>
    <td>🔍 <b>Local Embeddings — No External API</b></td>
    <td>sentence-transformers/all-MiniLM-L6-v2 runs entirely on CPU inside the container. No per-token embedding cost, no data leaving the environment — a critical requirement in financial and compliance-sensitive contexts.</td>
  </tr>
  <tr>
    <td>⚡ <b>Zero Cold-Start FAISS Index</b></td>
    <td>ingest.py executes during the Docker build step, downloading and indexing all source documents. The resulting index.faiss file is baked into the image layer so the very first query hits a warm, pre-built index with sub-millisecond retrieval latency.</td>
  </tr>
  <tr>
    <td>💬 <b>Multi-turn Conversational Memory</b></td>
    <td>LangChain ConversationBufferWindowMemory with per-session UUID isolation. Each browser tab maintains its own independent conversation history (k=6 turns). A condense prompt rewrites follow-up questions into standalone queries before retrieval, enabling coherent multi-turn conversations about regulatory documents.</td>
  </tr>
  <tr>
    <td>📊 <b>Retrieval Confidence Scoring</b></td>
    <td>A parallel FAISS similarity_search_with_score call converts L2 distances to cosine similarities using cos = 1 − d²/2, then applies exponential-decay weights (exp(−0.6·rank)) to give the top-ranked chunk disproportionate influence. Result is displayed as a colour-coded 🟢 HIGH / 🟡 MEDIUM / 🔴 LOW badge per answer.</td>
  </tr>
  <tr>
    <td>🛡️ <b>Domain Guardrails</b></td>
    <td>Two-stage filter runs before every retrieval call. Stage 1 fast-passes any query containing 1 or more of 70+ finance domain keywords in O(n) time. Stage 2 rejects queries matching 8 off-topic regex patterns (cooking, entertainment, sports, etc.) with a user-friendly explanatory message. Borderline follow-up queries pass through to the chain.</td>
  </tr>
  <tr>
    <td>📤 <b>Runtime PDF Upload</b></td>
    <td>Users upload proprietary financial PDFs through the UI. Documents are immediately chunked, embedded, and merged into the live singleton FAISS index via vectorstore.merge_from() — extending the knowledge base without restarting the application or invalidating existing memory sessions.</td>
  </tr>
  <tr>
    <td>📚 <b>Source Citations with Page References</b></td>
    <td>Every answer includes a deduplicated citation list rendered as markdown, showing the source filename and page number (where available) for each retrieved chunk that contributed to the answer.</td>
  </tr>
  <tr>
    <td>🧪 <b>RAGAS-Inspired Local Evaluation</b></td>
    <td>evaluation.py provides heuristic quality metrics — answer relevance (keyword overlap), context precision (chunk relevance fraction), faithfulness (speculation detection), and a composite score — all computed locally without calling the LLM or an external scoring API.</td>
  </tr>
  <tr>
    <td>🔒 <b>Secure by Design</b></td>
    <td>API key stored in gitignored config.py locally and as a HuggingFace Space Secret in production. Non-root Docker user (appuser, UID 1000). .dockerignore strips config.py from every image layer. Contributors use the committed config.example.py as their template. The key never travels to GitHub under any workflow.</td>
  </tr>
  <tr>
    <td>🐳 <b>Containerised Deployment — Docker SDK</b></td>
    <td>Single-stage Dockerfile using python:3.10-slim. Deployed to HuggingFace Spaces via the Docker SDK (not the Inference API), bypassing all HuggingFace model restrictions and allowing a fully self-contained open-source LLM stack. Exposes port 7860.</td>
  </tr>
</table>

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│              Financial Document Intelligence Platform — RAG Pipeline     │
│                                                                          │
│  ┌──────────────┐   ┌───────────────────┐   ┌────────────────────────┐  │
│  │  PDF Sources │──▶│    ingest.py      │──▶│   FAISS Index          │  │
│  │  FDIC · NAIC │   │  Download · Chunk │   │   (baked at Docker     │  │
│  │  SEC · User  │   │  Embed · Persist  │   │    build time)         │  │
│  └──────────────┘   └───────────────────┘   └──────────┬─────────────┘  │
│                                                         │                │
│  ┌──────────────┐   ┌───────────────────┐   ┌──────────▼─────────────┐  │
│  │ guardrails   │──▶│   rag_pipeline    │◀──│  MMR Retriever         │  │
│  │ 70+ keywords │   │   LangChain       │   │  top-k=5 · λ=0.7       │  │
│  │ 8 regex pats │   │   Conv.Retrieval  │   └────────────────────────┘  │
│  └──────────────┘   │   Chain           │                               │
│                     └─────────┬─────────┘                               │
│  ┌──────────────┐             │            ┌────────────────────────┐   │
│  │ memory_      │◀────────────┤            │  confidence.py         │   │
│  │ manager      │             │            │  L2 → cosine · exp     │   │
│  │ UUID · k=6   │             ▼            │  decay weighted mean   │   │
│  └──────────────┘   ┌───────────────────┐ └────────────────────────┘   │
│                     │  Groq API         │                               │
│                     │  LLaMA 3.3 70B    │                               │
│                     │  Versatile        │                               │
│                     └───────────────────┘                               │
└──────────────────────────────────────────────────────────────────────────┘
```

| Component | Technology | Role |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local CPU embeddings, 384-D normalised vectors, no external API |
| Vector Store | `faiss-cpu` IndexFlatL2 | Millisecond ANN search, pre-baked into Docker image at build time |
| LLM | Groq · LLaMA 3.3 70B Versatile | Sub-second generation via Groq API, temperature=0.1 |
| RAG Orchestration | LangChain 0.3 `ConversationalRetrievalChain` | Multi-turn condense prompt + grounded QA prompt |
| Memory | `ConversationBufferWindowMemory` (k=6) | Thread-safe per-session UUID registry |
| Guardrails | Keyword scan + off-topic regex | Domain relevance enforcement before every retrieval call |
| Confidence | Cosine similarity + exponential decay weights | Hallucination risk flagging per answer |
| Evaluation | RAGAS-inspired heuristics | Local quality metrics — relevance, precision, faithfulness |
| UI | Gradio 4.x Blocks + custom dark CSS | Dark financial theme, 3 tabs, per-tab session state |
| Deployment | Single-stage `Dockerfile` → HuggingFace Spaces Docker SDK | Port 7860, non-root `appuser`, GROQ_API_KEY injected at runtime |

### Knowledge Sources

| Source | Domain | Content |
|---|---|---|
| FDIC — Your Insured Deposits | Banking | Coverage limits, ownership categories, joint/retirement/trust accounts |
| NAIC — Life Insurance Buyer's Guide | Insurance | Term vs. whole vs. universal life, premiums, death benefit, cash value |
| SEC — Beginner's Guide to Financial Statements | Disclosure | Balance sheet, income statement, cash flow, EPS, MD&A, 10-K/10-Q/8-K |
| FDIC — Consumer News | Banking | Consumer protection, fraud awareness, deposit products |
| Synthetic Corpus (4 built-in .txt files) | Banking · Insurance · Risk | Fallback corpus covering banking fundamentals, insurance guide, financial statements, Basel III / AML / KYC |
| User Uploads (runtime) | Any financial domain | PDFs uploaded through the UI, merged into the live FAISS index instantly |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- A free [Groq API key](https://console.groq.com/) (free tier is sufficient)
- Git

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/financial-rag-intelligence-platform.git
cd financial-rag-intelligence-platform

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your private config
cp config.example.py config.py
# Open config.py and set: _GROQ_API_KEY_PRIVATE = "gsk_your_key_here"

# 5. Build the FAISS index
# This downloads public PDFs from FDIC / NAIC / SEC and writes the index
# Falls back to the built-in synthetic corpus if any download fails
# Takes approximately 1–3 minutes on first run
python ingest.py

# 6. Launch the application
python app.py
```

Open your browser at `http://localhost:7860` 🎉

> **Note:** The `data/` directory and all subdirectories (`docs/`, `uploads/`, `faiss_index/`) are created automatically by `ingest.py`. You do not need to create them manually.

---

## 🐳 Docker Deployment

```bash
# Build the image
# ingest.py runs during the build step — the FAISS index is baked into the image
docker build -t financial-rag-platform .

# Run with your Groq API key injected as an environment variable
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_your_key_here financial-rag-platform

# Or set the key in a .env file and use Docker Compose
echo "GROQ_API_KEY=gsk_your_key_here" > .env
docker compose up --build
```

Open your browser at `http://localhost:7860` 🎉

---

## 🤗 HuggingFace Spaces Deployment

This project uses the **Docker SDK** on HuggingFace Spaces, not the Inference API. This bypasses all HuggingFace model restrictions and allows the fully self-contained Groq + LLaMA 3.3 70B stack to run without modification.

```bash
# 1. Create a new Space at huggingface.co/new-space
#    → Select Docker as the SDK
#    → Set visibility to Public or Private

# 2. Add the HuggingFace Space as a git remote
git remote add space https://huggingface.co/spaces/mnoorchenar/Financial-RAG-Platform

# 3. Push (do NOT push config.py — it is gitignored)
git push space main
```

**4.** In your Space → **Settings → Variables and Secrets**, add:

| Key | Value |
|---|---|
| `GROQ_API_KEY` | `gsk_your_groq_api_key_here` |

The Space will automatically trigger a Docker build. During the build, `ingest.py` runs and the FAISS index is baked into the image. The app launches on port 7860 with zero cold-start latency on first query.

---

## 📊 Dashboard Modules

| Module | Description | Status |
|--------|-------------|--------|
| 💬 Multi-turn Chat | ConversationalRetrievalChain with per-session memory, condense + QA prompts, and source citations | ✅ Live |
| 📊 Confidence Panel | Per-answer retrieval confidence badge (🟢 HIGH / 🟡 MEDIUM / 🔴 LOW) with score and explanation | ✅ Live |
| 📚 Source Citations | Deduplicated markdown citation list with source filename and page reference per answer | ✅ Live |
| 🛡️ Domain Guardrails | Two-stage keyword + regex filter with user-friendly rejection messages for off-topic queries | ✅ Live |
| 📤 Runtime PDF Upload | User PDF ingestion merged into the live FAISS index via vectorstore.merge_from() — no restart | ✅ Live |
| 🏗️ Architecture Tab | Interactive system diagram and full component reference table rendered inside the UI | ✅ Live |

---

## 🧠 ML Models

```python
# Core Models and Techniques Used in the Financial Document Intelligence Platform
models = {
    "embeddings":       "sentence-transformers/all-MiniLM-L6-v2  (local CPU, 384-D, L2-normalised)",
    "llm":              "llama-3.3-70b-versatile  (via Groq API, temperature=0.1, max_tokens=1024)",
    "vector_index":     "FAISS IndexFlatL2  (CPU, pre-built at Docker build time, ~0 ms cold-start)",
    "chunking":         "LangChain RecursiveCharacterTextSplitter  (chunk_size=800, overlap=150)",
    "retrieval":        "MMR — Maximal Marginal Relevance  (lambda_mult=0.7, top_k=5, fetch_k=15)",
    "confidence":       "Exponential-decay weighted cosine similarity  (L2 → cosine: 1 − d²/2)",
    "evaluation":       "RAGAS-inspired heuristics  (answer relevance, context precision, faithfulness)",
}
```

---

## 📁 Project Structure

```
financial-rag-intelligence-platform/
│
├── 📄 app.py                    # Gradio UI + application entry point (3 tabs, dark theme)
├── 📄 config.py                 # ⚠️  GITIGNORED — private config, Groq API key lives here
├── 📄 config.example.py         # Committed template for contributors — copy → config.py
├── 📄 ingest.py                 # PDF download · chunking · embedding · FAISS index builder
├── 📄 rag_pipeline.py           # ConversationalRetrievalChain · MMR retriever · source formatter
├── 📄 memory_manager.py         # Thread-safe UUID-keyed ConversationBufferWindowMemory registry
├── 📄 guardrails.py             # Two-stage domain filter — keyword scan + off-topic regex
├── 📄 confidence.py             # FAISS L2 → cosine · exp-decay weighted score → HIGH/MEDIUM/LOW
├── 📄 evaluation.py             # RAGAS-inspired local quality metrics — no external API needed
│
├── 📄 requirements.txt          # Python dependencies with version bounds
├── 📄 Dockerfile                # python:3.10-slim · runs ingest.py at build · non-root appuser
├── 📄 .gitignore                # Excludes config.py · .env · data/faiss_index/ · data/docs/*.pdf
├── 📄 .dockerignore             # Strips config.py and secrets from every Docker image layer
├── 📄 README.md                 # This file
│
└── 📂 data/                     # Created automatically by ingest.py — do not create manually
    ├── 📂 docs/                 # Downloaded PDFs + built-in synthetic .txt corpus
    │   ├── FDIC_Your_Insured_Deposits.pdf
    │   ├── SEC_Beginners_Guide_To_Financial_Statements.pdf
    │   ├── NAIC_Life_Insurance_Buyers_Guide.pdf
    │   ├── FDIC_Consumer_News.pdf
    │   ├── banking_fundamentals.txt
    │   ├── insurance_guide.txt
    │   ├── financial_statements_guide.txt
    │   └── risk_management_banking.txt
    │
    ├── 📂 uploads/              # Runtime user-uploaded PDFs (merged into live FAISS index)
    │
    └── 📂 faiss_index/          # ⚠️  GITIGNORED — rebuilt by ingest.py at every build
        ├── index.faiss
        └── index.pkl
```

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mnoorchenar/Financial-RAG-Platform)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

> **Important:** Never commit `config.py` or any file containing API keys. Always use `config.example.py` as your template.

---

## Disclaimer

<span style="color:red">This project is developed strictly for educational and research purposes and does not constitute professional financial, legal, investment, or regulatory advice of any kind. All datasets used are either synthetically generated or publicly available from official government sources (FDIC, SEC, NAIC) — no real user data is stored or processed at any point. This software is provided "as is" without warranty of any kind; use at your own risk.</span>

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:4f46e5&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![GitHub Stars](https://img.shields.io/github/stars/mnoorchenar/financial-rag-intelligence-platform?style=social)](https://github.com/mnoorchenar/financial-rag-intelligence-platform)
[![GitHub Forks](https://img.shields.io/github/forks/mnoorchenar/financial-rag-intelligence-platform?style=social)](https://github.com/mnoorchenar/financial-rag-intelligence-platform/fork)

<sub>This project is developed for academic and research purposes only. Any similarity to existing company names, products, or trademarks is entirely coincidental and unintentional. This project has no affiliation with any commercial entity including FDIC, NAIC, SEC, Groq, Meta, or HuggingFace.</sub>

</div>