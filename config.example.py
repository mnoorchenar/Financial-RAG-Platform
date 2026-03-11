"""
config.example.py — Template configuration for FinRAG.

Copy this file to config.py and fill in your credentials.
config.py is excluded from version control via .gitignore.
"""

import os
import pathlib
from dotenv import load_dotenv

load_dotenv()

_GROQ_API_KEY_PRIVATE: str = ""  # <- paste your Groq key here for local dev
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY") or _GROQ_API_KEY_PRIVATE
GROQ_MODEL: str = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS: int = 1024
GROQ_TEMPERATURE: float = 0.1

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE: str = "cpu"

BASE_DIR: str = str(pathlib.Path(__file__).parent.resolve())
FAISS_INDEX_PATH: str = os.path.join(BASE_DIR, "data", "faiss_index")
DOCUMENTS_PATH: str = os.path.join(BASE_DIR, "data", "docs")
UPLOADED_DOCS_PATH: str = os.path.join(BASE_DIR, "data", "uploads")

CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 150
MIN_CHUNK_LENGTH: int = 50
TOP_K_DOCS: int = 5
MEMORY_WINDOW_K: int = 6
CONFIDENCE_HIGH: float = 0.58
CONFIDENCE_MEDIUM: float = 0.38

FINANCE_KEYWORDS: list = [
    "bank", "banking", "insurance", "policy", "premium", "claim", "deductible",
    "coverage", "loan", "mortgage", "credit", "interest", "investment", "portfolio",
    "asset", "liability", "equity", "dividend", "bond", "securities", "regulatory",
    "compliance", "underwriting", "reinsurance", "actuarial", "annuity", "beneficiary",
    "pension", "retirement", "fund", "audit", "balance sheet", "income statement",
    "financial", "risk", "disclosure", "sec", "fdic", "naic", "fiduciary", "deposit",
    "account", "transaction", "payment", "fraud", "money", "capital", "revenue",
    "profit", "loss", "tax", "cash", "market", "stock", "share", "indemnity",
    "surety", "guarantee", "collateral", "derivative", "hedge", "swap", "yield",
    "rate", "inflation", "liquidity", "solvency", "leverage", "default", "basel",
    "aml", "kyc", "fintech", "lending", "underwriter", "actuary", "reserve",
]
GUARDRAIL_MIN_KEYWORD_MATCH: int = 1

SOURCE_DOCUMENTS: list = [
    {"name": "FDIC_Your_Insured_Deposits", "url": "https://www.fdic.gov/resources/deposit-insurance/brochures/documents/your-insured-deposits-english.pdf", "domain": "banking"},
    {"name": "SEC_Beginners_Guide_To_Financial_Statements", "url": "https://www.sec.gov/investor/pubs/begfinstmtguide.pdf", "domain": "disclosure"},
    {"name": "NAIC_Life_Insurance_Buyers_Guide", "url": "https://content.naic.org/sites/default/files/publication-lbg-01-life-insurance-buyers-guide.pdf", "domain": "insurance"},
    {"name": "FDIC_Consumer_News", "url": "https://www.fdic.gov/consumers/consumer/news/cnfall08.pdf", "domain": "banking"},
]
