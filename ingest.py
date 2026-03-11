"""
ingest.py — Document ingestion pipeline.
Downloads public financial PDFs, falls back to a synthetic corpus,
chunks, embeds with MiniLM-L6-v2, and persists a FAISS index.

Run manually:  python ingest.py
Called automatically by app.py on first startup if the index is missing.
Also exports load_file() and chunk_documents() for use by pipeline.py.
"""

import time
import logging
from pathlib import Path
from typing import Optional

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)-8s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).parent.resolve()
FAISS_INDEX_PATH   = str(BASE_DIR / "data" / "faiss_index")
DOCUMENTS_PATH     = str(BASE_DIR / "data" / "docs")
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE   = "cpu"
CHUNK_SIZE         = 800
CHUNK_OVERLAP      = 150
MIN_CHUNK_LENGTH   = 50

SOURCE_DOCUMENTS = [
    {"name": "FDIC_Your_Insured_Deposits",             "url": "https://www.fdic.gov/resources/deposit-insurance/brochures/documents/your-insured-deposits-english.pdf"},
    {"name": "SEC_Beginners_Guide_To_Financial_Statements", "url": "https://www.sec.gov/investor/pubs/begfinstmtguide.pdf"},
    {"name": "NAIC_Life_Insurance_Buyers_Guide",        "url": "https://content.naic.org/sites/default/files/publication-lbg-01-life-insurance-buyers-guide.pdf"},
    {"name": "FDIC_Consumer_News",                      "url": "https://www.fdic.gov/consumers/consumer/news/cnfall08.pdf"},
]

_SYNTHETIC = {
    "banking_fundamentals.txt": """BANKING FUNDAMENTALS — CONSUMER REFERENCE GUIDE

FDIC Deposit Insurance
The Federal Deposit Insurance Corporation (FDIC) is an independent agency of the United States government that protects depositors against the loss of their insured deposits if an FDIC-insured bank fails. The standard deposit insurance amount is $250,000 per depositor, per insured bank, for each account ownership category. Ownership categories include single accounts, joint accounts, retirement accounts, and revocable trust accounts.

Types of Deposit Accounts
A checking account is a demand deposit account that allows unlimited withdrawals and deposits. A savings account is an interest-bearing deposit account. A certificate of deposit (CD) is a time deposit with a fixed maturity date and fixed interest rate. The annual percentage rate (APR) expresses the total yearly cost of borrowing as a percentage of the loan amount. Credit scores range from 300 to 850 and are calculated from payment history (35%), amounts owed (30%), length of credit history (15%), credit mix (10%), and new credit inquiries (10%).

Mortgage Lending
A mortgage is a secured loan used to purchase real property where the property itself serves as collateral. Fixed-rate mortgages carry the same interest rate for the life of the loan. Adjustable-rate mortgages (ARMs) have rates that reset periodically based on a reference index such as SOFR plus a margin. The loan-to-value (LTV) ratio is the loan balance divided by the appraised property value; most lenders require private mortgage insurance (PMI) when LTV exceeds 80 percent.""",

    "insurance_guide.txt": """INSURANCE CONSUMER GUIDE — NAIC STANDARDS AND DEFINITIONS

Life Insurance
Life insurance is a contract between a policyholder and an insurer in which the insurer pays a death benefit to beneficiaries upon the death of the insured. Term life insurance provides coverage for a specific period (10, 20, or 30 years) with no cash value accumulation. Whole life insurance provides lifetime coverage with a guaranteed cash value component that grows at a fixed rate. Universal life insurance offers permanent coverage with flexible premiums and an adjustable death benefit.

Health Insurance Key Terms
A premium is the monthly amount paid to maintain coverage regardless of whether services are used. A deductible is the out-of-pocket amount the insured must pay before the insurer begins paying for covered services. A copayment is a fixed dollar amount paid at the time of service. Coinsurance is the percentage of costs shared between insurer and insured after the deductible is met. The out-of-pocket maximum is the annual cap on cost-sharing; once reached, the insurer covers 100% of covered in-network services for the rest of the plan year.

Underwriting and Actuarial Science
Underwriting is the process by which an insurer evaluates applicants to determine eligibility and appropriate premium levels. Actuaries use statistical analysis, mortality tables, and loss experience data to price insurance products and establish loss reserves. Reinsurance is the practice of insurers transferring portions of their risk to other insurers to reduce exposure to large losses. The NAIC develops model laws and coordinates regulatory efforts across all U.S. states.""",

    "financial_statements_guide.txt": """UNDERSTANDING FINANCIAL STATEMENTS — SEC INVESTOR EDUCATION

The Balance Sheet
A balance sheet presents a company's assets, liabilities, and shareholders' equity at a specific point in time. The accounting equation states: Assets = Liabilities + Shareholders' Equity. Current assets include cash, accounts receivable, and inventory. Non-current assets include property, plant and equipment (PP&E) and intangibles. Shareholders' equity consists of common stock, retained earnings, and accumulated other comprehensive income (AOCI).

The Income Statement
The income statement shows revenues, expenses, and net income over a reporting period. Gross profit equals revenue minus cost of goods sold (COGS). Operating income equals gross profit minus operating expenses. Net income equals operating income minus interest expense and income taxes. Earnings per share (EPS) is net income divided by weighted average diluted shares outstanding.

SEC Reporting Requirements
The SEC requires public companies to file Form 10-K (annual report) within 60-90 days of fiscal year end. Form 10-Q (quarterly report) must be filed within 40-45 days of each quarter end. Form 8-K discloses material events promptly. Management's Discussion and Analysis (MD&A) is a required narrative section explaining financial results, trends, and risks.""",

    "risk_management_banking.txt": """RISK MANAGEMENT IN BANKING — REGULATORY FRAMEWORK REFERENCE

Basel III Capital Standards
Basel III is the global regulatory framework developed by the Basel Committee on Banking Supervision (BCBS). Common Equity Tier 1 (CET1) capital must represent at least 4.5% of risk-weighted assets (RWA). Tier 1 capital must be at least 6% of RWA. Total capital must be at least 8% of RWA. Banks must also maintain a Capital Conservation Buffer of 2.5% CET1. The Liquidity Coverage Ratio (LCR) requires banks to hold sufficient high-quality liquid assets (HQLA) to cover net cash outflows over 30 days. The Net Stable Funding Ratio (NSFR) requires available stable funding to exceed required stable funding over a one-year horizon.

Categories of Banking Risk
Credit risk is the risk of loss from a borrower failing to meet contractual obligations. Market risk is the risk of losses from movements in interest rates, equity prices, FX rates, and commodity prices. Operational risk is the risk of loss from inadequate processes, systems, people, or external events. Liquidity risk is the risk that a bank cannot meet cash obligations without unacceptable losses.

Anti-Money Laundering and Know Your Customer
The Bank Secrecy Act (BSA) requires U.S. financial institutions to file Currency Transaction Reports (CTRs) for cash transactions exceeding $10,000 and Suspicious Activity Reports (SARs) for suspicious transactions of $5,000 or more. KYC programs require institutions to verify client identity and assess risk profiles. Customer Due Diligence (CDD) includes identifying beneficial owners and monitoring transactions. Enhanced Due Diligence (EDD) applies to higher-risk customers such as politically exposed persons (PEPs).""",
}


# ── Helpers (also imported by pipeline.py) ────────────────────────────────────

def load_file(file_path: str) -> list:
    """Load a PDF or .txt into LangChain Document objects."""
    path = Path(file_path)
    try:
        if path.suffix.lower() == ".pdf":
            docs = PyPDFLoader(str(path)).load()
            for d in docs:
                d.metadata.setdefault("source", path.name)
        else:
            docs = [Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name, "page": 0})]
        logger.info("Loaded %d page(s) from %s", len(docs), path.name)
        return docs
    except Exception as e:
        logger.error("Failed to load %s: %s", path.name, e)
        return []


def chunk_documents(documents: list) -> list:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_documents(documents)
    return [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LENGTH]


def get_embeddings() -> HuggingFaceEmbeddings:
    """Shared embedding model constructor used here and in pipeline.py."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": EMBEDDING_DEVICE}, encode_kwargs={"normalize_embeddings": True})


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_ingestion() -> None:
    """Full pipeline: download → synthetic fallback → chunk → embed → persist."""
    logger.info("=" * 60)
    logger.info("  Ingestion Pipeline — Starting")
    logger.info("=" * 60)

    for p in (DOCUMENTS_PATH, FAISS_INDEX_PATH):
        Path(p).mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for doc in SOURCE_DOCUMENTS:
        dest = Path(DOCUMENTS_PATH) / f"{doc['name']}.pdf"
        if dest.exists():
            downloaded += 1
            continue
        try:
            r = requests.get(doc["url"], timeout=45, headers={"User-Agent": "FinRAG/1.0 (Educational)"})
            r.raise_for_status()
            dest.write_bytes(r.content)
            logger.info("Downloaded %s (%.1f KB)", dest.name, len(r.content) / 1024)
            downloaded += 1
        except Exception as e:
            logger.warning("Download failed for '%s': %s", doc["name"], e)
        time.sleep(0.5)

    if downloaded == 0:
        logger.warning("No PDFs downloaded — using synthetic corpus only.")
    for name, content in _SYNTHETIC.items():
        p = Path(DOCUMENTS_PATH) / name
        if not p.exists():
            p.write_text(content.strip(), encoding="utf-8")

    all_docs = []
    for fp in sorted(Path(DOCUMENTS_PATH).iterdir()):
        if fp.suffix.lower() in (".pdf", ".txt"):
            all_docs.extend(load_file(str(fp)))
    if not all_docs:
        raise RuntimeError("No documents loaded. Check data/docs/")

    chunks = chunk_documents(all_docs)
    logger.info("Building FAISS index from %d chunks...", len(chunks))
    vs = FAISS.from_documents(chunks, get_embeddings())
    vs.save_local(FAISS_INDEX_PATH)
    size_mb = (Path(FAISS_INDEX_PATH) / "index.faiss").stat().st_size / (1024 * 1024)
    logger.info("FAISS index saved (%.2f MB). Ingestion complete.", size_mb)
    logger.info("=" * 60)


if __name__ == "__main__":
    run_ingestion()