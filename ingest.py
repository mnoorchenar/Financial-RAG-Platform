"""
ingest.py — Financial Document Ingestion Pipeline for FinRAG.

Downloads publicly available financial PDFs (FDIC, NAIC, SEC), chunks them with
overlap-aware recursive splitting, generates local embeddings via
sentence-transformers/all-MiniLM-L6-v2, and persists a FAISS vector index for
zero-latency retrieval at inference time.

Run once before launching the app:
    python ingest.py
"""

import os
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

from config import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE, FAISS_INDEX_PATH,
    DOCUMENTS_PATH, UPLOADED_DOCS_PATH, CHUNK_SIZE,
    CHUNK_OVERLAP, MIN_CHUNK_LENGTH, SOURCE_DOCUMENTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Directory Setup ────────────────────────────────────────────────────────────

def ensure_directories() -> None:
    """Create all required data directories if they do not yet exist."""
    for path in (DOCUMENTS_PATH, UPLOADED_DOCS_PATH, FAISS_INDEX_PATH):
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info("Directory ready: %s", path)


# ── Document Download ──────────────────────────────────────────────────────────

def download_pdf(doc_info: dict, dest_dir: str) -> Optional[str]:
    """
    Download a single PDF to dest_dir with a descriptive User-Agent header.

    Args:
        doc_info: Dict with keys 'name', 'url', 'domain'.
        dest_dir: Local destination directory.

    Returns:
        Path to the saved file, or None if the download fails.
    """
    dest_path = Path(dest_dir) / f"{doc_info['name']}.pdf"
    if dest_path.exists():
        logger.info("Already downloaded, skipping: %s", dest_path.name)
        return str(dest_path)

    headers = {"User-Agent": "FinRAG/1.0 (Educational Portfolio)"}
    try:
        logger.info("Downloading [%s]: %s", doc_info["domain"], doc_info["url"])
        resp = requests.get(doc_info["url"], timeout=45, headers=headers)
        resp.raise_for_status()
        dest_path.write_bytes(resp.content)
        logger.info("Saved %s (%.1f KB)", dest_path.name, len(resp.content) / 1024)
        return str(dest_path)
    except requests.RequestException as exc:
        logger.warning("Download failed for '%s': %s", doc_info["name"], exc)
        return None


# ── Synthetic Fallback Corpus ──────────────────────────────────────────────────

SYNTHETIC_DOCS: dict = {
    "banking_fundamentals.txt": """
BANKING FUNDAMENTALS — CONSUMER REFERENCE GUIDE

FDIC Deposit Insurance
The Federal Deposit Insurance Corporation (FDIC) is an independent agency of the
United States government that protects depositors against the loss of their insured
deposits if an FDIC-insured bank or savings association fails. FDIC insurance is
backed by the full faith and credit of the United States government.
The standard deposit insurance amount is $250,000 per depositor, per insured bank,
for each account ownership category. Ownership categories include single accounts,
joint accounts, retirement accounts, revocable trust accounts, and certain other
account types that may qualify for higher coverage.

Types of Deposit Accounts
A checking account is a demand deposit account held at a financial institution that
allows unlimited withdrawals and deposits by check, debit card, or electronic transfer.
A savings account is an interest-bearing deposit account. Banks typically limit
withdrawals to six per month under Regulation D, though this was suspended in 2020.
A certificate of deposit (CD) is a time deposit with a fixed maturity date and fixed
interest rate, available in terms from 30 days to five years. Early withdrawal
penalties typically apply.

Credit and Lending
A loan is money borrowed from a lender with an agreement to repay the principal plus
interest over a defined period. The annual percentage rate (APR) expresses the total
yearly cost of borrowing, including interest and fees, as a percentage of the loan
amount. Credit scores range from 300 to 850 and are calculated from payment history
(35%), amounts owed (30%), length of credit history (15%), credit mix (10%), and
new credit inquiries (10%).

Mortgage Lending
A mortgage is a secured loan used to purchase real property, where the property
itself serves as collateral. Fixed-rate mortgages carry the same interest rate for
the life of the loan. Adjustable-rate mortgages (ARMs) have rates that reset
periodically based on a reference index such as SOFR plus a margin. The loan-to-value
(LTV) ratio is the loan balance divided by the appraised property value; most lenders
require private mortgage insurance (PMI) when LTV exceeds 80 percent.
""",

    "insurance_guide.txt": """
INSURANCE CONSUMER GUIDE — NAIC STANDARDS AND DEFINITIONS

Life Insurance
Life insurance is a contract between a policyholder and an insurer in which the
insurer agrees to pay a death benefit to named beneficiaries upon the death of the
insured, in exchange for premium payments.
Term life insurance provides coverage for a specific period (typically 10, 20, or 30
years). If the insured dies within the term, the insurer pays the face amount to
beneficiaries. There is no cash value accumulation.
Whole life insurance provides lifetime coverage and includes a guaranteed cash value
component that grows at a fixed rate. Premiums are fixed and higher than term.
Universal life insurance is permanent coverage with flexible premiums and an
adjustable death benefit. The policy accumulates cash value based on current interest
rates credited by the insurer.
Variable life insurance ties the cash value and potentially the death benefit to
investment sub-accounts chosen by the policyholder, introducing market risk.

Health Insurance Key Terms
A premium is the monthly amount paid to maintain insurance coverage, regardless of
whether services are used. A deductible is the out-of-pocket amount the insured must
pay before the insurance company begins paying for covered services. A copayment is a
fixed dollar amount paid at the time of service. Coinsurance is the percentage of
costs shared between insurer and insured after the deductible is met. The out-of-pocket
maximum is the annual cap on the insured's cost-sharing; once reached, the insurer
covers 100% of covered in-network services for the rest of the plan year.

Property and Casualty Insurance
Homeowner's insurance covers the dwelling structure, personal property, additional
living expenses, and personal liability. Standard policies exclude floods, earthquakes,
and sewer backup, which require separate riders or policies.
The National Association of Insurance Commissioners (NAIC) is the regulatory support
organization created by state insurance commissioners. It develops model laws and
regulations, maintains databases, and coordinates regulatory efforts across states.

Underwriting and Actuarial Science
Underwriting is the process by which an insurer evaluates and classifies applicants
to determine eligibility and appropriate premium levels. Actuaries use statistical
analysis, mortality tables, and loss experience data to price insurance products and
establish loss reserves. Reinsurance is the practice of insurers transferring portions
of their risk to other insurers to reduce exposure to large losses.
""",

    "financial_statements_guide.txt": """
UNDERSTANDING FINANCIAL STATEMENTS — SEC INVESTOR EDUCATION

The Balance Sheet
A balance sheet presents a company's assets, liabilities, and shareholders' equity at
a specific point in time, following the equation: Assets = Liabilities + Shareholders' Equity.
Current assets include cash, accounts receivable, and inventory. Non-current assets
include property, plant and equipment (PP&E) and intangible assets. Current liabilities
include accounts payable and short-term debt. Long-term liabilities include bonds payable
and deferred tax liabilities. Shareholders' equity consists of common stock, retained
earnings, and accumulated other comprehensive income (AOCI).

The Income Statement
The income statement shows revenues, expenses, and net income over a reporting period.
Gross profit equals revenue minus cost of goods sold (COGS). Operating income equals
gross profit minus operating expenses. Net income equals operating income minus interest
expense and income taxes. Earnings per share (EPS) is net income divided by weighted
average diluted shares outstanding.

The Cash Flow Statement
Operating activities reflect cash from core business operations. Investing activities
include capital expenditures (CAPEX) and asset disposals. Financing activities include
debt issuance, debt repayments, dividend payments, and share buybacks. Free cash flow
(FCF) is operating cash flow minus CAPEX.

SEC Reporting Requirements
The SEC requires public companies to file Form 10-K (annual report) within 60-90 days
of fiscal year end. Form 10-Q (quarterly report) must be filed within 40-45 days of
each quarter end. Form 8-K discloses material events. Management's Discussion and
Analysis (MD&A) is a required narrative section explaining financial results and risks.
""",

    "risk_management_banking.txt": """
RISK MANAGEMENT IN BANKING — REGULATORY FRAMEWORK REFERENCE

Basel III Capital Standards
Basel III is the global regulatory framework developed by the Basel Committee on Banking
Supervision (BCBS). Common Equity Tier 1 (CET1) capital must represent at least 4.5%
of risk-weighted assets (RWA). Tier 1 capital must be at least 6% of RWA. Total capital
must be at least 8% of RWA. Banks must also maintain a Capital Conservation Buffer of
2.5% CET1. The Liquidity Coverage Ratio (LCR) requires banks to hold sufficient
high-quality liquid assets (HQLA) to cover net cash outflows over 30 days at 100% or
more. The Net Stable Funding Ratio (NSFR) requires available stable funding to exceed
required stable funding over a one-year horizon.

Categories of Banking Risk
Credit risk is the risk of loss from a borrower failing to meet contractual obligations.
Market risk is the risk of losses from movements in interest rates, equity prices, FX
rates, and commodity prices. Value-at-Risk (VaR) is a standard measure of market risk.
Operational risk is the risk of loss from inadequate processes, systems, people, or
external events. Liquidity risk is the risk that a bank cannot meet cash obligations
without unacceptable losses. Systemic risk is the risk that the failure of one
institution triggers cascading failures across the financial system.

Anti-Money Laundering and Know Your Customer
The Bank Secrecy Act (BSA) requires U.S. financial institutions to file Currency
Transaction Reports (CTRs) for cash transactions exceeding $10,000 and Suspicious
Activity Reports (SARs) for suspicious transactions of $5,000 or more. Know Your
Customer (KYC) programs require financial institutions to verify client identity,
assess risk profiles, and understand the nature of business relationships. Customer
Due Diligence (CDD) includes identifying beneficial owners and monitoring transactions.
Enhanced Due Diligence (EDD) applies to higher-risk customers such as politically
exposed persons (PEPs).
""",
}


def create_synthetic_fallback(dest_dir: str) -> list:
    """
    Write synthetic financial text documents to dest_dir as a self-contained fallback.

    Args:
        dest_dir: Directory in which to write the .txt documents.

    Returns:
        List of file paths written.
    """
    created = []
    for filename, content in SYNTHETIC_DOCS.items():
        path = Path(dest_dir) / filename
        if not path.exists():
            path.write_text(content.strip(), encoding="utf-8")
            logger.info("Created synthetic document: %s", filename)
        created.append(str(path))
    return created


# ── Document Loading ───────────────────────────────────────────────────────────

def load_document(file_path: str) -> list:
    """
    Load a PDF or plaintext document into a list of LangChain Document objects.

    Args:
        file_path: Path to the file.

    Returns:
        List of Document objects with page_content and metadata populated.
    """
    path = Path(file_path)
    docs = []
    try:
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
            docs = loader.load()
            for doc in docs:
                doc.metadata.setdefault("source", path.name)
                doc.metadata["source_file"] = path.name
                doc.metadata["file_type"] = "pdf"
        elif path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8")
            docs = [Document(
                page_content=text,
                metadata={"source": path.name, "source_file": path.name, "file_type": "txt", "page": 0},
            )]
        logger.info("Loaded %d page(s) from %s", len(docs), path.name)
    except Exception as exc:
        logger.error("Failed to load %s: %s", path.name, exc)
    return docs


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_documents(documents: list) -> list:
    """
    Split raw documents into overlapping chunks for dense retrieval.

    Args:
        documents: List of Document objects.

    Returns:
        List of chunked Document objects filtered for minimum length.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LENGTH]
    logger.info("Produced %d chunks from %d source documents", len(chunks), len(documents))
    return chunks


# ── Index Building ─────────────────────────────────────────────────────────────

def build_and_save_index(chunks: list, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Embed all chunks, build a FAISS index, and persist it to disk.

    Args:
        chunks: List of Document chunks.
        embeddings: Initialised HuggingFaceEmbeddings instance.

    Returns:
        The constructed FAISS vectorstore.
    """
    logger.info("Building FAISS index from %d chunks...", len(chunks))
    t0 = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info("Index built in %.1fs", time.time() - t0)
    vectorstore.save_local(FAISS_INDEX_PATH)
    index_file = Path(FAISS_INDEX_PATH) / "index.faiss"
    size_mb = index_file.stat().st_size / (1024 * 1024)
    logger.info("FAISS index saved to %s (%.2f MB)", FAISS_INDEX_PATH, size_mb)
    return vectorstore


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def run_ingestion() -> None:
    """
    Execute the full ingestion pipeline end-to-end.

    Steps:
      1. Ensure directories exist.
      2. Attempt to download public financial PDFs.
      3. Always write synthetic docs as a guaranteed fallback corpus.
      4. Load all available documents from DOCUMENTS_PATH.
      5. Chunk, embed, and persist FAISS index.
    """
    logger.info("=" * 64)
    logger.info("  FinRAG Ingestion Pipeline  —  Starting")
    logger.info("=" * 64)

    ensure_directories()

    downloaded = 0
    for doc_info in SOURCE_DOCUMENTS:
        path = download_pdf(doc_info, DOCUMENTS_PATH)
        if path:
            downloaded += 1
        time.sleep(0.5)

    if downloaded == 0:
        logger.warning("No PDFs downloaded. Using synthetic documents only.")

    create_synthetic_fallback(DOCUMENTS_PATH)

    all_docs = []
    for fp in sorted(Path(DOCUMENTS_PATH).iterdir()):
        if fp.suffix.lower() in (".pdf", ".txt"):
            all_docs.extend(load_document(str(fp)))

    if not all_docs:
        raise RuntimeError("No documents loaded. Check that data/docs/ is populated.")

    chunks = chunk_documents(all_docs)

    logger.info("Loading embedding model: %s on %s", EMBEDDING_MODEL, EMBEDDING_DEVICE)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    build_and_save_index(chunks, embeddings)

    logger.info("Ingestion complete. %d chunks indexed.", len(chunks))
    logger.info("=" * 64)


if __name__ == "__main__":
    run_ingestion()