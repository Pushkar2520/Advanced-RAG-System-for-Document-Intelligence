"""
ingest.py — Document ingestion pipeline.

Run this script whenever you add, remove, or update PDFs in data/.
It rebuilds the FAISS vector index from scratch.

Usage
-----
    python ingest.py
"""

import sys
import time
import logging

from langchain_community.vectorstores import FAISS

import config
from utils import load_pdfs, split_documents, get_embedding_model

logger = logging.getLogger(__name__)


def ingest() -> None:
    """
    End-to-end ingestion:
      1. Load every PDF from data/
      2. Split into overlapping chunks
      3. Compute embeddings with a local model
      4. Build a FAISS index and persist it to disk
    """
    start = time.time()

    # ── Step 1: Load ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading PDFs from '%s'", config.DATA_DIR)
    logger.info("=" * 60)
    try:
        documents = load_pdfs()
    except Exception as exc:
        logger.error("Error loading PDFs: %s", str(exc))
        sys.exit(1)

    logger.info("Total documents loaded: %d", len(documents))

    # ── Step 2: Split ───────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2 — Splitting into chunks")
    logger.info("=" * 60)
    chunks = split_documents(documents)

    if not chunks:
        logger.error("No chunks produced — PDFs may be empty or unreadable.")
        sys.exit(1)

    # ── Step 3 + 4: Embed & store ───────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3 — Embedding chunks & building FAISS index")
    logger.info("=" * 60)
    embeddings = get_embedding_model()

    # FAISS.from_documents embeds every chunk and inserts into the index
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist to disk so retrieval.py can load without re-embedding
    if config.FAISS_INDEX_DIR.exists():
        logger.warning("Overwriting existing FAISS index...")
    
    config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(config.FAISS_INDEX_DIR))

    elapsed = time.time() - start
    logger.info("")
    logger.info("✅  Ingestion complete in %.1f s", elapsed)
    logger.info("   • Chunks indexed : %d", len(chunks))
    logger.info("   • Index saved to : %s/", config.FAISS_INDEX_DIR)

    # Quick sanity preview — show the first chunk
    logger.info("")
    logger.info("── Sample chunk (first 200 chars) ──")
    logger.info("Source : %s", chunks[0].metadata.get("source", "?"))
    logger.info("Text   : %s…", chunks[0].page_content[:200])


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    ingest()