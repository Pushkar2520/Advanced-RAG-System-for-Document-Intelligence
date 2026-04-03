"""
config.py — Central configuration for the RAG system.

All tuneable parameters live here so you never hard-code values
in business-logic files. API keys are loaded from a .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables from .env ────────────────────────────
load_dotenv()

# ── Mistral API ─────────────────────────────────────────────────────
MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
if not MISTRAL_API_KEY:
    raise EnvironmentError(
        "MISTRAL_API_KEY is not set. "
        "Create a .env file with: MISTRAL_API_KEY=your-mistral-key-here"
    )

# Model to use for answer generation
LLM_MODEL: str = "mistral-medium-latest"

# Temperature=0 → deterministic, no creative drift
LLM_TEMPERATURE: float = 0.0

# Maximum tokens the LLM may generate per answer
LLM_MAX_TOKENS: int = 1024

# ── Paths ───────────────────────────────────────────────────────────
# Folder that holds source PDF files
DATA_DIR: Path = Path("data")

# Folder where the FAISS index + metadata are persisted
FAISS_INDEX_DIR: Path = Path("faiss_index")

# ── Chunking parameters ────────────────────────────────────────────
# Number of characters per chunk (not tokens — LangChain measures chars)
CHUNK_SIZE: int = 1000

# Overlap between consecutive chunks so context is not lost at boundaries
CHUNK_OVERLAP: int = 200

# ── Retrieval parameters ────────────────────────────────────────────
# How many chunks to retrieve per query
TOP_K: int = 3

# ── Embedding model ─────────────────────────────────────────────────
# We use a local, free model via HuggingFace sentence-transformers.
# "all-MiniLM-L6-v2" is small (~80 MB), fast, and good enough for most
# English-language retrieval tasks.
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"