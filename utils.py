"""
utils.py — Reusable helper functions.

Keeps business logic out of ingest.py and retrieval.py so each file
stays focused on a single responsibility.

Pipeline:
  PDF → Text → classify_document() → Chunking → detect_section_embedding() → Embedding
                (file-level)                      (chunk-level)
"""

import logging
from pathlib import Path
from typing import Dict, List

from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

import config


# ────────────────────────────────────────────────────────────────────
# Section labels for embedding-based detection
# ────────────────────────────────────────────────────────────────────
# Keys   = clean labels stored in metadata (used for matching in retrieval)
# Values = rich descriptions embedded for cosine similarity (stronger signal)

SECTION_LABELS: Dict[str, str] = {
    "electrical": "electrical specifications voltage current power supply ratings",
    "mechanical": "mechanical dimensions size weight height width torque housing",
    "safety": "safety interlock emergency protection SIL PLE IP rating category",
    "installation": "installation setup mounting wiring assembly instructions connection",
    "network": "network ethernet ports switching connectivity protocol communication",
    "ordering": "ordering part number catalog reference model type designation SKU",
    "general": "general information overview introduction description",
}

# Minimum cosine similarity to assign a section label.
# Below this threshold → defaults to "general" to prevent wrong classification.
SECTION_SIMILARITY_THRESHOLD: float = 0.3


# ────────────────────────────────────────────────────────────────────
# Document-level classification (entire file)
# ────────────────────────────────────────────────────────────────────
def classify_document(text: str) -> str:
    """
    Classify the entire document into a type based on keyword matching.
    Returns one of: 'datasheet', 'manual', 'report', 'general'.
    """
    text_lower = text.lower()

    if any(word in text_lower for word in [
        "technical data", "specification", "ratings", "voltage"
    ]):
        return "datasheet"

    elif any(word in text_lower for word in [
        "user manual", "installation", "operation", "instructions"
    ]):
        return "manual"

    elif any(word in text_lower for word in [
        "report", "analysis", "summary"
    ]):
        return "report"

    return "general"


# ── Logging setup ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Embedding model (singleton)
# ────────────────────────────────────────────────────────────────────
_embedding_model = None  # module-level cache


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return a HuggingFaceEmbeddings instance, creating it only once.
    Uses the model name defined in config.EMBEDDING_MODEL_NAME.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s …", config.EMBEDDING_MODEL_NAME)
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model ready.")
    return _embedding_model


# ────────────────────────────────────────────────────────────────────
# Cached section embeddings
# ────────────────────────────────────────────────────────────────────
_section_embeddings = None  # module-level cache


def get_section_embeddings() -> Dict[str, List[float]]:
    """
    Embed each section label description once and cache the result.
    Returns dict: clean_label → embedding vector.
    """
    global _section_embeddings

    if _section_embeddings is None:
        logger.info("Building section label embeddings …")
        model = get_embedding_model()

        _section_embeddings = {
            label: model.embed_query(description)
            for label, description in SECTION_LABELS.items()
        }
        logger.info("Section embeddings cached for %d labels.", len(_section_embeddings))

    return _section_embeddings


# ────────────────────────────────────────────────────────────────────
# Embedding-based section detection (per CHUNK, not per page)
# ────────────────────────────────────────────────────────────────────
def detect_section_embedding(text: str) -> str:
    """
    Detect the section of a CHUNK by comparing its embedding against
    cached section label embeddings using cosine similarity.

    If the best score is below SECTION_SIMILARITY_THRESHOLD (0.3),
    returns 'general' to prevent wrong classification of unrelated text.

    Returns the clean label (e.g. 'electrical', 'mechanical').
    """
    model = get_embedding_model()
    section_embeds = get_section_embeddings()

    # Embed the chunk text
    text_embedding = model.embed_query(text)

    best_section = "general"
    best_score = -1.0

    for section_label, section_embed in section_embeds.items():
        score = cosine_similarity(
            [text_embedding], [section_embed]
        )[0][0]

        if score > best_score:
            best_score = score
            best_section = section_label

    # Threshold check: if best score is too low, text is not clearly
    # about any section → default to general
    if best_score < SECTION_SIMILARITY_THRESHOLD:
        logger.info(
            "  Section: general (best was '%s' at %.3f, below threshold %.2f)",
            best_section,
            best_score,
            SECTION_SIMILARITY_THRESHOLD,
        )
        return "general"

    logger.info(
        "  Section detected: %s (score=%.3f)",
        best_section,
        best_score,
    )
    return best_section


# ────────────────────────────────────────────────────────────────────
# 1. Load all PDFs from the data/ folder
# ────────────────────────────────────────────────────────────────────
def load_pdfs(data_dir: Path = config.DATA_DIR) -> List[Document]:
    """
    Iterate over every .pdf in *data_dir*, parse each page, and tag
    it with file-level metadata only:
      - source   : filename
      - doc_type : datasheet / manual / report / general

    Section detection happens LATER at chunk level in split_documents().

    Returns
    -------
    List[Document]
        One Document per PDF page, with source and doc_type set.
    """
    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{data_dir.resolve()}'. "
            "Add at least one .pdf and try again."
        )

    all_docs: List[Document] = []

    for pdf_path in pdf_files:
        logger.info("Loading: %s", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        # File-level classification (combine all pages)
        full_text = " ".join([page.page_content for page in pages])
        doc_type = classify_document(full_text)
        logger.info("  → Classified as: %s", doc_type)

        # Tag each page with source and doc_type only
        # Section is NOT set here — it's set per chunk after splitting
        for page in pages:
            page.metadata["source"] = pdf_path.name
            page.metadata["doc_type"] = doc_type

        all_docs.extend(pages)
        logger.info("  → %d page(s) extracted", len(pages))

    logger.info("Total pages loaded across all PDFs: %d", len(all_docs))
    if all_docs:
        logger.info("Metadata sample: %s", all_docs[0].metadata)
    return all_docs


# ────────────────────────────────────────────────────────────────────
# 2. Split documents into overlapping chunks + detect section per chunk
# ────────────────────────────────────────────────────────────────────
def split_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """
    1. Split pages into smaller chunks using RecursiveCharacterTextSplitter.
    2. Run embedding-based section detection on EACH CHUNK.

    This is better than per-page detection because a single page can
    contain multiple topics (electrical + mechanical). Chunking first
    means each chunk is more focused → more accurate section label.

    Metadata after this step:
      source, doc_type, page  (from load_pdfs)
      section                 (detected here per chunk)

    Parameters
    ----------
    documents : list of Document
    chunk_size : int   – max characters per chunk
    chunk_overlap : int – characters shared between consecutive chunks

    Returns
    -------
    List[Document]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # ── Section detection per chunk ─────────────────────────────────
    logger.info("Detecting sections for %d chunk(s) …", len(chunks))

    for chunk in chunks:
        section = detect_section_embedding(chunk.page_content)
        chunk.metadata["section"] = section

    # ── Log summary ─────────────────────────────────────────────────
    if chunks:
        # Count section distribution
        section_counts: Dict[str, int] = {}
        for chunk in chunks:
            sec = chunk.metadata.get("section", "general")
            section_counts[sec] = section_counts.get(sec, 0) + 1

        logger.info("First chunk preview: %s", chunks[0].page_content[:200])
        logger.info("First chunk metadata: %s", chunks[0].metadata)
        logger.info("Section distribution: %s", section_counts)

    logger.info(
        "Split %d page(s) into %d chunk(s)  "
        "(size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks