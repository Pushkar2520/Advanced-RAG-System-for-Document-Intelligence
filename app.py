"""
app.py — Interactive CLI for the RAG system.

Usage
-----
    python app.py              # interactive loop
    python app.py --once "What is X?"   # single-shot mode
"""

import argparse
import json
import logging
import sys
import time

import config
from retrieval import ask

logger = logging.getLogger(__name__)


# ── Pretty-print a result dict ──────────────────────────────────────
def _display(result: dict) -> None:
    """Print the answer and source metadata in a readable format."""
    print()
    print("─" * 60)
    print("📝  ANSWER")
    print("─" * 60)
    print(result["answer"])
    print()

    if result["sources"]:
        print("📄  SOURCES")
        for src in result["sources"]:
            print(f"   • {src}")
        print()

    # Optionally show retrieved chunk previews (useful for debugging)
    if result.get("chunks"):
        print("🔍  RETRIEVED CHUNKS (first 300 chars each)")
        for i, chunk in enumerate(result["chunks"], 1):
            print(f"   [{i}] {chunk}…")
            print()


# ── Pre-flight checks ───────────────────────────────────────────────
def _preflight() -> bool:
    """Verify that the FAISS index exists before starting."""
    index_file = config.FAISS_INDEX_DIR / "index.faiss"
    if not index_file.exists():
        print(
            "\n❌  FAISS index not found at '%s'.\n"
            "   Run ingestion first:\n\n"
            "       python ingest.py\n"
            % config.FAISS_INDEX_DIR
        )
        return False
    return True


# ── Interactive loop ─────────────────────────────────────────────────
def interactive() -> None:
    """Run an interactive question-answer loop until the user quits."""
    print()
    print("=" * 60)
    print("  📚  RAG System — Ask questions about your PDFs")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    print()

    while True:
        try:
            question = input("❓  Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if not question:
            print("⚠️  Please enter a question.\n")
            continue

        logger.info("User question: %s", question)
        start = time.time()
        result = ask(question, top_k=config.TOP_K)
        logger.info("Total response time: %.2fs", time.time() - start)
        _display(result)


# ── Single-shot mode ─────────────────────────────────────────────────
def single_shot(question: str, as_json: bool = False) -> None:
    """Answer one question and exit."""
    start = time.time()
    result = ask(question, top_k=config.TOP_K)
    logger.info("Total response time: %.2fs", time.time() - start)

    if as_json:
        # Machine-readable output (pipe-friendly)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _display(result)


# ── Entry point ──────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG CLI — ask questions grounded in your PDF documents."
    )
    parser.add_argument(
        "--once",
        type=str,
        default=None,
        help='Ask a single question, e.g. --once "What is the refund policy?"',
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the result as JSON (useful with --once).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.TOP_K,
        help=f"Number of chunks to retrieve (default {config.TOP_K}).",
    )
    args = parser.parse_args()

    # Override global top-k if the flag was passed
    if args.top_k != config.TOP_K:
        config.TOP_K = args.top_k

    if not _preflight():
        sys.exit(1)

    if args.once:
        single_shot(args.once, as_json=args.json)
    else:
        interactive()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    main()