# 📚 Intelligent RAG System — PDF Question Answering

A production-grade Retrieval-Augmented Generation (RAG) system that answers
questions **strictly** from PDF documents. Uses FAISS vector search, embedding-based
section detection, 3-component re-ranking, source diversity, and Mistral LLM —
going beyond basic RAG into **Intelligent RAG**.

---

## What Makes This "Intelligent RAG"?

| Feature | Basic RAG | This System |
|---------|-----------|-------------|
| Search | Embedding similarity only | Embedding + 3-component re-ranking |
| Section awareness | None | Embedding-based section detection per chunk |
| Intent detection | None | Embedding-based query intent matching |
| Document type | Ignored | Datasheets prioritized over general docs |
| Result diversity | Random top-k | Max 1 chunk per (source, page) pair |
| Hallucination control | Basic prompt | Strict prompt + temperature=0 + context-only |
| Section boost | None | Cosine similarity (partial relevance, not binary) |

---

## Architecture

```
┌                 INGESTION (run once)
│                                                             │
│  PDF files → PyPDFLoader → classify_document() (file-level) │
│                  │                                          │
│           RecursiveCharacterTextSplitter                    │
│           (1000 chars, 200 overlap)                         │
│                  │                                          │
│           detect_section_embedding() (chunk-level)          │
│           cosine vs 7 section label embeddings              │
│                  │                                          │
│           all-MiniLM-L6-v2 → FAISS index (saved to disk)   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────── QUERY (per question) ───────────────────┐
│                                                             │
│  Question → Embed → FAISS search (top_k × 3 candidates)    │
│                │                                            │
│         detect_intent_embedding()                           │
│         (cosine vs section labels → "electrical", etc.)     │
│                │                                            │
│         rank_chunks() — 3-component scoring:                │
│           50% similarity:  max(0, 1-distance)               │
│           20% doc_type:    datasheet > manual > report       │
│           30% section:     cosine(intent, section)          │
│                │                                            │
│         _deduplicate_sources()                              │
│         (1 chunk per source+page)                           │
│                │                                            │
│         Top-K selection → Prompt → Mistral (temp=0)         │
│                │                                            │
│         { answer, sources, doc_types, sections, chunks }    │
└─────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
rag-system/
├── config.py           # Settings: API key, paths, chunk size, top-k
├── utils.py            # PDF loading, chunking, embeddings, section detection
├── ingest.py           # Build FAISS index from PDFs
├── retrieval.py        # Query: search → intent → rank → diversity → LLM
├── app.py              # CLI interface (interactive + single-shot)
├── requirements.txt    # Python dependencies
├── .env.example        # Template for API key
├── .env                # ← YOU CREATE THIS (gitignored)
├── .gitignore          # Protects secrets and generated files
├── data/               # ← Drop your PDF files here
│   ├── datasheet.pdf
│   └── manual.pdf
├── faiss_index/        # Auto-generated after ingestion
│   ├── index.faiss
│   └── index.pkl
└── docs/               # Project documentation
    ├── 01_Project_Presentation.md
    ├── 02_Data_Flow_Architecture.md
    └── 03_Concepts_Guide.md
```

---

## Setup (Step by Step)

### 1. Create a virtual environment

```bash
cd rag-system
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Mistral API key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real key from [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys).

### 4. Add PDF files

Drop one or more `.pdf` files into the `data/` folder.

### 5. Build the index

```bash
python ingest.py
```

You will see:
```
Loading: schmersal_datasheet.pdf
  → Classified as: datasheet
  → 12 page(s) extracted
Detecting sections for 28 chunk(s) …
  Section detected: electrical (score=0.782)
  Section detected: mechanical (score=0.651)
  ...
Section distribution: {'electrical': 8, 'mechanical': 6, 'safety': 4, ...}
✅ Ingestion complete in 14.2s
   • Chunks indexed: 28
   • Index saved to: faiss_index/
```

### 6. Ask questions

```bash
# Interactive mode
python app.py

# Single question
python app.py --once "What is the operating voltage?"

# JSON output (for scripts/CI)
python app.py --once "What is the IP rating?" --json

# Custom top-k
python app.py --top-k 5
```

---

## Example Session

```
$ python app.py

============================================================
  📚  RAG System — Ask questions about your PDFs
  Type 'quit' or 'exit' to stop.
============================================================

❓  Your question: What is the operating voltage of AZM 161?

────────────────────────────────────────────────────────────
📝  ANSWER
────────────────────────────────────────────────────────────
The operating voltage of the AZM 161 is 24V DC.

📄  SOURCES
   • schmersal_azm161.pdf
   • safety_manual.pdf

📂  DOCUMENT TYPES
   • datasheet
   • manual

🏷️  SECTIONS
   • electrical
   • installation

🔍  RETRIEVED CHUNKS (first 300 chars each)
   [1] ELECTRICAL SPECIFICATIONS Operating voltage: 24V DC
       Current consumption: 50mA Output type: Safety relay…

   [2] When installing safety switches, ensure the supply
       voltage matches the device rating…
```

### JSON Output

```bash
$ python app.py --once "What is the IP rating?" --json
```

```json
{
  "answer": "The IP rating is IP67.",
  "sources": ["sensor_datasheet.pdf"],
  "doc_types": ["datasheet"],
  "sections": ["safety"],
  "chunks": ["Protection class: IP67. The sensor housing..."]
}
```

---

## How It Works

### Ingestion Pipeline

1. **Load PDFs** — `PyPDFLoader` extracts text per page
2. **Classify document** — keyword scan of full text → `datasheet` / `manual` / `report` / `general`
3. **Chunk** — `RecursiveCharacterTextSplitter` splits pages into ~1000-char chunks with 200-char overlap
4. **Detect section** — each chunk is embedded and compared via cosine similarity against 7 section label embeddings (`electrical`, `mechanical`, `safety`, `installation`, `network`, `ordering`, `general`)
5. **Index** — chunks are embedded with `all-MiniLM-L6-v2` and stored in FAISS

### Query Pipeline

1. **Embed query** — question becomes a 384-dim vector
2. **FAISS search** — fetch `top_k × 3` candidates (extra room for re-ranking)
3. **Filter** — discard chunks with FAISS distance ≥ 0.8
4. **Detect intent** — embed query, cosine similarity against section labels → best match
5. **Re-rank** — 3-component scoring (weights sum to 1.0):
   - **50%** — `max(0, 1 - distance)` — text similarity
   - **20%** — document type boost (datasheet=1.0, manual=0.7, report=0.4)
   - **30%** — cosine similarity between intent embedding and section embedding (partial credit)
6. **Diversity** — max 1 chunk per (source, page) pair
7. **Top-K** — select best k chunks
8. **Prompt** — build context with metadata labels
9. **LLM** — Mistral at temperature=0, strict system prompt
10. **Return** — answer + sources + doc_types + sections + chunks

---

## Configuration Reference

Edit `config.py` to change any of these:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | `mistral-medium-latest` | Mistral model for answer generation |
| `LLM_TEMPERATURE` | `0.0` | 0 = deterministic, no hallucination |
| `LLM_MAX_TOKENS` | `1024` | Max response length |
| `DATA_DIR` | `data/` | Where PDFs live |
| `FAISS_INDEX_DIR` | `faiss_index/` | Where the index is persisted |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `3` | Chunks retrieved per query |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Local sentence-transformer model |

Edit `retrieval.py` constants:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIM_WEIGHT` | `0.5` | Ranking weight for text similarity |
| `DOC_WEIGHT` | `0.2` | Ranking weight for document type |
| `SECTION_WEIGHT` | `0.3` | Ranking weight for section-intent match |
| `INTENT_SIMILARITY_THRESHOLD` | `0.3` | Min cosine score to assign non-general intent |

Edit `utils.py` constants:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SECTION_LABELS` | 7 categories | Section descriptions for embedding-based detection |
| `SECTION_SIMILARITY_THRESHOLD` | `0.3` | Min cosine score to assign non-general section |

---

## Code Map

### config.py
All settings in one place. API key loaded from `.env`.

### utils.py
| Function | Purpose |
|----------|---------|
| `classify_document()` | File-level: keywords → datasheet/manual/report/general |
| `get_embedding_model()` | Singleton: loads all-MiniLM-L6-v2 once |
| `get_section_embeddings()` | Singleton: embeds 7 section label descriptions, caches |
| `detect_section_embedding()` | Chunk-level: cosine similarity → section label |
| `load_pdfs()` | Reads PDFs, tags source + doc_type |
| `split_documents()` | Chunks pages, runs section detection per chunk |

### ingest.py
| Function | Purpose |
|----------|---------|
| `ingest()` | Full pipeline: load → split → embed → save FAISS index |

### retrieval.py
| Function | Purpose |
|----------|---------|
| `detect_intent_embedding()` | Query → embedding → cosine vs section labels → intent |
| `rank_chunks()` | 3-component scoring: similarity + doc_type + section cosine |
| `_deduplicate_sources()` | Max 1 chunk per (source, page) pair |
| `retrieve_chunks()` | Full pipeline: FAISS → filter → intent → rank → diversity |
| `_build_prompt()` | Format context + question with metadata labels |
| `_call_llm()` | Mistral API call with timing |
| `ask()` | Public API: question in → answer dict out |

### app.py
| Function | Purpose |
|----------|---------|
| `_display()` | Pretty-prints answer + sources + types + sections + chunks |
| `_preflight()` | Checks FAISS index exists before starting |
| `interactive()` | Loop: input → ask() → display |
| `single_shot()` | One question → answer → exit |
| `main()` | Arg parsing (--once, --json, --top-k) |

---

## Documentation

Three documents are included in `docs/`:

| Document | What It Covers |
|----------|---------------|
| `01_Project_Presentation.md` | Problem statement → architecture → approach → full demo walkthrough → cross-questions |
| `02_Data_Flow_Architecture.md` | Step-by-step data flow through both pipelines, code locations, ASCII architecture diagrams |
| `03_Concepts_Guide.md` | 15 topics explained with examples: embeddings, FAISS, chunking, cosine similarity, re-ranking, etc. |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `MISTRAL_API_KEY is not set` | Create `.env` with your key |
| `No PDF files found` | Add `.pdf` files to `data/` |
| `FAISS index not found` | Run `python ingest.py` first |
| Slow first run | Embedding model (~80 MB) downloads on first use |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| All chunks get section "general" | Check `SECTION_SIMILARITY_THRESHOLD` (default 0.3) |
| Answer says "No relevant information" | Try `--top-k 5` or check FAISS distance threshold (0.8) |
| Stale results after adding PDFs | Re-run `python ingest.py` |

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Vector DB | FAISS (CPU) | Free, fast, local, no server |
| Embeddings | all-MiniLM-L6-v2 | Free, local, 80MB, good quality |
| LLM | Mistral (API) | Good quality, affordable |
| Framework | LangChain | PDF loading, splitting, FAISS integration |
| Section Detection | Cosine similarity | Catches synonyms, partial relevance |
| Python | 3.10+ | Required by mistralai SDK |

---

## Notes

- Embeddings run **locally** — no API cost for indexing or section detection.
- FAISS index is CPU-only. For GPU acceleration on large corpora, switch to `faiss-gpu`.
- Re-run `python ingest.py` whenever you add, remove, or update PDFs in `data/`.
- The `docs/` folder contains beginner-friendly documentation for presenting and understanding the project.