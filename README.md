# RAG Football Players System

A Retrieval-Augmented Generation (RAG) system that answers questions about football players using Wikipedia articles, local LLMs via Ollama, and a ChromaDB vector store — running entirely offline on local hardware.

---

## Architecture

![System Architecture](System%20architecture.png)

### Stage 1 — Data Pipeline *(runs once, offline)*

```
HuggingFace Dataset                player_names.txt
(rcds/wikipedia-persons-masked)    (10,000 names)
68,729 Wikipedia articles               │
         │                              │
         ▼                              ▼
   filter_dataset.py  ─────────────────┘
   Title + text-pattern matching
   68,729 → 2,504 footballers
         │
         ▼
   export_to_txt.py
   2,504 .txt files → wiki_data_filtered/
         │
         ▼
   build_index.py
   • Chunk text (512 tokens, 50 overlap)
   • Embed with nomic-embed-text (Ollama, GPU)
   • Store vectors → wiki_rag/ (ChromaDB)
```

### Stage 2 — Query Pipeline *(runs on every user question)*

```
User Question (Streamlit UI)
         │
         ▼
   Query Embedding
   nomic-embed-text via Ollama (GPU)
         │
         ▼
   Vector Search
   ChromaDB cosine similarity → top 10 chunks
         │
         ▼
   SimilarityPostprocessor
   Drop chunks below score 0.50 → ~5 chunks
         │
         ▼
   LLM Reranker  (qwen2.5:3b via Ollama, GPU)
   Score each chunk 1–10 → top 2 chunks
         │
         ▼
   QA Synthesis  (qwen2.5:3b via Ollama, GPU)
   Custom strict prompt, temperature = 0
         │
         ▼
   Answer + Retrieved Context (Streamlit UI)
```

---

## Project Structure

```
RAG_Wikipedia/
├── main.py                  # Streamlit app — query pipeline
├── build_index.py           # One-time index builder
├── filter_dataset.py        # Filter HuggingFace dataset to footballers
├── export_to_txt.py         # Export filtered dataset to .txt files
├── player_names.txt         # 10,000 known football player names
├── requirements.txt         # Python dependencies
│
├── wiki_data_filtered/      # 2,504 exported Wikipedia .txt articles
├── wiki_rag/                # ChromaDB persistent vector index
├── dataset/                 # Cached HuggingFace dataset
│   └── football_players/    # Filtered HuggingFace dataset (Arrow format)
└── venv/                    # Python virtual environment
```

---

## Infrastructure

| Component       | Technology                          |
|----------------|--------------------------------------|
| Frontend        | Streamlit (Python)                  |
| LLM             | qwen2.5:3b via Ollama (GPU)         |
| Embedding model | nomic-embed-text via Ollama (GPU)   |
| Vector database | ChromaDB (local, persistent)        |
| RAG framework   | LlamaIndex                          |
| GPU             | NVIDIA RTX 3050 Laptop (4 GB VRAM)  |
| CPU             | Intel i5-12450H (12 cores)          |
| RAM             | 15.7 GB                             |

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- NVIDIA GPU with CUDA (recommended)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pull required Ollama models

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### 3. Download and filter the dataset

```bash
python filter_dataset.py
python export_to_txt.py
```

### 4. Build the vector index *(takes ~5–10 min on GPU)*

```bash
python build_index.py
```

### 5. Run the app

```bash
streamlit run main.py
```

Open `http://localhost:8501` in your browser.

---

## Dataset

**Source:** [`rcds/wikipedia-persons-masked`](https://huggingface.co/datasets/rcds/wikipedia-persons-masked)

**Filtering:** 68,729 total articles filtered down to **2,504 football players** using two strategies:
- **Title matching:** exact and fuzzy match against `player_names.txt`
- **Text pattern matching:** first-sentence patterns like *"professional footballer"*, *"plays as a midfielder"*, etc., with exclusion of American football, Gaelic football, and other sports

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| 512-token chunks | Smaller chunks give more focused embeddings per topic section, reducing cross-topic confusion during retrieval |
| SimilarityPostprocessor (cutoff 0.50) | Drops clearly irrelevant chunks before they reach the LLM |
| LLMRerank (top 2) | Reduces noise fed to QA synthesis, limiting hallucination |
| temperature = 0 | Makes answers fully deterministic — same question always gives same answer |
| qwen2.5:3b over phi3:mini | Better instruction-following and negation handling for strict QA prompts |
| Offline-only stack | No API keys or internet required at query time |

---

## Limitations

- **Not exhaustive:** For questions like *"list all players who retired before 30"*, the system returns the 2 most relevant matches — not all of them. RAG is a search tool, not a database query engine.
- **Dataset coverage:** Only players present in the HuggingFace dataset are covered. Players without a Wikipedia article are not included.
- **LLM accuracy:** Answers depend on the quality of retrieved chunks. Ambiguous or cross-topic chunks may still occasionally lead to incorrect answers.
