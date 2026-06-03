# HR Policy Chatbot

A retrieval-augmented (RAG) HR policy assistant built with **FastAPI**, **LangChain**, **ChromaDB**, and a **React** chat UI. Answers are grounded in PDF policy documents stored in a local vector database.

## Features

- **RAG over HR PDFs** — Chroma + embeddings retrieve relevant policy chunks before answering
- **LangChain chains** — `ChatOllama` or `ChatGroq` for search-query rewriting and final answers
- **Server-side conversation memory** — last 10 messages kept verbatim; older turns summarized incrementally
- **Session API** — frontend sends a single user message + `session_id`; history stays on the backend
- **PII redaction** — middleware scrubs emails, SSN-like patterns, and employee IDs from requests
- **Local or cloud LLM** — Ollama (local, GPU-capable) or Groq (API)

## Architecture

```
┌─────────────┐     POST /ask          ┌──────────────────────────────────┐
│  React UI   │ ──────────────────────► │  FastAPI (main.py)               │
│  (Vite)     │   message + session_id  │  • session store                 │
└─────────────┘                         │  • conversation_history (window) │
                                        │  • retriever (Chroma)            │
                                        └──────────┬───────────────────────┘
                                                   │
                     ┌─────────────────────────────┼─────────────────────────────┐
                     ▼                             ▼                             ▼
              ┌─────────────┐              ┌─────────────┐              ┌─────────────┐
              │   ChromaDB  │              │   Ollama    │              │    Groq     │
              │  chroma_db/ │              │ ChatOllama  │              │  (optional) │
              └─────────────┘              └─────────────┘              └─────────────┘
```

## Prerequisites

- **Python 3.10+** (3.11 recommended if using `torch-directml` for HuggingFace embeddings on AMD)
- **Node.js 18+** (for the frontend)
- **Ollama** — [ollama.com](https://ollama.com) (for local LLM / optional embeddings)
- **HR PDFs** in `private-docs/` (not committed; add your own files)

## Quick start

### 1. Clone and configure

```bash
git clone <repo-url>
cd chatbot-hrapp
cp .env.example .env
# Edit .env — set LLM_PROVIDER, models, keys as needed
```

### 2. Python backend

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Place policy PDFs in `private-docs/`, then index them:

```bash
cd app
python seed_docs.py
```

Start the API (from the `app` directory):

```bash
uvicorn main:app --port 8000 --reload
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Frontend

```bash
cd frontend/chatbot-ui
npm install
npm run dev
```

Open the URL Vite prints (usually [http://localhost:5173](http://localhost:5173)). The UI calls `http://localhost:8000/ask`.

## Environment variables

Copy `.env.example` to `.env` in the project root (or set variables in your shell / Docker).

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `ollama` or `groq` | `ollama` |
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model name for chat | `deepseek-coder:1.3b` |
| `OLLAMA_NUM_GPU` | Layers on GPU (`-1` = all) | — |
| `OLLAMA_NUM_THREAD` | CPU threads (fallback) | — |
| `GROQ_API_KEY` | Required if `LLM_PROVIDER=groq` | — |
| `GROQ_MODEL` | Groq model id | `llama-3.1-8b-instant` |
| `EMBEDDING_PROVIDER` | `ollama` or `huggingface` | `huggingface` |
| `OLLAMA_EMBED_MODEL` | Ollama embed model | `nomic-embed-text` |
| `CHROMA_EMBEDDING_MODEL` | HuggingFace model name | `all-MiniLM-L6-v2` |
| `EMBED_DEVICE` | `cpu`, `cuda`, `dml` (HF only) | `cpu` |
| `CHROMA_COLLECTION` | Chroma collection name | `hr_policies` |
| `RELEVANCE_THRESHOLD` | Min similarity to answer | `0.1` |
| `HISTORY_WINDOW` | Recent messages kept verbatim | `10` |
| `PORT` | API port (Docker) | `8000` |

**Important:** If you change `EMBEDDING_PROVIDER` or the embedding model, delete or recreate `chroma_db/` and run `seed_docs.py` again (vector dimensions must match).

## API

### `POST /ask`

Send one user message; history is loaded from the server session.

**Request**

```json
{
  "message": "How many vacation days do I get?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response**

```json
{
  "reply": "..."
}
```

The frontend generates and stores `session_id` in `localStorage` (`hr_chat_session_id`).

### `POST /session/reset`

Clears server-side history for a session (used by **New chat** in the UI).

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## AMD GPU + ChatOllama (Ollama)

**ChatOllama does not use the GPU directly.** The [Ollama](https://ollama.com) server runs the model; configure GPU there.

1. Set in `.env`:

   ```env
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=deepseek-coder:6.7b
   OLLAMA_NUM_GPU=-1
   EMBEDDING_PROVIDER=ollama
   OLLAMA_EMBED_MODEL=nomic-embed-text
   ```

2. **Windows + AMD:** ROCm in Ollama is mainly for Linux. On Windows, use **Vulkan**:
   - Update AMD Adrenalin drivers
   - Quit Ollama from the system tray
   - Run:

     ```powershell
     .\scripts\start-ollama-amd.ps1
     ```

     Or set user environment variables before starting Ollama:
     - `OLLAMA_VULKAN=1`
     - `OLLAMA_GPU_OVERIDE=vulkan`

3. Pull models:

   ```bash
   ollama pull deepseek-coder:6.7b
   ollama pull nomic-embed-text
   ```

4. Confirm GPU use in Task Manager while running `ollama run <model>`.

5. Restart the FastAPI app. Startup logs from `ollama_gpu.py` will list models and AMD hints if Vulkan is not set.

For HuggingFace embeddings on AMD Windows (without Ollama embeddings), use a Python 3.11 venv and `pip install torch-directml`, then `EMBED_DEVICE=dml`.

## Docker

```bash
docker compose up --build
```

Ollama should run on the **host**; the compose file points to `http://host.docker.internal:11434`. Seed `chroma_db` and mount `private-docs` as in `docker-compose.yml`.

## Project structure

```
chatbot-hrapp/
├── app/
│   ├── main.py                 # FastAPI routes, LangChain chains
│   ├── retriever.py            # Chroma + embeddings
│   ├── conversation_history.py # 10-message window + summary
│   ├── session_store.py        # In-memory sessions
│   ├── ollama_gpu.py           # Ollama / AMD startup checks
│   ├── app_security.py         # PII redaction middleware
│   └── seed_docs.py            # Index PDFs into Chroma
├── frontend/chatbot-ui/        # React + Vite chat UI
├── private-docs/               # HR PDFs (local only)
├── chroma_db/                  # Vector store (generated)
├── scripts/
│   └── start-ollama-amd.ps1    # Ollama with Vulkan for AMD on Windows
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| `No relevant information found` | Run `python seed_docs.py`; check PDFs in `private-docs/` |
| Ollama connection errors | `ollama serve`; verify `OLLAMA_URL` |
| Answers ignore policy | Lower `RELEVANCE_THRESHOLD` slightly or improve PDF chunking |
| Slow retrieval | Use `EMBEDDING_PROVIDER=ollama` so embed + chat share Ollama GPU |
| Groq used instead of Ollama | Set `LLM_PROVIDER=ollama` in `.env` and restart API |
| AMD GPU still on CPU | Vulkan env vars + latest drivers; try a smaller quant model |
| Session lost after restart | Sessions are in-memory; add Redis/DB for production |

## License

Add your license here if applicable.
