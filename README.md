# Edu Nexus — Tri-Hybrid GraphRAG Academic Engine

**Edu Nexus** is a zero-cost university semantic search engine powered by a **Tri-Hybrid RAG** strategy (BM25 + Qdrant + NetworkX). Upload research papers, query across three AI retrieval engines, and build knowledge graphs — all in one unified academic workspace.

---

## 🎥 Homepage Tour

![Homepage Tour](docs/screenshots/homepage_tour.webp)

---

## 🏠 Landing Page

### Hero Section — Spline 3D Interactive Background
![Homepage Hero](docs/screenshots/00_homepage_hero.png)

### Tri-Hybrid Intelligence — Three Specialized Retrieval Engines
![Tri-Hybrid Intelligence](docs/screenshots/00_homepage_tri_hybrid.png)

### How It Works — Upload → Process → Query → Answer
![How It Works](docs/screenshots/00_homepage_how_it_works.png)

### Features & Tech Stack
![Features & Tech Stack](docs/screenshots/00_homepage_features_techstack.png)

### Team PraxisX
![Team Section](docs/screenshots/00_homepage_team.png)

### Call to Action & Footer
![CTA & Footer](docs/screenshots/00_homepage_cta_footer.png)

---

## 🔐 Authentication

| Sign In | Sign Up |
|---------|---------|
| ![Sign In](docs/screenshots/00_sign_in.png) | ![Sign Up](docs/screenshots/00_sign_up.png) |

---

## 📊 Dashboard — Light Theme

### Source Documents
![Sources Light](docs/screenshots/01_sources_light.png)

### Tri-Hybrid RAG Chat
![Chat Light](docs/screenshots/02_chat_light.png)

### Knowledge Graph Explorer
![Graph Light](docs/screenshots/03_graph_light.png)

### Query History
![History Light](docs/screenshots/04_history_light.png)

### Settings & Appearance
![Settings Light](docs/screenshots/05_settings_light.png)

---

## 🌙 Dashboard — Dark Theme

### Source Documents
![Sources Dark](docs/screenshots/01_sources_dark.png)

### Tri-Hybrid RAG Chat
![Chat Dark](docs/screenshots/02_chat_dark.png)

### Knowledge Graph Explorer
![Graph Dark](docs/screenshots/03_graph_dark.png)

### Settings
![Settings Dark](docs/screenshots/05_settings_dark.png)

---

## Architecture

| Brain | Engine | Purpose |
|-------|--------|---------|
| **Semantic** | Qdrant + SentenceTransformers | Vector similarity search over document embeddings |
| **Keyword** | BM25 (Okapi) | Exact-match lexical retrieval |
| **Graph** | NetworkX + GLiNER | Knowledge graph traversal and relationship discovery |

A Groq-hosted LLM acts as the **intelligent router**, deciding which brain(s) to invoke per query, then fusing the results into a final grounded answer with chain-of-thought transparency.

## Tech Stack

| Layer | Technology |
|-------|-----------| 
| Backend | Python 3, FastAPI (dynamic PORT via env) |
| Frontend | Vite + React 18 + TypeScript + Tailwind CSS 3 |
| State | Zustand (auth, workspaces, sidebar, theme) |
| Data Fetching | TanStack Query + Axios |
| 3D Background | Spline (WebGL) |
| Animations | Framer Motion, MagicUI |
| LLM | Groq (query routing + answer generation) |
| Graph Engine | NetworkX (local, JSON-persisted per workspace) |
| NER | GLiNER (local model, no API calls) |
| Vector DB | Qdrant (local embedded mode, disk-persisted) |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |

## Features

- **Multi-format Upload**: PDF, DOCX, TXT, PPTX, XLSX, CSV, MD — drag-and-drop batch ingestion
- **Tri-Hybrid RAG Chat**: Ask questions and get answers grounded in your documents with visible chain-of-thought
- **Knowledge Graph Explorer**: Full-screen visualization with 4 layout modes (Force, Radial, Hierarchy, Grid)
- **Workspace Management**: Organize documents and chats into named workspaces
- **Document Viewer**: Read ingested chunks with inline AI chat
- **Search**: Cross-engine search with filter tabs (BM25 / Qdrant / Graph)
- **Query History**: Browse and manage past queries with engine tags
- **Engine Settings**: Tune retrieval weights per engine
- **Single-User Auth**: Secure local authentication with bcrypt (rounds=12) + session tokens
- **Theme Support**: Light, Dark, and System theme modes with 4 accent color options
- **Docling Integration** (opt-in): Higher quality PDF/PPTX extraction with OCR support
- **Deployment Ready**: Railway backend + Vercel frontend with IndexedDB browser storage

## Quick Start

### One-Time Setup
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Edu-Nexus.git
cd Edu-Nexus

# Run the setup script (creates venv, installs deps, creates .env)
setup.bat
```

Or manually:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd frontend && npm install && cd ..
copy .env.example .env
# Fill in API keys in .env
```

### Run
```bash
# Start both backend (8000) and frontend (5173)
run.bat
```

Or manually:
```bash
# Terminal 1 — Backend
venv\Scripts\activate
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

### Environment Variables (`.env`)
```
GROQ_API_KEY=your_groq_api_key
ALLOWED_ORIGINS=*                          # Comma-separated CORS origins
DOCLING_ENABLED=false                      # Set true for enhanced PDF extraction
```

> **Note**: No external database required. Qdrant runs in embedded mode and NetworkX graphs are stored as local JSON files. All data lives in the `data/` directory.

### Deploy to Railway + Vercel

```bash
# Backend (Railway)
# 1. Push repo to GitHub
# 2. Connect Railway to your repo
# 3. Set env vars: GROQ_API_KEY, ALLOWED_ORIGINS=https://your-frontend.vercel.app
# Railway auto-detects Procfile and runtime.txt

# Frontend (Vercel)
cd frontend
# Set env var: VITE_API_URL=https://your-backend.railway.app
npm run build
# Deploy dist/ to Vercel
```

## API Surface

REST endpoints on `http://localhost:8000`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register a new account |
| POST | `/api/auth/login` | Login with credentials |
| POST | `/api/auth/logout` | End session |
| GET | `/api/auth/status` | Check auth state |
| POST | `/api/auth/delete-account` | Wipe all data and account |
| GET | `/api/status` | Engine readiness + document counts |
| GET | `/api/sources` | List all ingested documents |
| POST | `/api/sources/upload` | Upload files → full ingestion pipeline |
| DELETE | `/api/sources/{name}` | Remove a source + rebuild indices |
| GET | `/api/sources/{name}/content` | Get parsed text chunks |
| POST | `/api/chat` | RAG query → answer + chain-of-thought |
| GET | `/api/history` | Past queries |
| DELETE | `/api/history/{id}` | Delete a history entry |
| GET | `/api/graph/nodes` | All graph nodes (supports `min_frequency` filter) |
| GET | `/api/graph/edges` | All graph edges (supports `min_weight` filter) |
| GET | `/api/graph/node/{name}` | Node detail + connections |
| GET | `/api/search?q=&engine=` | Targeted search |
| GET | `/api/settings/engines` | Load engine weights |
| POST | `/api/settings/engines` | Save engine weights |
| GET | `/api/jobs/{job_id}` | Poll ingestion job status |
| POST | `/api/process-and-return` | Stateless: extract chunks + graph, return to client |
| POST | `/api/query-with-context` | Stateless: answer query with client-provided chunks |

## Project Structure

> See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the full directory tree and [MODULE_DETAILS.md](MODULE_DETAILS.md) for a functional breakdown.

```
Edu-Nexus/
├── frontend/          # Vite + React + TypeScript SPA
├── src/               # Python backend — retrieval engines
│   ├── auth/          # Single-user auth (bcrypt + session tokens)
│   ├── graph_engine/  # NetworkX knowledge graph pipeline + GLiNER NER
│   ├── ingest/        # Document loading, extraction, cleaning, Docling
│   ├── orchestrator/  # Central query router (manager.py)
│   ├── pipeline/      # Full ingestion workflow glue
│   ├── retrieval/     # BM25 keyword index
│   ├── splitter/      # Text chunking logic
│   └── vector_engine/ # Qdrant vector store
├── server.py          # FastAPI REST API
├── config.py          # Centralized configuration
├── Procfile           # Railway deployment
├── railway.toml       # Railway config
├── runtime.txt        # Python version spec
├── setup.bat          # One-time environment setup
└── run.bat            # Start backend + frontend
```

## Team

| Role | Member | Focus |
|------|--------|-------|
| Arch / Core | Sarvesh | Orchestrator, Graph Logic, Frontend, API |
| Data Eng | Swaraj | PDF Cleaning Pipeline |
| Vector Eng | Saatvik | Chunking & Vector Store |
| QA / Ops | Kulvansh | Data Collection |

## License

See [LICENSE](LICENSE).
