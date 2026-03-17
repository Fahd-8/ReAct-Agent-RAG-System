# ReAct Agent RAG System

A full-stack RAG system with a ReAct (Reasoning + Acting) agent, 
conversational UI, and real-time Knowledge Base visualization. 
Delivers intelligent, document-augmented responses with full 
reasoning transparency.

## What It Does

- Ingests documents via URL into a live Knowledge Base
- Uses ReAct agent loop — reasons step-by-step before answering
- Retrieves relevant context from ingested documents
- Shows the full ReAct reasoning flow in a collapsible sidebar
- Responsive conversational UI built with React

## Tech Stack

- **ReAct Agent** — reasoning + acting loop for intelligent responses
- **RAG Pipeline** — document ingestion + semantic retrieval
- **React Frontend** — conversational UI with Knowledge Base panel
- **FastAPI Backend** — API layer and agent orchestration
- **Python** — core agent and retrieval logic

## Architecture
```
User Query → ReAct Agent Loop → Reasoning Step
→ RAG Retrieval (Knowledge Base) → Action
→ Final Answer + Reasoning Trace
```

## Key Features

- **ReAct Flow Visualization** — see exactly how the agent reasons
- **Collapsible Knowledge Base** — manage ingested documents live
- **Document URL ingestion** — add any document to context instantly
- **Full-stack** — frontend + backend in one repo

## Setup
```bash
git clone https://github.com/Fahd-8/ReAct-Agent-RAG-System
cd ReAct-Agent-RAG-System

# Backend
cd Backend
pip install -r requirements.txt
python main.py

# Frontend
cd frontend
npm install
npm start
```

## Usage

1. Enter a document URL in the Knowledge Base to ingest content
2. Ask questions in the chat area
3. Toggle ReAct Flow sidebar to see agent reasoning steps
4. Toggle Knowledge Base sidebar to manage documents

---
Built by [Fahad Zaman](https://github.com/Fahd-8) — AI Engineer
