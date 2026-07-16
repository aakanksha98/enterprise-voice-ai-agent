# Enterprise Voice AI Agent

An enterprise-style voice AI assistant that handles service enquiries and appointment workflows through a browser voice interface and a graph-orchestrated AI backend.

## Project Goal

Build a production-minded flagship AI project while learning LangGraph through explicit state, routing, retrieval, and tool execution.

## Planned Stack

- Frontend: HTML, CSS, and vanilla JavaScript
- Voice: Browser Speech Recognition API and Speech Synthesis API
- Backend: FastAPI
- AI orchestration: LangGraph and LangChain
- LLM and embeddings: OpenAI
- RAG storage: Neon PostgreSQL with pgvector
- Deployment: Vercel

## Planned Capabilities

- Voice and text conversations
- Three demo business profiles: Dental Clinic, Salon, and Auto Repair Shop
- Profile-scoped service, pricing, policy, hours, and FAQ retrieval
- Mock appointment booking, cancellation, and rescheduling
- Human escalation routing
- One active appointment per conversation session

## Local Knowledge Setup

Index the bundled demo knowledge files into Neon after configuring `.env`:

```powershell
.\.venv\Scripts\python.exe -m backend.scripts.index_knowledge backend\data\knowledge
```

