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
- Required business type and required custom business name before the conversation starts
- Two-stage LLM design: one model plans intent and slots, a second model turns graph output into a natural receptionist reply
- Profile-scoped service, pricing, policy, hours, and FAQ retrieval
- Mock appointment booking, cancellation, and rescheduling
- Human escalation routing
- One active appointment per conversation session

## Business Name Contract

The user can enter any business name. The name personalizes the greeting only.

The selected business type controls the demo knowledge base, supported services, pricing, policies, and transactional workflow behavior:

- `dental`
- `salon`
- `auto_repair`

Example names such as `BrightSmile Dental`, `Luxe Hair Studio`, and `TurboFix Garage` are placeholders only. They are not defaults and do not affect retrieval or appointment logic.

## Response Generation Contract

The first LLM call is the planner. It classifies intent and extracts slots.

LangGraph handles workflow routing, state updates, missing field detection, RAG retrieval, service validation, and tool execution.

The second LLM call receives structured graph output and generates the final receptionist-style response. It must not execute tools, invent business facts, override workflow state, or make appointment decisions.

## Local Knowledge Setup

Index the bundled demo knowledge files into Neon after configuring `.env`:

```powershell
.\.venv\Scripts\python.exe -m backend.scripts.index_knowledge backend\data\knowledge
```
