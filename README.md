# Healthcare AI Assistant

A RAG-based AI assistant built for the Mindbowser AI Engineer Hackathon. It answers questions from a set of healthcare documents using a retrieval-augmented generation pipeline.

Stack: FastAPI + LangChain + ChromaDB + Ollama (phi3:mini)

---

## How it works

User sends a question → the agent classifies intent → either routes to the RAG pipeline (for healthcare knowledge questions) or to a mock appointment tool (for booking-related questions).

RAG pipeline:
1. Embeds the question using `all-MiniLM-L6-v2`
2. Searches ChromaDB for the most relevant document chunks
3. Passes retrieved context + question to `phi3:mini` via Ollama
4. Returns answer with source citations and a confidence score

---

## Project Structure

```
healthcare-ai-assistant/
├── app/
│   ├── main.py          # FastAPI app and endpoints
│   ├── config.py        # Settings loaded from .env
│   ├── rag.py           # RAG pipeline
│   ├── embeddings.py    # Embedding model
│   ├── llm.py           # Ollama client
│   ├── agent.py         # Intent router + mock appointment tool
│   ├── ingest.py        # Document ingestion
│   └── prompts.py       # Prompt templates
├── data/                # Synthetic healthcare documents
├── frontend/            # Simple chat UI
├── tests/
├── vector_store/        # ChromaDB storage
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup (Local)

**Requirements:**
- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running
- `phi3:mini` pulled: `ollama pull phi3:mini`

```bash
git clone https://github.com/yourusername/healthcare-ai-assistant.git
cd healthcare-ai-assistant

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

copy .env.example .env         # Windows
# cp .env.example .env         # macOS/Linux
```

Start the server:
```bash
uvicorn app.main:app --reload --port 8000
```

Ingest documents (run once):
```bash
# Windows PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/ingest" -Method POST -ContentType "application/json" -Body '{"data_dir": "./data"}'
```

Open the chat UI at `http://localhost:8000/` or the API docs at `http://localhost:8000/docs`.

---

## Docker

```bash
docker-compose up --build

# Pull model into Ollama container (first time)
docker exec healthcare_ollama ollama pull phi3:mini

# Then ingest
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"data_dir": "./data"}'
```

---

## API

### GET /health
Returns status of the API, LLM connection, and vector store.

### POST /ingest
```json
{ "data_dir": "./data" }
```
Loads documents, generates embeddings, stores in ChromaDB.

### POST /ask
```json
{ "question": "Can I get a medication refill through telehealth?" }
```

Response:
```json
{
  "answer": "Yes, patients can request medication refills through telehealth...",
  "sources": [
    {
      "document": "telehealth_consultation_guidelines.txt",
      "chunk": "Medication refill requests may be reviewed during telehealth visits...",
      "relevance_score": 0.92
    }
  ],
  "confidence": "high",
  "route": "rag"
}
```

If the answer isn't in the documents:
```json
{
  "answer": "I could not find this information in the provided documents.",
  "sources": [],
  "confidence": "low",
  "route": "rag"
}
```

Appointment question (routed to mock tool):
```json
{
  "answer": "Available slots for Cardiology on Monday: 9:00 AM, 11:30 AM...",
  "confidence": "high",
  "route": "appointment_tool"
}
```

---

## Dataset

Six synthetic healthcare documents — no real patient data or PHI used:

- `patient_discharge_instructions.txt`
- `appointment_scheduling_policy.txt`
- `insurance_eligibility_faq.txt`
- `hipaa_privacy_guidelines.txt`
- `medication_refill_policy.txt`
- `telehealth_consultation_guidelines.txt`

All documents are fictional and created for this prototype.

---

## Prompt Design

The system prompt enforces strict grounding:

```
You are a professional healthcare AI assistant. Answer ONLY from the provided context.
If the answer is not in the documents, say so clearly.
Do not guess, speculate, or provide medical diagnoses.
```

Temperature is set to 0.1 to minimize hallucination. The prompt explicitly handles the "not found" case so the model never makes up an answer.

---

## Agent Routing

Questions go through a two-step classifier:
1. Keyword check (fast) — if appointment-related keywords are found, route to mock tool
2. LLM classification (fallback) — for ambiguous questions

The mock appointment tool (`check_available_slots`) simulates a scheduling system. In production this would connect to an EHR API like Epic or Cerner.

---

## Confidence Scoring

| Level | Score | Meaning |
|---|---|---|
| high | > 0.75 | Strong document match |
| medium | 0.50–0.75 | Partial match |
| low | < 0.50 | Weak match — answer may be incomplete |

---

## Limitations

- Only `.txt` files supported (no PDF/DOCX)
- phi3:mini is small — larger models would improve answer quality
- Appointment tool is mocked, not connected to a real system
- No auth — production would need JWT or API key protection
- Single-turn only — no conversation history

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Choices

| Component | Choice | Reason |
|---|---|---|
| LLM | Ollama + phi3:mini | Runs locally, no API cost, on-premise for healthcare |
| Embeddings | all-MiniLM-L6-v2 | Lightweight, fast, good retrieval quality |
| Vector DB | ChromaDB | Embedded, no separate server, easy to prototype with |
| Framework | FastAPI | Async, auto-docs, good for ML APIs |
| Orchestration | LangChain | Mature RAG tooling |

---

Built by **[Your Name]** — Mindbowser AI Engineer Hackathon Assignment
