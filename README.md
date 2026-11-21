# Bilingual Clinic WhatsApp FAQ Bot (RAG + FastAPI + Gradio)

A modular Retrieval-Augmented Generation (RAG) application for answering bilingual (English, Modern Standard Arabic, and Egyptian Arabic) medical clinic FAQs through a WhatsApp-style Gradio interface and a FastAPI backend.

## Features

* **Bilingual & Dialect Support:** Handles queries in English, Modern Standard Arabic, and Egyptian dialect, normalizing Arabic script variations and matching the user's language in replies.
* **9-Category Clinic Knowledge Base (`data/clinic_faq.json`):** Covers working hours and location, specialties and doctors' schedules, consultation fees and follow-ups, payment methods, medical insurance, appointments/walk-ins/cancellations, visit preparation and medical records, facilities/accessibility/parking, and medical safety/emergencies.
* **Hybrid / Resilient Retrieval (`app/rag.py`):** Uses Gemini embeddings with cosine similarity and score boosting, falling back automatically to synonym-aware lexical retrieval if embeddings are unavailable.
* **Grounding & Safety Guardrails (`app/chat_service.py`):** Grounds responses in retrieved `<faq_context>` entries, avoids inventing contact details or phone numbers, refrains from diagnosing or prescribing medication, and directs urgent symptoms to emergency care.
* **WhatsApp-Style UI & REST API (`app/whatsapp_ui.py`, `app/main.py`):** Mounts a WhatsApp-inspired chat UI at `/` with clickable recommended prompts and a **RAG Retrieval Inspector**, alongside `/health` and `/api/chat` endpoints.

## Project Structure

```text
appgain-clinic-bot/
├── .env                  # Local secrets (git-ignored)
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── config.py             # Environment and model configuration
├── data/
│   └── clinic_faq.json   # 9-category bilingual clinic FAQ knowledge base
└── app/
    ├── __init__.py
    ├── rag.py            # Retriever with embeddings, boosting, and lexical fallback
    ├── chat_service.py   # Prompt orchestration, grounding, and language matching
    ├── whatsapp_ui.py    # WhatsApp-style Gradio chat interface
    └── main.py           # FastAPI app and mounted Gradio interface

```

## Setup & Local Run

1. **Create and activate a virtual environment (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```


2. **Install dependencies:**
```powershell
pip install -r requirements.txt

```


3. **Configure environment variables:**
Copy `.env.example` to `.env` and add your Gemini API key (`.env` is ignored by Git):
```env
GEMINI_API_KEY=your_gemini_api_key_here
EMBEDDING_MODEL=gemini-embedding-2
GENERATION_MODEL=gemini-3.5-flash-lite
FALLBACK_GENERATION_MODEL=gemini-3.8-flash

```


4. **Start the server:**
```powershell
uvicorn app.main:app --reload --port 8000

```


* **Chat UI:** `[http://127.0.0.1:8000](http://127.0.0.1:8000)`
* **Health Check:** `[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)`
* **API Docs:** `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`



## API Endpoints

* `GET /health`: Returns server status and the number of indexed FAQ chunks.
* `POST /api/chat`: Accepts `{"message": "..."}` and returns `{"reply": "...", "rag_debug": "..."}`.

## Notes on Security & Medical Disclaimer

* Never commit `.env` or raw API keys to GitHub. Store secrets in `.env` locally and in repository secrets (e.g., `GEMINI_API_KEY`) for CI/CD workflows.
* This bot provides general clinic FAQ information only and is not a substitute for professional medical advice, diagnosis, or emergency care.
