# 🎬 AutoStream AI Agent
### Social-to-Lead Agentic Workflow — ML Intern Assignment | ServiceHive × Inflx

A production-grade conversational AI agent that converts social media conversations into qualified business leads. Built for **AutoStream**, a fictional SaaS platform for automated video editing.

---

## 📁 Project Structure

```
autostream_agent/
│
├── knowledge_base/
│   ├── autostream_kb.md        ← Primary knowledge base (RAG source)
│   └── autostream_kb.json      ← Structured reference copy
│
├── faiss_index/                ← Auto-generated on first run (cached embeddings)
├── data/
│   └── autostream.db           ← SQLite database (auto-created on first run)
│
├── rag_pipeline.py             ← RAG pipeline (FAISS + HuggingFace embeddings)
├── intent_classifier.py        ← Intent classification (Gemini 2.5 Flash)
├── tools.py                    ← mock_lead_capture() tool
├── agent.py                    ← LangGraph state graph (core agent logic)
├── database.py                 ← SQLite layer (sessions, leads, messages, intents)
├── main.py                     ← CLI entry point
├── streamlit_app.py            ← Streamlit UI (Chat + Admin Panel)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/mohdfahad20/autostream-agent.git
cd autostream-agent
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
ADMIN_PASSWORD=your_admin_password_here
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 5. Run the Streamlit UI (recommended)
```bash
streamlit run streamlit_app.py
```

### 6. Or run via CLI
```bash
python main.py
```

> **Note:** On first run, the FAISS index will be built and saved to `./faiss_index/`. All subsequent runs load from disk and start significantly faster.

---

## 🏗️ Architecture Explanation 

The agent is built on **LangGraph**, chosen over AutoGen for its explicit state graph model each conversation turn flows through defined nodes with typed state, making the pipeline transparent, debuggable, and easy to extend.

The graph has four nodes: **Intent Classifier**, **Response Agent**, **Lead Collector**, and **Lead Capture**. Every user message enters at the classifier, which uses Gemini 2.5 Flash to label intent as `greeting`, `inquiry`, or `high_intent`. Conditional edges then route to the correct node greetings and inquiries go to the Response Agent (RAG-powered), while high-intent signals route to the Lead Collector.

**State management** is handled by a `TypedDict`-based `AgentState` that persists across all turns via LangGraph's `add_messages` reducer. This retains the full conversation history, all collected lead fields, intent progression, and session metadata across 5–6+ turns with zero manual memory management.

**RAG** uses HuggingFace `all-MiniLM-L6-v2` embeddings with a FAISS vector store, built once from a local Markdown knowledge base and cached to disk. The retriever fetches the top 3 relevant chunks per query, which are injected into the LLM prompt as context ensuring answers are grounded in actual product data, not hallucinated.

All events messages, intents, leads, session outcomes — are persisted to a **SQLite database** with four relational tables, queryable via the password-protected Admin Panel.

---

## 📱 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, the integration would use the **WhatsApp Business API (Cloud API)** provided by Meta, connected to the agent via a webhook.

### How it works

```
WhatsApp User
     │
     ▼
Meta WhatsApp Cloud API
     │  (POST webhook event)
     ▼
Your Webhook Server (FastAPI / Flask)
     │  parses message, calls agent
     ▼
AutoStream LangGraph Agent
     │  returns reply
     ▼
Meta API → send_message()
     │
     ▼
WhatsApp User receives reply
```

### Implementation Steps

**1. Set up a webhook server** using FastAPI:
```python
from fastapi import FastAPI, Request
from agent import build_graph, initial_state, chat

app   = FastAPI()
graph = build_graph()

# In-memory session store (use Redis in production)
sessions = {}

@app.post("/webhook")
async def webhook(request: Request):
    data       = await request.json()
    message    = data["entry"][0]["changes"][0]["value"]["messages"][0]
    phone      = message["from"]
    user_text  = message["text"]["body"]

    # Get or create session per phone number
    if phone not in sessions:
        from database import create_session
        sid = create_session()
        sessions[phone] = initial_state(sid)

    reply, sessions[phone] = chat(user_text, sessions[phone], graph)

    # Send reply back via WhatsApp Cloud API
    send_whatsapp_message(phone, reply)
    return {"status": "ok"}
```

**2. Register the webhook** in the Meta Developer Portal:
- Go to your App → WhatsApp → Configuration
- Set webhook URL to `https://yourdomain.com/webhook`
- Subscribe to `messages` events
- Set a verify token and handle the `GET /webhook` verification handshake

**3. Session persistence per phone number:**
- Use the caller's phone number as the session key
- Store `AgentState` in Redis or a database so conversations survive server restarts
- Each phone number maintains its own independent LangGraph state

**4. Deploy** the FastAPI server on any cloud provider (Railway, Render, AWS, GCP) with HTTPS required by Meta for webhook delivery.

---

## 🧪 Agent Capabilities

| Capability | Implementation |
|---|---|
| Intent Classification | Gemini 2.5 Flash → `greeting` / `inquiry` / `high_intent` |
| RAG Knowledge Retrieval | FAISS + `all-MiniLM-L6-v2` + local Markdown KB |
| Lead Capture Tool | `mock_lead_capture()` — fires only after all 3 fields validated |
| Email Validation | Inline regex check — re-asks if invalid, never stores bad email |
| Platform Pre-detection | Extracts platform from high-intent message — skips asking again |
| State Management | LangGraph `AgentState` — persists across 5–6+ turns |
| Session Logging | SQLite — 4 tables: sessions, leads, messages, intents |
| Admin Panel | Password-protected Streamlit tab with lead table + transcript viewer |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Agent Framework | LangGraph 0.4.1 |
| LLM | Gemini 2.5 Flash (google-genai SDK) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector Store | FAISS (disk-cached after first run) |
| Database | SQLite via Python `sqlite3` |
| UI | Streamlit |
| Retry Logic | Tenacity (exponential backoff on API errors) |

---

## 📊 Admin Panel

Access at `http://localhost:8501` → **📊 Admin Panel** tab.

Default password: set `ADMIN_PASSWORD` in your `.env` file.

Features:
- Live stats: total sessions, leads captured, conversion rate, avg turns to convert
- Full leads table with timestamps and session duration
- "Showed Interest — Not Captured" table for warm lead follow-up
- Session transcript viewer with intent badges per message

---

## 🎬 Demo Video

[Demo Video Link](https://www.loom.com/share/c37ef89596d74a728a203439b5914c97)

The demo covers:
1. Agent answering a pricing question via RAG
2. Agent detecting high-intent and pre-filling platform
3. Agent collecting name and email with inline validation
4. Successful lead capture and confirmation
5. Admin Panel showing the captured lead
