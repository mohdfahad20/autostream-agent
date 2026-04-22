"""
agent.py
--------
LangGraph agent for AutoStream — writes every event to SQLite via database.py.

Nodes:
  🔍 Intent Classifier  — classifies + logs intent
  💬 Response Agent     — greetings & RAG answers
  📋 Lead Collector     — collects name / email / platform
  🎯 Lead Capture       — fires mock_lead_capture() + saves lead to DB
"""

import os
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from google import genai
from google.genai.errors import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from rag_pipeline import build_retriever, retrieve_context
from intent_classifier import classify_intent
from tools import mock_lead_capture
from database import save_intent, save_lead, save_message

load_dotenv()

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:        Annotated[list, add_messages]
    intent:          str
    lead_name:       Optional[str]
    lead_email:      Optional[str]
    lead_platform:   Optional[str]
    lead_captured:   bool
    collecting_lead: bool
    session_id:      str          # DB session key threaded through state
    turn_number:     int          # increments each user message
    showed_interest: bool         # True if high_intent ever seen


# ── Gemini + retriever ────────────────────────────────────────────────────────

CLIENT    = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
RETRIEVER = build_retriever()

RESPONSE_SYSTEM = """You are a friendly and knowledgeable sales assistant for AutoStream,
an AI-powered video editing SaaS for content creators.

Guidelines:
- Answer ONLY using the provided knowledge base context.
- If context does not cover the question, say you will connect them with the team.
- Keep responses concise (2-4 sentences max).
- Never make up pricing, features, or policies.
- Be warm, helpful, and encouraging without being pushy.
- IMPORTANT: When mentioning prices, NEVER use the dollar sign ($). 
  Always write prices as plain numbers with USD, e.g. "29 USD/month" or "79 USD per month".
  This is a strict formatting requirement.
"""

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(ClientError),
)
def call_llm(messages) -> str:
    prompt = "\n".join([m.content for m in messages])
    response = CLIENT.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
    text = response.text.strip()
    # Post-process: replace $NUMBER with NUMBER USD so Streamlit doesn't
    # misinterpret $ as a LaTeX delimiter — LLM instructions alone are unreliable
    import re
    text = re.sub(r'\$(\d)', r'\1 USD', text)
    return text

def _print_agent(label: str):
    print(f"\n  ┌─ {label}")

# ── Nodes ─────────────────────────────────────────────────────────────────────

def classify_node(state: AgentState) -> AgentState:
    _print_agent("🔍 Intent Classifier")
    last = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last:
        return {**state, "intent": "greeting"}

    result  = classify_intent(last.content)
    intent  = result["intent"]
    conf    = result["confidence"]
    turn    = state.get("turn_number", 1)

    print(f"  └─ Intent: [{intent}] (confidence: {conf:.2f})")

    # Persist intent event
    save_intent(state["session_id"], intent, conf, turn)

    # Persist user message with intent annotation
    save_message(
        session_id  = state["session_id"],
        turn_number = turn,
        role        = "user",
        content     = last.content,
        intent      = intent,
        confidence  = conf,
    )

    showed = state.get("showed_interest", False) or (intent == "high_intent")
    return {**state, "intent": intent, "showed_interest": showed}


def respond_node(state: AgentState) -> AgentState:
    _print_agent("💬 Response Agent")
    last = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    user_text = last.content if last else ""

    if state["intent"] == "greeting":
        print("  └─ Sending greeting response")
        reply = (
            "Hey there! 👋 Welcome to AutoStream — your AI-powered video editing platform. "
            "I can help you with pricing, features, or anything else. What would you like to know?"
        )
    else:
        print("  └─ Retrieving from knowledge base (RAG)...")
        context = retrieve_context(user_text, RETRIEVER)
        msgs = [
            SystemMessage(content=RESPONSE_SYSTEM),
            SystemMessage(content=f"Knowledge Base Context:\n\n{context}"),
            *state["messages"],
        ]
        print("  └─ Generating response...")
        reply = call_llm(msgs)

    # Persist agent reply
    save_message(state["session_id"], state.get("turn_number", 1), "agent", reply)

    return {**state, "messages": state["messages"] + [AIMessage(content=reply)]}


PLATFORM_KEYWORDS = [
    "youtube", "instagram", "tiktok", "facebook", "twitter", "x",
    "snapchat", "linkedin", "twitch", "shorts", "reels", "threads",
    "marketplace", "fb", "yt",
]

def _extract_platform_from_text(text: str) -> str | None:
    """
    Detect platform from user message using substring matching.
    Handles variants like "instagram account", "youtube channel",
    "facebook page", "my tiktok", etc.
    Returns a clean platform name or None.
    """
    lower = text.lower()

    # Order matters — check more specific terms first
    platform_map = [
        (["youtube", "shorts", " yt "],              "YouTube"),
        (["instagram", "reels", " ig ", "insta"],    "Instagram"),
        (["tiktok", "tik tok"],                      "TikTok"),
        (["facebook", "marketplace", " fb "],        "Facebook"),
        (["twitter", "x.com", " on x "],             "Twitter/X"),
        (["twitch"],                                  "Twitch"),
        (["linkedin"],                                "LinkedIn"),
        (["snapchat", "snap"],                        "Snapchat"),
        (["threads"],                                 "Threads"),
        (["pinterest"],                               "Pinterest"),
    ]

    for keywords, platform in platform_map:
        if any(kw in lower for kw in keywords):
            return platform

    # LLM fallback for unusual phrasing
    try:
        prompt = (
            "Extract the social media or content platform name from this message "
            "(e.g. YouTube, Instagram, TikTok, Facebook, Twitter, LinkedIn, Twitch). "
            "Reply with ONLY the platform name, or 'none' if not mentioned.\n\n"
            f"Message: {text}"
        )
        response = CLIENT.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        result = response.text.strip().strip("\"'").strip()
        return None if result.lower() in ("none", "n/a", "") else result
    except Exception:
        return None


def _is_valid_email(email: str) -> bool:
    """Basic email validation — must have @ and a dot after it."""
    import re
    return bool(re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email.strip()))


def collect_lead_node(state: AgentState) -> AgentState:
    _print_agent("📋 Lead Collector")
    last = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    user_text = (last.content if last else "").strip()

    name     = state.get("lead_name")
    email    = state.get("lead_email")
    platform = state.get("lead_platform")
    reply    = ""

    if state.get("collecting_lead"):
        # Store answer in the correct next empty field
        if name is None:
            name = user_text
            print(f"  └─ Name collected: {name}")

        elif email is None:
            # Validate email before accepting it
            if _is_valid_email(user_text):
                email = user_text
                print(f"  └─ Email collected: {email}")
            else:
                # Reject and re-ask — do NOT advance to platform
                reply = (
                    f"⚠️ **'{user_text}'** doesn't look like a valid email address. "
                    f"Please enter a valid email (e.g. `name@example.com`)."
                )
                print(f"  └─ Invalid email rejected: {user_text}")
                if reply:
                    save_message(state["session_id"], state.get("turn_number", 1), "agent", reply)
                return {
                    **state,
                    "messages": state["messages"] + [AIMessage(content=reply)],
                    "collecting_lead": True,
                    "lead_name":     name,
                    "lead_email":    None,   # keep email as None — re-ask next turn
                    "lead_platform": platform,
                }

        elif platform is None:
            # Also try platform extraction here in case user says "instagram account" at this stage
            extracted = _extract_platform_from_text(user_text)
            platform  = extracted if extracted else user_text
            print(f"  └─ Platform collected: {platform}")

    else:
        # First entry — high intent just detected
        # Try to pre-fill platform from the original message
        print("  └─ High-intent detected — starting lead collection")
        extracted = _extract_platform_from_text(user_text)
        if extracted:
            platform = extracted
            print(f"  └─ Platform pre-filled from message: {platform}")

    if name is None:
        reply = "That's awesome! 🎉 Could I start with your **full name**?"
    elif email is None:
        reply = f"Great, {name}! What's the best **email address** to reach you?"
    elif platform is None:
        reply = "Perfect! Which platform do you primarily create content on? (YouTube, Instagram, TikTok…)"
    else:
        reply = ""

    if reply:
        save_message(state["session_id"], state.get("turn_number", 1), "agent", reply)

    new_messages = state["messages"] + ([AIMessage(content=reply)] if reply else [])
    return {
        **state,
        "messages":        new_messages,
        "collecting_lead": True,
        "lead_name":       name,
        "lead_email":      email,
        "lead_platform":   platform,
    }


def capture_lead_node(state: AgentState) -> AgentState:
    _print_agent("🎯 Lead Capture")
    print(f"  └─ Saving lead to DB: {state['lead_name']}")

    result = mock_lead_capture(state["lead_name"], state["lead_email"], state["lead_platform"])

    if result["status"] == "success":
        save_lead(
            session_id       = state["session_id"],
            name             = state["lead_name"],
            email            = state["lead_email"],
            platform         = state["lead_platform"],
            turns_to_convert = state.get("turn_number", 1),
        )
        reply = (
            f"🎉 You're all set, {state['lead_name']}! "
            f"Our team will reach out to you shortly. Welcome aboard! 🚀"
        )
    else:
        # Fallback — should rarely happen now that email is validated earlier
        reply = (
            f"⚠️ Something went wrong saving your details. "
            f"Please try again or contact support."
        )

    save_message(state["session_id"], state.get("turn_number", 1), "agent", reply)

    return {
        **state,
        "messages":      state["messages"] + [AIMessage(content=reply)],
        "lead_captured": result["status"] == "success",
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_classify(state: AgentState) -> str:
    if state.get("lead_captured"):   return "respond"
    if state.get("collecting_lead"): return "collect_lead"
    return "collect_lead" if state["intent"] == "high_intent" else "respond"

def route_after_collect(state: AgentState) -> str:
    if state.get("lead_name") and state.get("lead_email") and state.get("lead_platform"):
        return "capture_lead"
    return END


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify",     classify_node)
    g.add_node("respond",      respond_node)
    g.add_node("collect_lead", collect_lead_node)
    g.add_node("capture_lead", capture_lead_node)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route_after_classify,
                            {"respond": "respond", "collect_lead": "collect_lead"})
    g.add_edge("respond",      END)
    g.add_edge("capture_lead", END)
    g.add_conditional_edges("collect_lead", route_after_collect,
                            {"capture_lead": "capture_lead", END: END})
    return g.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def initial_state(session_id: str) -> dict:
    return {
        "messages":        [],
        "intent":          "greeting",
        "lead_name":       None,
        "lead_email":      None,
        "lead_platform":   None,
        "lead_captured":   False,
        "collecting_lead": False,
        "session_id":      session_id,
        "turn_number":     0,
        "showed_interest": False,
    }


def chat(user_input: str, state: dict, graph) -> tuple[str, dict]:
    state = {
        **state,
        "messages":    state["messages"] + [HumanMessage(content=user_input)],
        "turn_number": state.get("turn_number", 0) + 1,
    }
    new_state = graph.invoke(state)
    last = next((m for m in reversed(new_state["messages"]) if isinstance(m, AIMessage)), None)
    return (last.content if last else ""), new_state