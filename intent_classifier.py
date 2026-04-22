"""
intent_classifier.py
--------------------
Classifies each user message into one of three intent categories
using Gemini (google.genai SDK).

Intents:
  - greeting
  - inquiry
  - high_intent
"""

import os
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────

VALID_INTENTS = {"greeting", "inquiry", "high_intent"}

SYSTEM_PROMPT = """You are an intent classifier for AutoStream, a SaaS video editing platform.

Classify the user's message into EXACTLY one of these three intents:

1. greeting     — casual hello, hi, hey, small talk, or no clear question
2. inquiry      — asking about features, pricing, plans, refunds, support
3. high_intent  — clear intent to sign up, buy, or start using

Rules:
- Respond ONLY with JSON
- Format: {"intent": "...", "confidence": 0.0-1.0}
- If unsure → use "inquiry"
"""

# ── Shared client (IMPORTANT optimization) ───────────

CLIENT = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


# ── Core function ───────────────────────────────────

def classify_intent(user_message: str) -> dict:
    prompt = f"""
{SYSTEM_PROMPT}

User message:
{user_message}
"""

    try:
        response = CLIENT.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        raw = response.text.strip()

        # Clean markdown if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)

        intent = result.get("intent", "inquiry").lower()
        if intent not in VALID_INTENTS:
            intent = "inquiry"

        return {
            "intent": intent,
            "confidence": float(result.get("confidence", 0.9))
        }

    except Exception:
        return {"intent": "inquiry", "confidence": 0.5}


# ── Test block ──────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        ("Hey there!", "greeting"),
        ("What is the price?", "inquiry"),
        ("Do you offer refunds?", "inquiry"),
        ("I want to try this", "high_intent"),
        ("Sign me up!", "high_intent"),
    ]

    print("\n🧠 Intent Classifier Test\n" + "=" * 50)

    for msg, expected in test_cases:
        result = classify_intent(msg)
        status = "✅" if result["intent"] == expected else "❌"

        print(f"{status} {msg}")
        print(f"   Expected: {expected} | Got: {result['intent']} ({result['confidence']:.2f})\n")