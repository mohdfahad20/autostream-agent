"""
main.py — CLI entry point.
Run with: python main.py
"""

from dotenv import load_dotenv
load_dotenv()

from database import init_db, create_session, close_session
from agent import build_graph, initial_state, chat

def main():
    init_db()

    print("\n" + "=" * 60)
    print("🎬  AutoStream AI Assistant")
    print("    Powered by Gemini 2.5 Flash + LangGraph")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end.\n")

    graph      = build_graph()
    session_id = create_session()
    state      = initial_state(session_id)

    print(f"📋 Session ID: {session_id}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nAgent: Thanks for chatting! Have a great day! 👋\n")
            break

        reply, state = chat(user_input, state, graph)
        print(f"\nAgent: {reply}\n")

        if state.get("lead_captured"):
            print("─" * 60)
            print("✅ Lead captured and saved to database.")
            print("─" * 60 + "\n")
            break

    close_session(
        session_id      = session_id,
        outcome         = "lead_captured" if state.get("lead_captured") else "abandoned",
        showed_interest = state.get("showed_interest", False),
        total_turns     = state.get("turn_number", 0),
    )
    print(f"📝 Session {session_id} saved to database.")

if __name__ == "__main__":
    main()