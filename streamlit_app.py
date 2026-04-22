"""
streamlit_app.py
----------------
Two-tab Streamlit app:
  Tab 1 — 💬 Chat       : live conversation with the agent
  Tab 2 — 📊 Admin Panel: all captured leads, interested-not-captured, session stats

Run with: streamlit run streamlit_app.py
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from database import (
    init_db, create_session, close_session,
    get_all_leads, get_interested_not_captured,
    get_session_transcript, get_stats,
)
from agent import build_graph, initial_state, chat


def safe_md(text: str) -> str:
    """Escape $ so Streamlit doesn't treat prices as LaTeX math delimiters."""
    return text.replace("$", r"\$")


# ── Init DB on startup ────────────────────────────────────────────────────────
init_db()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoStream AI",
    page_icon="🎬",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background-color: #0f1117; }

.as-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 22px 28px; border-radius: 14px; margin-bottom: 20px;
    border: 1px solid #1e3a5f;
}
.as-header h1 { color: #fff; font-size: 1.6rem; margin: 0; }
.as-header p  { color: #8ab4d4; margin: 5px 0 0; font-size: 0.85rem; }

.stat-card {
    background: #1a1f2e; border: 1px solid #2d3748;
    border-radius: 12px; padding: 18px; text-align: center;
}
.stat-num   { font-size: 2rem; font-weight: 700; color: #60a5fa; }
.stat-label { font-size: 0.8rem; color: #8ab4d4; text-transform: uppercase;
              letter-spacing: 0.06em; margin-top: 4px; }

.badge { display:inline-block; padding:3px 10px; border-radius:20px;
         font-size:0.75rem; font-weight:600; }
.badge-green  { background:#1a3a2a; color:#4ade80; border:1px solid #166534; }
.badge-blue   { background:#1a2a3a; color:#60a5fa; border:1px solid #1d4ed8; }
.badge-yellow { background:#3a2a00; color:#fbbf24; border:1px solid #92400e; }
.badge-gray   { background:#1f2937; color:#9ca3af; border:1px solid #374151; }
.badge-red    { background:#3a1a1a; color:#f87171; border:1px solid #991b1b; }

.agent-label { font-size:0.7rem; color:#60a5fa; font-weight:600;
               letter-spacing:0.05em; text-transform:uppercase; margin-bottom:2px; }

.section-title { color:#8ab4d4; font-size:0.78rem; font-weight:700;
                 text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="as-header">
  <h1>🎬 AutoStream AI Assistant</h1>
  <p>Powered by Gemini 2.5 Flash · LangGraph · RAG · SQLite</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_admin = st.tabs(["💬 Chat", "📊 Admin Panel"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:
    col_chat, col_side = st.columns([2, 1])

    # ── Init session state ────────────────────────────────────────────────
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()
    if "session_id" not in st.session_state:
        st.session_state.session_id = create_session()
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = initial_state(st.session_state.session_id)
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "session_closed" not in st.session_state:
        st.session_state.session_closed = False

    # ── Sidebar info ──────────────────────────────────────────────────────
    with col_side:
        st.markdown("#### 🧠 Live Session")

        astate = st.session_state.agent_state
        intent = astate.get("intent", "—")
        bc_map = {"greeting":"badge-gray","inquiry":"badge-blue","high_intent":"badge-yellow"}
        bc     = bc_map.get(intent, "badge-gray")

        st.markdown(f"""
        <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;padding:14px;margin-bottom:12px">
            <div class="section-title">Current Intent</div>
            <span class="badge {bc}">{intent.replace('_',' ').title()}</span>
        </div>
        """, unsafe_allow_html=True)

        def fbadge(val):
            return f'<span class="badge badge-green">✓ {val}</span>' if val else '<span class="badge badge-gray">pending</span>'

        name     = astate.get("lead_name")
        email    = astate.get("lead_email")
        platform = astate.get("lead_platform")
        captured = astate.get("lead_captured", False)
        turns    = astate.get("turn_number", 0)
        interest = astate.get("showed_interest", False)

        st.markdown(f"""
        <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;padding:14px;margin-bottom:12px">
            <div class="section-title">Lead Details</div>
            <div style="margin:5px 0"><b style="color:#60a5fa">Name</b><br>{fbadge(name)}</div>
            <div style="margin:5px 0"><b style="color:#60a5fa">Email</b><br>{fbadge(email)}</div>
            <div style="margin:5px 0"><b style="color:#60a5fa">Platform</b><br>{fbadge(platform)}</div>
        </div>
        """, unsafe_allow_html=True)

        outcome_label = "✅ Lead Captured" if captured else ("🔥 Interested" if interest else "💬 Chatting")
        outcome_bc    = "badge-green" if captured else ("badge-yellow" if interest else "badge-blue")
        st.markdown(f"""
        <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;padding:14px;margin-bottom:12px">
            <div class="section-title">Status</div>
            <span class="badge {outcome_bc}">{outcome_label}</span>
            <div style="color:#9ca3af;font-size:0.8rem;margin-top:8px">Turns: {turns}</div>
            <div style="color:#9ca3af;font-size:0.8rem">Session: {st.session_state.session_id}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🔄 New Conversation", use_container_width=True):
            if not st.session_state.session_closed:
                close_session(
                    session_id      = st.session_state.session_id,
                    outcome         = "lead_captured" if captured else "abandoned",
                    showed_interest = interest,
                    total_turns     = turns,
                )
            for k in ["session_id","agent_state","chat_messages","session_closed"]:
                del st.session_state[k]
            st.rerun()

    # ── Chat messages ─────────────────────────────────────────────────────
    with col_chat:
        NODE_LABEL = {
            "greeting":  "💬 Response Agent",
            "inquiry":   "💬 Response Agent · RAG",
            "high_intent":"📋 Lead Collector",
            "capture":   "🎯 Lead Capture",
        }

        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    node = msg.get("node", "")
                    if node:
                        st.markdown(f'<div class="agent-label">{node}</div>', unsafe_allow_html=True)
                    st.markdown(safe_md(msg["content"]))

        if not captured:
            user_input = st.chat_input("Type your message…")
            if user_input:
                st.session_state.chat_messages.append({"role":"user","content":user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        reply, new_state = chat(
                            user_input,
                            st.session_state.agent_state,
                            st.session_state.graph,
                        )

                    new_intent   = new_state.get("intent","inquiry")
                    new_captured = new_state.get("lead_captured", False)
                    was_collecting = st.session_state.agent_state.get("collecting_lead", False)
                    now_collecting = new_state.get("collecting_lead", False)

                    if new_captured:
                        node_label = NODE_LABEL["capture"]
                    elif now_collecting and not was_collecting:
                        node_label = NODE_LABEL["high_intent"]
                    elif now_collecting:
                        node_label = "📋 Lead Collector"
                    else:
                        node_label = NODE_LABEL.get(new_intent, "💬 Response Agent")

                    st.markdown(f'<div class="agent-label">{node_label}</div>', unsafe_allow_html=True)
                    st.markdown(safe_md(reply))

                st.session_state.agent_state = new_state
                st.session_state.chat_messages.append({
                    "role":"assistant","content":reply,"node":node_label
                })

                if new_captured and not st.session_state.session_closed:
                    close_session(
                        session_id      = st.session_state.session_id,
                        outcome         = "lead_captured",
                        showed_interest = True,
                        total_turns     = new_state.get("turn_number", 0),
                    )
                    st.session_state.session_closed = True

                st.rerun()
        else:
            st.success("✅ Lead captured! Our team will be in touch with you shortly. 🚀")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ADMIN PANEL (password protected)
# ════════════════════════════════════════════════════════════════════════════
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")   # set in .env

with tab_admin:
    # ── Auth gate ─────────────────────────────────────────────────────────
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        st.markdown("### 🔒 Admin Access Required")
        st.markdown("<br>", unsafe_allow_html=True)

        col_form, _, _ = st.columns([1, 1, 1])
        with col_form:
            st.markdown("""
            <div style="background:#1a1f2e;border:1px solid #2d3748;
                        border-radius:12px;padding:28px;">
                <div style="color:#8ab4d4;font-size:0.85rem;margin-bottom:16px;">
                    Enter the admin password to access lead data and session analytics.
                </div>
            """, unsafe_allow_html=True)

            pwd_input = st.text_input("Password", type="password", key="admin_pwd_input",
                                      placeholder="Enter admin password…")

            if st.button("🔓 Login", use_container_width=True):
                if pwd_input == ADMIN_PASSWORD:
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect password. Please try again.")

            st.markdown("</div>", unsafe_allow_html=True)

        st.stop()   # stop rendering anything below for this tab

    # ── Authenticated — show logout + panel ───────────────────────────────
    col_title, col_logout = st.columns([5, 1])
    with col_title:
        st.markdown("### 📊 Admin Panel")
    with col_logout:
        if st.button("🔒 Logout", use_container_width=True):
            st.session_state.admin_authenticated = False
            st.rerun()

    if st.button("🔄 Refresh Data"):
        st.rerun()

    # ── Stats row ─────────────────────────────────────────────────────────
    stats = get_stats()
    c1, c2, c3, c4, c5 = st.columns(5)

    def stat_card(col, num, label):
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-num">{num}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    stat_card(c1, stats["total_sessions"],          "Total Sessions")
    stat_card(c2, stats["total_leads"],              "Leads Captured")
    stat_card(c3, f"{stats['conversion_rate']}%",    "Conversion Rate")
    stat_card(c4, stats["avg_turns_to_convert"],     "Avg Turns to Convert")
    stat_card(c5, stats["interested_not_captured"],  "Interested, Not Captured")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Captured Leads table ──────────────────────────────────────────────
    st.markdown("#### ✅ Captured Leads")
    leads = get_all_leads()
    if leads:
        df = pd.DataFrame(leads)
        df = df.rename(columns={
            "name":             "Name",
            "email":            "Email",
            "platform":         "Platform",
            "captured_at":      "Captured At",
            "turns_to_convert": "Turns to Convert",
            "session_id":       "Session ID",
            "duration_seconds": "Session Duration (s)",
        })
        df["Captured At"] = pd.to_datetime(df["Captured At"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            df[["Name","Email","Platform","Captured At","Turns to Convert","Session ID","Session Duration (s)"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No leads captured yet. Start a chat to capture your first lead.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Interested but not captured ───────────────────────────────────────
    st.markdown("#### 🔥 Showed Interest — Not Captured")
    warm = get_interested_not_captured()
    if warm:
        df2 = pd.DataFrame(warm)
        df2 = df2.rename(columns={
            "session_id":        "Session ID",
            "started_at":        "Started At",
            "duration_seconds":  "Duration (s)",
            "total_turns":       "Total Turns",
            "high_intent_count": "High-Intent Signals",
        })
        df2["Started At"] = pd.to_datetime(df2["Started At"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(df2, use_container_width=True, hide_index=True)
    else:
        st.info("No abandoned high-intent sessions yet.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Transcript viewer ─────────────────────────────────────────────────
    st.markdown("#### 🗂 Session Transcript Viewer")
    all_leads_raw = get_all_leads()
    all_sids = list({r["session_id"] for r in all_leads_raw})
    warm_sids = [r["session_id"] for r in warm] if warm else []
    all_sids_combined = list(dict.fromkeys(all_sids + warm_sids))

    if all_sids_combined:
        selected_sid = st.selectbox("Select a Session ID", all_sids_combined)
        if selected_sid:
            transcript = get_session_transcript(selected_sid)
            if transcript:
                for row in transcript:
                    role    = row["role"]
                    content = row["content"]
                    intent  = row.get("intent")
                    conf    = row.get("confidence")
                    ts      = row.get("timestamp", "")[:19]

                    with st.chat_message("user" if role == "user" else "assistant"):
                        if role == "user" and intent:
                            bc = {"greeting":"badge-gray","inquiry":"badge-blue",
                                  "high_intent":"badge-yellow"}.get(intent,"badge-gray")
                            st.markdown(
                                f'<span class="badge {bc}">{intent}</span> '
                                f'<span style="color:#6b7280;font-size:0.75rem">{conf:.2f} · {ts}</span>',
                                unsafe_allow_html=True,
                            )
                        elif role == "agent":
                            st.markdown(
                                f'<span style="color:#6b7280;font-size:0.75rem">{ts}</span>',
                                unsafe_allow_html=True,
                            )
                        st.markdown(safe_md(content))
            else:
                st.info("No messages found for this session.")
    else:
        st.info("No sessions available yet.")