"""
database.py
-----------
SQLite database layer for AutoStream agent.

Tables:
  sessions  — one row per conversation
  leads     — one row per captured lead (linked to session)
  messages  — every user + agent message with timestamps
  intents   — every intent classification with turn tracking
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "data" / "autostream.db"


# ── Connection ────────────────────────────────────────────────────────────────

@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row          # access columns by name
    conn.execute("PRAGMA journal_mode=WAL") # safe concurrent writes
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist. Safe to call on every startup."""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id       TEXT PRIMARY KEY,
            started_at       TEXT NOT NULL,
            ended_at         TEXT,
            duration_seconds REAL,
            outcome          TEXT DEFAULT 'abandoned',   -- abandoned | lead_captured
            showed_interest  INTEGER DEFAULT 0,          -- 1 if high_intent ever seen
            total_turns      INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS leads (
            lead_id          TEXT PRIMARY KEY,
            session_id       TEXT NOT NULL,
            name             TEXT NOT NULL,
            email            TEXT NOT NULL,
            platform         TEXT NOT NULL,
            captured_at      TEXT NOT NULL,
            turns_to_convert INTEGER,                   -- how many user turns before capture
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS messages (
            message_id  TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            turn_number INTEGER NOT NULL,               -- 1-indexed per session
            role        TEXT NOT NULL,                  -- 'user' | 'agent'
            content     TEXT NOT NULL,
            intent      TEXT,                           -- set for user turns only
            confidence  REAL,                           -- set for user turns only
            timestamp   TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS intents (
            intent_id   TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            intent      TEXT NOT NULL,
            confidence  REAL NOT NULL,
            turn_number INTEGER NOT NULL,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        """)
    print(f"✅ Database ready at {DB_PATH}")


# ── Session ops ───────────────────────────────────────────────────────────────

def create_session() -> str:
    """Insert a new session row and return the session_id."""
    session_id = str(uuid.uuid4())[:8]
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, started_at) VALUES (?, ?)",
            (session_id, datetime.now().isoformat()),
        )
    return session_id


def close_session(session_id: str, outcome: str, showed_interest: bool, total_turns: int):
    """Finalise session with end time, duration, and outcome."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT started_at FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return

        started = datetime.fromisoformat(row["started_at"])
        ended   = datetime.now()
        duration = round((ended - started).total_seconds(), 1)

        conn.execute("""
            UPDATE sessions
            SET ended_at=?, duration_seconds=?, outcome=?, showed_interest=?, total_turns=?
            WHERE session_id=?
        """, (ended.isoformat(), duration, outcome, int(showed_interest), total_turns, session_id))


# ── Message ops ───────────────────────────────────────────────────────────────

def save_message(session_id: str, turn_number: int, role: str,
                 content: str, intent: str = None, confidence: float = None):
    """Insert a single message (user or agent) into the messages table."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO messages
              (message_id, session_id, turn_number, role, content, intent, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4())[:8], session_id, turn_number,
            role, content, intent, confidence,
            datetime.now().isoformat(),
        ))


# ── Intent ops ────────────────────────────────────────────────────────────────

def save_intent(session_id: str, intent: str, confidence: float, turn_number: int):
    """Log an intent classification event."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO intents (intent_id, session_id, intent, confidence, turn_number, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4())[:8], session_id, intent,
            confidence, turn_number, datetime.now().isoformat(),
        ))


# ── Lead ops ──────────────────────────────────────────────────────────────────

def save_lead(session_id: str, name: str, email: str,
              platform: str, turns_to_convert: int):
    """Insert a captured lead and update the parent session outcome."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO leads (lead_id, session_id, name, email, platform, captured_at, turns_to_convert)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4())[:8], session_id, name, email,
            platform, datetime.now().isoformat(), turns_to_convert,
        ))
        conn.execute(
            "UPDATE sessions SET outcome='lead_captured', showed_interest=1 WHERE session_id=?",
            (session_id,),
        )


# ── Query helpers (for admin panel) ──────────────────────────────────────────

def get_all_leads() -> list[dict]:
    """Return all captured leads joined with their session data."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                l.lead_id, l.name, l.email, l.platform,
                l.captured_at, l.turns_to_convert,
                s.session_id, s.duration_seconds, s.total_turns
            FROM leads l
            JOIN sessions s ON l.session_id = s.session_id
            ORDER BY l.captured_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_interested_not_captured() -> list[dict]:
    """Sessions where user showed high_intent but lead was NOT captured."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT s.session_id, s.started_at, s.duration_seconds, s.total_turns,
                   COUNT(i.intent_id) as high_intent_count
            FROM sessions s
            JOIN intents i ON s.session_id = i.session_id AND i.intent = 'high_intent'
            WHERE s.outcome = 'abandoned'
            GROUP BY s.session_id
            ORDER BY s.started_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_session_transcript(session_id: str) -> list[dict]:
    """Return all messages for a given session in order."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT role, content, intent, confidence, timestamp, turn_number
            FROM messages
            WHERE session_id = ?
            ORDER BY turn_number, role DESC
        """, (session_id,)).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    """Aggregate stats for the admin dashboard."""
    with get_conn() as conn:
        total_sessions   = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        total_leads      = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        interested       = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE showed_interest=1 AND outcome='abandoned'"
        ).fetchone()[0]
        avg_turns        = conn.execute(
            "SELECT AVG(turns_to_convert) FROM leads"
        ).fetchone()[0]
        avg_duration     = conn.execute(
            "SELECT AVG(duration_seconds) FROM sessions WHERE outcome='lead_captured'"
        ).fetchone()[0]
    return {
        "total_sessions":    total_sessions,
        "total_leads":       total_leads,
        "interested_not_captured": interested,
        "conversion_rate":   round(total_leads / total_sessions * 100, 1) if total_sessions else 0,
        "avg_turns_to_convert": round(avg_turns, 1) if avg_turns else 0,
        "avg_duration_secs": round(avg_duration, 1) if avg_duration else 0,
    }