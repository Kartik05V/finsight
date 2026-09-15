"""
SQLite persistence: saves transactions and chat history across sessions.
DB at FINSIGHT_DB_PATH (default ~/.finsight/finsight.db). No extra dependencies.
Schema: transactions(session_id, data JSON, saved_at) | chat_messages(session_id, role, content, ts)
"""
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from finsight.logging_setup import get_logger
from finsight.models import Transaction

logger = get_logger(__name__)

_DEFAULT_DB = Path.home() / ".finsight" / "finsight.db"
DB_PATH = Path(os.getenv("FINSIGHT_DB_PATH", str(_DEFAULT_DB)))


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            session_id TEXT NOT NULL,
            data       TEXT NOT NULL,
            saved_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            ts         TEXT NOT NULL
        );
        """
    )
    conn.commit()


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_transactions(session_id: str, transactions: list[Transaction]) -> None:
    """Persist the transaction list for a session (replaces existing data)."""
    data = json.dumps([t.model_dump(mode="json") for t in transactions])
    with _connect() as conn:
        _ensure_tables(conn)
        conn.execute(
            "DELETE FROM transactions WHERE session_id = ?", (session_id,)
        )
        conn.execute(
            "INSERT INTO transactions (session_id, data, saved_at) VALUES (?, ?, ?)",
            (session_id, data, _now_iso()),
        )
        conn.commit()
    logger.info(f"saved {len(transactions)} transactions for session={session_id!r}")


def load_transactions(session_id: str) -> list[Transaction]:
    """Load transactions for a session. Returns [] if none saved yet."""
    with _connect() as conn:
        _ensure_tables(conn)
        row = conn.execute(
            "SELECT data FROM transactions WHERE session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone()

    if not row:
        return []

    raw = json.loads(row["data"])
    transactions = [Transaction.model_validate(r) for r in raw]
    logger.info(f"loaded {len(transactions)} transactions for session={session_id!r}")
    return transactions


def save_message(session_id: str, role: str, content: str) -> None:
    """Append one chat message to the history for a session."""
    with _connect() as conn:
        _ensure_tables(conn)
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now_iso()),
        )
        conn.commit()


def load_chat_history(session_id: str) -> list[dict]:
    """Return full ordered chat history for a session as list of {role, content} dicts."""
    with _connect() as conn:
        _ensure_tables(conn)
        rows = conn.execute(
            "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def clear_session(session_id: str) -> None:
    """Delete all persisted data for a session (transactions + chat history)."""
    with _connect() as conn:
        _ensure_tables(conn)
        conn.execute("DELETE FROM transactions WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        conn.commit()
    logger.info(f"cleared session={session_id!r}")
