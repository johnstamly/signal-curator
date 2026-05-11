"""SQLite-backed label store with WAL mode for multi-process concurrent access.

Schema is intentionally MORPHO-flavoured (panel / freq_khz / actuator /
receiver / cycle / rep_idx) since those columns are useful even for many
non-MORPHO datasets. If your dataset doesn't have actuator/receiver,
just store dummy values (e.g., 0); the labelling workflow uses only the
signal_id primary key and the label column.
"""
from __future__ import annotations
import sqlite3
import threading

_local = threading.local()

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS labels (
        signal_id TEXT PRIMARY KEY,
        panel TEXT,
        freq_khz INTEGER,
        actuator INTEGER,
        receiver INTEGER,
        cycle INTEGER,
        rep_idx INTEGER,
        label TEXT,
        source TEXT DEFAULT 'user',
        p_score REAL,
        labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""


def _new_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def _get_conn(db_path: str) -> sqlite3.Connection:
    """One connection per thread per db_path, table created on first use."""
    conns = getattr(_local, "conns", None)
    if conns is None:
        conns = {}
        _local.conns = conns
    conn = conns.get(db_path)
    if conn is None:
        conn = _new_conn(db_path)
        conns[db_path] = conn
    return conn


def _reset() -> None:
    """Close and clear all thread-local connections (for testing)."""
    conns = getattr(_local, "conns", None)
    if conns is not None:
        for conn in conns.values():
            conn.close()
        _local.conns = {}


def init_db(db_path: str) -> None:
    """Create the labels table if it does not exist."""
    _get_conn(db_path)


def signal_id(
    panel: str,
    freq_khz: int,
    actuator: int,
    receiver: int,
    cycle: int,
    rep_idx: int,
) -> str:
    """Deterministic unique ID for a signal across the dataset."""
    return f"{panel}_{freq_khz}kHz_act{actuator}_recv{receiver}_c{cycle}_r{rep_idx}"


def insert_label(
    db_path: str,
    sig_id: str,
    panel: str,
    freq_khz: int,
    actuator: int,
    receiver: int,
    cycle: int,
    rep_idx: int,
    label: str,
    source: str = "user",
    p_score: float | None = None,
) -> bool:
    """Insert a label. Returns True if inserted, False if duplicate (idempotent)."""
    try:
        conn = _get_conn(db_path)
        conn.execute(
            "INSERT INTO labels "
            "(signal_id, panel, freq_khz, actuator, receiver, cycle, rep_idx, "
            "label, source, p_score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sig_id, panel, freq_khz, actuator, receiver, cycle, rep_idx,
             label, source, p_score),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def insert_labels_batch(db_path: str, rows: list[tuple]) -> int:
    """Insert many labels in a single transaction; duplicates silently skipped."""
    conn = _get_conn(db_path)
    count = 0
    for row in rows:
        try:
            conn.execute(
                "INSERT INTO labels "
                "(signal_id, panel, freq_khz, actuator, receiver, cycle, rep_idx, "
                "label, source, p_score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            count += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return count


def update_label(db_path: str, sig_id: str, new_label: str) -> bool:
    """Update an existing label. Returns True if updated, False if not found."""
    conn = _get_conn(db_path)
    cur = conn.execute(
        "UPDATE labels SET label = ?, labeled_at = CURRENT_TIMESTAMP "
        "WHERE signal_id = ?",
        (new_label, sig_id),
    )
    conn.commit()
    return cur.rowcount > 0


def get_all_labels(db_path: str, freq_khz: int | None = None) -> list[dict]:
    """Return all label rows as dicts, optionally filtered by frequency."""
    conn = _get_conn(db_path)
    if freq_khz is not None:
        rows = conn.execute(
            "SELECT * FROM labels WHERE freq_khz = ?", (freq_khz,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM labels").fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM labels LIMIT 0").description]
    return [dict(zip(cols, r)) for r in rows]


def get_labeled_ids(db_path: str, freq_khz: int | None = None) -> set[str]:
    """Return the set of signal_ids that have been labelled."""
    conn = _get_conn(db_path)
    if freq_khz is not None:
        rows = conn.execute(
            "SELECT signal_id FROM labels WHERE freq_khz = ?", (freq_khz,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT signal_id FROM labels").fetchall()
    return {r[0] for r in rows}


def get_stats(db_path: str, freq_khz: int | None = None) -> dict:
    """Return {total, good, bad} label counts."""
    conn = _get_conn(db_path)
    if freq_khz is not None:
        total = conn.execute(
            "SELECT COUNT(*) FROM labels WHERE freq_khz = ?", (freq_khz,)
        ).fetchone()[0]
        good = conn.execute(
            "SELECT COUNT(*) FROM labels WHERE freq_khz = ? AND label = 'GOOD'",
            (freq_khz,),
        ).fetchone()[0]
        bad = conn.execute(
            "SELECT COUNT(*) FROM labels WHERE freq_khz = ? AND label = 'BAD'",
            (freq_khz,),
        ).fetchone()[0]
    else:
        total = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        good = conn.execute(
            "SELECT COUNT(*) FROM labels WHERE label = 'GOOD'"
        ).fetchone()[0]
        bad = conn.execute(
            "SELECT COUNT(*) FROM labels WHERE label = 'BAD'"
        ).fetchone()[0]
    return {"total": total, "good": good, "bad": bad}
