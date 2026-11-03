"""
ticket_logger.py — writes one row per ticket to logs/tickets.csv.

Error handling
- The logs/ folder and the CSV header are created automatically.
- If the CSV is locked (for example open in Excel on Windows), the write is
  retried 3 times, then saved to logs/tickets_backup.csv so no ticket is lost.
- A damaged CSV never crashes the app: read_log() returns an empty table and
  reports the problem.
"""
from __future__ import annotations

import csv
import io
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

COLUMNS = [
    "ticket_id", "timestamp", "customer_name", "order_id", "complaint",
    "category", "urgency", "sentiment", "summary", "kb_sources", "response",
    "confidence", "status", "escalation_reason",
]


class TicketLogError(Exception):
    pass


def _ensure_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLUMNS)


def next_ticket_id(path: Path) -> str:
    """TKT-0001, TKT-0002, ... based on the highest id already in the log."""
    highest = 0
    for p in (path, path.with_name(path.stem + "_backup.csv")):
        if not p.exists():
            continue
        try:
            for m in re.finditer(r"TKT-(\d+)", p.read_text(encoding="utf-8", errors="ignore")):
                highest = max(highest, int(m.group(1)))
        except OSError:
            pass
    return f"TKT-{highest + 1:04d}"


def append_ticket(path: Path, row: dict) -> Path:
    """Append a row. Returns the file actually written to."""
    clean = {col: _clean(row.get(col, "")) for col in COLUMNS}
    clean["timestamp"] = clean["timestamp"] or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            _ensure_file(path)
            with path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=COLUMNS).writerow(clean)
            return path
        except PermissionError as exc:          # file open in Excel, etc.
            last_error = exc
            time.sleep(0.5 * (attempt + 1))

    backup = path.with_name(path.stem + "_backup.csv")
    try:
        _ensure_file(backup)
        with backup.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow(clean)
        return backup
    except OSError as exc:
        raise TicketLogError(f"Could not write ticket to {path} or backup: {last_error or exc}") from exc


def _clean(value) -> str:
    if isinstance(value, (list, tuple, set)):
        value = ", ".join(str(v) for v in value)
    return str(value if value is not None else "").strip()


def read_log(path: Path) -> tuple[pd.DataFrame, str | None]:
    """Return (table, error_message_or_None)."""
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS), None
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip")
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[COLUMNS], None
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(columns=COLUMNS), f"Could not read ticket log: {exc}"


def find_ticket(path: Path, ticket_id: str) -> dict | None:
    df, _ = read_log(path)
    match = df[df["ticket_id"] == ticket_id]
    if match.empty:
        backup = path.with_name(path.stem + "_backup.csv")
        if backup.exists():
            df, _ = read_log(backup)
            match = df[df["ticket_id"] == ticket_id]
    return None if match.empty else match.iloc[-1].to_dict()


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Tickets")
    return buf.getvalue()
