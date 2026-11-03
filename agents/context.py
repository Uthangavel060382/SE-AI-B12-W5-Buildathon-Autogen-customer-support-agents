"""
agents/context.py — the "ticket folder" that travels with one complaint.

The raw facts (complaint text, customer name, order id, which documents were
found) are stored here by Python, so they are logged exactly as entered and
cannot be changed or forgotten by an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TicketContext:
    ticket_id: str
    complaint: str
    customer_name: str = ""
    order_id: str = ""

    # filled by the search_kb tool
    kb_searched: bool = False
    kb_sources: list[str] = field(default_factory=list)

    # filled by the log_ticket tool
    logged: bool = False
    log_row: dict | None = None
    log_file_used: str = ""

    @property
    def kb_found(self) -> bool:
        return len(self.kb_sources) > 0
