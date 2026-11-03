"""
run_test_cases.py — runs the 8 test complaints through the real agents
(uses your API key) and prints a results table.

    python run_test_cases.py
"""
import asyncio

from agents import TicketContext
from config import get_settings
from rag import KnowledgeBase
from team import run_ticket
from ticket_logger import next_ticket_id

CASES = [
    ("Priya", "ORD-4512", "Where is my order? It was supposed to arrive yesterday.", "Resolved"),
    ("Daniel", "ORD-3301", "I want to return the shoes I bought last week. How do I get a refund?", "Resolved"),
    ("Aisha", "", "I forgot my password and can't log in.", "Resolved"),
    ("Marco", "ORD-2219", "The blender I received is broken. Can I get a replacement?", "Resolved"),
    ("Kevin", "ORD-5120", "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer.", "Escalated"),
    ("Linda", "", "Do you sell gift cards for a company of 500 employees with custom branding?", "Escalated"),
    ("Sam", "ORD-6001", "I just placed an order but typed the wrong street. Can I change the shipping address?", "Resolved"),
    ("Lucía", "ORD-7788", "El seguimiento dice que mi paquete fue entregado, pero no lo he recibido.", "Resolved"),
]


async def main() -> None:
    settings = get_settings()
    kb = KnowledgeBase(settings)
    print(f"Knowledge base: {len(kb.chunks)} chunks, backend={kb.backend}")
    for w in kb.warnings:
        print("  warning:", w)
    passed = 0
    for i, (name, order, text, expected) in enumerate(CASES, 1):
        ctx = TicketContext(ticket_id=next_ticket_id(settings.log_file), complaint=text,
                            customer_name=name, order_id=order)
        out = await run_ticket(ctx, kb, settings)
        ok = out.status == expected
        passed += ok
        t = out.triage
        print(f"\n#{i} {ctx.ticket_id} {'PASS' if ok else 'CHECK'}  expected={expected} got={out.status}")
        print(f"   {t.get('category')} · {t.get('urgency')} · {t.get('sentiment')} | sources: {out.kb_sources}")
        if out.reason:
            print(f"   reason: {out.reason}")
        if out.error:
            print(f"   error: {out.error}")
    print(f"\n{passed}/{len(CASES)} matched the expected status.")


if __name__ == "__main__":
    asyncio.run(main())
