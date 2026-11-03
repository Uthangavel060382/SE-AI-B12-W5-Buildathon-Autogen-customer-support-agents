"""
tests/offline_check.py — checks the project WITHOUT an API key or internet.

    python -m tests.offline_check

1. Builds the FAISS index with local embeddings and checks each test complaint
   retrieves the right document (and the gift-card question retrieves nothing).
2. Checks the escalation rule against the 6 assignment test cases.
3. Runs the full 4-agent AutoGen team with a scripted ("replay") model, to
   prove the order, the tools, the termination and the log row all work.
4. Checks the safety net: if the model fails, the ticket is still logged as Escalated.
"""
import asyncio
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

os.environ["EMBEDDING_PROVIDER"] = "local"

from autogen_core import FunctionCall  # noqa: E402
from autogen_core.models import CreateResult, RequestUsage  # noqa: E402
from autogen_ext.models.replay import ReplayChatCompletionClient  # noqa: E402

from agents import TicketContext  # noqa: E402
from config import get_settings  # noqa: E402
from escalation import decide_escalation  # noqa: E402
from rag import KnowledgeBase  # noqa: E402
from team import run_ticket  # noqa: E402
from ticket_logger import read_log  # noqa: E402

TMP = Path(tempfile.mkdtemp())
SETTINGS = replace(get_settings(), log_file=TMP / "tickets.csv", index_dir=TMP / "vs")
USAGE = RequestUsage(prompt_tokens=0, completion_tokens=0)
INFO = {"vision": False, "function_calling": True, "json_output": True,
        "family": "unknown", "structured_output": False}


def check(cond, label):
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        raise SystemExit(1)


def test_retrieval(kb):
    print("\n[1] FAISS retrieval (local embeddings)")
    cases = [
        ("Shipping: Where is my order? It was supposed to arrive yesterday.", "shipping_policy.md"),
        ("Refund: return shoes bought last week, how do I get a refund", "refund_policy.md"),
        ("Account: forgot my password and can't log in", "account_and_login.md"),
        ("Product: broken blender replacement", "product_faq.md"),
        ("Billing: charged my card three times duplicate charge", "billing_and_payments.md"),
    ]
    for query, expected in cases:
        results = kb.search(query)
        found = [r["source"] for r in results]
        check(expected in found, f"{query[:45]:45s} -> {found}")
    none = kb.search("Other: corporate gift cards for 500 employees custom branding")
    check(len(none) == 0, f"gift-card question returns no match (got {[r['source'] for r in none]})")


def test_escalation():
    print("\n[2] Escalation rule")
    rows = [
        ("Medium", "Frustrated", True, "High", "Where is my order?", False),
        ("Low", "Neutral", True, "High", "Return shoes", False),
        ("Low", "Neutral", True, "High", "Forgot password", False),
        ("Medium", "Frustrated", True, "High", "Blender broken", False),
        ("High", "Angry", True, "Medium",
         "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer.", True),
        ("Low", "Neutral", False, "Low", "Gift cards for 500 employees", True),
    ]
    for i, (urg, sent, found, conf, text, expected) in enumerate(rows, 1):
        esc, reason = decide_escalation(urgency=urg, sentiment=sent, kb_found=found,
                                        confidence=conf, complaint=text)
        check(esc == expected, f"case {i}: escalate={esc} {('(' + reason + ')') if reason else ''}")


def scripted_client(triage, query, reply, confidence, sources):
    def call(name, args):
        return CreateResult(finish_reason="function_calls", usage=USAGE, cached=False,
                            content=[FunctionCall(id="c1", name=name, arguments=json.dumps(args))])
    return ReplayChatCompletionClient([
        json.dumps(triage),
        call("search_kb", {"query": query, "category": triage["category"]}),
        json.dumps({"reply": reply, "confidence": confidence, "sources_used": sources}),
        call("log_ticket", {**{k: triage[k] for k in ("category", "urgency", "sentiment", "summary")},
                            "response": reply, "confidence": confidence,
                            "risk_flags": ",".join(triage["risk_flags"]) or "none"}),
    ], model_info=INFO)


async def test_team(kb):
    print("\n[3] Full 4-agent team (scripted model)")
    ctx = TicketContext("TKT-0001", "I want to return the shoes I bought last week. How do I get a refund?",
                        "Daniel", "ORD-3301")
    client = scripted_client(
        {"category": "Refund", "urgency": "Low", "sentiment": "Neutral",
         "summary": "Customer wants to return shoes", "risk_flags": [], "language": "English"},
        "return shoes refund window", "Hi Daniel, you can return within 30 days…", "High",
        ["refund_policy.md"])
    out = await run_ticket(ctx, kb, SETTINGS, model_client=client)
    order = [s.agent for s in out.steps if s.kind == "message"]
    check(order == ["query_agent", "knowledge_agent", "resolution_agent", "log_agent"],
          f"agent order {order}")
    check("LOGGED" in out.stop_reason or "LOGGED" in out.steps[-1].content, "terminated on LOGGED")
    check(out.status == "Resolved", f"status Resolved (got {out.status}, reason={out.reason!r})")
    check("refund_policy.md" in out.kb_sources, f"kb sources {out.kb_sources}")

    ctx2 = TicketContext("TKT-0002", "You charged my card three times! Fix this now or I'm "
                         "contacting my bank and a lawyer.", "Kevin", "ORD-5120")
    client2 = scripted_client(
        {"category": "Billing", "urgency": "High", "sentiment": "Angry",
         "summary": "Charged three times, threatens bank and lawyer",
         "risk_flags": ["legal_threat", "payment_dispute"], "language": "English"},
        "duplicate charge card charged three times", "Hi Kevin, a Billing specialist…", "Medium",
        ["billing_and_payments.md"])
    out2 = await run_ticket(ctx2, kb, SETTINGS, model_client=client2)
    check(out2.status == "Escalated", f"billing case Escalated ({out2.reason})")

    df, err = read_log(SETTINGS.log_file)
    check(err is None and len(df) == 2, f"log has 2 rows, columns ok={list(df.columns)[:4]}…")


async def test_safety_net(kb):
    print("\n[4] Safety net when the model fails")
    ctx = TicketContext("TKT-0003", "Where is my order?", "Priya", "ORD-4512")
    broken = ReplayChatCompletionClient([], model_info=INFO)   # no responses -> raises
    out = await run_ticket(ctx, kb, SETTINGS, model_client=broken)
    check(out.status == "Escalated" and out.error, f"escalated with error: {out.error[:70]}…")
    df, _ = read_log(SETTINGS.log_file)
    check("TKT-0003" in set(df["ticket_id"]), "failed ticket still written to the log")


async def main():
    kb = KnowledgeBase(SETTINGS)
    print(f"Knowledge base: {len(kb.document_names)} docs, {len(kb.chunks)} chunks, backend={kb.backend}")
    test_retrieval(kb)
    test_escalation()
    await test_team(kb)
    await test_safety_net(kb)
    print("\nAll offline checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
