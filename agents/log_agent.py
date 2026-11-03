"""
Agent 4 — Log Agent
Job: call log_ticket to save the ticket and decide Resolved vs Escalated.

The tool returns a line starting with "LOGGED", which is the word the team's
TextMentionTermination listens for. Because reflect_on_tool_use=False, that
line becomes the agent's final message and the conversation stops cleanly.
"""
from typing import Annotated

from autogen_agentchat.agents import AssistantAgent

from agents.context import TicketContext
from escalation import decide_escalation
from ticket_logger import append_ticket

NAME = "log_agent"

SYSTEM_MESSAGE = """You are the Log Agent for NorthPeak Goods customer support.
Your ONLY job is to call the log_ticket tool exactly once, using values from the conversation:
- category, urgency, sentiment, summary and risk_flags from the query_agent JSON
- response = the "reply" text from the resolution_agent JSON, copied exactly
- confidence from the resolution_agent JSON
- escalate_suggestion = "yes" if you think a human should take over, otherwise "no"
Do not write anything else. The tool decides the final status."""


def make_log_ticket_tool(ctx: TicketContext, log_file):
    def log_ticket(
        category: Annotated[str, "Shipping, Refund, Account, Product, Billing or Other"],
        urgency: Annotated[str, "Low, Medium or High"],
        sentiment: Annotated[str, "Positive, Neutral, Frustrated or Angry"],
        summary: Annotated[str, "One-line summary of the problem"],
        response: Annotated[str, "The exact reply written by the Resolution Agent"],
        confidence: Annotated[str, "High, Medium or Low (from the Resolution Agent)"],
        risk_flags: Annotated[str, "Comma-separated: legal_threat, fraud, payment_dispute, or none"] = "none",
        escalate_suggestion: Annotated[str, "yes or no"] = "no",
    ) -> str:
        """Save the ticket to the log file and decide whether it must be escalated to a human."""
        escalate, reason = decide_escalation(
            urgency=urgency, sentiment=sentiment, kb_found=ctx.kb_found,
            confidence=confidence, complaint=ctx.complaint, risk_flags=risk_flags,
        )
        if escalate_suggestion.strip().lower() == "yes" and not escalate:
            escalate, reason = True, "Log Agent judged the case complex"

        row = {
            "ticket_id": ctx.ticket_id,
            "customer_name": ctx.customer_name,
            "order_id": ctx.order_id,
            "complaint": ctx.complaint,
            "category": category, "urgency": urgency, "sentiment": sentiment,
            "summary": summary,
            "kb_sources": ", ".join(ctx.kb_sources) if ctx.kb_sources else "none",
            "response": response, "confidence": confidence,
            "status": "Escalated" if escalate else "Resolved",
            "escalation_reason": reason,
        }
        try:
            written_to = append_ticket(log_file, row)
        except Exception as exc:  # noqa: BLE001
            # Still terminate cleanly; team.py will show the error and retry the write.
            return f"LOGGED | ticket_id={ctx.ticket_id} | status=ERROR | could not write log: {exc}"

        ctx.logged, ctx.log_row, ctx.log_file_used = True, row, str(written_to)
        return (f"LOGGED | ticket_id={ctx.ticket_id} | status={row['status']}"
                f"{' | reason=' + reason if reason else ''}")

    return log_ticket


def create_log_agent(model_client, ctx: TicketContext, log_file) -> AssistantAgent:
    return AssistantAgent(
        name=NAME,
        description="Records the ticket to the log file and escalates complex cases to human support.",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
        tools=[make_log_ticket_tool(ctx, log_file)],
        reflect_on_tool_use=False,
    )
