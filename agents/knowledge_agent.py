"""
Agent 2 — Knowledge Base Agent (RAG)
Job: search the company documents (FAISS vector store) and return only the
relevant passages, each labelled with its source file name.

reflect_on_tool_use=False means the agent's message IS the tool's output,
so the passages reach the Resolution Agent word for word (no paraphrasing,
no chance to invent a policy).
"""
from typing import Annotated

from autogen_agentchat.agents import AssistantAgent

from agents.context import TicketContext

NAME = "knowledge_agent"
NO_MATCH = "KB_RESULTS: NO_MATCH - no relevant company document was found for this issue."

SYSTEM_MESSAGE = """You are the Knowledge Base Agent for NorthPeak Goods.
Your ONLY job is to call the search_kb tool exactly once.
Build the query from the Query Agent's summary and category (for example:
"refund window and how to return shoes"). Pass the category too.
Do not answer the customer and do not add any text of your own."""


def make_search_kb_tool(ctx: TicketContext, kb):
    """Create the search_kb tool bound to this ticket and knowledge base."""

    def search_kb(
        query: Annotated[str, "Search query built from the complaint summary"],
        category: Annotated[str, "Category from the Query Agent, e.g. Shipping"] = "",
    ) -> str:
        """Search the company knowledge base (FAISS) and return matching passages with their source file."""
        ctx.kb_searched = True
        full_query = f"{category}: {query}" if category else query
        try:
            results = kb.search(full_query)
        except Exception as exc:  # noqa: BLE001 — a search error must not stop the ticket
            ctx.kb_sources = []
            return f"{NO_MATCH} (search error: {exc})"

        if not results:
            ctx.kb_sources = []
            return NO_MATCH

        ctx.kb_sources = sorted({r["source"] for r in results})
        lines = [f"KB_RESULTS: {len(results)} matching passage(s)"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] source: {r['source']} | section: {r['section']} | score: {r['score']}")
            lines.append(r["text"])
        return "\n".join(lines)

    return search_kb


def create_knowledge_agent(model_client, ctx: TicketContext, kb) -> AssistantAgent:
    return AssistantAgent(
        name=NAME,
        description="Searches the company documents with FAISS and returns relevant passages with file names.",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
        tools=[make_search_kb_tool(ctx, kb)],
        reflect_on_tool_use=False,
    )
