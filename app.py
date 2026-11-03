"""
app.py — Streamlit UI for the AI Customer Support Desk.
Run with:   streamlit run app.py
"""
from __future__ import annotations

import asyncio
import html

import pandas as pd
import streamlit as st

from agents import TicketContext
from config import ConfigError, get_settings
from rag import KnowledgeBase, KnowledgeBaseError
from team import AGENT_LABELS, Step, run_ticket
from ticket_logger import next_ticket_id, read_log, to_excel_bytes

MAX_COMPLAINT_CHARS = 2000

SAMPLE_COMPLAINTS = {
    "— choose a sample complaint —": ("", "", ""),
    "1 · Late order": ("Priya", "ORD-4512", "Where is my order? It was supposed to arrive yesterday."),
    "2 · Return shoes": ("Daniel", "ORD-3301", "I want to return the shoes I bought last week. How do I get a refund?"),
    "3 · Forgot password": ("Aisha", "", "I forgot my password and can't log in."),
    "4 · Broken blender": ("Marco", "ORD-2219", "The blender I received is broken. Can I get a replacement?"),
    "5 · Charged three times": ("Kevin", "ORD-5120", "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer."),
    "6 · Corporate gift cards": ("Linda", "", "Do you sell gift cards for a company of 500 employees with custom branding?"),
    "7 · (custom) Change address": ("Sam", "ORD-6001", "I just placed an order but typed the wrong street. Can I change the shipping address?"),
    "8 · (custom) Spanish – package marked delivered": ("Lucía", "ORD-7788", "El seguimiento dice que mi paquete fue entregado, pero no lo he recibido."),
}

st.set_page_config(page_title="AI Support Desk", page_icon="🎧", layout="wide")

st.markdown("""
<style>
.badge {display:inline-block;padding:4px 14px;border-radius:999px;font-weight:700;font-size:0.95rem;}
.badge-resolved {background:#16a34a;color:#fff;}
.badge-escalated {background:#dc2626;color:#fff;}
.reply-box {border-left:6px solid #2563eb;background:rgba(37,99,235,0.08);
            padding:16px 20px;border-radius:8px;white-space:pre-wrap;line-height:1.55;}
.meta {color:#6b7280;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Startup: settings + knowledge base (cached so FAISS is built only once)
# --------------------------------------------------------------------------
try:
    settings = get_settings()
except ConfigError as exc:
    st.error(f"⚙️ Configuration problem: {exc}")
    st.stop()


@st.cache_resource(show_spinner="Building the FAISS knowledge index…")
def load_kb(version: int) -> KnowledgeBase:  # version busts the cache on "Rebuild"
    return KnowledgeBase(settings)


if "kb_version" not in st.session_state:
    st.session_state.kb_version = 0

kb, kb_error = None, None
try:
    kb = load_kb(st.session_state.kb_version)
except KnowledgeBaseError as exc:
    kb_error = str(exc)
except Exception as exc:  # noqa: BLE001
    kb_error = f"Unexpected error while loading the knowledge base: {exc}"


# --------------------------------------------------------------------------
# Sidebar: health checks
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🩺 System status")
    st.write(("✅" if settings.has_api_key else "❌") + " API key " +
             ("found" if settings.has_api_key else "missing — add it to .env"))
    st.write(f"🤖 Model: `{settings.model}`")
    if kb:
        st.write(f"📚 {len(kb.document_names)} documents · {len(kb.chunks)} chunks")
        st.write(f"🧭 FAISS backend: `{kb.backend}` (min similarity {kb.min_similarity})")
        for w in kb.warnings:
            st.warning(w)
        with st.expander("Documents in the knowledge base"):
            for name in kb.document_names:
                st.write(f"• {name}")
    else:
        st.error(f"📚 Knowledge base error: {kb_error}")

    if st.button("🔄 Rebuild knowledge index", use_container_width=True,
                 help="Use after you edit or add files in knowledge_base/"):
        st.session_state.kb_version += 1
        load_kb.clear()
        st.rerun()

    st.divider()
    st.caption("Agents run in order: Query → Knowledge → Resolution → Log")


# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------
st.title("🎧 AI Customer Support Desk")
st.caption("NorthPeak Goods · 4 AutoGen agents · FAISS RAG over company documents")

tab_new, tab_log, tab_dash = st.tabs(["📝 New ticket", "📒 Ticket log", "📊 Dashboard"])


def render_step(container, step: Step) -> None:
    label = AGENT_LABELS.get(step.agent, step.agent)
    icon = {"tool_call": "🔧", "tool_result": "📄", "message": "💬"}[step.kind]
    title = {"tool_call": "calls a tool", "tool_result": "tool result",
             "message": "output"}[step.kind]
    with container.expander(f"{icon} {label} — {title}", expanded=step.kind == "message"):
        text = step.content
        if text.strip().startswith("{"):
            st.code(text, language="json")
        else:
            st.text(text)


def render_outcome(outcome) -> None:
    badge = "badge-resolved" if outcome.status == "Resolved" else "badge-escalated"
    icon = "✅" if outcome.status == "Resolved" else "🚨"
    st.markdown(f"### Result for {outcome.ticket_id} "
                f"<span class='badge {badge}'>{icon} {outcome.status}</span>",
                unsafe_allow_html=True)
    if outcome.error:
        st.error(f"⚠️ {outcome.error}")
    if outcome.status == "Escalated":
        st.markdown(f"<p class='meta'><b>Escalation reason:</b> "
                    f"{html.escape(outcome.reason or 'n/a')}</p>", unsafe_allow_html=True)

    t = outcome.triage or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Category", t.get("category", "—"))
    c2.metric("Urgency", t.get("urgency", "—"))
    c3.metric("Sentiment", t.get("sentiment", "—"))
    c4.metric("Confidence", (outcome.resolution or {}).get("confidence", "—"))

    st.markdown("**Reply to customer**")
    reply = outcome.reply or "_No reply was generated — a human agent will respond._"
    st.markdown(f"<div class='reply-box'>{html.escape(reply)}</div>", unsafe_allow_html=True)
    st.markdown(f"<p class='meta'>📚 Sources: {', '.join(outcome.kb_sources) or 'none found'}</p>",
                unsafe_allow_html=True)


with tab_new:
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        sample = st.selectbox("Try a test complaint", list(SAMPLE_COMPLAINTS))
        s_name, s_order, s_text = SAMPLE_COMPLAINTS[sample]
        with st.form("ticket_form", clear_on_submit=False):
            name = st.text_input("Customer name (optional)", value=s_name)
            order = st.text_input("Order ID (optional)", value=s_order, placeholder="ORD-1234")
            complaint = st.text_area("Complaint", value=s_text, height=170,
                                     max_chars=MAX_COMPLAINT_CHARS,
                                     placeholder="Describe the customer's problem…")
            submitted = st.form_submit_button("🚀 Submit ticket", use_container_width=True,
                                              type="primary")

    with right:
        if submitted:
            # ---- input validation ----
            if not complaint or len(complaint.strip()) < 5:
                st.warning("Please type a complaint of at least a few words.")
            elif kb is None:
                st.error(f"Cannot run: the knowledge base is not available ({kb_error}).")
            elif not settings.has_api_key:
                st.error("Cannot run: OPENAI_API_KEY is missing. Add it to your .env file "
                         "and restart the app.")
            else:
                ctx = TicketContext(ticket_id=next_ticket_id(settings.log_file),
                                    complaint=complaint.strip()[:MAX_COMPLAINT_CHARS],
                                    customer_name=name.strip(), order_id=order.strip())
                st.subheader(f"Agent timeline · {ctx.ticket_id}")
                timeline = st.container()
                with st.spinner("Agents are working on the ticket…"):
                    outcome = asyncio.run(run_ticket(
                        ctx, kb, settings, on_step=lambda s: render_step(timeline, s)))
                st.session_state.last_outcome = outcome
                st.divider()
                render_outcome(outcome)
        elif "last_outcome" in st.session_state:
            o = st.session_state.last_outcome
            st.subheader(f"Agent timeline · {o.ticket_id}")
            box = st.container()
            for step in o.steps:
                render_step(box, step)
            st.divider()
            render_outcome(o)
        else:
            st.info("Pick a sample or type a complaint, then press **Submit ticket**. "
                    "Each agent's work will appear here step by step.")


with tab_log:
    df, err = read_log(settings.log_file)
    if err:
        st.error(err)
    if df.empty:
        st.info("No tickets yet. Submit one from the **New ticket** tab.")
    else:
        status_filter = st.multiselect("Filter by status", ["Resolved", "Escalated"],
                                       default=["Resolved", "Escalated"])
        view = df[df["status"].isin(status_filter)].iloc[::-1]  # newest first
        st.dataframe(view, use_container_width=True, hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
                           "tickets.csv", "text/csv", use_container_width=True)
        try:
            d2.download_button("⬇️ Download Excel", to_excel_bytes(df), "tickets.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            d2.warning(f"Excel export unavailable: {exc}")


with tab_dash:
    df, _ = read_log(settings.log_file)
    if df.empty:
        st.info("The dashboard appears once tickets have been logged.")
    else:
        total = len(df)
        escalated = int((df["status"] == "Escalated").sum())
        urgency_score = df["urgency"].map({"Low": 1, "Medium": 2, "High": 3}).dropna()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total tickets", total)
        m2.metric("Resolved", total - escalated)
        m3.metric("Escalated", escalated, f"{escalated / total:.0%} of tickets", delta_color="off")
        m4.metric("Avg urgency (1–3)", f"{urgency_score.mean():.2f}" if len(urgency_score) else "—")
        g1, g2 = st.columns(2)
        g1.markdown("**Tickets per category**")
        g1.bar_chart(df["category"].value_counts())
        g2.markdown("**Resolved vs escalated**")
        g2.bar_chart(df["status"].value_counts())
        st.markdown("**Most common escalation reasons**")
        reasons = (df.loc[df["status"] == "Escalated", "escalation_reason"]
                   .str.split(r" \+ ").explode().str.strip())
        reasons = reasons[reasons != ""].value_counts()
        st.dataframe(pd.DataFrame({"reason": reasons.index, "tickets": reasons.values}),
                     hide_index=True, use_container_width=True)
