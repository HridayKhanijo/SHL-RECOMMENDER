"""
app/agent/chat_agent.py — Main agent orchestrator.

WHY: This is the brain of the system. It decides:
  1. What the user wants (intent classification)
  2. What catalog data to retrieve
  3. Which prompt template to use
  4. How to validate and format the response

DESIGN DECISION: We use a simple intent → action routing pattern rather
than a multi-agent framework (no LangGraph, no LangChain agents). This
is intentional:
  - Fewer moving parts = easier to debug and defend in interviews
  - Deterministic control flow = more reliable schema compliance
  - LangChain overhead would slow responses toward the 30s limit

ARCHITECTURE:
  Intent Classifier (LLM, fast small call)
       ↓
  [clarify | recommend | refine | compare | off_topic | injection]
       ↓
  Retriever (FAISS + BM25)
       ↓
  Prompt Builder
       ↓
  LLM (main call)
       ↓
  Response Validator + Schema Enforcer
"""

import json
import logging
import re
from typing import Optional

from app.api.schemas import Message, Recommendation
from app.agent.llm_client import call_llm, parse_llm_json
from app.agent.prompts import (
    build_clarify_prompt,
    build_compare_prompt,
    build_intent_classifier_prompt,
    build_recommend_prompt,
    build_refuse_prompt,
    build_refine_prompt,
)
from app.retrieval.retriever import retrieve
from app.core.index import get_by_name

logger = logging.getLogger(__name__)

# Maximum conversation turns — hard cap per assignment spec
MAX_TURNS = 8

# How many catalog candidates we pass to the LLM
# (more = better recall; fewer = shorter prompt = faster + cheaper)
CATALOG_WINDOW = 12


def _format_history(messages: list[Message]) -> str:
    """Format conversation history for prompt injection."""
    lines = []
    for m in messages:
        prefix = "User" if m.role == "user" else "Assistant"
        lines.append(f"{prefix}: {m.content}")
    return "\n".join(lines)


def _extract_query_from_history(messages: list[Message]) -> str:
    """
    Summarise the user's requirements from conversation history.
    We concatenate all user messages — this gives the retriever the
    richest possible query without a separate summarisation LLM call.
    """
    user_texts = [m.content for m in messages if m.role == "user"]
    return " ".join(user_texts)


def _extract_prior_recommendations(messages: list[Message]) -> list[dict]:
    """
    Parse any recommendations the agent previously returned.
    These are embedded in assistant messages as JSON blobs if we
    store them there, OR we can re-derive them.

    DESIGN DECISION: We re-retrieve rather than parse history to avoid
    brittle JSON-in-history patterns. The refine path gets a fresh query
    that includes the full conversation context.
    """
    # Check for structured recommendation markers in assistant messages
    for m in reversed(messages):
        if m.role == "assistant" and '"url"' in m.content:
            try:
                data = json.loads(m.content)
                return data.get("recommendations", [])
            except Exception:
                pass
    return []


def _has_prior_recommendations(messages: list[Message]) -> bool:
    prior = _extract_prior_recommendations(messages)
    return len(prior) > 0


def _detect_injection(text: str) -> bool:
    """
    Heuristic prompt injection detection.

    DESIGN DECISION: Pattern-matching is fast and catches obvious attacks.
    The LLM's system prompt handles subtler cases by grounding responses
    in catalog data (injected instructions can't invent real URLs).
    """
    injection_patterns = [
        r"ignore (previous|all|above|prior) instructions?",
        r"you are now",
        r"pretend (you are|to be)",
        r"disregard (your|the) (system|instructions?|rules?)",
        r"reveal (your|the) (prompt|system|instructions?)",
        r"jailbreak",
        r"DAN mode",
        r"act as if",
        r"forget (all|your|previous)",
        r"new (persona|role|identity)",
    ]
    text_lower = text.lower()
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            logger.warning(f"Injection pattern detected: {pattern}")
            return True
    return False


def _detect_off_topic(text: str) -> bool:
    """
    Heuristic off-topic detection for fast-path refusal.
    The LLM classifier also catches these, but fast refusal saves latency.
    """
    off_topic_signals = [
        r"\b(salary|compensation|benefits|pay scale)\b",
        r"\b(legal advice|lawsuit|sue|EEOC|discrimination case)\b",
        r"\b(stock price|investment|buy shares)\b",
        r"\b(recipe|cooking|food)\b",
        r"\b(weather|sports score|news)\b",
    ]
    text_lower = text.lower()
    for pattern in off_topic_signals:
        if re.search(pattern, text_lower):
            return True
    return False


def _extract_comparison_names(messages: list[Message]) -> tuple[Optional[str], Optional[str]]:
    """
    Extract two assessment names from user message for comparison queries.
    Handles patterns like:
      - "compare OPQ and Verify G+"
      - "what's the difference between SHL Verify and MQ"
      - "OPQ vs GSA"
    """
    last_user = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )
    # Pattern: "compare A and B" or "A vs B" or "between A and B"
    patterns = [
        r"compare\s+(.+?)\s+(?:and|with|vs\.?)\s+(.+?)(?:\?|$)",
        r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
        r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"difference.*?between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, last_user, re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None, None


def _extract_required_types(messages: list[Message]) -> list[str]:
    """
    Detect test type preferences from conversation.
    Handles: "personality test", "ability assessment", "knowledge test", etc.
    """
    type_keywords = {
        "P": ["personality", "behaviour", "behavioral", "opq", "trait"],
        "A": ["ability", "aptitude", "cognitive", "reasoning", "numerical", "verbal"],
        "K": ["knowledge", "skill", "technical", "java", "python", "coding"],
        "B": ["situational judgement", "sjt", "biodata"],
        "M": ["motivation", "motivational", "career"],
        "S": ["simulation", "exercise"],
        "C": ["competency", "competencies"],
    }
    found = []
    full_text = " ".join(m.content for m in messages).lower()
    for code, keywords in type_keywords.items():
        if any(kw in full_text for kw in keywords):
            found.append(code)
    return found


def _catalog_to_prompt_text(items: list[dict]) -> str:
    """Format catalog items for inclusion in LLM prompt."""
    lines = []
    for i, item in enumerate(items, 1):
        lines.append(
            f"{i}. Name: {item['name']}\n"
            f"   URL: {item['url']}\n"
            f"   Type: {item.get('test_type_label', item.get('test_type', ''))}\n"
            f"   Description: {item.get('description', 'N/A')[:200]}"
        )
    return "\n\n".join(lines)


def _validate_recommendations(
    raw_recs: list[dict],
) -> list[Recommendation]:
    """
    Validate LLM-generated recommendations against catalog.

    CRITICAL: This is our hallucination firewall. We only pass through
    recommendations whose URLs exist in the scraped catalog.

    DESIGN DECISION: We log rejected items so we can diagnose prompt
    failures in evaluation runs.
    """
    from app.core.index import get_all_metadata
    valid_urls = {item["url"] for item in get_all_metadata()}

    validated = []
    for rec in raw_recs:
        url = rec.get("url", "")
        name = rec.get("name", "")
        test_type = rec.get("test_type", "K")

        if not url or not name:
            logger.warning(f"Recommendation missing name/url: {rec}")
            continue

        if url not in valid_urls:
            logger.warning(f"Hallucinated URL rejected: {url} ({name})")
            # Try to find the real URL for this name
            catalog_item = get_by_name(name)
            if catalog_item:
                url = catalog_item["url"]
                test_type = catalog_item.get("test_type", test_type)
                logger.info(f"Corrected URL for '{name}' → {url}")
            else:
                continue  # drop this recommendation

        validated.append(
            Recommendation(name=name, url=url, test_type=test_type)
        )
        if len(validated) >= 10:
            break

    return validated


def _safe_fallback_response(
    messages: list[Message],
) -> tuple[str, list[Recommendation], bool]:
    """
    Rule-based fallback when the LLM fails or returns invalid JSON.
    Returns the top catalog items without any LLM involvement.
    """
    query = _extract_query_from_history(messages)
    items = retrieve(query, top_k=5)
    recs = [
        Recommendation(
            name=i["name"],
            url=i["url"],
            test_type=i.get("test_type", "K"),
        )
        for i in items
    ]
    reply = (
        "Here are some SHL assessments that may match your needs. "
        "Let me know if you'd like to refine these further."
    )
    return reply, recs, False


async def _classify_intent(messages: list[Message]) -> str:
    """
    Use a fast LLM call to classify user intent.

    DESIGN DECISION: Separate intent classification call keeps the main
    LLM prompt focused and avoids "do everything" mega-prompts that
    degrade instruction-following.

    TRADEOFF: Extra LLM call adds ~0.5s latency. Acceptable given 30s budget.
    For ultra-low latency, replace with a fine-tuned BERT classifier.
    """
    last_user = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )

    # Fast heuristic checks before LLM classification
    if _detect_injection(last_user):
        return "injection"
    if _detect_off_topic(last_user):
        return "off_topic"

    history = _format_history(messages)
    has_prior = _has_prior_recommendations(messages)

    prompt = build_intent_classifier_prompt(last_user, history, has_prior)
    raw = await call_llm(prompt, temperature=0.0, max_tokens=200)
    parsed = parse_llm_json(raw)

    if parsed and "intent" in parsed:
        intent = parsed["intent"]
        logger.info(f"Intent: {intent} (conf={parsed.get('confidence', '?')})")
        return intent

    # Fallback: simple heuristics
    user_lower = last_user.lower()
    if any(w in user_lower for w in ["compare", "vs", "versus", "difference between"]):
        return "compare"
    if any(w in user_lower for w in ["also add", "actually", "remove", "change", "instead"]):
        return "refine"
    if len(last_user.split()) > 8:  # rich enough to recommend
        return "recommend"
    return "clarify_needed"


async def run_agent(
    messages: list[Message],
) -> tuple[str, list[Recommendation], bool]:
    """
    Main agent entry point.

    Args:
        messages: Full conversation history (stateless).

    Returns:
        (reply, recommendations, end_of_conversation)
    """
    # --- Turn cap guard ---
    turn_count = len(messages)
    if turn_count >= MAX_TURNS:
        logger.info("Turn cap reached — forcing final recommendation.")
        query = _extract_query_from_history(messages)
        req_types = _extract_required_types(messages)
        items = retrieve(query, top_k=10, required_types=req_types)
        recs = _validate_recommendations(
            [{"name": i["name"], "url": i["url"], "test_type": i.get("test_type", "K")}
             for i in items]
        )
        return (
            "Based on our conversation, here are my final assessment recommendations:",
            recs,
            True,
        )

    # --- Classify intent ---
    intent = await _classify_intent(messages)
    history_str = _format_history(messages)
    query = _extract_query_from_history(messages)
    req_types = _extract_required_types(messages)

    # --- Route to action ---
    if intent == "injection":
        logger.warning("Prompt injection attempt detected.")
        raw = await call_llm(
            build_refuse_prompt(history_str, "prompt injection attempt"),
            temperature=0.0,
        )
        parsed = parse_llm_json(raw) or {}
        return parsed.get("reply", "I can only help with SHL assessment selection."), [], False

    if intent == "off_topic":
        raw = await call_llm(
            build_refuse_prompt(history_str, "off-topic question"),
            temperature=0.0,
        )
        parsed = parse_llm_json(raw) or {}
        return (
            parsed.get(
                "reply",
                "I can only assist with SHL assessment recommendations. "
                "What role are you hiring for?",
            ),
            [],
            False,
        )

    if intent == "clarify_needed":
        raw = await call_llm(
            build_clarify_prompt(history_str),
            temperature=0.1,
        )
        parsed = parse_llm_json(raw) or {}
        return parsed.get("reply", "Could you tell me more about the role?"), [], False

    if intent == "compare":
        name_a, name_b = _extract_comparison_names(messages)
        item_a = get_by_name(name_a) if name_a else None
        item_b = get_by_name(name_b) if name_b else None

        if not item_a or not item_b:
            # Fall back to retrieval if we can't find exact names
            items = retrieve(query, top_k=5)
            if len(items) >= 2:
                item_a, item_b = items[0], items[1]
            else:
                return (
                    "I couldn't find those specific assessments to compare. "
                    "Could you provide the exact names?",
                    [],
                    False,
                )

        raw = await call_llm(
            build_compare_prompt(
                history_str,
                json.dumps(item_a, indent=2),
                json.dumps(item_b, indent=2),
            )
        )
        parsed = parse_llm_json(raw) or {}
        recs = _validate_recommendations(parsed.get("recommendations", []))
        return parsed.get("reply", ""), recs, parsed.get("end_of_conversation", False)

    # recommend or refine
    items = retrieve(query, top_k=CATALOG_WINDOW, required_types=req_types)
    catalog_text = _catalog_to_prompt_text(items)

    if intent == "refine":
        last_user_msg = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        prior_recs = _extract_prior_recommendations(messages)
        raw = await call_llm(
            build_refine_prompt(
                history_str,
                catalog_text,
                json.dumps(prior_recs, indent=2),
                last_user_msg,
            )
        )
    else:  # recommend
        raw = await call_llm(
            build_recommend_prompt(
                history_str,
                catalog_text,
                query,
            )
        )

    parsed = parse_llm_json(raw)
    if not parsed:
        logger.warning("LLM returned unparseable JSON — using fallback.")
        return _safe_fallback_response(messages)

    raw_recs = parsed.get("recommendations", [])
    recs = _validate_recommendations(raw_recs)

    # If validation rejected all recs (hallucination), use retrieval fallback
    if raw_recs and not recs:
        logger.warning("All recommendations rejected — using retrieval fallback.")
        recs = [
            Recommendation(
                name=i["name"],
                url=i["url"],
                test_type=i.get("test_type", "K"),
            )
            for i in items[:5]
        ]

    reply = parsed.get("reply", "")
    eoc = parsed.get("end_of_conversation", False) and bool(recs)

    return reply, recs, eoc
