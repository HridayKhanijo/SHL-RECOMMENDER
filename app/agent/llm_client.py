import asyncio
import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

def _get_client_and_model():
    from openai import OpenAI
    if os.getenv("GROQ_API_KEY"):
        client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return client, model
    if os.getenv("GEMINI_API_KEY"):
        client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return client, model
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return client, model
    raise EnvironmentError("No LLM API key found. Set GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY.")

def _call_llm_sync(prompt: str, temperature: float = 0.1, max_tokens: int = 1500) -> str:
    client, model = _get_client_and_model()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

async def call_llm(prompt: str, temperature: float = 0.1, max_tokens: int = 1500) -> str:
    return await asyncio.to_thread(_call_llm_sync, prompt, temperature, max_tokens)

def parse_llm_json(raw: str) -> Optional[dict]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    logger.warning(f"Failed to parse LLM JSON. Raw: {raw[:300]}")
    return None