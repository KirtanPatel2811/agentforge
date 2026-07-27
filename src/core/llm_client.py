"""
src/core/llm_client.py — Unified LLM Client
All agent LLM calls go through this single interface.

Design decisions:
1. UNIFIED INTERFACE: Hides provider differences — agents just call llm.complete().
2. AUTOMATIC FALLBACK: Groq fails → transparently retry with Gemini.
3. RETRY WITH BACKOFF: Transient errors retried before triggering fallback.
4. JSON PARSING: parse_json() strips markdown fences and validates JSON.
"""

import json
import re
import time
from typing import Optional
from loguru import logger
from config import settings


class LLMError(Exception):
    pass

class LLMParseError(Exception):
    pass


_groq_client = None
_gemini_client = None


def _get_groq():
    global _groq_client
    if _groq_client is None:
        if not settings.groq_api_key:
            raise LLMError("GROQ_API_KEY not set in .env")
        from groq import Groq
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        if not settings.google_api_key:
            raise LLMError("GOOGLE_API_KEY not set in .env")
        import google.generativeai as genai
        genai.configure(api_key=settings.google_api_key)
        _gemini_client = genai.GenerativeModel(settings.fallback_model)
    return _gemini_client


def _groq_complete(messages, system_prompt=None, max_tokens=None, temperature=None):
    client = _get_groq()
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)
    response = client.chat.completions.create(
        model=settings.primary_model,
        messages=full_messages,
        max_tokens=max_tokens or settings.max_tokens,
        temperature=temperature if temperature is not None else settings.temperature,
        timeout=settings.request_timeout,
    )
    return response.choices[0].message.content


def _gemini_complete(messages, system_prompt=None, max_tokens=None, temperature=None):
    model = _get_gemini()
    history = []
    last_user_message = None
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        if msg == messages[-1] and role == "user":
            last_user_message = msg["content"]
        else:
            history.append({"role": role, "parts": [msg["content"]]})
    if last_user_message is None:
        last_user_message = messages[-1]["content"]
    if system_prompt and not history:
        last_user_message = f"{system_prompt}\n\n{last_user_message}"
    elif system_prompt:
        history[0]["parts"][0] = f"{system_prompt}\n\n{history[0]['parts'][0]}"
    chat = model.start_chat(history=history)
    response = chat.send_message(last_user_message)
    return response.text


class LLMClient:
    """
    Unified LLM interface for all agents.
    Usage:
        client = LLMClient(system_prompt="You are a research agent...")
        response = client.complete([{"role": "user", "content": "Find X"}])
        data = client.complete_json([{"role": "user", "content": "Return JSON..."}])
    """

    def __init__(self, agent_name: str = "agent", system_prompt: Optional[str] = None):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self._call_count = 0
        self._groq_failures = 0

    def complete(self, messages, max_tokens=None, temperature=None, force_fallback=False):
        self._call_count += 1
        kwargs = dict(messages=messages, system_prompt=self.system_prompt,
                      max_tokens=max_tokens, temperature=temperature)

        if not force_fallback and settings.groq_api_key:
            for attempt in range(settings.max_retries):
                try:
                    logger.debug(f"[{self.agent_name}] Groq call #{self._call_count} attempt {attempt+1}")
                    result = _groq_complete(**kwargs)
                    logger.debug(f"[{self.agent_name}] Groq OK — {len(result)} chars")
                    return result
                except Exception as e:
                    err_str = str(e).lower()
                    logger.warning(f"[{self.agent_name}] Groq attempt {attempt+1} failed: {e}")
                    if "rate_limit" in err_str or "429" in err_str:
                        wait = settings.retry_delay * (2 ** attempt)
                        logger.info(f"[{self.agent_name}] Rate limited, waiting {wait}s")
                        time.sleep(wait)
                    elif attempt < settings.max_retries - 1:
                        time.sleep(settings.retry_delay)
                    else:
                        logger.warning(f"[{self.agent_name}] Groq exhausted — falling back to Gemini")
                        self._groq_failures += 1

        if settings.google_api_key:
            for attempt in range(settings.max_retries):
                try:
                    logger.debug(f"[{self.agent_name}] Gemini fallback attempt {attempt+1}")
                    result = _gemini_complete(**kwargs)
                    logger.debug(f"[{self.agent_name}] Gemini OK — {len(result)} chars")
                    return result
                except Exception as e:
                    logger.warning(f"[{self.agent_name}] Gemini attempt {attempt+1} failed: {e}")
                    if attempt < settings.max_retries - 1:
                        time.sleep(settings.retry_delay)

        raise LLMError(
            f"[{self.agent_name}] All LLM providers failed. "
            "Check your API keys in .env and your network connection."
        )

    def complete_json(self, messages, max_tokens=None, temperature=None):
        raw = self.complete(messages, max_tokens=max_tokens, temperature=temperature)
        return parse_json(raw, agent_name=self.agent_name)

    def simple(self, prompt: str, **kwargs) -> str:
        return self.complete([{"role": "user", "content": prompt}], **kwargs)

    def simple_json(self, prompt: str, **kwargs) -> dict:
        return self.complete_json([{"role": "user", "content": prompt}], **kwargs)

    @property
    def stats(self):
        return {"agent": self.agent_name, "total_calls": self._call_count,
                "groq_failures": self._groq_failures}


def parse_json(text: str, agent_name: str = "agent") -> dict:
    """
    Extract and parse JSON from LLM output.
    Handles markdown fences, preambles, and raw JSON.
    """
    if not text or not text.strip():
        raise LLMParseError(f"[{agent_name}] LLM returned empty response")
    stripped = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", stripped)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        raise LLMParseError(
            f"[{agent_name}] Could not parse JSON.\n"
            f"Error: {e}\nRaw output: {text[:500]}"
        )


llm = LLMClient(agent_name="default")


if __name__ == "__main__":
    from rich import print as rprint
    rprint("[bold cyan]Testing LLM Client...[/bold cyan]")
    client = LLMClient(agent_name="test", system_prompt="You are a helpful assistant. Be brief.")
    response = client.simple("Say exactly: 'AgentForge LLM client works!'")
    rprint(f"Response: {response}")
    json_response = client.simple_json(
        "Return a JSON object with keys 'status' (value: 'ok') and 'agent' (value: 'test'). "
        "Return ONLY the JSON object, no other text."
    )
    rprint(f"Parsed JSON: {json_response}")
    rprint(f"Stats: {client.stats}")
    rprint("[bold green]✓ LLM Client test passed![/bold green]")
