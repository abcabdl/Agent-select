from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Iterable, Optional

import httpx


def _first_env(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


class LLMClient:
    """Minimal OpenAI-compatible chat client using httpx."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 60.0,
        organization: Optional[str] = None,
        max_retries: int = 2,
        retry_backoff_s: float = 2.0,
    ) -> None:
        self.api_key = api_key or _first_env("LLM_API_KEY", "OPENAI_API_KEY")
        if self.api_key:
            self.api_key = self.api_key.strip("'\"")
        self.base_url = base_url or _first_env(
            "LLM_API_BASE",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            default="https://api.openai.com/v1",
        )
        if self.base_url:
            self.base_url = self.base_url.strip("'\"")
        self.model = model or _first_env("LLM_MODEL", "OPENAI_MODEL", default="gpt-4o-mini")
        self.timeout_s = timeout_s
        self.organization = organization or _first_env("OPENAI_ORG")
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

    def chat(
        self,
        messages: Iterable[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY/OPENAI_API_KEY is not set")
        base = (self.base_url or "").rstrip("/")
        if base.endswith("/chat/completions"):
            url = base
        elif base.endswith("/responses"):
            url = base
        else:
            url = f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        sys.stderr.write(f"[LLMClient] Requesting {url} (model={self.model})\n")

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                break
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                sys.stderr.write(f"[LLMClient] Error (attempt {attempt+1}): {exc}\n")
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
                     # Attempt to fix double path if any
                     pass
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_backoff_s * (attempt + 1))
        else:
            if last_exc:
                raise last_exc
            raise RuntimeError("LLM request failed without response")
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"Unexpected LLM response: {json.dumps(data)[:500]}")
