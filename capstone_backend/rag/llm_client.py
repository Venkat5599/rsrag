import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import RagConfig

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = {"openai", "gemini", "ollama"}


class LLMUnavailableError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    raw: Dict[str, Any]


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:400]
        raise LLMUnavailableError(f"provider returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise LLMUnavailableError(f"provider unreachable: {error.reason}") from error
    except (TimeoutError, json.JSONDecodeError) as error:
        raise LLMUnavailableError(f"provider response invalid: {error}") from error


class LLMClient:
    def __init__(self, config: RagConfig) -> None:
        self._config = config

    @property
    def provider(self) -> str:
        return self._config.llm_provider

    @property
    def model(self) -> str:
        return self._config.llm_model

    def is_available(self) -> bool:
        provider = self._config.llm_provider

        if provider not in SUPPORTED_PROVIDERS:
            return False
        if provider in {"openai", "gemini"} and not self._config.llm_api_key:
            return False
        if not self._config.llm_base_url:
            return False

        return True

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if not self.is_available():
            raise LLMUnavailableError(
                f"llm provider '{self._config.llm_provider}' is not configured"
            )

        provider = self._config.llm_provider

        if provider == "openai":
            return self._generate_openai(system_prompt, user_prompt)
        if provider == "gemini":
            return self._generate_gemini(system_prompt, user_prompt)
        return self._generate_ollama(system_prompt, user_prompt)

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = f"{self._config.llm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self._config.llm_model,
            "temperature": self._config.llm_temperature,
            "max_tokens": self._config.llm_max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.llm_api_key}",
        }

        data = _post_json(url, payload, headers, self._config.llm_timeout)
        choices = data.get("choices") or []

        if not choices:
            raise LLMUnavailableError("openai response contained no choices")

        text = str(choices[0].get("message", {}).get("content", "")).strip()
        return LLMResponse(text=text, model=self._config.llm_model, provider="openai", raw=data)

    def _generate_gemini(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        base = self._config.llm_base_url.rstrip("/")
        url = f"{base}/models/{self._config.llm_model}:generateContent?key={self._config.llm_api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self._config.llm_temperature,
                "maxOutputTokens": self._config.llm_max_output_tokens,
            },
        }
        headers = {"Content-Type": "application/json"}

        data = _post_json(url, payload, headers, self._config.llm_timeout)
        candidates = data.get("candidates") or []

        if not candidates:
            raise LLMUnavailableError("gemini response contained no candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(str(part.get("text", "")) for part in parts).strip()
        return LLMResponse(text=text, model=self._config.llm_model, provider="gemini", raw=data)

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = f"{self._config.llm_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self._config.llm_model,
            "stream": False,
            "options": {"temperature": self._config.llm_temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Content-Type": "application/json"}

        data = _post_json(url, payload, headers, self._config.llm_timeout)
        text = str(data.get("message", {}).get("content", "")).strip()

        if not text:
            raise LLMUnavailableError("ollama response contained no message content")

        return LLMResponse(text=text, model=self._config.llm_model, provider="ollama", raw=data)


def build_client(config: Optional[RagConfig] = None) -> LLMClient:
    from .config import load_config

    return LLMClient(config or load_config())
