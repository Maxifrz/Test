"""
Schlanker asynchroner Client für einen LOKALEN Ollama-Server (DSGVO: kein
Datenabfluss — OLLAMA_BASE_URL zeigt ins interne Netz, z. B. Compose-Service).

Bewusst minimal gehalten: Embeddings + nicht-streamende Generierung.
"""
from __future__ import annotations

import httpx

from app.core.config import get_settings

DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str | None = None):
        settings = get_settings()
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.llm_model = settings.KI_LLM_MODEL
        self.embed_model = settings.KI_EMBED_MODEL

    async def embed(self, text: str) -> list[float]:
        """Embedding für einen Text (KI_EMBED_MODEL)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
            )
        if resp.status_code != 200:
            raise OllamaError(f"Ollama embeddings HTTP {resp.status_code}: {resp.text[:200]}")
        embedding = resp.json().get("embedding")
        if not embedding:
            raise OllamaError("Ollama lieferte kein Embedding")
        return embedding

    async def generate(self, prompt: str, *, temperature: float = 0.1) -> str:
        """Nicht-streamende Textgenerierung (KI_LLM_MODEL). Niedrige Temperatur
        für faktennahe, quellengebundene Antworten."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
            )
        if resp.status_code != 200:
            raise OllamaError(f"Ollama generate HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json().get("response", "")

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
