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
            return await self._embed_with(client, text)

    async def _embed_with(self, client: httpx.AsyncClient, text: str) -> list[float]:
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

    async def embed_many(self, texts: list[str], *, concurrency: int = 4) -> list[list[float]]:
        """
        Embeddings für mehrere Texte über EINE HTTP-Verbindung, begrenzt
        nebenläufig.

        Vorher wurde je Chunk ein eigener AsyncClient (und damit eine eigene
        TCP-/TLS-Verbindung) aufgebaut und die Anfragen liefen streng
        sequenziell — ein ganzes Gesetz bedeutete hunderte Verbindungsaufbauten.
        Die Reihenfolge bleibt erhalten, weil die Chunk-Zuordnung daran hängt.
        """
        if not texts:
            return []
        import asyncio

        semaphore = asyncio.Semaphore(max(1, concurrency))
        limits = httpx.Limits(max_connections=max(1, concurrency))
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, limits=limits) as client:
            async def one(text: str) -> list[float]:
                async with semaphore:
                    return await self._embed_with(client, text)

            return list(await asyncio.gather(*(one(t) for t in texts)))

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
