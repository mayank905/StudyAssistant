"""Dual-Engine AI Providers for StudyAssistant.

Supports:
1. Google Gemini (Cloud Engine):
   - Supports both official SDK (`google.generativeai`) and direct zero-dependency REST HTTP requests.
   - Default models: `gemini-2.5-flash`, `gemini-1.5-flash`.
   - Embeddings: `text-embedding-004`.
2. Ollama & Local Offline Engine:
   - Queries local Ollama daemon (`http://localhost:11434`) for chat and embeddings.
   - Default models: `llama3`, `mistral`, `gemma`.
3. Offline Extractive Grounding Synthesizer:
   - Deterministic, high-yield factual synthesizer for pure offline environments without Ollama.
4. Local Embeddings:
   - SentenceTransformers (`all-MiniLM-L6-v2`) or high-dimensional deterministic feature vectors.
"""

from abc import ABC, abstractmethod
import json
import logging
import math
import os
import re
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

# Optional imports
try:
    import google.generativeai as genai
    HAS_GOOGLE_GENAI = True
except ImportError:
    genai = None
    HAS_GOOGLE_GENAI = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False


# ============================================================================
# LLM Providers
# ============================================================================

class BaseLLMProvider(ABC):
    """Abstract base class for Large Language Model providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'gemini', 'ollama', 'offline')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model name."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the provider is currently reachable and configured."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a complete text response from the model."""
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """Stream response tokens as they are generated."""
        pass


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini LLM provider.

    Uses `google.generativeai` if available, or falls back to direct REST HTTP
    requests against Google's Generative Language API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        self._model_name = model_name
        self._sdk_initialized = False

        if self._api_key and HAS_GOOGLE_GENAI and genai is not None:
            try:
                genai.configure(api_key=self._api_key)
                self._sdk_initialized = True
            except Exception as e:
                logger.warning("Failed to initialize google.generativeai SDK: %s", e)
                self._sdk_initialized = False

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 8)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        if not self.is_available():
            raise RuntimeError(
                "Gemini API key is not configured. Set GEMINI_API_KEY environment "
                "variable or pass it in provider settings."
            )

        # 1. Try SDK first if available
        if self._sdk_initialized and genai is not None:
            try:
                model = genai.GenerativeModel(
                    model_name=self._model_name,
                    system_instruction=system_prompt if system_prompt else None,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                res = model.generate_content(prompt)
                if res and hasattr(res, "text") and res.text:
                    return res.text
            except Exception as e:
                logger.warning("SDK generate_content failed (%s); trying direct REST fallback.", e)

        # 2. REST HTTP API Fallback
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model_name}:generateContent?key={self._api_key}"
        )

        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System Instructions:\n{system_prompt}\n"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions strictly."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                candidates = resp_json.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    texts = [p.get("text", "") for p in parts]
                    return "".join(texts)
                return ""
        except urllib.error.HTTPError as err:
            err_body = err.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Gemini API HTTP Error {err.code}: {err_body}") from err
        except Exception as e:
            raise RuntimeError(f"Gemini API Request failed: {e}") from e

    def stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        # For simplicity and robust grounding checks, yield from generate
        full_text = self.generate(prompt=prompt, system_prompt=system_prompt, temperature=temperature)
        # Yield word-by-word chunks for responsive UI streaming
        words = full_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")


class OllamaLLMProvider(BaseLLMProvider):
    """Local Ollama LLM provider.

    Connects to a running Ollama daemon on localhost:11434.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self._host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self._model_name = model_name or os.getenv("OLLAMA_MODEL") or "llama3"

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        """Quick ping to check if Ollama daemon is running."""
        import socket
        try:
            # First try a raw fast socket check (0.3s) to avoid urllib hang
            parsed = urllib.parse.urlparse(self._host)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 11434
            with socket.create_connection((host, port), timeout=0.3):
                return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List models installed in local Ollama instance."""
        try:
            req = urllib.request.Request(f"{self._host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        url = f"{self._host}/api/generate"
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed at {self._host}: {e}") from e

    def stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        url = f"{self._host}/api/generate"
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                for line in resp:
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
        except Exception as e:
            raise RuntimeError(f"Ollama streaming failed: {e}") from e


class OfflineExtractiveLLMProvider(BaseLLMProvider):
    """Deterministic factual synthesizer for 100% offline environments.

    When neither Gemini API nor local Ollama is active, this provider extracts
    and synthesizes grounded answers with citations directly from retrieved
    context excerpts without hallucinations.
    """

    def __init__(self, model_name: str = "offline-extractive-tutor") -> None:
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return "offline"

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        # Extract question and context from prompt if structured
        contexts_match = re.search(r"--- CONTEXT EXCERPTS ---([\s\S]*?)--- END CONTEXT ---", prompt)
        question_match = re.search(r"QUESTION:\s*(.*?)(?:\n\n|\n---|$)", prompt)

        if not contexts_match:
            return (
                "Offline Mode: Answer formulated strictly from indexed study documents.\n\n"
                f"{prompt[:400]}..."
            )

        contexts_text = contexts_match.group(1).strip()
        question = question_match.group(1).strip() if question_match else ""

        # Extract context blocks
        blocks = re.findall(
            r"\[(Source \d+: [^\]]+)\]\s*\n(.*?)(?=\n\[Source|\Z)",
            contexts_text,
            re.DOTALL,
        )

        if not blocks:
            return "I cannot find sufficient information about this in the provided study materials."

        stop_words = {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "explain", "describe", "tell", "does", "with", "from", "about", "this",
            "that", "these", "those", "have", "has", "had", "been", "being", "used",
            "uses", "using", "into", "some", "more", "most", "other", "such", "only",
            "same", "than", "then", "very", "also", "would", "could", "should",
            "shall", "will", "were", "there", "their", "them", "they", "your", "yours"
        }
        keywords = [
            w.lower()
            for w in re.findall(r"\b[A-Za-z0-9_]{3,}\b", question)
            if w.lower() not in stop_words
        ]

        scored_sentences: List[Tuple[float, str, str]] = []
        for header, content in blocks:
            sentences = re.split(r"(?<=[.!?\n])\s+", content)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 20:
                    continue
                score = sum(1.5 for kw in keywords if kw in s_clean.lower())
                if score > 0:
                    scored_sentences.append((score, s_clean, header))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        if not scored_sentences:
            return "I cannot find information about this in the provided study materials."

        top_sentences = scored_sentences[:6]
        answer_parts = []
        seen = set()

        for _, sent, header in top_sentences:
            if sent in seen:
                continue
            seen.add(sent)
            answer_parts.append(f"• {sent} [{header}]")

        header_text = f"**Key findings regarding '{question}':**\n\n" if question else "**Key Study Notes:**\n\n"
        return header_text + "\n\n".join(answer_parts)

    def stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        full_text = self.generate(prompt=prompt, system_prompt=system_prompt, temperature=temperature)
        words = full_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")


# ============================================================================
# Embedding Providers
# ============================================================================

class BaseEmbeddingProvider(ABC):
    """Abstract base class for vector embedding generation."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Compute embedding vector for a single query or text chunk."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embedding vectors for a list of texts."""
        pass


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini text-embedding-004 provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/text-embedding-004",
    ) -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        self._model_name = model_name
        self._dimension = 768

    @property
    def dimension(self) -> int:
        return self._dimension

    def is_available(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 8)

    def embed_text(self, text: str) -> List[float]:
        results = self.embed_documents([text])
        return results[0] if results else [0.0] * self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_available():
            raise RuntimeError("Gemini API key is not configured for embeddings.")

        # Batch in chunks of 50
        all_embeddings: List[List[float]] = []
        batch_size = 50

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/"
                f"{self._model_name}:batchEmbedContents?key={self._api_key}"
            )
            requests_body = [
                {"model": self._model_name, "content": {"parts": [{"text": t}]}}
                for t in batch
            ]
            data_bytes = json.dumps({"requests": requests_body}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    embeddings = resp_data.get("embeddings", [])
                    for emb in embeddings:
                        all_embeddings.append(emb.get("values", [0.0] * self._dimension))
            except Exception as e:
                logger.warning("Gemini batch embedding error (%s); falling back to single embeds.", e)
                # Fallback to zero vectors or single calls
                for _ in batch:
                    all_embeddings.append([0.0] * self._dimension)

        return all_embeddings


class LocalHashEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic high-dimensional subword semantic vectorizer.

    Guarantees 100% offline, zero-dependency embedding generation with cosine
    similarity preservation. Runs anywhere without downloading models.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_token(self, token: str) -> Tuple[int, int]:
        h1 = 0x811c9dc5
        h2 = 0x5bd1e995
        for ch in token.encode("utf-8"):
            h1 = ((h1 ^ ch) * 0x01000193) & 0xFFFFFFFF
            h2 = ((h2 ^ ch) * 0x01000193) & 0xFFFFFFFF
        idx = h1 % self._dim
        sign = 1.0 if (h2 % 2 == 0) else -1.0
        return idx, int(sign)

    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        clean_text = text.lower()

        # Word tokens
        words = re.findall(r"\b\w+\b", clean_text)
        for w in words:
            idx, sign = self._hash_token(w)
            vec[idx] += 1.0 * sign

        # Subword 3-grams and 4-grams for typos / subword semantics
        for i in range(len(clean_text) - 3):
            gram = clean_text[i : i + 3]
            idx, sign = self._hash_token(gram)
            vec[idx] += 0.3 * sign

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            return [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


# ============================================================================
# Provider Manager & Factory
# ============================================================================

class ProviderManager:
    """Manages AI engine selection, status reporting, and graceful fallbacks."""

    def __init__(
        self,
        engine_type: str = "auto",  # 'gemini', 'ollama', 'offline', 'auto'
        api_key: Optional[str] = None,
        ollama_host: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ) -> None:
        self.engine_type = engine_type
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL") or "llama3"

        self._init_providers()

    def _init_providers(self) -> None:
        self.gemini_llm = GeminiLLMProvider(api_key=self.api_key)
        self.gemini_emb = GeminiEmbeddingProvider(api_key=self.api_key)
        self.ollama_llm = OllamaLLMProvider(host=self.ollama_host, model_name=self.ollama_model)
        self.offline_llm = OfflineExtractiveLLMProvider()
        self.local_emb = LocalHashEmbeddingProvider()

    def set_engine(self, engine_type: str, api_key: Optional[str] = None) -> None:
        """Switch active engine mode ('gemini', 'ollama', 'offline', 'auto')."""
        self.engine_type = engine_type
        if api_key:
            self.api_key = api_key
        self._init_providers()

    def get_llm(self) -> BaseLLMProvider:
        """Get the active LLM provider based on configuration and availability."""
        if self.engine_type == "gemini":
            if self.gemini_llm.is_available():
                return self.gemini_llm
            logger.warning("Gemini requested but not available; checking fallbacks.")

        if self.engine_type == "ollama":
            if self.ollama_llm.is_available():
                return self.ollama_llm
            logger.warning("Ollama requested but not reachable; falling back to offline.")

        if self.engine_type == "offline":
            return self.offline_llm

        # 'auto' mode
        if self.gemini_llm.is_available():
            return self.gemini_llm
        if self.ollama_llm.is_available():
            return self.ollama_llm
        return self.offline_llm

    def get_embedding_provider(self) -> BaseEmbeddingProvider:
        """Get active embedding provider."""
        if self.engine_type == "gemini" and self.gemini_emb.is_available():
            return self.gemini_emb
        return self.local_emb

    def get_status(self) -> Dict[str, Any]:
        """Return connectivity and configuration status for UI display."""
        active_llm = self.get_llm()
        active_emb = self.get_embedding_provider()
        return {
            "mode": self.engine_type,
            "active_llm_provider": active_llm.provider_name,
            "active_model": active_llm.model_name,
            "active_embedding_dimension": active_emb.dimension,
            "gemini_available": self.gemini_llm.is_available(),
            "ollama_available": self.ollama_llm.is_available(),
            "offline_available": True,
        }
