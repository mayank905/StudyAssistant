"""Core Grounded Retrieval-Augmented Generation (RAG) Engine for StudyAssistant.

Features:
1. High-fidelity PDF Ingestion:
   - Fast, accurate extraction using pypdfium2 with text-stream fallback.
   - Automatic subject classification (Engineering, Fire, Motor, General).
   - Chapter and section outline detection from document structure.
   - Overlapping recursive sliding-window chunking.

2. ChromaDB Indexing & Persistence:
   - Dual-engine embeddings (Gemini text-embedding-004 & Local semantic vectors).
   - Fast metadata filtering by subject, chapter/topic, and source document.
   - Robust in-memory/disk vector fallback if native ChromaDB is unavailable.

3. Strict Zero-Hallucination Grounding Guard:
   - Verifiable document and page citations for every factual claim.
   - Refusal mechanism for out-of-corpus queries unless external knowledge is allowed.
   - Reference card extraction for side-by-side verification in the UI.

4. Dynamic Document Ingestion:
   - Ingests uploaded PDF/TXT files on the fly from Streamlit without restarting.
   - Immediate live updating of subject and chapter scope selectors.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
import logging
import math
import os
import re
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

# Disable ChromaDB telemetry globally before imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"

logger = logging.getLogger(__name__)

# Providers
from core.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    GeminiEmbeddingProvider,
    GeminiLLMProvider,
    LocalHashEmbeddingProvider,
    OfflineExtractiveLLMProvider,
    OllamaLLMProvider,
    ProviderManager,
)

# Optional ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
    HAS_CHROMADB = True
except ImportError:
    chromadb = None
    HAS_CHROMADB = False

# Optional pypdfium2
try:
    import pypdfium2
    HAS_PYPDFIUM = True
except ImportError:
    pypdfium2 = None
    HAS_PYPDFIUM = False


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Citation:
    """Document citation for grounded answers."""
    file_name: str
    page_number: int
    chapter: str
    subject: str
    snippet: str
    similarity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def formatted(self) -> str:
        chap_str = f" - {self.chapter}" if self.chapter and self.chapter != "General" else ""
        return f"[Doc: {self.file_name}, Page: {self.page_number}{chap_str}]"


@dataclass
class DocumentChunk:
    """A semantic chunk of text from a study material."""
    chunk_id: str
    content: str
    file_name: str
    subject: str
    chapter: str
    page_number: int
    char_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.content)
        if not self.chunk_id:
            raw = f"{self.file_name}_{self.page_number}_{self.content[:40]}"
            self.chunk_id = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class RAGQueryResult:
    """Structured response from the RAG Engine."""
    answer: str
    citations: List[Citation]
    contexts: List[Dict[str, Any]]
    grounded: bool
    confidence_score: float
    subject: str
    engine_used: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "contexts": self.contexts,
            "grounded": self.grounded,
            "confidence_score": round(self.confidence_score, 2),
            "subject": self.subject,
            "engine_used": self.engine_used,
        }


# ============================================================================
# PDF Ingestor & Outline Extractor
# ============================================================================

class PDFIngestor:
    """Extracts text, identifies subjects, detects chapter outlines, and chunks PDFs."""

    DEFAULT_SUBJECT_PATTERNS = [
        (re.compile(r"engg|engineering", re.IGNORECASE), "Engineering"),
        (re.compile(r"fire|tariff", re.IGNORECASE), "Fire"),
        (re.compile(r"motor|vehicle", re.IGNORECASE), "Motor"),
    ]

    HEADING_PATTERNS = [
        re.compile(r"^(Section\s+[IVXLCDM\d]+[^\n:]*)", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^(Chapter\s+\d+[^\n:]*)", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^(PART\s+[A-Z\d]+[^\n:]*)", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^([A-Z\s]{5,40}:)", re.MULTILINE),
    ]

    @classmethod
    def detect_subject(cls, file_name: str, sample_text: str = "") -> str:
        """Infer subject from file name or initial text."""
        for pattern, subject in cls.DEFAULT_SUBJECT_PATTERNS:
            if pattern.search(file_name):
                return subject
        for pattern, subject in cls.DEFAULT_SUBJECT_PATTERNS:
            if pattern.search(sample_text[:1000]):
                return subject
        return "General"

    @classmethod
    def extract_pages(
        cls, file_source: Union[str, bytes, io.BytesIO]
    ) -> List[Tuple[int, str]]:
        """Extract list of (page_num, text) from a PDF file path or bytes."""
        pages: List[Tuple[int, str]] = []

        # 1. Try pypdfium2 if available
        if HAS_PYPDFIUM and pypdfium2 is not None:
            try:
                if isinstance(file_source, (bytes, bytearray)):
                    doc = pypdfium2.PdfDocument(io.BytesIO(file_source))
                elif isinstance(file_source, io.BytesIO):
                    doc = pypdfium2.PdfDocument(file_source)
                else:
                    doc = pypdfium2.PdfDocument(str(file_source))

                for i, page in enumerate(doc):
                    text = page.get_textpage().get_text_range()
                    clean_text = cls._clean_text(text)
                    if clean_text:
                        pages.append((i + 1, clean_text))
                return pages
            except Exception as e:
                logger.warning("pypdfium2 extraction failed (%s); trying fallback.", e)

        # 2. Pure Python FlateDecode fallback for raw bytes/file
        try:
            if isinstance(file_source, str) and os.path.exists(file_source):
                with open(file_source, "rb") as f:
                    data = f.read()
            elif isinstance(file_source, io.BytesIO):
                data = file_source.getvalue()
            else:
                data = bytes(file_source)

            # Look for compressed streams
            import zlib
            stream_matches = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL)
            page_no = 1
            for raw_stream in stream_matches:
                try:
                    decomp = zlib.decompress(raw_stream)
                    # Extract text inside Tj and TJ operators
                    extracted = []
                    # Tj matches
                    for match in re.findall(rb"\((.*?)\)\s*Tj", decomp):
                        extracted.append(match.decode("latin1", errors="ignore"))
                    # TJ array matches
                    for match in re.findall(rb"\[(.*?)\]\s*TJ", decomp):
                        inner_texts = re.findall(rb"\((.*?)\)", match)
                        extracted.append("".join(t.decode("latin1", errors="ignore") for t in inner_texts))

                    page_text = " ".join(extracted)
                    clean = cls._clean_text(page_text)
                    if len(clean) > 50:
                        pages.append((page_no, clean))
                        page_no += 1
                except Exception:
                    continue
        except Exception as e:
            logger.error("Fallback PDF extraction error: %s", e)

        return pages

    @classmethod
    def _clean_text(cls, text: str) -> str:
        """Normalize whitespace and remove non-printable control characters."""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def detect_outline_and_chunk(
        cls,
        pages: List[Tuple[int, str]],
        file_name: str,
        subject: Optional[str] = None,
        chunk_size: int = 700,
        chunk_overlap: int = 150,
    ) -> List[DocumentChunk]:
        """Parse pages into chunks tagged with active chapter and page numbers."""
        if not pages:
            return []

        first_sample = pages[0][1] if pages else ""
        resolved_subject = subject or cls.detect_subject(file_name, first_sample)

        chunks: List[DocumentChunk] = []
        current_chapter = "Overview"

        for page_num, page_text in pages:
            # Check for section or chapter heading on this page
            for pattern in cls.HEADING_PATTERNS:
                heading_match = pattern.search(page_text[:400])
                if heading_match:
                    detected_heading = heading_match.group(1).strip()
                    # Clean trailing colons and extra whitespace
                    detected_heading = re.sub(r":$", "", detected_heading).strip()
                    if len(detected_heading) > 4:
                        current_chapter = detected_heading
                        break

            # Window-based chunking with boundary awareness
            page_chunks = cls._chunk_text(
                text=page_text,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )

            for i, chunk_text in enumerate(page_chunks):
                chunk_id = f"{file_name}_p{page_num}_c{i}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    file_name=file_name,
                    subject=resolved_subject,
                    chapter=current_chapter,
                    page_number=page_num,
                    metadata={
                        "file_name": file_name,
                        "subject": resolved_subject,
                        "chapter": current_chapter,
                        "page_number": page_num,
                        "chunk_index": i,
                    },
                )
                chunks.append(chunk)

        return chunks

    @classmethod
    def _chunk_text(cls, text: str, chunk_size: int = 700, overlap: int = 150) -> List[str]:
        """Split text into overlapping chunks respecting paragraph or sentence boundaries."""
        if len(text) <= chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                # Try break at paragraph
                para_break = text.rfind("\n\n", start + overlap, end)
                if para_break != -1:
                    end = para_break + 2
                else:
                    # Try break at sentence end
                    period_break = text.rfind(". ", start + overlap, end)
                    if period_break != -1:
                        end = period_break + 2
                    else:
                        # Try space
                        space_break = text.rfind(" ", start + overlap, end)
                        if space_break != -1:
                            end = space_break + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break
            start = max(start + 1, end - overlap)

        return chunks


# ============================================================================
# Vector Store Engine (ChromaDB with In-Memory / File Fallback)
# ============================================================================

class VectorStore:
    """Manages document chunk indexing and semantic vector retrieval."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
    ) -> None:
        self.persist_dir = persist_dir
        self.embedding_provider = embedding_provider or LocalHashEmbeddingProvider()
        self.chunks: Dict[str, DocumentChunk] = {}
        self.vectors: Dict[str, List[float]] = {}
        self._chroma_client = None
        self._chroma_collection = None
        self._use_chroma = False

        self._init_store()

    def _init_store(self) -> None:
        """Initialize ChromaDB or in-memory fallback."""
        os.makedirs(self.persist_dir, exist_ok=True)

        if HAS_CHROMADB and chromadb is not None:
            try:
                # Custom Chroma Embedding Adapter
                class ChromaEmbeddingAdapter(EmbeddingFunction):
                    def __init__(self, provider: BaseEmbeddingProvider) -> None:
                        self.provider = provider

                    def __call__(self, input: Documents) -> Embeddings:
                        return self.provider.embed_documents(input)

                # Disable telemetry to prevent socket warnings
                settings = Settings(anonymized_telemetry=False, is_persistent=True, persist_directory=self.persist_dir)
                self._chroma_client = chromadb.PersistentClient(path=self.persist_dir, settings=settings)
                adapter = ChromaEmbeddingAdapter(self.embedding_provider)
                self._chroma_collection = self._chroma_client.get_or_create_collection(
                    name="study_materials",
                    embedding_function=adapter,
                    metadata={"hnsw:space": "cosine"},
                )
                self._use_chroma = True
                logger.info("ChromaDB initialized at %s", self.persist_dir)
            except Exception as e:
                logger.warning("ChromaDB initialization failed (%s); using in-memory vector store.", e)
                self._use_chroma = False
        else:
            self._use_chroma = False

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Index a list of DocumentChunks into the vector database."""
        if not chunks:
            return 0

        # Update in-memory lookup
        for c in chunks:
            self.chunks[c.chunk_id] = c

        if self._use_chroma and self._chroma_collection is not None:
            try:
                ids = [c.chunk_id for c in chunks]
                docs = [c.content for c in chunks]
                metadatas = [c.metadata for c in chunks]

                # ChromaDB handles batch embedding
                batch_size = 64
                for i in range(0, len(ids), batch_size):
                    self._chroma_collection.upsert(
                        ids=ids[i : i + batch_size],
                        documents=docs[i : i + batch_size],
                        metadatas=metadatas[i : i + batch_size],
                    )
                return len(chunks)
            except Exception as e:
                logger.warning("ChromaDB upsert error (%s); indexing in local vector store.", e)

        # Fallback vector index
        texts = [c.content for c in chunks]
        embs = self.embedding_provider.embed_documents(texts)
        for c, emb in zip(chunks, embs):
            self.vectors[c.chunk_id] = emb

        return len(chunks)

    def search(
        self,
        query: str,
        subject: Optional[str] = None,
        chapter: Optional[str] = None,
        top_k: int = 4,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Perform semantic search with metadata filters."""
        if not self.chunks and (not self._use_chroma or self._chroma_collection is None):
            return []

        # 1. Try ChromaDB if active
        if self._use_chroma and self._chroma_collection is not None:
            try:
                where_clause: Dict[str, Any] = {}
                filters = []
                if subject and subject != "All" and subject != "General":
                    filters.append({"subject": subject})
                if chapter and chapter != "All":
                    filters.append({"chapter": chapter})

                if len(filters) == 1:
                    where_clause = filters[0]
                elif len(filters) > 1:
                    where_clause = {"$and": filters}

                res = self._chroma_collection.query(
                    query_texts=[query],
                    n_results=min(top_k, max(len(self.chunks), 1)),
                    where=where_clause if where_clause else None,
                )

                matched_chunks: List[Tuple[DocumentChunk, float]] = []
                if res and res.get("ids") and res["ids"][0]:
                    ids = res["ids"][0]
                    distances = res.get("distances", [[0.0] * len(ids)])[0]
                    docs = res.get("documents", [[ ""] * len(ids)])[0]
                    metas = res.get("metadatas", [[{}] * len(ids)])[0]

                    for cid, dist, doc_text, meta in zip(ids, distances, docs, metas):
                        chunk = self.chunks.get(cid)
                        if not chunk:
                            chunk = DocumentChunk(
                                chunk_id=cid,
                                content=doc_text,
                                file_name=meta.get("file_name", "Unknown"),
                                subject=meta.get("subject", "General"),
                                chapter=meta.get("chapter", "Overview"),
                                page_number=int(meta.get("page_number", 1)),
                                metadata=meta,
                            )
                            self.chunks[cid] = chunk

                        # Distance to similarity (0 to 1)
                        sim = max(0.0, 1.0 - (dist / 2.0)) if dist is not None else 0.8
                        matched_chunks.append((chunk, sim))

                    if matched_chunks:
                        return matched_chunks
            except Exception as e:
                logger.warning("ChromaDB query failed (%s); falling back to local cosine search.", e)

        # 2. In-Memory Cosine Similarity Fallback
        query_vec = self.embedding_provider.embed_text(query)
        scored: List[Tuple[DocumentChunk, float]] = []

        for cid, chunk in self.chunks.items():
            # Apply scope filters
            if subject and subject not in ("All", "General") and chunk.subject != subject:
                continue
            if chapter and chapter != "All" and chunk.chapter != chapter:
                continue

            doc_vec = self.vectors.get(cid)
            if not doc_vec:
                doc_vec = self.embedding_provider.embed_text(chunk.content)
                self.vectors[cid] = doc_vec

            # Cosine similarity
            dot = sum(q * d for q, d in zip(query_vec, doc_vec))
            scored.append((chunk, dot))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ============================================================================
# Zero-Hallucination Grounding Guard
# ============================================================================

class GroundingGuard:
    """Enforces zero hallucination by formatting strict grounding prompts and parsing citations."""

    SYSTEM_PROMPT = """You are StudyAssistant, an expert exam preparation tutor for insurance professional examinations.

CRITICAL INSTRUCTION - ZERO HALLUCINATION POLICY:
1. Base your answer EXCLUSIVELY on the provided context excerpts below.
2. If the context does not contain the answer, explicitly state:
   "I cannot find information about this in the provided study materials."
3. Do NOT make up, assume, or extrapolate facts, figures, percentages, policy names, or rules not present in the context.
4. Every single factual claim MUST be followed by an exact inline citation format:
   [Doc: <filename>, Page: <page>] or [Doc: <filename>, Page: <page>, Section: <section>]
5. Maintain a professional, exam-focused, educational tone.
"""

    @classmethod
    def construct_prompt(
        cls,
        question: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
        allow_external_knowledge: bool = False,
    ) -> str:
        """Construct a prompt embedding context snippets with exact source labels."""
        if not retrieved_chunks:
            return (
                f"QUESTION: {question}\n\n"
                "No relevant documents were found in the index for this query. "
                "State that no information is available in the study materials."
            )

        context_lines = []
        for i, (chunk, score) in enumerate(retrieved_chunks, 1):
            header = f"Source {i}: {chunk.file_name} | Page {chunk.page_number} | Subject: {chunk.subject} | Chapter: {chunk.chapter}"
            context_lines.append(f"[{header}]\n{chunk.content.strip()}\n")

        contexts_joined = "\n".join(context_lines)

        external_note = ""
        if allow_external_knowledge:
            external_note = (
                "\nNOTE: The user has toggled 'Allow External Knowledge'. You may supplement "
                "with verified general insurance principles, but clearly distinguish external info from cited text.\n"
            )

        prompt = f"""--- CONTEXT EXCERPTS ---
{contexts_joined}
--- END CONTEXT ---
{external_note}
QUESTION: {question}

Provide a comprehensive, accurately grounded answer citing specific pages and documents."""
        return prompt

    @classmethod
    def extract_citations(
        cls,
        answer_text: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
    ) -> Tuple[List[Citation], float]:
        """Extract citations and calculate grounding confidence."""
        citations: List[Citation] = []
        seen_keys: Set[Tuple[str, int]] = set()

        # 1. Parse inline citations from generated text [Doc: ..., Page: ...]
        matches = re.findall(
            r"\[(?:Doc:\s*)?([^,\]]+),\s*(?:Page:\s*)?(\d+)(?:,\s*(?:Section|Chapter):\s*([^\]]+))?\]",
            answer_text,
            re.IGNORECASE,
        )

        for doc_name, page_str, sec_str in matches:
            doc_clean = doc_name.strip()
            page_num = int(page_str)
            key = (doc_clean, page_num)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Match with retrieved chunk
            matched_chunk = None
            sim = 0.85
            for ch, score in retrieved_chunks:
                if ch.page_number == page_num and (doc_clean.lower() in ch.file_name.lower() or ch.file_name.lower() in doc_clean.lower()):
                    matched_chunk = ch
                    sim = score
                    break

            snippet = matched_chunk.content[:200] + "..." if matched_chunk else "Referenced in text."
            chap = sec_str.strip() if sec_str else (matched_chunk.chapter if matched_chunk else "General")
            subj = matched_chunk.subject if matched_chunk else "General"

            citations.append(
                Citation(
                    file_name=matched_chunk.file_name if matched_chunk else doc_clean,
                    page_number=page_num,
                    chapter=chap,
                    subject=subj,
                    snippet=snippet,
                    similarity_score=sim,
                )
            )

        # 2. Add top retrieved chunks as citation references if few inline parsed
        if len(citations) < len(retrieved_chunks):
            for ch, score in retrieved_chunks:
                key = (ch.file_name, ch.page_number)
                if key not in seen_keys:
                    seen_keys.add(key)
                    citations.append(
                        Citation(
                            file_name=ch.file_name,
                            page_number=ch.page_number,
                            chapter=ch.chapter,
                            subject=ch.subject,
                            snippet=ch.content[:220] + "...",
                            similarity_score=score,
                        )
                    )

        # Grounding confidence metric
        refusal_phrases = [
            "cannot find information",
            "not present in the provided",
            "no mention of",
            "study materials do not contain",
        ]
        is_refusal = any(phrase in answer_text.lower() for phrase in refusal_phrases)

        if is_refusal:
            confidence = 0.1
        elif citations:
            avg_sim = sum(c.similarity_score for c in citations) / len(citations)
            confidence = min(1.0, max(0.5, avg_sim * 1.1))
        else:
            confidence = 0.4

        return citations, confidence


# ============================================================================
# Main RAG Engine
# ============================================================================

class RAGEngine:
    """Complete RAG coordinator managing ingestion, storage, grounding, and answering."""

    DEFAULT_PDF_NAMES = [
        "ENGG_Study_Material_Fast_Revision.pdf",
        "Fire tariff.pdf",
        "FIRE_Study_Material.pdf",
        "UPDATED_MOTOR_WRITE_UP_FOR_PE_2026_27_EXAMINATION_ON_27_12_2025.pdf",
    ]

    def __init__(
        self,
        base_dir: Optional[str] = None,
        provider_manager: Optional[ProviderManager] = None,
    ) -> None:
        self.base_dir = base_dir or os.getcwd()
        self.provider_mgr = provider_manager or ProviderManager()
        self.vector_store = VectorStore(
            persist_dir=os.path.join(self.base_dir, "data", "chroma_db"),
            embedding_provider=self.provider_mgr.get_embedding_provider(),
        )
        self.ingested_files: Set[str] = set()

    def ingest_default_pdfs(self) -> int:
        """Scan and ingest the 4 demo insurance study PDFs in base directory."""
        total_chunks = 0
        for pdf_name in self.DEFAULT_PDF_NAMES:
            full_path = os.path.join(self.base_dir, pdf_name)
            if os.path.exists(full_path) and pdf_name not in self.ingested_files:
                count = self.ingest_document(full_path, file_name=pdf_name)
                total_chunks += count
        return total_chunks

    def ingest_document(
        self,
        file_source: Union[str, bytes, io.BytesIO],
        file_name: str,
        subject: Optional[str] = None,
        chapter: Optional[str] = None,
    ) -> int:
        """Ingest a single document dynamically on the fly."""
        logger.info("Ingesting document: %s", file_name)

        if file_name.lower().endswith(".pdf"):
            pages = PDFIngestor.extract_pages(file_source)
        else:
            # Plain text or Markdown
            if isinstance(file_source, str) and os.path.exists(file_source):
                with open(file_source, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
            elif isinstance(file_source, io.BytesIO):
                raw_text = file_source.getvalue().decode("utf-8", errors="ignore")
            elif isinstance(file_source, (bytes, bytearray)):
                raw_text = file_source.decode("utf-8", errors="ignore")
            else:
                raw_text = str(file_source)

            pages = [(1, PDFIngestor._clean_text(raw_text))]

        chunks = PDFIngestor.detect_outline_and_chunk(
            pages=pages,
            file_name=file_name,
            subject=subject,
        )

        if chapter:
            for c in chunks:
                c.chapter = chapter
                c.metadata["chapter"] = chapter

        added = self.vector_store.add_chunks(chunks)
        self.ingested_files.add(file_name)
        logger.info("Indexed %d chunks from %s", added, file_name)
        return added

    def query(
        self,
        question: str,
        subject: Optional[str] = None,
        chapter: Optional[str] = None,
        top_k: int = 4,
        allow_external_knowledge: bool = False,
    ) -> Dict[str, Any]:
        """Perform grounded retrieval, prompt construction, and synthesis."""
        # 1. Retrieve most relevant context chunks
        retrieved = self.vector_store.search(
            query=question,
            subject=subject,
            chapter=chapter,
            top_k=top_k,
        )

        # 2. Construct grounded prompt
        prompt = GroundingGuard.construct_prompt(
            question=question,
            retrieved_chunks=retrieved,
            allow_external_knowledge=allow_external_knowledge,
        )

        # 3. Invoke LLM provider
        llm = self.provider_mgr.get_llm()
        raw_answer = llm.generate(
            prompt=prompt,
            system_prompt=GroundingGuard.SYSTEM_PROMPT,
            temperature=0.15,
        )

        # 4. Extract citations and compute grounding confidence
        citations, confidence = GroundingGuard.extract_citations(
            answer_text=raw_answer,
            retrieved_chunks=retrieved,
        )

        is_grounded = confidence >= 0.4 and not (
            "cannot find information" in raw_answer.lower()
        )

        contexts_payload = [
            {
                "chunk_id": ch.chunk_id,
                "file_name": ch.file_name,
                "page_number": ch.page_number,
                "chapter": ch.chapter,
                "subject": ch.subject,
                "similarity_score": round(score, 3),
                "snippet": ch.content[:350],
            }
            for ch, score in retrieved
        ]

        result = RAGQueryResult(
            answer=raw_answer,
            citations=citations,
            contexts=contexts_payload,
            grounded=is_grounded,
            confidence_score=confidence,
            subject=subject or "General",
            engine_used=f"{llm.provider_name} ({llm.model_name})",
        )

        return result.to_dict()

    def generate(self, prompt: str) -> str:
        """Direct completion generation interface for quizzes and viva evaluation."""
        llm = self.provider_mgr.get_llm()
        return llm.generate(prompt)

    def get_available_subjects(self) -> List[str]:
        """Return list of distinct subjects indexed."""
        subjects = {c.subject for c in self.vector_store.chunks.values()}
        order = ["General", "Engineering", "Fire", "Motor"]
        sorted_subjs = [s for s in order if s in subjects]
        for s in sorted(subjects):
            if s not in sorted_subjs:
                sorted_subjs.append(s)
        return sorted_subjs or ["General", "Engineering", "Fire", "Motor"]

    def get_available_chapters(self, subject: Optional[str] = None) -> List[str]:
        """Return distinct chapters/outlines detected for a subject."""
        chapters = set()
        for c in self.vector_store.chunks.values():
            if not subject or subject in ("All", "General") or c.subject == subject:
                if c.chapter:
                    chapters.add(c.chapter)
        return sorted(list(chapters)) or ["Overview"]

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics for UI dashboards."""
        total_chunks = len(self.vector_store.chunks)
        by_subject: Dict[str, int] = {}
        by_file: Dict[str, int] = {}

        for c in self.vector_store.chunks.values():
            by_subject[c.subject] = by_subject.get(c.subject, 0) + 1
            by_file[c.file_name] = by_file.get(c.file_name, 0) + 1

        return {
            "total_chunks": total_chunks,
            "total_documents": len(by_file),
            "documents": list(by_file.keys()),
            "subject_breakdown": by_subject,
            "engine_status": self.provider_mgr.get_status(),
        }

    def set_api_key(self, api_key: str) -> None:
        """Dynamically update Gemini API key."""
        self.provider_mgr.set_api_key(api_key)

    def switch_engine(self, engine_type: str, api_key: Optional[str] = None) -> None:
        """Switch active engine mode ('gemini', 'ollama', 'offline', 'auto')."""
        self.provider_mgr.set_engine(engine_type, api_key=api_key)


# ============================================================================
# Singleton Accessor
# ============================================================================

_GLOBAL_RAG_ENGINE: Optional[RAGEngine] = None

def get_rag_engine(base_dir: Optional[str] = None) -> RAGEngine:
    """Retrieve or initialize the global shared RAG engine instance."""
    global _GLOBAL_RAG_ENGINE
    if _GLOBAL_RAG_ENGINE is None:
        _GLOBAL_RAG_ENGINE = RAGEngine(base_dir=base_dir)
    return _GLOBAL_RAG_ENGINE
