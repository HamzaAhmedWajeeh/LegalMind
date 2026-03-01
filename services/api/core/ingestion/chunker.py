"""
core/ingestion/chunker.py
=========================
Text chunker — splits parsed document text into overlapping chunks.

Design Pattern: STRATEGY PATTERN
─────────────────────────────────
The chunking algorithm is encapsulated behind a common interface
(`ChunkingStrategy`). The pipeline selects a strategy at runtime
without changing any other code.

Two concrete strategies:
  1. RecursiveChunkingStrategy  — fixed-size token windows with overlap.
                                   Fast, predictable, matches the spec
                                   (512 tokens, 10% overlap).
  2. SemanticChunkingStrategy   — splits on meaningful content transitions
                                   by comparing sentence embedding similarity.
                                   Keeps related clauses together.

Usage:
    strategy = RecursiveChunkingStrategy()          # or SemanticChunkingStrategy()
    chunker  = Chunker(strategy=strategy)
    chunks   = chunker.chunk(parsed_doc)
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import structlog
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = structlog.get_logger(__name__)

# tiktoken encoding for Claude / GPT-4 compatible token counting
_ENCODING = tiktoken.get_encoding("cl100k_base")


# ──────────────────────────────────────────────────────────────────
# Output data class
# ──────────────────────────────────────────────────────────────────
@dataclass
class TextChunk:
    """
    A single chunk of text ready for embedding and storage.

    Attributes:
        text        : The chunk content
        chunk_index : Position in the document (0-based)
        token_count : Approximate token count
        metadata    : Page number, strategy used, etc.
    """
    text: str
    chunk_index: int
    token_count: int
    metadata: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Abstract Strategy Interface
# ──────────────────────────────────────────────────────────────────
class ChunkingStrategy(ABC):
    """
    Abstract base class for all chunking strategies.
    Any new strategy just implements `split()` — the rest of
    the pipeline is completely unaware of which strategy is active.
    """

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """
        Split raw text into a list of string chunks.

        Args:
            text: Full document text (pages joined by \\f)

        Returns:
            List of raw text chunks (not yet wrapped in TextChunk)
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name for logging/metadata."""
        ...


# ──────────────────────────────────────────────────────────────────
# Strategy 1: Recursive Fixed-Size Chunking
# ──────────────────────────────────────────────────────────────────
class RecursiveChunkingStrategy(ChunkingStrategy):
    """
    Splits text using LangChain's RecursiveCharacterTextSplitter.

    Tries to split on paragraph boundaries first (\\n\\n), then
    sentence boundaries (\\n), then word boundaries ( ), ensuring
    chunks never exceed `chunk_size` tokens.

    This matches the spec: 512 tokens, ~10% overlap (51 tokens).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 51,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Use token-based length function for accuracy
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=_count_tokens,
            separators=[
                "\n\n\n",   # Section breaks
                "\n\n",     # Paragraph breaks
                "\n",       # Line breaks
                ". ",       # Sentence ends
                "; ",       # Clause separators (common in legal text)
                ", ",       # Sub-clause separators
                " ",        # Word boundaries
                "",         # Character-level fallback
            ],
        )

    @property
    def name(self) -> str:
        return f"recursive(size={self._chunk_size}, overlap={self._chunk_overlap})"

    def split(self, text: str) -> list[str]:
        # Replace form-feed page separators with paragraph breaks
        normalised = text.replace("\f", "\n\n")
        chunks = self._splitter.split_text(normalised)
        # Filter out empty or whitespace-only chunks
        return [c.strip() for c in chunks if c.strip()]


# ──────────────────────────────────────────────────────────────────
# Strategy 2: Semantic Chunking
# ──────────────────────────────────────────────────────────────────
class SemanticChunkingStrategy(ChunkingStrategy):
    """
    Splits text at points where the semantic meaning shifts significantly.

    Algorithm:
      1. Split text into sentences.
      2. Embed each sentence using a lightweight local model.
      3. Calculate cosine similarity between adjacent sentence embeddings.
      4. Insert a chunk boundary wherever similarity drops below `breakpoint_threshold`.
      5. Merge tiny chunks into the previous one to respect `min_chunk_tokens`.

    This keeps related legal clauses (e.g., an indemnification clause
    and its sub-clauses) in the same chunk even if they span multiple
    sentences, which improves retrieval precision.
    """

    def __init__(
        self,
        breakpoint_threshold: float = 0.75,
        min_chunk_tokens: int = 100,
        max_chunk_tokens: int = 600,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self._threshold = breakpoint_threshold
        self._min_tokens = min_chunk_tokens
        self._max_tokens = max_chunk_tokens
        self._model_name = embedding_model
        self._model = None   # Lazy-loaded on first use

    @property
    def name(self) -> str:
        return f"semantic(threshold={self._threshold})"

    def _get_model(self):
        """Lazy-load the sentence transformer to avoid startup overhead."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info("Loaded semantic chunking model", model=self._model_name)
        return self._model

    def split(self, text: str) -> list[str]:
        import numpy as np

        normalised = text.replace("\f", "\n\n")
        sentences = _split_into_sentences(normalised)

        if len(sentences) <= 1:
            return [normalised.strip()] if normalised.strip() else []

        # Embed all sentences in one batch
        model = self._get_model()
        embeddings = model.encode(sentences, batch_size=64, show_progress_bar=False)

        # Find breakpoints: adjacent sentence pairs with low similarity
        breakpoints: list[int] = []
        for i in range(len(embeddings) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self._threshold:
                breakpoints.append(i + 1)   # Break before sentence i+1

        # Build raw chunks from breakpoints
        chunks: list[str] = []
        start = 0
        for bp in breakpoints:
            chunk_text = " ".join(sentences[start:bp]).strip()
            if chunk_text:
                chunks.append(chunk_text)
            start = bp
        # Append the final chunk
        final = " ".join(sentences[start:]).strip()
        if final:
            chunks.append(final)

        # Merge chunks that are too short into the previous one
        merged = _merge_short_chunks(chunks, self._min_tokens)

        # Split chunks that are too long using the recursive strategy as fallback
        finalised: list[str] = []
        recursive = RecursiveChunkingStrategy(
            chunk_size=self._max_tokens,
            chunk_overlap=int(self._max_tokens * 0.10),
        )
        for chunk in merged:
            if _count_tokens(chunk) > self._max_tokens:
                finalised.extend(recursive.split(chunk))
            else:
                finalised.append(chunk)

        return finalised


# ──────────────────────────────────────────────────────────────────
# Chunker — Context class that uses a Strategy
# ──────────────────────────────────────────────────────────────────
class Chunker:
    """
    Context class in the Strategy Pattern.

    Accepts any ChunkingStrategy and applies it to a ParsedDocument.
    Wraps raw string chunks in TextChunk dataclasses with metadata.

    Swapping strategies:
        chunker = Chunker(strategy=SemanticChunkingStrategy())
        # Everything downstream is unaffected.
    """

    def __init__(self, strategy: Optional[ChunkingStrategy] = None):
        self._strategy = strategy or RecursiveChunkingStrategy()

    @property
    def strategy(self) -> ChunkingStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy: ChunkingStrategy) -> None:
        """Allow hot-swapping the strategy at runtime."""
        logger.info(
            "Chunking strategy changed",
            old=self._strategy.name,
            new=new_strategy.name,
        )
        self._strategy = new_strategy

    def chunk(self, parsed_doc) -> list[TextChunk]:
        """
        Split a ParsedDocument into TextChunks.

        Args:
            parsed_doc: ParsedDocument from parser.py

        Returns:
            Ordered list of TextChunk objects
        """
        logger.info(
            "Starting chunking",
            filename=parsed_doc.filename,
            strategy=self._strategy.name,
            raw_chars=len(parsed_doc.raw_text),
        )

        raw_chunks = self._strategy.split(parsed_doc.raw_text)

        text_chunks = [
            TextChunk(
                text=chunk_text,
                chunk_index=idx,
                token_count=_count_tokens(chunk_text),
                metadata={
                    "strategy": self._strategy.name,
                    "filename": parsed_doc.filename,
                    "file_type": parsed_doc.file_type,
                    "ocr_used": parsed_doc.ocr_used,
                },
            )
            for idx, chunk_text in enumerate(raw_chunks)
        ]

        logger.info(
            "Chunking complete",
            filename=parsed_doc.filename,
            chunk_count=len(text_chunks),
            avg_tokens=int(
                sum(c.token_count for c in text_chunks) / max(len(text_chunks), 1)
            ),
        )

        return text_chunks


# ──────────────────────────────────────────────────────────────────
# Factory function — builds the right strategy from config
# ──────────────────────────────────────────────────────────────────
def get_chunker(strategy_name: str = "recursive") -> Chunker:
    """
    Factory function to create a Chunker from a strategy name string.

    Design Pattern: Factory — callers request a chunker by name,
    decoupled from concrete class instantiation.

    Args:
        strategy_name: 'recursive' | 'semantic'

    Returns:
        Configured Chunker instance
    """
    from core.config import get_settings
    settings = get_settings()

    if strategy_name == "semantic":
        strategy = SemanticChunkingStrategy()
    else:
        strategy = RecursiveChunkingStrategy(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    return Chunker(strategy=strategy)


# ──────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────
def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base encoding)."""
    return len(_ENCODING.encode(text))


def _split_into_sentences(text: str) -> list[str]:
    """
    Naive but effective sentence splitter for legal text.
    Splits on '. ', '? ', '! ' but avoids splitting on
    common legal abbreviations like 'Inc.', 'Co.', 'Ltd.', 'v.', 'No.'.
    """
    # Replace known abbreviations with placeholders
    abbrevs = ["Inc.", "Corp.", "Ltd.", "Co.", "No.", "Sec.", "Art.", "vs.", "v."]
    placeholder = "<<DOT>>"
    protected = text
    for abbrev in abbrevs:
        protected = protected.replace(abbrev, abbrev.replace(".", placeholder))

    # Split on sentence-ending punctuation
    raw_sentences = re.split(r'(?<=[.!?])\s+', protected)

    # Restore placeholders
    sentences = [s.replace(placeholder, ".") for s in raw_sentences]
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two numpy vectors."""
    import numpy as np
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _merge_short_chunks(chunks: list[str], min_tokens: int) -> list[str]:
    """Merge chunks below min_tokens into the preceding chunk."""
    if not chunks:
        return chunks
    merged: list[str] = [chunks[0]]
    for chunk in chunks[1:]:
        if _count_tokens(chunk) < min_tokens:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)
    return merged
