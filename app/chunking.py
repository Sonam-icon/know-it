"""Heading-aware, token-bounded splitting for technical documentation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.documents import DocumentChunk, LoadedDocument

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_CODE_FENCE = chr(96) * 3


@dataclass(frozen=True, slots=True)
class _Section:
    heading: str
    text: str


class TechnicalDocumentChunker:
    """Split Markdown-like text while retaining headings and nearby code."""

    def __init__(self, chunk_size_tokens: int, chunk_overlap_tokens: int) -> None:
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size_tokens,
            chunk_overlap=chunk_overlap_tokens,
            separators=[
                "\n" + _CODE_FENCE,
                "\n## ",
                "\n### ",
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )

    def split(self, document: LoadedDocument) -> list[DocumentChunk]:
        sections = self._sections(document.text, document.title)
        pieces: list[tuple[str, str]] = []
        for section in sections:
            if self._token_count(section.text) <= self.chunk_size_tokens:
                pieces.append((section.heading, section.text))
                continue
            pieces.extend(
                (section.heading, piece)
                for piece in self._splitter.split_text(section.text)
                if piece.strip()
            )

        packed = self._pack(pieces)
        return [
            DocumentChunk(
                chunk_id=self._chunk_id(document, index),
                document_id=document.document_id,
                title=document.title,
                source=document.source,
                source_kind=document.source_kind,
                heading=heading,
                index=index,
                text=text,
                content_hash=document.content_hash,
            )
            for index, (heading, text) in enumerate(packed)
        ]

    def _sections(self, text: str, fallback_heading: str) -> list[_Section]:
        heading_stack: list[str] = []
        current_heading = fallback_heading
        current_lines: list[str] = []
        sections: list[_Section] = []
        in_fence = False
        fence_marker = ""

        def flush() -> None:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append(_Section(current_heading, content))
            current_lines.clear()

        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith((_CODE_FENCE, "~~~")):
                marker = stripped[:3]
                if not in_fence:
                    in_fence = True
                    fence_marker = marker
                elif marker == fence_marker:
                    in_fence = False

            match = None if in_fence else _HEADING_PATTERN.match(line)
            if match:
                flush()
                level = len(match.group(1))
                title = match.group(2).strip()
                heading_stack[level - 1 :] = [title]
                current_heading = " / ".join(heading_stack)
            current_lines.append(line)
        flush()
        return sections

    def _pack(self, pieces: list[tuple[str, str]]) -> list[tuple[str, str]]:
        packed: list[tuple[str, str]] = []
        current_headings: list[str] = []
        current_texts: list[str] = []

        def flush() -> None:
            if not current_texts:
                return
            unique_headings = list(dict.fromkeys(current_headings))
            packed.append(("; ".join(unique_headings), "\n\n".join(current_texts).strip()))
            current_headings.clear()
            current_texts.clear()

        for heading, text in pieces:
            candidate = "\n\n".join([*current_texts, text])
            if current_texts and self._token_count(candidate) > self.chunk_size_tokens:
                flush()
            current_headings.append(heading)
            current_texts.append(text)
        flush()
        return packed

    def _token_count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    @staticmethod
    def _chunk_id(document: LoadedDocument, index: int) -> str:
        value = f"{document.document_id}:{document.content_hash}:{index}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
