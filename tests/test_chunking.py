import tiktoken

from app.chunking import TechnicalDocumentChunker
from app.documents import LoadedDocument, SourceKind


def _document(text: str) -> LoadedDocument:
    return LoadedDocument(
        document_id="doc-1",
        title="Guide",
        source="guide.md",
        source_kind=SourceKind.FILE,
        text=text,
        content_hash="content-v1",
    )


def test_chunks_are_token_bounded_and_keep_heading_metadata() -> None:
    text = "# Guide\n\n" + " ".join(["overview"] * 20)
    text += "\n\n## Install\n\n" + " ".join(["install"] * 80)
    chunker = TechnicalDocumentChunker(chunk_size_tokens=40, chunk_overlap_tokens=5)

    chunks = chunker.split(_document(text))
    encoding = tiktoken.get_encoding("cl100k_base")

    assert len(chunks) >= 2
    assert all(len(encoding.encode(chunk.text)) <= 40 for chunk in chunks)
    assert any("Install" in chunk.heading for chunk in chunks)


def test_chunk_ids_are_deterministic_and_change_with_document_version() -> None:
    chunker = TechnicalDocumentChunker(chunk_size_tokens=40, chunk_overlap_tokens=5)
    original = _document("# Guide\n\nSome useful content.")
    changed = LoadedDocument(
        document_id=original.document_id,
        title=original.title,
        source=original.source,
        source_kind=original.source_kind,
        text="# Guide\n\nChanged useful content.",
        content_hash="content-v2",
    )

    first = chunker.split(original)
    repeated = chunker.split(original)
    updated = chunker.split(changed)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in repeated]
    assert [chunk.chunk_id for chunk in first] != [chunk.chunk_id for chunk in updated]
