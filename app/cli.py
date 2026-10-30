"""Command-line operations for corpus ingestion and inspection."""

import argparse
import json
from pathlib import Path
from typing import Any

from app.bootstrap import (
    build_document_chunker,
    build_document_loader,
    build_document_store,
    build_ingestion_service,
)
from app.config import get_settings


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(
        isinstance(item, dict) and isinstance(item.get("url"), str) for item in value
    ):
        raise ValueError("Corpus manifest must be a list of objects containing a URL")
    return value


def seed(manifest_path: Path) -> None:
    service = build_ingestion_service(get_settings())
    for entry in _load_manifest(manifest_path):
        result = service.ingest_url(entry["url"])
        print(
            f"{result.status.value:9} {result.document.title} "
            f"({result.document.chunk_count} chunks)"
        )


def list_documents() -> None:
    store = build_document_store(get_settings())
    for document in store.list_documents():
        print(f"{document.title}\t{document.chunk_count}\t{document.source}")


def inspect_manifest(manifest_path: Path) -> None:
    settings = get_settings()
    loader = build_document_loader(settings)
    chunker = build_document_chunker(settings)
    for entry in _load_manifest(manifest_path):
        document = loader.load_url(entry["url"])
        chunks = chunker.split(document)
        print(f"{document.title}\t{len(chunks)} chunks\t{len(document.text)} characters")


def main() -> None:
    parser = argparse.ArgumentParser(description="Know-It corpus administration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    seed_parser = subparsers.add_parser("seed", help="Ingest URLs from a corpus manifest")
    seed_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("corpus/seed_urls.json"),
    )
    subparsers.add_parser("list", help="List indexed documents")
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Load and chunk a manifest without embedding it",
    )
    inspect_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("corpus/seed_urls.json"),
    )
    arguments = parser.parse_args()

    if arguments.command == "seed":
        seed(arguments.manifest)
    elif arguments.command == "inspect":
        inspect_manifest(arguments.manifest)
    else:
        list_documents()


if __name__ == "__main__":
    main()
