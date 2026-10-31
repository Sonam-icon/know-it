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
    build_query_service,
)
from app.config import get_settings
from app.evaluation import (
    load_evaluation_cases,
    run_evaluation,
    write_evaluation_summary,
)


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


def query(question: str) -> None:
    result = build_query_service(get_settings()).run(question)
    print(result.answer)
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"- [{source.citation_id}] {source.title} — {source.heading} ({source.source})")


def evaluate(dataset_path: Path, output_path: Path) -> None:
    cases = load_evaluation_cases(dataset_path)
    summary = run_evaluation(build_query_service(get_settings()), cases)
    write_evaluation_summary(summary, output_path)
    print(
        f"Passed {summary.passed}/{summary.total} cases "
        f"({summary.pass_rate:.1%}); average latency "
        f"{summary.average_latency_ms:.2f} ms"
    )
    print(f"Report: {output_path}")


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
    query_parser = subparsers.add_parser("query", help="Run a question through the RAG graph")
    query_parser.add_argument("question")
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run the versioned RAG evaluation dataset",
    )
    evaluate_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/questions.json"),
    )
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation.json"),
    )
    arguments = parser.parse_args()

    if arguments.command == "seed":
        seed(arguments.manifest)
    elif arguments.command == "inspect":
        inspect_manifest(arguments.manifest)
    elif arguments.command == "query":
        query(arguments.question)
    elif arguments.command == "evaluate":
        evaluate(arguments.dataset, arguments.output)
    else:
        list_documents()


if __name__ == "__main__":
    main()
