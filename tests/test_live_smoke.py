import os

import pytest

from app.bootstrap import build_runtime_services
from app.config import Settings
from app.rag import AnswerStatus

_live_settings = Settings()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_TESTS") != "1" or not _live_settings.provider_configured,
        reason="Set RUN_LIVE_TESTS=1 and GOOGLE_API_KEY to run live provider tests",
    ),
]


def test_live_ingestion_and_grounded_query(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        google_api_key=_live_settings.google_api_key,
        chat_model=_live_settings.chat_model,
        embedding_model=_live_settings.embedding_model,
        embedding_dimension=_live_settings.embedding_dimension,
        chroma_path=tmp_path / "chroma",
        chroma_collection="live_smoke_test",
    )
    runtime = build_runtime_services(settings)
    runtime.ingestion.ingest_file(
        "smoke.md",
        (
            b"# Widget Configuration\n\n"
            b"Configure a Widget by calling configure_widget with a name argument."
        ),
    )

    result = runtime.query.run("How do I configure a Widget?")

    assert result.status is AnswerStatus.ANSWERED
    assert result.sources
    assert result.sources[0].source == "smoke.md"
