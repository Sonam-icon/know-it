from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_have_safe_local_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.retrieval_top_k == 5
    assert settings.max_query_rewrites == 2
    assert settings.chroma_path == Path("data/chroma")
    assert settings.provider_configured is False


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="must be smaller"):
        Settings(
            _env_file=None,
            chunk_size_tokens=100,
            chunk_overlap_tokens=100,
        )


def test_comma_separated_google_keys_are_trimmed_and_deduplicated() -> None:
    settings = Settings(
        _env_file=None,
        google_api_key=" key-a, key-b, key-a, ,key-c ",
    )

    assert settings.provider_configured is True
    assert [key.get_secret_value() for key in settings.google_api_keys] == [
        "key-a",
        "key-b",
        "key-c",
    ]
