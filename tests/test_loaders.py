import pytest

from app.documents import SourceKind
from app.errors import DocumentTooLargeError, UnsafeUrlError, UnsupportedDocumentError
from app.loaders import DocumentLoader, extract_html, validate_public_url


@pytest.fixture
def loader() -> DocumentLoader:
    return DocumentLoader(
        max_upload_bytes=1024,
        max_fetch_bytes=2048,
        fetch_timeout_seconds=1,
        max_fetch_redirects=1,
    )


def test_markdown_file_is_normalized_with_stable_identity(loader: DocumentLoader) -> None:
    first = loader.load_file("guide.md", b"# Guide\n\n  Install   the package.  ")
    second = loader.load_file("guide.md", b"# Guide\n\n  Install   the package.  ")

    assert first == second
    assert first.title == "Guide"
    assert first.source_kind is SourceKind.FILE
    assert first.text == "# Guide\n\n Install the package."


def test_html_extraction_keeps_content_and_code_but_drops_navigation() -> None:
    html = """
    <html><head><title>Fallback</title></head><body>
      <nav>Skip this link</nav>
      <main>
        <h1>API Guide</h1>
        <p>Call the endpoint.</p>
        <pre><code>client.get("/items")</code></pre>
      </main>
    </body></html>
    """

    text, title = extract_html(html, "fallback")

    assert title == "API Guide"
    assert "Skip this link" not in text
    assert "# API Guide" in text
    assert 'client.get("/items")' in text


def test_file_loader_rejects_unsupported_and_oversized_files(loader: DocumentLoader) -> None:
    with pytest.raises(UnsupportedDocumentError):
        loader.load_file("guide.pdf", b"not a PDF parser")
    with pytest.raises(DocumentTooLargeError):
        loader.load_file("guide.txt", b"x" * 1025)


def test_url_validation_rejects_loopback_hosts() -> None:
    with pytest.raises(UnsafeUrlError, match="public network"):
        validate_public_url("http://127.0.0.1/internal")
