"""Safe document loading and lightweight HTML-to-Markdown extraction."""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup, Tag

from app.documents import LoadedDocument, SourceKind
from app.errors import (
    DocumentLoadError,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsafeUrlError,
    UnsupportedDocumentError,
)

_ALLOWED_FILE_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm"}
_ALLOWED_CONTENT_TYPES = {"text/html", "text/plain", "text/markdown"}
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_SPACE_PATTERN = re.compile(r"[ \t]+")
_CODE_FENCE = chr(96) * 3


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        marker = line.lstrip()[:3]
        if marker in {_CODE_FENCE, "~~~"}:
            lines.append(line.rstrip())
            if marker == fence:
                fence = None
            elif fence is None:
                fence = marker
        elif fence is not None:
            lines.append(line)
        else:
            lines.append(_SPACE_PATTERN.sub(" ", line).rstrip())
    return "\n".join(lines).strip()


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        candidate = line.lstrip("#").strip()
        if candidate:
            return candidate[:200]
    return fallback


def _build_loaded_document(
    *,
    text: str,
    title: str,
    source: str,
    source_kind: SourceKind,
) -> LoadedDocument:
    normalized = _normalize_text(text)
    if not normalized:
        raise EmptyDocumentError(f"No indexable content was found in {source}")
    canonical_source = source.strip()
    return LoadedDocument(
        document_id=_sha256(f"{source_kind.value}:{canonical_source}"),
        title=title.strip()[:200] or canonical_source,
        source=canonical_source,
        source_kind=source_kind,
        text=normalized,
        content_hash=_sha256(normalized),
    )


def extract_html(html: str, fallback_title: str) -> tuple[str, str]:
    """Extract the main technical content as Markdown-like text."""

    soup = BeautifulSoup(html, "html.parser")
    root = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )
    for element in root.select(
        "script, style, nav, footer, aside, form, noscript, svg, [aria-hidden='true'], .headerlink"
    ):
        element.decompose()
    first_heading = root.find("h1")
    title = (
        (first_heading.get_text(" ", strip=True) if first_heading else "")
        or (soup.title.get_text(" ", strip=True) if soup.title else "")
        or fallback_title
    )

    output: list[str] = []
    block_names = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "ul", "ol", "table"}
    container_names = {"p", "pre", "ul", "ol", "table"}
    for element in root.find_all(block_names):
        if not isinstance(element, Tag) or element.find_parent(container_names) is not None:
            continue

        if element.name and element.name.startswith("h"):
            level = int(element.name[1])
            text = element.get_text(" ", strip=True)
            if text:
                output.append(f"{'#' * level} {text}")
        elif element.name == "pre":
            code = element.get_text("\n", strip=False).strip("\n")
            if code:
                output.append(f"{_CODE_FENCE}\n{code}\n{_CODE_FENCE}")
        elif element.name in {"ul", "ol"}:
            items: list[str] = []
            for number, item in enumerate(element.find_all("li", recursive=False), start=1):
                prefix = f"{number}." if element.name == "ol" else "-"
                items.append(f"{prefix} {item.get_text(' ', strip=True)}")
            if items:
                output.append("\n".join(items))
        elif element.name == "table":
            rows = [
                [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                for row in element.find_all("tr")
            ]
            rows = [row for row in rows if row]
            if rows:
                output.append("\n".join(" | ".join(row) for row in rows))
        else:
            text = element.get_text(" ", strip=True)
            if text:
                output.append(text)

    return "\n\n".join(output), title


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise UnsafeUrlError("Only absolute HTTP and HTTPS URLs are accepted")
    hostname = parsed.hostname.lower()
    host_for_netloc = f"[{hostname}]" if ":" in hostname else hostname
    default_port = (parsed.scheme.lower() == "http" and parsed.port == 80) or (
        parsed.scheme.lower() == "https" and parsed.port == 443
    )
    netloc = (
        host_for_netloc
        if parsed.port is None or default_port
        else f"{host_for_netloc}:{parsed.port}"
    )
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""))


def validate_public_url(url: str) -> str:
    """Reject URLs resolving to local, private, or otherwise non-public addresses."""

    canonical = canonicalize_url(url)
    parsed = urlsplit(canonical)
    assert parsed.hostname is not None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = {
            ipaddress.ip_address(item[4][0])
            for item in socket.getaddrinfo(
                parsed.hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        }
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"Could not resolve URL host: {parsed.hostname}") from exc

    if not addresses or any(not address.is_global for address in addresses):
        raise UnsafeUrlError("URL host must resolve only to public network addresses")
    return canonical


class DocumentLoader:
    """Load bounded files and public URLs into a common document representation."""

    def __init__(
        self,
        *,
        max_upload_bytes: int,
        max_fetch_bytes: int,
        fetch_timeout_seconds: float,
        max_fetch_redirects: int,
    ) -> None:
        self.max_upload_bytes = max_upload_bytes
        self.max_fetch_bytes = max_fetch_bytes
        self.fetch_timeout_seconds = fetch_timeout_seconds
        self.max_fetch_redirects = max_fetch_redirects

    def load_file(self, filename: str, data: bytes) -> LoadedDocument:
        if len(data) > self.max_upload_bytes:
            raise DocumentTooLargeError(
                f"File exceeds the {self.max_upload_bytes}-byte upload limit"
            )
        suffix = Path(filename).suffix.lower()
        if suffix not in _ALLOWED_FILE_SUFFIXES:
            raise UnsupportedDocumentError(
                f"Unsupported file type {suffix or '(none)'}; use Markdown, text, or HTML"
            )
        try:
            decoded = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentLoadError("Document must use UTF-8 text encoding") from exc

        fallback_title = Path(filename).stem or "Uploaded document"
        if suffix in {".html", ".htm"}:
            text, title = extract_html(decoded, fallback_title)
        else:
            text = decoded
            title = _title_from_text(decoded, fallback_title)
        return _build_loaded_document(
            text=text,
            title=title,
            source=filename,
            source_kind=SourceKind.FILE,
        )

    def load_url(self, url: str) -> LoadedDocument:
        current_url = validate_public_url(url)
        with httpx.Client(
            follow_redirects=False,
            timeout=self.fetch_timeout_seconds,
            headers={"User-Agent": "Know-It/0.1 documentation indexer"},
        ) as client:
            for redirect_count in range(self.max_fetch_redirects + 1):
                with client.stream("GET", current_url) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if location is None:
                            raise DocumentLoadError("Redirect response did not include a location")
                        if redirect_count >= self.max_fetch_redirects:
                            raise DocumentLoadError("URL exceeded the redirect limit")
                        current_url = validate_public_url(urljoin(current_url, location))
                        continue

                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        raise DocumentLoadError(
                            f"URL returned HTTP {response.status_code}"
                        ) from exc

                    media_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if media_type not in _ALLOWED_CONTENT_TYPES:
                        raise UnsupportedDocumentError(
                            f"Unsupported URL content type: {media_type or '(missing)'}"
                        )
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > self.max_fetch_bytes:
                        raise DocumentTooLargeError(
                            f"URL exceeds the {self.max_fetch_bytes}-byte fetch limit"
                        )
                    body = bytearray()
                    for part in response.iter_bytes():
                        body.extend(part)
                        if len(body) > self.max_fetch_bytes:
                            raise DocumentTooLargeError(
                                f"URL exceeds the {self.max_fetch_bytes}-byte fetch limit"
                            )
                    final_url = canonicalize_url(str(response.url))
                    encoding = response.encoding or "utf-8"
                    break
            else:  # pragma: no cover - the bounded loop always exits or raises
                raise DocumentLoadError("Unable to load URL")

        try:
            decoded = bytes(body).decode(encoding)
        except (UnicodeDecodeError, LookupError) as exc:
            raise DocumentLoadError("URL content must use a supported text encoding") from exc

        fallback_title = urlsplit(final_url).path.rstrip("/").rsplit("/", 1)[-1] or final_url
        if media_type == "text/html":
            text, title = extract_html(decoded, fallback_title)
        else:
            text = decoded
            title = _title_from_text(decoded, fallback_title)
        return _build_loaded_document(
            text=text,
            title=title,
            source=final_url,
            source_kind=SourceKind.URL,
        )
