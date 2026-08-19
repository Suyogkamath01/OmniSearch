"""Explicit, source-pinned Flickr30k metadata acquisition utilities."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from .manifest import CaptionRecord, DatasetManifest, ImageRecord

OFFICIAL_DATA_PAGE = "https://shannon.cs.illinois.edu/DenotationGraph/data/index.html"
OFFICIAL_CAPTION_HTML_URL = (
    "https://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.html"
)
OFFICIAL_TOKEN_ARCHIVE_URL = (
    "https://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.tar.gz"
)
OFFICIAL_IMAGE_ARCHIVE_URL = "https://uofi.box.com/s/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl"
FLICKR_TERMS_URL = "https://www.flickr.com/terms.gne"
CAPTION_LICENSE_NOTE = "The official caption page states Creative Commons Attribution-ShareAlike; image rights remain with Flickr contributors and Flickr terms apply."

_IMAGE_LINK = re.compile(r"(?:^|/)([0-9]+)\.jpg$", re.IGNORECASE)


@dataclass(frozen=True)
class SourceCheck:
    name: str
    url: str
    status_code: int | None
    content_type: str | None
    content_length: int | None
    last_modified: str | None
    accessible: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _head(url: str, name: str) -> SourceCheck:
    request = Request(
        url, headers={"User-Agent": "OmniSearch/0.1 Phase1 preflight"}, method="HEAD"
    )
    try:
        with urlopen(request, timeout=30) as response:
            headers = response.headers
            content_length = headers.get("Content-Length")
            return SourceCheck(
                name=name,
                url=url,
                status_code=getattr(response, "status", None),
                content_type=headers.get_content_type(),
                content_length=int(content_length)
                if content_length and content_length.isdigit()
                else None,
                last_modified=headers.get("Last-Modified"),
                accessible=200 <= response.status < 400,
            )
    except HTTPError as exc:
        return SourceCheck(
            name,
            url,
            exc.code,
            exc.headers.get_content_type(),
            None,
            None,
            False,
            str(exc),
        )
    except (OSError, URLError) as exc:
        return SourceCheck(name, url, None, None, None, None, False, str(exc))


def preflight_flickr30k() -> dict[str, Any]:
    """Check official terms and distribution endpoints using headers only."""

    checks = (
        _head(OFFICIAL_DATA_PAGE, "official_data_page"),
        _head(FLICKR_TERMS_URL, "flickr_terms"),
        _head(OFFICIAL_CAPTION_HTML_URL, "caption_html_metadata"),
        _head(OFFICIAL_TOKEN_ARCHIVE_URL, "tokenized_caption_archive"),
        _head(OFFICIAL_IMAGE_ARCHIVE_URL, "official_image_archive"),
    )
    by_name = {check.name: check for check in checks}
    return {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "legal_note": CAPTION_LICENSE_NOTE,
        "official_data_page": OFFICIAL_DATA_PAGE,
        "checks": [check.to_dict() for check in checks],
        "metadata_accessible": by_name["caption_html_metadata"].accessible,
        "image_archive_accessible": by_name["official_image_archive"].accessible,
        "clean_image_acquisition": by_name["official_image_archive"].accessible,
    }


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_official_metadata(
    output: Path | str, url: str = OFFICIAL_CAPTION_HTML_URL
) -> dict[str, Any]:
    """Download only the official caption metadata after an explicit preflight."""

    if url not in {OFFICIAL_CAPTION_HTML_URL, OFFICIAL_TOKEN_ARCHIVE_URL}:
        raise ValueError(
            "download URL is not an allowlisted official Flickr30k metadata source"
        )
    source_check = _head(url, "requested_metadata_source")
    if not source_check.accessible:
        raise RuntimeError(
            f"official metadata source is not accessible: {source_check.to_dict()}"
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = Request(
        url, headers={"User-Agent": "OmniSearch/0.1 Phase1 metadata acquisition"}
    )
    with urlopen(request, timeout=60) as response, temporary.open("wb") as file:
        while chunk := response.read(1024 * 1024):
            file.write(chunk)
    temporary.replace(output)
    return {
        "url": url,
        "output": str(output),
        "sha256": sha256_file(output),
        "content_length": output.stat().st_size,
        "last_modified": source_check.last_modified,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
    }


class _FlickrCaptionParser(HTMLParser):
    def __init__(self, source_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source_url = source_url
        self.records: list[ImageRecord] = []
        self._current_image_id: str | None = None
        self._current_source_id_available = True
        self._captions: list[str] = []
        self._in_li = False
        self._li_parts: list[str] = []

    def _flush(self) -> None:
        if self._current_image_id is None:
            return
        filename = (
            f"{self._current_image_id}.jpg"
            if self._current_source_id_available
            else None
        )
        captions = tuple(
            CaptionRecord(f"{self._current_image_id}#{index}", text)
            for index, text in enumerate(self._captions)
        )
        self.records.append(
            ImageRecord(
                image_id=self._current_image_id,
                filename=filename,
                captions=captions,
                image_url=urljoin(self.source_url, filename) if filename else None,
                source_image_id_available=self._current_source_id_available,
            )
        )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag.lower() == "a":
            href = attrs_dict.get("href") or ""
            match = _IMAGE_LINK.search(urlparse(href).path)
            if match:
                self._flush()
                self._current_image_id = match.group(1)
                self._current_source_id_available = True
                self._captions = []
        elif tag.lower() == "li" and self._current_image_id is not None:
            self._in_li = True
            self._li_parts = []

    def handle_data(self, data: str) -> None:
        if data.strip().casefold() == "image not found":
            self._flush()
            self._current_image_id = f"missing-source-image-{len(self.records) + 1:06d}"
            self._current_source_id_available = False
            self._captions = []
        if self._in_li:
            self._li_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "li" and self._in_li:
            text = " ".join("".join(self._li_parts).split())
            if text:
                self._captions.append(text)
            self._li_parts = []
            self._in_li = False

    def finish(self) -> tuple[ImageRecord, ...]:
        self._flush()
        return tuple(self.records)


def parse_caption_html(
    source: Path | str, source_url: str = OFFICIAL_CAPTION_HTML_URL
) -> tuple[ImageRecord, ...]:
    parser = _FlickrCaptionParser(source_url)
    parser.feed(Path(source).read_text(encoding="utf-8"))
    parser.close()
    return parser.finish()


def build_manifest(
    records: Iterable[ImageRecord],
    source_path: Path | str,
    source_url: str = OFFICIAL_CAPTION_HTML_URL,
    terms_url: str = FLICKR_TERMS_URL,
    source_last_modified: str | None = None,
) -> DatasetManifest:
    source_path = Path(source_path)
    records = tuple(records)
    return DatasetManifest(
        dataset_id="flickr30k",
        dataset_version="flickr30k-caption-release",
        source_url=source_url,
        terms_url=terms_url,
        # Keep the canonical manifest reproducible. Run timestamps belong in
        # the acquisition artifact, while the manifest uses the source's
        # stable Last-Modified marker (or an explicit sentinel).
        source_snapshot_marker=source_last_modified or "not_recorded",
        source_sha256=sha256_file(source_path),
        records=records,
        metadata={
            "caption_license_note": CAPTION_LICENSE_NOTE,
            "distribution_page": OFFICIAL_DATA_PAGE,
            "source_last_modified": source_last_modified,
            "parsed_record_count": len(records),
            "parsed_caption_count": sum(len(record.captions) for record in records),
            "expected_captions_per_image": 5,
            "image_archive_url": OFFICIAL_IMAGE_ARCHIVE_URL,
            "image_archive_status_at_audit": "HTTP 404; no automatic fallback permitted",
        },
    )
