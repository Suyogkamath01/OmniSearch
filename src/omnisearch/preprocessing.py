"""Dataset-level preprocessing interfaces.

Model-specific tokenizers, image normalization, crops, and augmentations are
intentionally out of scope for Phase 1.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Protocol

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Apply conservative, deterministic normalization to caption text.

    Case and whitespace are normalized, while punctuation and word order are
    preserved. The original caption must remain in the manifest.
    """

    if not isinstance(text, str):
        raise TypeError("caption text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", normalized).strip()


class ImagePreprocessor(Protocol):
    """Interface reserved for later model-specific image transforms."""

    def __call__(self, path: Path) -> object: ...


class IdentityImagePreprocessor:
    """Phase 1-safe image interface that does not alter image bytes."""

    def __call__(self, path: Path) -> Path:
        return Path(path)
