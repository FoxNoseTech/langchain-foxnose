"""One-shot DeprecationWarning helpers for renamed input fields.

Each ``old_name`` is warned at most once per process so noisy clients (e.g.
notebooks calling the constructor in a loop) don't repeat the message.
"""

from __future__ import annotations

import warnings

_warned: set[str] = set()


def warn_deprecated_field(old_name: str, new_name: str, *, removal: str = "1.0") -> None:
    """Emit a :class:`DeprecationWarning` at most once per process per ``old_name``.

    Args:
        old_name: The kwarg/field being deprecated (e.g. ``"folder_path"``).
        new_name: The replacement name (e.g. ``"collection_path"``).
        removal: The library version where the old name will be removed.
    """
    if old_name in _warned:
        return
    _warned.add(old_name)
    warnings.warn(
        f"langchain-foxnose: {old_name} is deprecated; use {new_name} instead. "
        f"{old_name} will be removed in langchain-foxnose {removal}.",
        DeprecationWarning,
        stacklevel=3,
    )
