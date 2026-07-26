from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class KnowledgeSourceDocument:
    """One source document supplied to a knowledge synchronization run."""

    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeSyncReport:
    """Operational result of reconciling source files and retrieval indexes."""

    indexed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    scheduled: list[str] = field(default_factory=list)
    expired: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    invalidated_cache_entries: int = 0

    @property
    def changed_count(self) -> int:
        return len(self.indexed) + len(self.expired) + len(self.deleted)
