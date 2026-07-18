from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from customer_service_app.domain.schemas import KnowledgeChunk


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.])\s+(?=[A-Z0-9])")
_FAQ_QUESTION_RE = re.compile(r"^(?:Q(?:uestion)?\s*[:：]|问\s*[:：])", re.IGNORECASE)
_FAQ_ANSWER_RE = re.compile(r"^(?:A(?:nswer)?\s*[:：]|答\s*[:：])", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """控制 Markdown 切分粒度和相邻块冗余。"""

    max_chars: int = 900
    min_chars: int = 160
    overlap_chars: int = 120

    def __post_init__(self) -> None:
        if self.max_chars < 200:
            raise ValueError("max_chars must be at least 200")
        if self.min_chars < 0 or self.min_chars >= self.max_chars:
            raise ValueError("min_chars must be non-negative and smaller than max_chars")
        if self.overlap_chars < 0 or self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be non-negative and smaller than max_chars")


@dataclass(slots=True)
class _Section:
    heading_path: list[str]
    heading_marks: list[str]
    body: str
    index: int


@dataclass(slots=True)
class _DraftChunk:
    title: str
    content: str
    section_index: int
    heading_path: list[str]
    overlap_chars: int
    metadata: dict[str, Any] = field(default_factory=dict)


class MarkdownKnowledgeChunker:
    """按 Markdown 结构、段落和句子边界生成可检索知识块。"""

    schema_version = 2

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(
        self,
        *,
        text: str,
        source: str,
        document_metadata: dict[str, Any] | None = None,
    ) -> list[KnowledgeChunk]:
        """把一份 Markdown 文档转换为带稳定 ID 和结构元数据的知识块。"""

        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        front_matter, body = self._extract_front_matter(normalized)
        metadata = {**front_matter, **(document_metadata or {})}
        document_type = self._detect_document_type(source, body, metadata)
        sections = self._parse_sections(body)
        drafts: list[_DraftChunk] = []
        for section in sections:
            drafts.extend(self._chunk_section(section, source=source, document_type=document_type))

        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source))
        result: list[KnowledgeChunk] = []
        for index, draft in enumerate(drafts):
            chunk_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{source}:{draft.section_index}:{index}:{draft.content}",
                )
            )
            chunk_metadata = {
                **metadata,
                **draft.metadata,
                "schema_version": self.schema_version,
                "document_id": document_id,
                "document_type": document_type,
                "section_index": draft.section_index,
                "chunk_index": index,
                "heading_path": draft.heading_path,
                "overlap_chars": draft.overlap_chars,
                "char_count": len(draft.content),
            }
            result.append(
                KnowledgeChunk(
                    id=chunk_id,
                    source=source,
                    title=draft.title,
                    content=draft.content,
                    score=1.0,
                    metadata=chunk_metadata,
                )
            )
        return result

    @staticmethod
    def _extract_front_matter(text: str) -> tuple[dict[str, str], str]:
        """读取简单 YAML front matter；不引入 YAML 依赖。"""

        if not text.startswith("---\n"):
            return {}, text
        end = text.find("\n---\n", 4)
        if end < 0:
            return {}, text
        metadata: dict[str, str] = {}
        for line in text[4:end].splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip():
                metadata[key.strip()] = value.strip().strip('"\'')
        return metadata, text[end + 5 :].strip()

    @staticmethod
    def _parse_sections(text: str) -> list[_Section]:
        """按 ATX 标题拆 section，并保留完整标题路径。"""

        sections: list[_Section] = []
        heading_stack: list[tuple[int, str]] = []
        current_path: list[str] = []
        current_marks: list[str] = []
        body_lines: list[str] = []
        in_fence = False

        def flush() -> None:
            body = "\n".join(body_lines).strip()
            if body:
                sections.append(
                    _Section(
                        heading_path=list(current_path),
                        heading_marks=list(current_marks),
                        body=body,
                        index=len(sections),
                    )
                )
            body_lines.clear()

        for line in text.splitlines():
            if line.strip().startswith("```") or line.strip().startswith("~~~"):
                in_fence = not in_fence
            match = None if in_fence else _HEADING_RE.match(line)
            if match:
                flush()
                level = len(match.group(1))
                title = match.group(2).strip().strip("#").strip()
                heading_stack = [item for item in heading_stack if item[0] < level]
                heading_stack.append((level, title))
                current_path = [item[1] for item in heading_stack]
                current_marks = ["#" * item[0] + " " + item[1] for item in heading_stack]
            else:
                body_lines.append(line)
        flush()

        if sections:
            return sections
        return [_Section(heading_path=[], heading_marks=[], body=text.strip(), index=0)]

    def _chunk_section(
        self,
        section: _Section,
        *,
        source: str,
        document_type: str,
    ) -> list[_DraftChunk]:
        units = self._semantic_units(section.body)
        if not units:
            return []
        title = " / ".join(section.heading_path) or source.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        heading_prefix = "\n".join(section.heading_marks).strip()
        prefix_size = len(heading_prefix) + 2 if heading_prefix else 0
        body_limit = max(self.config.max_chars - prefix_size, 120)

        expanded: list[str] = []
        for unit in units:
            expanded.extend(self._split_oversized_unit(unit, body_limit))

        packed: list[tuple[list[str], int]] = []
        current: list[str] = []
        carried_chars = 0
        for unit in expanded:
            proposed = "\n\n".join([*current, unit])
            if current and len(proposed) > body_limit:
                packed.append((current, carried_chars))
                current = self._overlap_tail(current)
                carried_chars = len("\n\n".join(current))
            current.append(unit)
        if current:
            packed.append((current, carried_chars))

        packed = self._merge_small_tail(packed, body_limit)
        result: list[_DraftChunk] = []
        for units_in_chunk, overlap_chars in packed:
            body = "\n\n".join(units_in_chunk).strip()
            content = f"{heading_prefix}\n\n{body}".strip() if heading_prefix else body
            result.append(
                _DraftChunk(
                    title=title,
                    content=content,
                    section_index=section.index,
                    heading_path=list(section.heading_path),
                    overlap_chars=overlap_chars,
                    metadata={"contains_faq": self._contains_faq(body), "document_type": document_type},
                )
            )
        return result

    def _semantic_units(self, body: str) -> list[str]:
        """先按空行得到段落，再把 FAQ 的问和答合并为不可拆单元。"""

        blocks = [item.strip() for item in re.split(r"\n\s*\n", body) if item.strip()]
        result: list[str] = []
        index = 0
        while index < len(blocks):
            block = blocks[index]
            if _FAQ_QUESTION_RE.match(block) and index + 1 < len(blocks):
                answer = blocks[index + 1]
                if _FAQ_ANSWER_RE.match(answer):
                    result.append(f"{block}\n\n{answer}")
                    index += 2
                    continue
            result.append(block)
            index += 1
        return result

    def _split_oversized_unit(self, unit: str, limit: int) -> list[str]:
        """超长段落优先按句子切；单句仍超长时才使用带重叠的字符窗口。"""

        if len(unit) <= limit:
            return [unit]
        if self._is_atomic_markdown(unit):
            return self._window_split(unit, limit)

        sentences = [item.strip() for item in _SENTENCE_BOUNDARY_RE.split(unit) if item.strip()]
        if len(sentences) <= 1:
            return self._window_split(unit, limit)

        result: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > limit:
                if current:
                    result.append(current)
                    current = ""
                result.extend(self._window_split(sentence, limit))
                continue
            proposed = f"{current}{sentence}" if current else sentence
            if current and len(proposed) > limit:
                result.append(current)
                current = sentence
            else:
                current = proposed
        if current:
            result.append(current)
        return result

    def _window_split(self, text: str, limit: int) -> list[str]:
        step = max(limit - self.config.overlap_chars, 1)
        result: list[str] = []
        for start in range(0, len(text), step):
            item = text[start : start + limit].strip()
            if item:
                result.append(item)
            if start + limit >= len(text):
                break
        return result

    def _overlap_tail(self, units: list[str]) -> list[str]:
        if self.config.overlap_chars == 0:
            return []
        selected: list[str] = []
        size = 0
        for unit in reversed(units):
            remaining = self.config.overlap_chars - size
            if remaining <= 0:
                break
            if len(unit) > remaining:
                selected.append(self._semantic_tail(unit, remaining))
                size = self.config.overlap_chars
                break
            selected.append(unit)
            size += len(unit)
            if size >= self.config.overlap_chars:
                break
        return list(reversed(selected))

    @staticmethod
    def _semantic_tail(text: str, limit: int) -> str:
        """尽量从完整句子边界取得重叠尾部，必要时才截取字符。"""

        sentences = [item.strip() for item in _SENTENCE_BOUNDARY_RE.split(text) if item.strip()]
        selected: list[str] = []
        size = 0
        for sentence in reversed(sentences):
            if selected and size + len(sentence) > limit:
                break
            if len(sentence) > limit:
                return sentence[-limit:]
            selected.append(sentence)
            size += len(sentence)
        return "".join(reversed(selected)) or text[-limit:]

    def _merge_small_tail(
        self,
        packed: list[tuple[list[str], int]],
        limit: int,
    ) -> list[tuple[list[str], int]]:
        if len(packed) < 2:
            return packed
        tail_units, _ = packed[-1]
        if len("\n\n".join(tail_units)) >= self.config.min_chars:
            return packed
        previous_units, previous_overlap = packed[-2]
        merged = "\n\n".join([*previous_units, *tail_units])
        if len(merged) <= limit + self.config.overlap_chars:
            packed[-2] = ([*previous_units, *tail_units], previous_overlap)
            packed.pop()
        return packed

    @staticmethod
    def _is_atomic_markdown(unit: str) -> bool:
        stripped = unit.lstrip()
        lines = unit.splitlines()
        is_fence = stripped.startswith("```") or stripped.startswith("~~~")
        is_table = len(lines) >= 2 and "|" in lines[0] and re.search(r"\|?\s*:?-{3,}", lines[1])
        return bool(is_fence or is_table)

    @staticmethod
    def _contains_faq(text: str) -> bool:
        lines = [line.strip() for line in text.splitlines()]
        return any(_FAQ_QUESTION_RE.match(line) for line in lines) and any(
            _FAQ_ANSWER_RE.match(line) for line in lines
        )

    @staticmethod
    def _detect_document_type(
        source: str,
        body: str,
        metadata: dict[str, Any],
    ) -> str:
        configured = str(metadata.get("type") or metadata.get("document_type") or "").strip()
        if configured:
            return configured.lower()
        sample = f"{source}\n{body[:3000]}".lower()
        lines = [line.strip() for line in body.splitlines()]
        if any(_FAQ_QUESTION_RE.match(line) for line in lines) and any(
            _FAQ_ANSWER_RE.match(line) for line in lines
        ):
            return "faq"
        if any(word in sample for word in ("政策", "规则", "policy")):
            return "policy"
        if any(word in sample for word in ("操作流程", "处理流程", "sop", "步骤")):
            return "sop"
        if any(word in sample for word in ("产品手册", "使用指南", "guide", "manual")):
            return "guide"
        return "knowledge"
