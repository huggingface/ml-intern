"""Small Tantivy wrapper for local, snippet-first research search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tantivy


@dataclass(frozen=True)
class SearchHit:
    score: float
    fields: dict[str, Any]


class TantivyTextIndex:
    """A compact text index with stored metadata fields.

    The wrapper keeps Tantivy-specific details out of tool handlers and gives the
    research tools one simple operation: add stored documents, then search them
    with BM25 ranking and field boosts.
    """

    def __init__(
        self,
        *,
        text_fields: list[str],
        stored_fields: list[str] | None = None,
        field_boosts: dict[str, float] | None = None,
        path: Path | None = None,
    ) -> None:
        self.text_fields = text_fields
        self.stored_fields = list(dict.fromkeys([*(stored_fields or []), *text_fields]))
        self.field_boosts = field_boosts or {}

        builder = tantivy.SchemaBuilder()
        for field in self.stored_fields:
            tokenizer = "en_stem" if field in text_fields else "default"
            builder.add_text_field(field, stored=True, tokenizer_name=tokenizer)
        self.schema = builder.build()

        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            self.index = tantivy.Index(self.schema, path=str(path))
        else:
            self.index = tantivy.Index(self.schema)

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            return

        writer = self.index.writer(heap_size=30_000_000, num_threads=1)
        for item in documents:
            doc = tantivy.Document()
            for field in self.stored_fields:
                value = item.get(field, "")
                if value is None:
                    value = ""
                doc.add_text(field, str(value))
            writer.add_document(doc)
        writer.commit()
        writer.wait_merging_threads()
        self.index.reload()

    def search(self, query: str, *, limit: int = 10) -> tuple[list[SearchHit], list[str]]:
        clean_query = (query or "").strip()
        if not clean_query:
            return [], []

        parsed, errors = self.index.parse_query_lenient(
            clean_query,
            self.text_fields,
            field_boosts=self.field_boosts,
        )
        searcher = self.index.searcher()
        results = searcher.search(parsed, limit)
        hits: list[SearchHit] = []
        for score, address in results.hits:
            doc = searcher.doc(address).to_dict()
            hits.append(SearchHit(score=float(score), fields=_flatten_doc(doc)))
        return hits, [str(err) for err in errors]


def _flatten_doc(doc: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in doc.items():
        if isinstance(value, list):
            flattened[key] = value[0] if value else ""
        else:
            flattened[key] = value
    return flattened
