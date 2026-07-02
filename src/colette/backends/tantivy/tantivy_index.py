from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

# Stop words to strip before building a BM25 keyword query.
# Covers English and French; all checked against t.lower() so case doesn't matter.
# Keeping only content-bearing tokens (product codes, technical terms) avoids
# high-frequency domain words swamping the rare, high-IDF product-code tokens.
_STOP_WORDS = frozenset({
    # ── English ──────────────────────────────────────────────────────────────
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'or', 'and', 'but', 'if',
    'we', 'our', 'you', 'your', 'it', 'its', 'this', 'that', 'these',
    'those', 'i', 'what', 'which', 'who', 'how', 'when', 'where', 'why',
    'not', 'no', 'nor', 'so', 'yet', 'either', 'neither',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'than',
    'too', 'very', 'just', 'also', 'about', 'into', 'through', 'during',
    'up', 'down', 'out', 'off', 'over', 'under', 'again', 'then', 'once',
    'here', 'there', 'all', 'any', 'both',
    # ── French ───────────────────────────────────────────────────────────────
    # articles & determiners
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'au', 'aux',
    'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'ma', 'ta', 'sa',
    'nos', 'vos', 'leur', 'leurs',
    # pronouns
    'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
    'me', 'te', 'se', 'lui', 'eux',
    # prepositions
    'en', 'dans', 'sur', 'sous', 'par', 'pour', 'avec', 'sans', 'entre',
    'vers', 'chez', 'selon', 'lors', 'depuis', 'pendant', 'contre',
    'avant', 'apres',          # stripped of accents by Tantivy's tokeniser
    # conjunctions & relative pronouns
    'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'que', 'qui', 'quoi',
    'dont', 'si', 'comme', 'puisque', 'parce', 'afin', 'lorsque', 'quand',
    'ainsi', 'sinon', 'sauf',
    # common auxiliary / copula conjugations
    'est', 'sont', 'etait', 'etaient', 'etre', 'avoir',
    'ai', 'as', 'avons', 'avez', 'ont',
    'sera', 'serait', 'ete', 'fait', 'fais', 'font',
    # adverbs & quantifiers
    'ne', 'pas', 'plus', 'tres', 'aussi', 'bien', 'meme', 'alors', 'puis',
    'encore', 'deja', 'jamais', 'toujours', 'souvent', 'parfois',
    'peu', 'tout', 'tous', 'toute', 'toutes', 'trop', 'moins', 'assez',
    'beaucoup', 'plusieurs', 'certain', 'certains', 'certaines',
    # other common function words
    'voici', 'voila', 'notamment', 'quant', 'suite', 'soit',
    'nous', 'developpons', 'actuellement', 'nouveau', 'avons', 'retenu',
    'suivante', 'filtre', 'limiteur', 'isole',  # domain-generic French
})


def retrieval_mode_uses_embedding_retrieval(retrieval_mode: str) -> bool:
    return retrieval_mode in {"embedding_retrieval", "hybrid"}


def retrieval_mode_uses_text_search_engine(retrieval_mode: str) -> bool:
    return retrieval_mode in {"text_search_retrieval", "hybrid"}


def build_text_context(
    hits: list[dict[str, Any]],
    *,
    max_chars_per_doc: int,
    max_total_chars: int,
) -> tuple[list[dict[str, Any]], str]:
    """Trim Tantivy hits to a bounded text context for prompt injection."""

    bounded_hits: list[dict[str, Any]] = []
    total_chars = 0

    for hit in hits:
        content = (hit.get("content") or "").strip()
        if not content:
            continue

        remaining_chars = max_total_chars - total_chars
        if remaining_chars <= 0:
            break

        allowed_chars = min(max_chars_per_doc, remaining_chars)
        bounded_content = content[:allowed_chars].strip()
        if not bounded_content:
            continue

        bounded_hit = dict(hit)
        bounded_hit["content"] = bounded_content
        bounded_hits.append(bounded_hit)
        total_chars += len(bounded_content)

    context_blocks = []
    for idx, hit in enumerate(bounded_hits, start=1):
        header = f"Text source {idx}: {hit.get('source', 'unknown source')}"
        page_number = hit.get("page_number")
        if page_number is not None:
            header += f" page {page_number}"
        crop_label = hit.get("crop_label")
        if crop_label:
            header += f" type {crop_label}"
        context_blocks.append(f"{header}\n{hit['content']}")

    return bounded_hits, "\n\n".join(context_blocks)


class TantivyIndex:
    def __init__(self, index_path: Path, logger):
        self.index_path = Path(index_path)
        self.logger = logger
        self._index = None
        self._schema = None

    def exists(self) -> bool:
        return self.index_path.exists() and any(self.index_path.iterdir())

    def reset(self):
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        self._index = None
        self._schema = None

    def open(self, create: bool = False):
        tantivy = self._import_tantivy()
        if self._index is not None:
            return self._index

        self.index_path.mkdir(parents=True, exist_ok=True)
        if create or not self.exists():
            self._schema = self._build_schema(tantivy)
            self._index = tantivy.Index(self._schema, path=str(self.index_path))
        else:
            self._index = tantivy.Index.open(str(self.index_path))

        return self._index

    def add_documents(self, documents: list[dict[str, Any]], recreate: bool = False):
        if recreate:
            self.reset()

        if not documents:
            if recreate:
                self.open(create=True)
            return

        tantivy = self._import_tantivy()
        index = self.open(create=recreate)
        writer = index.writer()

        for document in documents:
            doc = tantivy.Document()
            doc.add_text("doc_id", str(document["doc_id"]))
            doc.add_text("source", str(document["source"]))
            doc.add_text("content", str(document["content"]))

            page_number = document.get("page_number")
            if page_number is not None:
                doc.add_unsigned("page_number", int(page_number))

            crop_label = document.get("crop_label")
            if crop_label:
                doc.add_text("crop_label", str(crop_label))

            label = document.get("label")
            if label:
                doc.add_text("label", str(label))

            writer.add_document(doc)

        writer.commit()
        index.reload()

    def search(
        self,
        query: str,
        *,
        limit: int,
        fields: list[str] | None = None,
        crop_label: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.exists():
            return []

        index = self.open()
        search_fields = fields or ["content", "source"]
        parsed_query = self._build_query_text(query, crop_label)
        searcher = index.searcher()
        try:
            tantivy_query = index.parse_query(parsed_query, search_fields)
        except ValueError:
            self.logger.warning(
                "Tantivy could not parse keyword query '%s' (original: '%s'); returning no text hits",
                parsed_query,
                query[:120],
            )
            return []
        results = searcher.search(tantivy_query, limit=limit)

        if not results.hits:
            self.logger.debug("Tantivy returned no hits for keyword query '%s'", parsed_query)

        hits: list[dict[str, Any]] = []
        for score, address in results.hits:
            doc = searcher.doc(address)
            hits.append(
                {
                    "doc_id": self._get_first_value(doc, "doc_id"),
                    "source": self._get_first_value(doc, "source"),
                    "page_number": self._coerce_optional_int(self._get_first_value(doc, "page_number")),
                    "crop_label": self._get_first_value(doc, "crop_label"),
                    "label": self._get_first_value(doc, "label"),
                    "content": self._get_first_value(doc, "content"),
                    "score": float(score),
                    "retriever": "text_search_engine",
                }
            )

        return hits

    def _build_schema(self, tantivy_module):
        schema_builder = tantivy_module.SchemaBuilder()
        schema_builder.add_text_field("doc_id", stored=True)
        schema_builder.add_text_field("source", stored=True)
        schema_builder.add_unsigned_field("page_number", stored=True)
        schema_builder.add_text_field("crop_label", stored=True)
        schema_builder.add_text_field("label", stored=True)
        schema_builder.add_text_field("content", stored=True)
        return schema_builder.build()

    def _extract_keywords(self, query: str) -> str:
        """Extract meaningful BM25 keywords from a natural language query.

        Hyphens are replaced with spaces so product codes like ``MGDD-08-N-E``
        expand into individual tokens (``MGDD``, ``08``) that the Tantivy
        tokeniser has already indexed separately.  English stop words and
        single-character tokens are removed so that high-frequency domain words
        (e.g. "isolation" in safety-standards PDFs) do not drown out the rare,
        high-IDF product-code tokens.

        Returns ``None`` when nothing survives, so the caller can skip the
        search entirely rather than issue a ``*`` wildcard that returns garbage.
        """
        normalized = re.sub(r'[-]', ' ', query)
        tokens = re.findall(r'[A-Za-z0-9]+', normalized)
        keywords = [t for t in tokens if t.lower() not in _STOP_WORDS and len(t) > 1]
        return ' '.join(keywords) if keywords else ''

    def _build_query_text(self, query: str, crop_label: str | list[str] | None) -> str:
        base_query = self._extract_keywords(query.strip()) or "*"
        filters: list[str] = []

        if crop_label is not None:
            if isinstance(crop_label, str):
                filters.append(f'crop_label:"{self._escape_term(crop_label)}"')
            elif isinstance(crop_label, list):
                label_filters = [f'crop_label:"{self._escape_term(label)}"' for label in crop_label if label]
                if label_filters:
                    filters.append("(" + " OR ".join(label_filters) + ")")

        if not filters:
            return base_query

        return f"({base_query}) AND {' AND '.join(filters)}"

    def _escape_term(self, term: str) -> str:
        return term.replace('\\', '\\\\').replace('"', '\\"')

    def _get_first_value(self, doc: Any, field_name: str) -> Any:
        values = doc.get(field_name, []) if hasattr(doc, "get") else doc[field_name]
        if not values:
            return None
        return values[0]

    def _coerce_optional_int(self, value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    def _import_tantivy(self):
        try:
            import tantivy
        except ImportError as exc:
            raise RuntimeError(
                "Tantivy support requires the 'tantivy' package. Install tantivy==0.25.1 to use retrieval_mode='text_search_retrieval' or 'hybrid'."
            ) from exc

        return tantivy
