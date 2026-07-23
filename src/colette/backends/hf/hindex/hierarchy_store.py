"""Persistence layer for DocumentHierarchy trees.

Each document gets its own JSON file under:
    <app_dir>/mm_index/hindex/<doc_name_slug>.json

The store is read-only after indexing — no concurrent writes occur because
hierarchy building is a sequential step in the offline indexing pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

from .hierarchy_builder import CollectionIndex, DocumentHierarchy


# Slug: keep letters, digits, hyphen, underscore; replace everything else with '_'
_SLUG_RE = re.compile(r"[^A-Za-z0-9_\-]")
_COLLECTION_INDEX_FILE = "_collection_index.json"


def _slug(name: str) -> str:
    return _SLUG_RE.sub("_", name)[:200]


class HierarchyStore:
    """Stores and loads DocumentHierarchy objects as JSON files.

    Parameters
    ----------
    index_dir:
        Root index directory (typically ``<app_dir>/mm_index``).
        The store creates a ``hindex/`` subdirectory inside it.
    """

    def __init__(self, index_dir: Path):
        self.hindex_dir = Path(index_dir) / "hindex"
        self.hindex_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, hierarchy: DocumentHierarchy) -> Path:
        """Persist *hierarchy* to disk.  Returns the path of the written file."""
        path = self._path_for(hierarchy.doc_name)
        path.write_text(json.dumps(hierarchy.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, doc_name: str) -> DocumentHierarchy | None:
        """Load the hierarchy for *doc_name*, or return None if not found."""
        path = self._path_for(doc_name)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return DocumentHierarchy.from_dict(data)

    def load_all(self) -> list[DocumentHierarchy]:
        """Load every document hierarchy in the store (skips internal files)."""
        hierarchies: list[DocumentHierarchy] = []
        for p in sorted(self.hindex_dir.glob("*.json")):
            if p.name.startswith("_"):  # skip _collection_index.json and other internals
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                hierarchies.append(DocumentHierarchy.from_dict(data))
            except Exception as exc:
                logger.warning("[hindex] Skipping corrupted hierarchy file '%s': %s", p, exc)
        return hierarchies

    # ------------------------------------------------------------------
    # Collection index (corpus-scale clustering)
    # ------------------------------------------------------------------

    def save_collection_index(self, index: CollectionIndex) -> Path:
        """Persist the collection index to disk."""
        path = self.hindex_dir / _COLLECTION_INDEX_FILE
        path.write_text(
            json.dumps(index.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def load_collection_index(self) -> CollectionIndex | None:
        """Load the collection index, or return None if not yet built."""
        path = self.hindex_dir / _COLLECTION_INDEX_FILE
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CollectionIndex.from_dict(data)
        except Exception as exc:
            logger.warning("[hindex] Failed to load collection index: %s", exc)
            return None

    def collection_index_exists(self) -> bool:
        return (self.hindex_dir / _COLLECTION_INDEX_FILE).exists()

    def exists(self, doc_name: str) -> bool:
        return self._path_for(doc_name).exists()

    def delete(self, doc_name: str) -> None:
        path = self._path_for(doc_name)
        if path.exists():
            path.unlink()

    def list_doc_names(self) -> list[str]:
        return [p.stem for p in sorted(self.hindex_dir.glob("*.json"))]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path_for(self, doc_name: str) -> Path:
        return self.hindex_dir / f"{_slug(doc_name)}.json"
