"""Persistence layer for DocumentHierarchy trees.

Each document gets its own JSON file under:
    <app_dir>/mm_index/hindex/<doc_name_slug>.json

The store is read-only after indexing — no concurrent writes occur because
hierarchy building is a sequential step in the offline indexing pipeline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .hierarchy_builder import DocumentHierarchy


# Slug: keep letters, digits, hyphen, underscore; replace everything else with '_'
_SLUG_RE = re.compile(r"[^A-Za-z0-9_\-]")


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
        """Load every hierarchy in the store."""
        hierarchies: list[DocumentHierarchy] = []
        for p in sorted(self.hindex_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                hierarchies.append(DocumentHierarchy.from_dict(data))
            except Exception:
                pass  # corrupted file — skip silently
        return hierarchies

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
