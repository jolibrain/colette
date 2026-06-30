from .hierarchy_builder import (
    HierarchyBuilder,
    DocumentHierarchy,
    PageRecord,
    TopicNode,
    HierarchyNode,
)
from .hierarchy_store import HierarchyStore
from .hierarchy_retriever import HierarchyRetriever, _page_key


def retrieval_mode_uses_hindex(retrieval_mode: str, hindex_cfg=None) -> bool:
    """Return True when hindex retrieval should run."""
    if retrieval_mode == "vision_hierarchical":
        return True
    if retrieval_mode == "hybrid" and hindex_cfg is not None and getattr(hindex_cfg, "enable_hindex", False):
        return True
    return False


__all__ = [
    "HierarchyBuilder",
    "HierarchyStore",
    "HierarchyRetriever",
    "DocumentHierarchy",
    "PageRecord",
    "TopicNode",
    "HierarchyNode",
    "retrieval_mode_uses_hindex",
    "_page_key",
]
