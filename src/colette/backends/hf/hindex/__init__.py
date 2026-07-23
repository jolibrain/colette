from .hierarchy_builder import (
    HierarchyBuilder,
    DocumentHierarchy,
    PageRecord,
    TopicNode,
    HierarchyNode,
    DocumentCluster,
    CollectionIndex,
    build_collection_index,
    COLLECTION_THRESHOLD,
)
from .hierarchy_store import HierarchyStore
from .hierarchy_retriever import HierarchyRetriever, _page_key


_HINDEX_ELIGIBLE_MODES = {"embedding_retrieval", "text_search_retrieval", "hybrid"}


def retrieval_mode_uses_hindex(retrieval_mode: str, hindex_cfg=None) -> bool:
    """Return True when hindex retrieval should run."""
    if retrieval_mode == "vision_hierarchical":
        return True
    if (
        retrieval_mode in _HINDEX_ELIGIBLE_MODES
        and hindex_cfg is not None
        and getattr(hindex_cfg, "enable_hindex", False)
    ):
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
    "DocumentCluster",
    "CollectionIndex",
    "build_collection_index",
    "COLLECTION_THRESHOLD",
    "retrieval_mode_uses_hindex",
    "_page_key",
]
