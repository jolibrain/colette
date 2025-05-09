import os
from typing import Union

import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor

from .collection import Collection
from .infra.config import ColBERTConfig
from .infra.run import Run
from .modeling.checkpoint import Checkpoint
from .search.index_storage import IndexScorer

TextQueries = Union[str, "list[str]", "dict[int, str]"]


class Searcher:
    def __init__(
        self,
        embedding_lib,
        embedding_model,
        index,
        checkpoint=None,
        collection=None,
        config=None,
        index_root=None,
        verbose: int = 3,
        texts=None,
        metadatas=None,
        logger=None,
        gpu_id=None,
    ):
        self.verbose = verbose
        self.logger = logger
        self.embedding_lib = embedding_lib
        self.embedding_model = embedding_model

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        index_root = index_root if index_root else default_index_root
        self.index = os.path.join(index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.texts = texts
        self.metadatas = metadatas
        self.gpu_id = gpu_id
        if gpu_id is not None:
            dev_str = "cuda:" + str(gpu_id)
        else:
            dev_str = "cpu"

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)

        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        use_gpu = self.gpu_id is not None
        if self.embedding_lib == "colbert":
            self.checkpoint = Checkpoint(
                self.checkpoint,
                colbert_config=self.config,
                verbose=self.verbose,
                gpu_id=self.gpu_id,
            )
            if use_gpu:
                self.checkpoint = self.checkpoint.cuda(self.gpu_id)
        elif self.embedding_lib == "huggingface":
            if "colpali" in self.embedding_model:
                self.processor = ColPaliProcessor.from_pretrained(self.embedding_model, cache_dir=self.checkpoint)
                self.model = ColPaliForRetrieval.from_pretrained(
                    self.embedding_model,
                    torch_dtype=torch.float16,  # colbert idnexing requires float16 and not torch.bfloat16
                    device_map="cuda:0",  # or "mps" if on Apple Silicon
                    cache_dir=self.checkpoint,
                ).eval()
            elif "colqwen2" in self.embedding_model:
                try:
                    from colpali_engine.models import ColQwen2, ColQwen2Processor

                    self.processor = ColQwen2Processor.from_pretrained(self.embedding_model, cache_dir=self.checkpoint)
                    self.model = ColQwen2.from_pretrained(
                        self.embedding_model,
                        torch_dtype=torch.float16,
                        device_map=dev_str,  # or "mps" if on Apple Silicon
                        cache_dir=self.checkpoint,
                    ).eval()
                except (ImportError, ModuleNotFoundError) as e:
                    self.logger.error("colpali-engine is not installed")
                    raise e
            elif "colSmol" in self.embedding_model:
                try:
                    from colpali_engine.models import ColIdefics3, ColIdefics3Processor

                    self.processor = ColIdefics3Processor.from_pretrained(
                        self.embedding_model, cache_dir=self.checkpoint
                    )
                    self.model = ColIdefics3.from_pretrained(
                        self.embedding_model,
                        torch_dtype=torch.float16,
                        device_map=dev_str,
                        attn_implementation="flash_attention_2",  # or eager
                        cache_dir=self.checkpoint,
                    ).eval()
                except (ImportError, ModuleNotFoundError) as e:
                    self.logger.error("colpali-engine is not installed")
                    raise e

            else:
                self.logger.error("unknown embedding model in coldb/searcher")
        load_index_with_mmap = self.config.load_index_with_mmap
        if load_index_with_mmap and use_gpu:
            raise ValueError("Memory-mapped index can only be used with CPU!")
        self.ranker = IndexScorer(self.index, False, load_index_with_mmap, None)

        # print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(self, text: TextQueries, full_length_search=False):
        queries = text if type(text) is list else [text]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True, full_length_search=full_length_search)

        return Q

    def search(self, text: str, k=10, filter_fn=None, full_length_search=False, pids=None):
        if self.embedding_lib == "colbert":
            Q = self.encode(text, full_length_search=full_length_search)
        elif self.embedding_lib in ["huggingface"]:
            queries = self.processor.process_queries([text]).to(self.model.device)
            if "colpali" in self.embedding_model:
                Q = self.model(**queries).embeddings
            else:
                Q = self.model(**queries)
        return self.dense_search(Q.to("cpu").float(), k, filter_fn=filter_fn, pids=pids)

    def dense_search(self, Q: torch.Tensor, k=10, filter_fn=None, pids=None):
        if k <= 10:
            if self.config.ncells is None:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None:
                self.configure(ndocs=256)
        elif k <= 100:
            if self.config.ncells is None:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None:
                self.configure(ndocs=1024)
        else:
            if self.config.ncells is None:
                self.configure(ncells=4)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.4)
            if self.config.ndocs is None:
                self.configure(ndocs=max(k * 4, 4096))

        pids, scores = self.ranker.rank(self.config, Q, filter_fn=filter_fn, pids=pids)

        return pids[:k], list(range(1, k + 1)), scores[:k]
