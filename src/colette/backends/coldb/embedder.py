import os
import tarfile
import time
import urllib

import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor

from .indexing.collection_encoder import CollectionEncoder
from .infra.config.config import ColBERTConfig
from .infra.run import Run
from .modeling.checkpoint import Checkpoint
from .utils.utils import batch


class Embedder:
    def __init__(
        self,
        app_directory,
        embedding_lib,
        embedding_model,
        model_path,
        logger,
        index_bsize=-1,
        gpu_id=None,
        image_width=512,
        image_height=512,
        kvstore=None,
    ):
        self.app_directory = app_directory
        self.embedding_lib = embedding_lib  # colbert or colpali
        self.embedding_model = embedding_model
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.image_width = image_width
        self.image_height = image_height
        self.kvstore = kvstore
        if self.gpu_id is not None:
            self.device_str = "cuda:" + str(gpu_id)
        else:
            self.device_str = "cpu"

        self.logger = logger

        if self.embedding_lib == "colbert":
            if index_bsize == -1:
                self.index_bsize = 64
            else:
                self.index_bsize = index_bsize
            if not os.path.exists(self.model_path):
                self.logger.info("downloading colbertv2.0 model")
                # url = "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz"
                url = "https://www.jolibrain.com/stuff/models/colbertv2.0.tar.gz"
                filehandle, _ = urllib.request.urlretrieve(url)
                tf = tarfile.open(filehandle, "r")
                os.makedirs(self.model_path, exist_ok=True)
                tf.extractall(self.model_path + "/..")
                os.remove(filehandle)

            self.checkpoint_path = str(self.model_path)
            self.checkpoint_config = ColBERTConfig.load_from_checkpoint(
                self.checkpoint_path
            )
            self.config = ColBERTConfig.from_existing(
                self.checkpoint_config, None, Run().config
            )
        elif self.embedding_lib in ["huggingface"]:
            self.config = ColBERTConfig()
            self.config.nbits = 2
            if index_bsize == -1:
                self.index_bsize = 2
            else:
                self.index_bsize = index_bsize
            # self.configure(checkpoint=self.checkpoint)
        self.config.model_name = self.embedding_lib
        self.use_gpu = self.gpu_id is not None
        self.logger.debug(f"coldb : indexing with batch size {self.index_bsize}")

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def init_model(self):  # done in each thread
        if self.embedding_lib == "colbert":
            self.checkpoint = Checkpoint(
                self.checkpoint_path, colbert_config=self.config, gpu_id=self.gpu_id
            )
            if self.use_gpu:
                self.checkpoint = self.checkpoint.cuda(self.gpu_id)

            self.encoder = CollectionEncoder(self.checkpoint, self.index_bsize)
        elif self.embedding_lib == "huggingface":
            if "colpali" in self.embedding_model:
                self.processor = ColPaliProcessor.from_pretrained(
                    self.embedding_model, cache_dir=self.model_path
                )
                self.model = ColPaliForRetrieval.from_pretrained(
                    self.embedding_model,
                    torch_dtype=torch.bfloat16,  # colbert idnexing requires float16 and not torch.bfloat16
                    device_map=self.device_str,  # or "mps" if on Apple Silicon
                    cache_dir=self.model_path,
                ).eval()
            elif "colqwen2" in self.embedding_model:
                try:
                    from colpali_engine.models import ColQwen2, ColQwen2Processor

                    self.processor = ColQwen2Processor.from_pretrained(
                        self.embedding_model, cache_dir=self.model_path
                    )
                    self.model = ColQwen2.from_pretrained(
                        self.embedding_model,
                        torch_dtype=torch.bfloat16,
                        device_map=self.device_str,  # or "mps" if on Apple Silicon
                        cache_dir=self.model_path,
                    ).eval()
                except (ImportError, ModuleNotFoundError) as e:
                    self.logger.error(
                        "colpali-engine (mandatory for colqwen2 for indexing) is not installed"
                    )
                    raise e
            elif "colSmol" in self.embedding_model:
                try:
                    from colpali_engine.models import ColIdefics3, ColIdefics3Processor

                    self.logger.debug(f"using {self.embedding_model} embedding_model")
                    self.processor = ColIdefics3Processor.from_pretrained(
                        self.embedding_model, cache_dir=self.model_path
                    )
                    self.model = ColIdefics3.from_pretrained(
                        self.embedding_model,
                        torch_dtype=torch.bfloat16,
                        device_map=self.device_str,
                        attn_implementation="flash_attention_2",  # or eager
                        cache_dir=self.model_path,
                    ).eval()

                except (ImportError, ModuleNotFoundError) as e:
                    self.logger.error(
                        "colpali-engine (mandatory for idefics3 for indexing) is not installed"
                    )
                    raise e
            else:
                self.logger.error("unknown embedding model in coldb/embedder")

    def encode_passages(self, passages):
        if self.embedding_lib == "colbert":
            return self.encoder.encode_passages(passages)
        if self.embedding_lib in ["huggingface"]:
            embs, doclens = [], []
            count = 0
            self.logger.debug(
                f"coldb: embedding {len(passages)} passages {self.index_bsize} at a time"
            )
            for batchp in batch(passages, self.index_bsize):
                count += self.index_bsize
                # if count > len(passages):
                #     count = passages
                self.logger.debug(
                    f"coldb: embedding {count}/{len(passages)} : {str(batchp)}"
                )
                start = time.time()
                batchi = []
                for p in batchp:
                    im = self.kvstore.retrieve_image(p)
                    w, h = im.size
                    if w > self.image_width:
                        h = round(h * self.image_width / w)
                        w = self.image_width
                    if h > self.image_height:
                        w = round(w / h * self.image_height)
                        h = self.image_height
                    batchi.append(im.resize((w, h)))

                batch_images = self.processor.process_images(batchi).to(
                    torch.device(self.device_str)
                )
                self.logger.debug(f"coldb preprocess took {time.time()-start} s")

                start = time.time()
                if "colpali" in self.embedding_model:
                    embeddings = self.model(**batch_images).embeddings.to(torch.float16)
                else:
                    embeddings = self.model(**batch_images).to(torch.float16)
                self.logger.debug(f"coldb embedding took {time.time()-start} s")
                for i in range(embeddings.shape[0]):
                    embs.append(embeddings[i])
                    doclens.append(embeddings.shape[1])

            embs = torch.cat(embs)
            return embs, doclens
