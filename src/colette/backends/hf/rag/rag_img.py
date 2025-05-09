import base64
import hashlib
import json
import logging
from io import BytesIO
from pathlib import Path
from time import time
from typing import Any, cast

import chromadb
import torch
from chromadb.api.types import (
    Embedding,
    EmbeddingFunction,
    Embeddings,
)
from chromadb.config import Settings
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from vllm import LLM as VLLM

from colette.apidata import InputConnectorObj
from colette.backends.coldb import ColDB
from colette.inputconnector import InputConnectorBadParamException

from ..layout_detector import LayoutDetector
from ..model_cache import ModelCache
from ..preprocessing import DocumentProcessor, ImageProcessor


# XXX: chromadb image handling is broken/embryonary
# we hack their image validation in order to pass arbitrary
# data to our custom embedding function
def is_image(target: Any) -> bool:
    return True


def compute_sha256_hash(file_path):
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def transform_pil_image_to_base64(image):
    #
    # Transform PIL image to base64
    # @param image: PIL image
    # @return: str
    #
    buffered = BytesIO()
    # extract image format from image
    image_type = image.format
    image.save(buffered, format=image_type)
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{image_type.lower()};base64,{encoded_string}"


chromadb.api.types.is_image = is_image


def get_md5sum(file_path, kvstore):
    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Open the file in binary read mode
    img = kvstore.retrieve_image(file_path.encode("utf-8", "replace").decode())
    md5_hash.update(img.tobytes())

    # Return the hexadecimal digest of the hash
    return md5_hash.hexdigest()


def sort_and_select_top_k(
    data: dict[str, list[list[Any]]], k: int, remove_duplicates: bool, kvstore, logger
) -> dict[str, list[list[Any]]]:
    """
    Sorts the dictionary based on the 'distances' key and reorders all other keys accordingly.
    Returns a new dictionary containing only the top k elements.

    Args:
        data (dict[str, list[list[Any]]]): The input dictionary with lists of lists.
        k (int): The number of top elements to return.

    Returns:
        dict[str, list[list[Any]]]: A new dictionary containing the top k elements.
    """
    if "distances" not in data or not isinstance(data["distances"], list) or not data["distances"]:
        raise KeyError("'distances' key is missing or improperly formatted.")

    distances = data["distances"][0]  # Extract the first inner list
    if not isinstance(distances, list):
        raise ValueError("'distances' should be a list of lists.")

    # Ensure that distances contains valid numeric values
    if not all(isinstance(d, int | float) for d in distances):
        raise ValueError("All elements in 'distances' must be numeric.")

    # Sort indices based on distance values
    sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])

    # Gather sources
    sorted_sources = []
    for key, nested_list in data.items():
        if key != "ids":
            continue

        # Skip None values
        if nested_list is None:
            # sorted_data[key] = None
            continue

        # Check if the key holds a list of lists
        if isinstance(nested_list, list) and nested_list and isinstance(nested_list[0], list):
            flat_list = nested_list[0]
            for s in flat_list:
                sorted_sources.append(s)

    # check for successive duplicates and remove them
    if remove_duplicates:
        duplicate_indices = []
        for i in range(len(sorted_sources) - 1):
            if get_md5sum(sorted_sources[i], kvstore) == get_md5sum(sorted_sources[i + 1], kvstore):
                duplicate_indices.append(i)
        if len(duplicate_indices) > 0:
            logger.debug("Found %d duplicates", len(duplicate_indices))

        # sorted_indices_bak = sorted_indices.copy()
        for i in sorted(duplicate_indices, reverse=True):
            del sorted_indices[i]

    top_k_indices = sorted_indices[:k]
    sorted_data = {}

    for key, nested_list in data.items():
        # Skip None values
        if nested_list is None:
            sorted_data[key] = None
            continue

        # Check if the key holds a list of lists
        if isinstance(nested_list, list) and nested_list and isinstance(nested_list[0], list):
            flat_list = nested_list[0]
            sorted_list = [flat_list[i] for i in top_k_indices if i < len(flat_list)]
            sorted_data[key] = [sorted_list]
        else:
            # Preserve non-list values as they are
            sorted_data[key] = nested_list

    return sorted_data


class ImageEmbeddingFunction(EmbeddingFunction):
    def __init__(self, ad: InputConnectorObj, models_repository, logger):
        self.device = ad.rag.gpu_id
        # embedder
        if ad.rag.embedding_model is not None:
            self.rag_embedding_model = ad.rag.embedding_model
        else:
            self.rag_embedding_model = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
        self.shared = ad.rag.shared_model
        self.rag_image_width = ad.rag.ragm.image_width
        self.rag_image_height = ad.rag.ragm.image_height
        self.rag_auto_scale_for_font = ad.rag.ragm.auto_scale_for_font
        self.logger = logger
        self.vllm = ad.rag.embedding_lib == "vllm"
        self.vllm_memory_utilization = ad.rag.vllm_rag_memory_utilization
        self.vllm_quantization = ad.rag.vllm_rag_quantization
        self.vllm_enforce_eager = ad.rag.vllm_rag_enforce_eager

        # min/max image size
        min_pixels = 1 * 28 * 28
        max_pixels = 2560 * 28 * 28

        # load model
        ## only qwen2vl-based embedder for now
        expected_prefix = "Alibaba-NLP/gme-Qwen2-VL"
        if not self.vllm and not self.rag_embedding_model.startswith(expected_prefix):
            self.logger.warning("rag.embedding_model should be " + expected_prefix)
        self.model = None
        if self.shared:
            cache_key = ("vllm_embed" if self.vllm else "huggingface", self.rag_embedding_model)
            cached_model = ModelCache.get(cache_key)
            if cached_model:
                self.model, self.processor, self.llm_lib = cached_model
                self.logger.info("Reusing cached LLM embedder for %s", self.rag_embedding_model)
                return

        # not cached or not shared, load the model up
        if not self.model:
            if self.vllm:
                if self.vllm_quantization == "bitsandbytes":
                    self.vllm_load_format = "bitsandbytes"
                else:
                    self.vllm_load_format = "auto"
                self.model = VLLM(
                    model=self.rag_embedding_model,
                    download_dir=models_repository,
                    load_format=self.vllm_load_format,
                    quantization=self.vllm_quantization,
                    max_model_len=4096,
                    max_num_seqs=5,
                    task="embed",
                    mm_processor_kwargs={
                        "min_pixels": 28 * 28,
                        "max_pixels": 1280 * 28 * 28,
                        "fps": 1,
                    },
                    disable_mm_preprocessor_cache=False,
                    gpu_memory_utilization=self.vllm_memory_utilization,
                    limit_mm_per_prompt={"image": 1},
                    enforce_eager=self.vllm_enforce_eager,
                )
            else:
                self.processor = AutoProcessor.from_pretrained(
                    self.rag_embedding_model,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    cache_dir=str(models_repository),
                )
                self.model = (
                    Qwen2VLForConditionalGeneration.from_pretrained(
                        self.rag_embedding_model,
                        attn_implementation="flash_attention_2",
                        torch_dtype=torch.bfloat16,
                        cache_dir=str(models_repository),
                    )
                    .to("cuda:" + str(self.device))
                    .eval()
                )

        if not self.vllm:
            # https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct/blob/main/gme_inference.py#L39
            self.processor.tokenizer.padding_side = "right"
            self.model.padding_side = "left"

        # Cache the embedder for future use
        if self.shared:
            if self.vllm:
                ModelCache.add(cache_key, self.model, None, self.rag_embedding_model)
            else:
                ModelCache.add(cache_key, self.model, self.processor, "qwen2vl")

    def get_embedding(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        reps = last_hidden_state[:, -1]
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def get_embedding_vllm(self, embeddings: torch.Tensor) -> torch.Tensor:
        reps = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return reps

    def __call_vllm__(self, input: dict) -> Embeddings:
        mm_inputs = []
        for item in input:
            if type(item) is dict:
                label = item.get("label", None)
                item = item.get("doc", None)
            else:
                label = None
            if "PIL" in str(type(item)):  # or "Png" in str(type(item)) or "Jpeg" in str(type(item)):
                if not self.rag_auto_scale_for_font and self.rag_image_width and self.rag_image_height:
                    width, height = item.size
                    if width > self.rag_image_width or height > self.rag_image_height:
                        item.thumbnail((self.rag_image_width, self.rag_image_height))
                if self.rag_auto_scale_for_font and (self.rag_image_width or self.rag_image_height):
                    self.logger.warn("Auto scaling for font is enabled. image_width and image_height are ignored")
                question = "What is the content of this image?"
                if label:
                    question = "This image has a " + label + ". " + question
                placeholder = "<|image_pad|>"
                prompt = (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                    f"{question}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                mm_inputs.append(
                    {
                        "data": item,
                        "prompt": prompt,
                    }
                )
            else:
                question = item
                placeholder = "<|image_pad|>"
                prompt = (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                    f"{question}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )

                mm_inputs.append({"data": Image.new("RGB", (28, 28)), "prompt": prompt})

        inputs = [{"prompt": mmi["prompt"], "multi_modal_data": {"image": mmi["data"]}} for mmi in mm_inputs]
        outputs = self.model.embed(inputs)
        embeddings = [o.outputs.embedding for o in outputs]
        doc_embeddings = self.get_embedding_vllm(torch.tensor(embeddings))
        db_embeddings = cast(Embedding, doc_embeddings.squeeze().tolist())  ##TODO: beware multiple docs...
        return db_embeddings

    def __call__(self, input: dict) -> Embeddings:
        if self.vllm:
            return self.__call_vllm__(input)
        # prepare data
        doc_messages = []

        for item in input:
            # if is_image(input): ##XXX: fails to detect PIL image, checks for np array...
            if type(item) is dict:
                label = item.get("label", None)
                item = item.get("doc", None)
            else:
                label = None
            if "PIL" in str(type(item)):  # or "Png" in str(type(item)) or "Jpeg" in str(type(item)):
                if not self.rag_auto_scale_for_font and self.rag_image_width and self.rag_image_height:
                    width, height = item.size
                    if width > self.rag_image_width or height > self.rag_image_height:
                        item.thumbnail((self.rag_image_width, self.rag_image_height))

                embed_prompt = ""  # gme-Qwen2-VL does not have a user prompt by default
                if label:
                    embed_prompt = "This image has a " + label + ". " + embed_prompt
                message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": item,
                            },
                            #'resized_height':xxx , 'resized_width':xxx
                            # adjust the image size for efficiency trade-off
                            {
                                "type": "text",
                                "text": embed_prompt,
                            },
                        ],
                    }
                ]
            else:
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": item},
                        ],
                    }
                ]
        doc_messages.append(message)

        # print("doc_messages=", doc_messages)

        doc_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
            for msg in doc_messages
        ]

        # print("doc_texts=", doc_texts)

        doc_image_inputs, doc_video_inputs = process_vision_info(doc_messages)
        doc_inputs = self.processor(
            text=doc_texts,
            images=doc_image_inputs,
            videos=doc_video_inputs,
            # https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct/blob/main/gme_inference.py#L116
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda:" + str(self.device))
        cache_position = torch.arange(0, len(doc_texts))
        doc_inputs = self.model.prepare_inputs_for_generation(
            **doc_inputs, cache_position=cache_position, use_cache=False
        )

        # call on the model
        with torch.no_grad():
            output = self.model(**doc_inputs, return_dict=True, output_hidden_states=True)

        doc_embeddings = self.get_embedding(output.hidden_states[-1])
        del output
        # output = None
        db_embeddings = cast(Embedding, doc_embeddings.squeeze().tolist())  ##TODO: beware multiple docs...

        return db_embeddings


class RAGImgRetriever:
    def __init__(
        self,
        indexlib,
        indexdb,
        top_k,
        remove_duplicates,
        filter_width,
        filter_height,
        app_repository,
        kvstore,
        logger,
    ):
        self.indexlib = indexlib
        self.indexdb = indexdb
        self.top_k = top_k
        self.remove_duplicates = remove_duplicates
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.query_depth_mult = (
            200  # XXX: chroma has inconsistent best results wrt depth, so we push depth artificially
        )
        self.app_repository = app_repository
        self.kvstore = kvstore
        self.logger = logger
        if self.indexlib == "coldb":
            self.colretriever = self.indexdb.as_retriever(search_type="similarity", search_kwargs={"k": self.top_k})

    def invoke(self, question: str, query_depth_mult: int):
        ##- call on DB
        if query_depth_mult is None:
            query_depth_mult = self.query_depth_mult

        if self.indexlib == "chromadb":
            docs = self.indexdb.query(query_texts=[question], n_results=self.top_k * query_depth_mult)
        else:
            docs = self.colretriever.invoke(question, query_depth_mult)

        self.logger.debug(f"retrieved documents: {json.dumps(docs, indent=2)}")
        ##- filter docs and add images
        docs = self.filter(docs)
        self.logger.debug(f"filtered documents: {json.dumps(docs, indent=2)}")

        return docs

    def filter(self, docs):
        # filter only top_k docs based on distance
        # we do not filter based on the distance, as it is already done by the embedder
        return sort_and_select_top_k(docs, self.top_k, self.remove_duplicates, self.kvstore, self.logger)


class RAGImg:
    def init(
        self,
        ad: InputConnectorObj,
        app_repository: Path,
        models_repository: Path,
        cpu: bool,
        logger: logging.Logger,
        kvstore,
    ):
        self.ad = ad
        self.kvstore = kvstore
        self.app_repository = app_repository
        self.models_repository = models_repository
        self.cpu = cpu
        self.logger = logger

        ##XXX: no preprocessing for now, only get images

        if ad.rag is not None:
            self.rag = True
            self.rag_embf = None
            self.rag_reindex = ad.rag.reindex
            self.rag_index_protection = ad.rag.index_protection
            self.rag_top_k = ad.rag.top_k
            self.rag_remove_duplicates = ad.rag.remove_duplicates
            self.rag_chunk_num = ad.rag.chunk_num
            self.rag_chunk_overlap = ad.rag.chunk_overlap
            self.rag_indexdb_lib = ad.rag.indexdb_lib
            self.rag_num_partitions = ad.rag.num_partitions
            self.gpu_id = ad.rag.gpu_id

            if ad.rag.ragm is not None:
                self.rag_layout_detection = ad.rag.ragm.layout_detection
                self.rag_layout_detector_gpu_id = ad.rag.ragm.layout_detector_gpu_id
                if self.rag_layout_detection and self.rag_layout_detector_gpu_id is None:
                    self.rag_layout_detector_gpu_id = ad.rag.gpu_id
                self.rag_layout_detector_model_path = ad.rag.ragm.layout_detector_model_path
                self.rag_filter_width = ad.rag.ragm.filter_width
                self.rag_filter_height = ad.rag.ragm.filter_height
                self.rag_index_overview = ad.rag.ragm.index_overview
                self.rag_auto_scale_for_font = ad.rag.ragm.auto_scale_for_font
                self.rag_min_font_size = ad.rag.ragm.min_font_size
                self.rag_word_detector_gpu_id = ad.rag.gpu_id

        # indexdb
        self.indexpath = self.app_repository / "mm_index"
        if self.rag_indexdb_lib == "chromadb":
            self.rag_indexdb_client = chromadb.PersistentClient(
                str(self.indexpath), Settings(anonymized_telemetry=False)
            )
        else:
            self.rag_embedding_model = ad.rag.embedding_model
            self.rag_embedding_lib = ad.rag.embedding_lib
            self.rag_indexdb_client = None

        self.logger.info(f"self.indexpath exists: {self.indexpath.exists()}")

        if self.rag_indexdb_lib == "chromadb":
            self.logger.info(f"# collections: {self.rag_indexdb_client.count_collections()}")

        self.reload_index_if_any(ad)

        # layout detection
        # XXX: URL and models are hardcoded for now since this is a custom
        #      that accomodates 'smart' crops for the multimodal rag documents.

        # retriever
        self.rag_retriever = None

    def __del__(self):
        if self.rag_retriever is not None:
            del self.rag_retriever
            self.rag_retriever = None
        if self.rag_indexdb_collection is not None:
            del self.rag_indexdb_collection
        if self.rag_indexdb_client is not None:
            del self.rag_indexdb_client

    def reload_index_if_any(self, ad):
        # Decide whether to load an existing index or create a new one.
        self.has_existing_index = False
        if self.rag_indexdb_lib == "chromadb":
            collection_names = self.rag_indexdb_client.list_collections()
            if collection_names and not isinstance(collection_names[0], str):
                collection_names = [col.name for col in collection_names]
            self.logger.debug(f"Existing collections: {collection_names}")
            self.has_existing_index = "mm_db" in collection_names and (self.indexpath / "chroma.sqlite3").exists()
        else:
            self.has_existing_index = True  # Assuming ColDB always has a persistent index
        self.logger.info(f"has_existing_index: {self.has_existing_index}")

        if self.has_existing_index:
            # Initialize embedding function if using chromadb
            if ad.rag.gpu_id == -1:
                msg = "ad.rag.gpu_id is mandatory when reloading db at service creation or using coldb"
                self.logger.error(msg)
                raise InputConnectorBadParamException(msg)
            if self.rag_indexdb_lib == "chromadb":
                self.rag_embf = ImageEmbeddingFunction(ad, self.models_repository, self.logger)
            else:
                self.rag_embf = None
            self.logger.info("Loading existing index")
            if self.rag_indexdb_lib == "chromadb":
                self.rag_indexdb_collection = self.rag_indexdb_client.get_collection(
                    name="mm_db", embedding_function=self.rag_embf
                )
            else:
                self.rag_indexdb_collection = ColDB(
                    persist_directory=self.indexpath,
                    embedding_model_path=self.models_repository,
                    embedding_function=None,
                    embedding_lib=self.rag_embedding_lib,
                    embedding_model=self.rag_embedding_model,
                    collection_name="mm_db",
                    logger=self.logger,
                    gpu_id=ad.rag.gpu_id,
                    num_partitions=self.rag_num_partitions,
                    index_bsize=ad.rag.index_bsize,
                    image_width=ad.rag.ragm.image_width,
                    image_height=ad.rag.ragm.image_height,
                    kvstore=self.kvstore,
                )
            self.logger.info("Existing index loaded successfully")

    def index(self, ad: InputConnectorObj, sorted_documents: dict[str, list[str]]):
        # Check whether this is an update to an existing index
        self.gpu_id = ad.rag.gpu_id if ad.rag.gpu_id != -1 else self.gpu_id
        self.preproc_dpi = ad.preprocessing.dpi
        self.rag_update_index = ad.rag.update_index
        self.rag_reindex = ad.rag.reindex
        self.rag_index_protection = ad.rag.index_protection
        self.rag_layout_detector_gpu_id = ad.rag.gpu_id

        if self.rag_layout_detection:
            self.rag_layout_detector = LayoutDetector(
                model_path=self.rag_layout_detector_model_path,
                resize_width=768,
                resize_height=1024,
                models_repository=self.models_repository,
                logger=self.logger,
                device=self.rag_layout_detector_gpu_id,
            )
        else:
            self.rag_layout_detector = None

        # Ensure index directory exists
        if not self.indexpath.exists():
            self.indexpath.mkdir(parents=True, exist_ok=True)
            self.logger.debug("Created app index dir %s", self.indexpath)

        if self.has_existing_index and not self.rag_reindex:
            # index reload : already done at service creation
            pass
        else:
            if self.has_existing_index and self.rag_reindex:
                if self.rag_index_protection:
                    msg = "Index already exists and is protected. To reindex, disable index_protection."
                    self.logger.error(msg)
                    raise InputConnectorBadParamException(msg)
                # Delete the existing index if protection is off.
                if self.rag_indexdb_lib == "chromadb":
                    self.rag_indexdb_client.delete_collection(name="mm_db")

            self.logger.info("Creating new index")
            if self.rag_indexdb_lib == "chromadb":
                self.rag_embf = ImageEmbeddingFunction(ad, self.models_repository, self.logger)
                self.rag_indexdb_collection = self.rag_indexdb_client.create_collection(
                    name="mm_db",
                    embedding_function=self.rag_embf,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                self.rag_embf = None
                self.rag_indexdb_collection = ColDB(
                    persist_directory=self.indexpath,
                    embedding_model_path=self.models_repository,
                    embedding_function=None,
                    embedding_lib=self.rag_embedding_lib,
                    embedding_model=self.rag_embedding_model,
                    collection_name="mm_db",
                    logger=self.logger,
                    num_partitions=self.rag_num_partitions,
                    gpu_id=ad.rag.gpu_id,
                    index_bsize=ad.rag.index_bsize,
                    image_width=ad.rag.ragm.image_width,
                    image_height=ad.rag.ragm.image_height,
                    kvstore=self.kvstore,
                )

        # save state for for multiple index queries
        self.has_existing_index = True

        if self.rag_update_index:
            files, offset = set(), 0
            if self.rag_indexdb_lib == "chromadb":
                # Get all data from the collection by batch of 10_000
                while (result := self.rag_indexdb_collection.get(offset=offset, limit=10_000)) and (
                    len(result["ids"]) > 0
                ):
                    files.update([f["source"] for f in result["metadatas"]])
                    offset += len(result["ids"][0])
            self.logger.info(f"Existing index contains {len(files)} elements")
        elif self.rag_reindex:
            files = set()
        else:
            self.logger.info(f"{self.rag_indexdb_collection.count()} elements in index")

        if self.rag_reindex or self.rag_update_index:
            if self.rag_update_index:
                self.logger.info("Updating an existing index")

            # Process and add documents/images to the index
            processor = DocumentProcessor(
                app_repository=self.app_repository,
                logger=self.logger,
                dpi=self.preproc_dpi,
            )
            self.rag_word_detector_gpu_id = ad.rag.gpu_id

            image_processor = ImageProcessor(
                self.rag_layout_detector,
                self.rag_chunk_num,
                self.rag_chunk_overlap,
                self.rag_index_overview,
                self.rag_auto_scale_for_font,
                self.rag_min_font_size,
                self.rag_word_detector_gpu_id,
                self.rag_filter_width,
                self.rag_filter_height,
                self.logger,
            )

            doclist = []
            metadatalist = []
            t1 = time()
            for fext, docs in sorted_documents.items():
                for doc in tqdm(docs, desc="Indexing documents"):
                    self.logger.info(f"Indexing document {doc}")
                    if doc in files:
                        self.logger.info(f"Document {doc} already indexed")
                        continue

                    document = dict(source=Path(doc), ext=fext.lower(), images=list())
                    # Augment document with images
                    processor.transform_documents_to_images([document])
                    self.logger.info(f"\t{len(document['images'])} images extracted")
                    document["npages"] = len(document["images"])
                    # Augment document with crops/chunks
                    image_processor.preprocess_images([document])
                    self.logger.info(f"\t{len(document['parts'])} parts generated")

                    # store images in the index & vector store
                    for part in tqdm(document["parts"]):
                        metadatas = part["metadata"]

                        self.kvstore.store_image(
                            part["name"].encode("utf-8", "replace").decode(),
                            part["img"],
                        )

                        if self.rag_indexdb_lib == "chromadb":
                            # also pass the crop_label to the embedder
                            image_dict = {
                                "doc": part["img"],
                                "label": metadatas.get("crop_label"),
                            }
                            # store document in the vector store
                            self.rag_indexdb_collection.add(
                                images=[image_dict],
                                ids=[part["name"].encode("utf-8", "replace").decode()],
                                metadatas=[metadatas],
                            )
                        else:
                            doclist.append(part["name"])
                            metadatalist.append(metadatas)

            if self.rag_indexdb_lib == "coldb":
                self.rag_indexdb_collection.add_imgs(doclist, metadatalist, "mm_db")

            self.logger.info(f"{self.rag_indexdb_collection.count()} elements in store [{time() - t1:.2f}]")

        # Release layout detector resources
        if self.rag_layout_detection:
            del self.rag_layout_detector.model

    # returns docs
    def retrieve(self, rag_question, query_depth_mult):
        self.logger.debug("retrieving " + rag_question)
        if self.rag_retriever is None:
            self.rag_retriever = RAGImgRetriever(
                self.rag_indexdb_lib,
                self.rag_indexdb_collection,
                self.rag_top_k,
                self.rag_remove_duplicates,
                self.rag_filter_width,
                self.rag_filter_height,
                self.app_repository,
                self.kvstore,
                self.logger,
            )

        return self.rag_retriever.invoke(rag_question, query_depth_mult)

    def delete_embedder(self):
        if self.rag_embf:
            if self.rag_embf.vllm:
                cache_key = ("vllm_embed", self.rag_embf.rag_embedding_model)
            else:
                cache_key = ("huggingface", self.rag_embf.rag_embedding_model)
            if self.rag_embf.shared:
                ModelCache.acquire_lock(cache_key)
            if not self.rag_embf.shared or not ModelCache.is_in_use(cache_key, 1):
                # shared = True
                del self.rag_embf.model
                self.rag_embf.model = None

            if self.rag_embf.shared:
                ModelCache.release_lock(cache_key)
                ModelCache.release(cache_key)

            self.rag_embf = None

            import gc

            gc.collect()
            torch.cuda.empty_cache()
