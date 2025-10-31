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
from transformers import AutoProcessor, AutoModel, Qwen2VLForConditionalGeneration
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


def take_top_k(data: dict[str, list], k: int) -> dict[str, list]:
    """
    Extracts the first k elements from each list in the data dictionary and reverses them.
    Each value in data is expected to be a list (or a list of lists).

    Args:
        data: dictionary with keys such as 'distances', 'documents', etc.
        k: number of elements to extract

    Returns:
        A new dictionary containing only the first k elements for each key, in reversed order.
    """
    result = {}
    for key, value in data.items():
        if value is None:
            result[key] = None
            continue

        if isinstance(value, list) and value and isinstance(value[0], list):
            result[key] = [value[0][:k][::-1]]
        elif isinstance(value, list):
            result[key] = value[:k][::-1]
        else:
            result[key] = value
    return result


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
            self.rag_embedding_model = "jinaai/jina-embeddings-v4-vllm-retrieval"
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
        max_pixels = 768 * 28 * 28

        # load model
        ## only qwen2vl-based embedder for now
        expected_prefix = "jinaai/jina-embeddings"
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
                    download_dir=str(models_repository),
                    load_format=self.vllm_load_format,
                    quantization=self.vllm_quantization,
                    max_model_len=2048,
                    max_num_seqs=5,
                    task="embed",
                    mm_processor_kwargs={
                        "min_pixels": 28 * 28,
                        "max_pixels": 768 * 28 * 28,
                        "fps": 1,
                    },
                    disable_mm_preprocessor_cache=False,
                    gpu_memory_utilization=self.vllm_memory_utilization,
                    limit_mm_per_prompt={"image": 1},
                    enforce_eager=self.vllm_enforce_eager,
                    trust_remote_code=True,  # ADD THIS LINE
                )
            else:
                self.processor = AutoProcessor.from_pretrained(
                    self.rag_embedding_model,
                    trust_remote_code=True,
                    # torch_dtype=torch.float16,
                    # min_pixels=min_pixels,
                    # max_pixels=max_pixels,
                    cache_dir=str(models_repository),
                )

                # Configure the image processor separately
                self.processor.image_processor.min_pixels = min_pixels
                self.processor.image_processor.max_pixels = max_pixels

                # Load model
                loaded_model = AutoModel.from_pretrained(
                    self.rag_embedding_model,
                    trust_remote_code=True,
                    # attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    cache_dir=str(models_repository),
                )

                # Check if it's a PEFT model and get the base model
                if hasattr(loaded_model, 'base_model'):
                    self.logger.info("PEFT model detected, extracting base model for vision processing")
                    # Get the actual base model without PEFT wrappers
                    self.model = loaded_model.get_base_model()
                else:
                    self.model = loaded_model

                self.model = self.model.to("cuda:" + str(self.device)).eval()

                # Add diagnostic logging
                self.logger.info(f"Model type: {type(self.model)}")
                self.logger.info(f"Model class: {self.model.__class__.__name__}")
                if hasattr(self.model, 'peft_config'):
                    self.logger.info(f"PEFT config found: {self.model.peft_config}")
                else:
                    self.logger.info("No PEFT config found - ready for vision processing")

                # CRITICAL FIX: Check vision encoder configuration
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
                    visual = self.model.model.visual
                    self.logger.info(f"Vision encoder found")
                    self.logger.info(f"  spatial_merge_unit: {getattr(visual, 'spatial_merge_unit', 'N/A')}")
                    self.logger.info(f"  patch_size: {getattr(visual.patch_embed, 'patch_size', 'N/A') if hasattr(visual, 'patch_embed') else 'N/A'}")

        # Set chat template if not present (only for HuggingFace, not vLLM)
        if not self.vllm:
            if not hasattr(self.processor, 'chat_template') or self.processor.chat_template is None:
                self.processor.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'}}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' %}{{'<|vision_start|><|image_pad|><|vision_end|>'}}{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{'<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
            
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
        
        # Separate images and texts
        images_list = []
        texts_list = []
        
        for item in input:
            if type(item) is dict:
                label = item.get("label", None)
                item = item.get("doc", None)
            else:
                label = None
                
            if "PIL" in str(type(item)):
                # Handle PIL images
                if not self.rag_auto_scale_for_font and self.rag_image_width and self.rag_image_height:
                    width, height = item.size
                    if width > self.rag_image_width or height > self.rag_image_height:
                        item.thumbnail((self.rag_image_width, self.rag_image_height))
                images_list.append(item)
            else:
                # Handle text
                texts_list.append(item)
        
        all_embeddings = []
        
        # Process images using the correct format for Jina v4
        if images_list:
            for idx, img in enumerate(images_list):
                self.logger.info(f"Processing image {idx + 1}/{len(images_list)}")
                self.logger.info(f"Image type: {type(img)}, size: {img.size if hasattr(img, 'size') else 'N/A'}")
                
                try:
                    # CRITICAL: Resize image to dimensions that work well with the vision encoder
                    # The vision encoder expects images that result in patch grids divisible by spatial_merge_unit (4)
                    width, height = img.size
                    self.logger.info(f"Original image size: {width}x{height}")
                    
                    # Calculate target size that will result in patch counts divisible by 4
                    # With patch_size=14 and spatial_merge_unit=4, we need dimensions where:
                    # (h/14) % 4 == 0 and (w/14) % 4 == 0
                    # This means h and w should be multiples of 56 (14*4)
                    
                    target_width = ((width // 56) * 56)
                    target_height = ((height // 56) * 56)
                    
                    # Ensure we don't make it too small
                    if target_width < 224:
                        target_width = 224
                    if target_height < 224:
                        target_height = 224
                        
                    if (width, height) != (target_width, target_height):
                        self.logger.info(f"Resizing image from {width}x{height} to {target_width}x{target_height}")
                        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    
                    # Calculate if we need to resize based on min/max pixels
                    total_pixels = img.size[0] * img.size[1]
                    min_pixels = self.processor.image_processor.min_pixels
                    max_pixels = self.processor.image_processor.max_pixels
                    
                    self.logger.info(f"Image pixels: {total_pixels}, min: {min_pixels}, max: {max_pixels}")
                    
                    # Use the message format that Jina v4 expects
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                            ],
                        }
                    ]
                    
                    # Use process_vision_info to extract images properly
                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    self.logger.info(f"Number of images from process_vision_info: {len(image_inputs) if image_inputs else 0}")
                    
                    # Apply chat template 
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    
                    self.logger.info(f"Generated text: {text[:200]}")
                    
                    # Process - the key is here
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    
                    # Check what we got back
                    self.logger.info(f"After processor - pixel_values shape: {inputs.get('pixel_values', 'missing').shape if 'pixel_values' in inputs else 'missing'}")
                    self.logger.info(f"After processor - image_grid_thw: {inputs.get('image_grid_thw', 'missing')}")
                    
                    # Move to device
                    inputs = {k: v.to("cuda:" + str(self.device)) if isinstance(v, torch.Tensor) else v 
                            for k, v in inputs.items()}
                    
                    self.logger.info(f"Processor output keys: {inputs.keys()}")
                    for key in inputs.keys():
                        if hasattr(inputs[key], 'shape'):
                            self.logger.info(f"{key} shape: {inputs[key].shape}")
                    
                    # Validate inputs
                    if 'pixel_values' not in inputs or inputs['pixel_values'].numel() == 0:
                        self.logger.error("No pixel_values generated")
                        all_embeddings.append([0.0] * 2048)
                        continue
                        
                    if inputs.get('input_ids') is None or inputs['input_ids'].shape[1] == 0:
                        self.logger.error("Invalid input_ids generated")
                        all_embeddings.append([0.0] * 2048)
                        continue
                    
                    # Check if pixel_values shape is correct
                    # Should be [num_patches, hidden_dim] or similar
                    pv_shape = inputs['pixel_values'].shape
                    if len(pv_shape) != 2:
                        self.logger.error(f"Unexpected pixel_values dimensions: {pv_shape}")
                        all_embeddings.append([0.0] * 2048)
                        continue
                        
                except Exception as e:
                    self.logger.error(f"Error in processor: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    all_embeddings.append([0.0] * 2048)
                    continue
                
                # Call model
                with torch.no_grad():
                    try:
                        batch_size = inputs['input_ids'].shape[0]
                        # task_label=1 for image embeddings
                        task_label = torch.ones(batch_size, dtype=torch.long, device="cuda:" + str(self.device))
                        
                        self.logger.info(f"Calling model.forward() with:")
                        self.logger.info(f"  batch_size: {batch_size}")
                        self.logger.info(f"  input_ids: {inputs['input_ids'].shape}")
                        self.logger.info(f"  pixel_values: {inputs['pixel_values'].shape}, dtype: {inputs['pixel_values'].dtype}")
                        self.logger.info(f"  image_grid_thw: {inputs['image_grid_thw']}, shape: {inputs['image_grid_thw'].shape}")
                        self.logger.info(f"  task_label: {task_label}")
                        
                        # Convert pixel_values to match model dtype if needed
                        if inputs['pixel_values'].dtype != self.model.dtype:
                            self.logger.info(f"Converting pixel_values from {inputs['pixel_values'].dtype} to {self.model.dtype}")
                            inputs['pixel_values'] = inputs['pixel_values'].to(self.model.dtype)
                        
                        # Log model's expected input format
                        self.logger.info(f"Model dtype: {self.model.dtype}")
                        self.logger.info(f"Model device: {next(self.model.parameters()).device}")
                        
                        output = self.model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            pixel_values=inputs['pixel_values'],
                            image_grid_thw=inputs['image_grid_thw'],
                            task_label=task_label,
                            return_dict=True,
                            output_hidden_states=True
                        )
                        
                        embedding = self.get_embedding(output.hidden_states[-1])
                        all_embeddings.append(embedding.squeeze().cpu().tolist())
                        self.logger.info("Successfully generated embedding")
                        
                        del output
                        del inputs
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        self.logger.error(f"Error in model forward: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        all_embeddings.append([0.0] * 2048)

        # Process texts
        if texts_list:
            for text_item in texts_list:
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_item},
                        ],
                    }
                ]
                
                text = self.processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                ) + "<|endoftext|>"
                
                inputs = self.processor(
                    text=[text],
                    images=None,
                    videos=None,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to("cuda:" + str(self.device))
                
                with torch.no_grad():
                    batch_size = inputs['input_ids'].shape[0]
                    task_label = torch.zeros(batch_size, dtype=torch.long, device="cuda:" + str(self.device))
                    output = self.model(**inputs, task_label=task_label, return_dict=True, output_hidden_states=True)
                
                embedding = self.get_embedding(output.hidden_states[-1])
                all_embeddings.append(embedding.squeeze().cpu().tolist())
                del output
        
        # Return single embedding or list
        if len(all_embeddings) == 1:
            return cast(Embedding, all_embeddings[0])
        return cast(Embeddings, all_embeddings)

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

        return docs

    def filter(self, docs):
        # filter only top_k docs based on distance
        # we do not filter based on the distance, as it is already done by the embedder
        if self.remove_duplicates :
            return sort_and_select_top_k(docs, self.top_k, self.remove_duplicates, self.kvstore, self.logger)
        else:
            return take_top_k(docs, self.top_k)


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
