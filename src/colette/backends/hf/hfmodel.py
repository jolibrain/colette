import base64
from io import BytesIO
from threading import Thread

import PIL
import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    TextIteratorStreamer,
)
from vllm import LLM as VLLM
from vllm import SamplingParams

from colette.apidata import LLMModelObj
from colette.llmmodel import LLMModel, LLMModelBadParamException

from .model_cache import ModelCache
from .vllm_client import VllmClient


def stitch_images_vertically(image_list):
    # Get total height and maximum width for the new image
    total_height = sum(img.height for img in image_list)
    max_width = max(img.width for img in image_list)

    # Create a new blank image
    stitched_image = PIL.Image.new("RGB", (max_width, total_height))

    # Paste each image into the new blank image
    y_offset = 0
    for img in image_list:
        stitched_image.paste(img, (0, y_offset))
        y_offset += img.height

    return stitched_image


class HFModel(LLMModel):
    def __init__(self):
        super().__init__()

    def __del__(self):
        self.llm = None

    def init(self, ad: LLMModelObj, kvstore):
        super().init(ad)
        self.kvstore = kvstore
        self.llm = None
        self.shared = ad.shared_model
        self.image_width = ad.image_width
        self.image_height = ad.image_height
        if not self.llm_obj.gpu_ids:
            self.cpu = True
        else:
            self.gpu_ids = self.llm_obj.gpu_ids
            self.cpu = False
        self.device = torch.device(self.gpu_ids[0] if not self.cpu else -1)
        self.stitch_crops = ad.stitch_crops
        self.load_in_8bit = ad.load_in_8bit
        self.query_rephrasing = ad.query_rephrasing
        self.vllm_memory_utilization = ad.vllm_memory_utilization
        self.vllm_quantization = ad.vllm_quantization
        self.vllm_context_size = ad.context_size
        self.vllm_enforce_eager = ad.vllm_enforce_eager
        if ad.external_vllm_server is not None:
            self.server_url = ad.external_vllm_server.url
            self.api_key = ad.external_vllm_server.api_key

        # Initialize the LLM
        # self.initialize_llm()
        self.logger.info("HFModel initialized")

    def initialize_llm(self):
        # Check if model is already instantiated
        if self.shared:
            cache_key = (self.llm_lib, self.llm_source)
            cached_model = ModelCache.get(cache_key)
            if cached_model:
                self.llm, self.processor, self.llm_type = cached_model
                self.logger.info("Reusing cached LLM model: " + self.llm_source)
                return

        bandb_config = None
        if self.load_in_8bit:
            bandb_config = BitsAndBytesConfig(
                load_in_8bit=True,  # XXX: not applicable to 24Gb GPUs otherwise
            )

        # - define self.llm, used by hflib for inference
        if self.llm_lib == "huggingface":
            self.logger.debug(
                "Using HuggingFace model: "
                + self.llm_source
                + " with models repository: "
                + str(self.models_repository)
            )
            if "Qwen2-VL" in self.llm_source:
                self.llm = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.llm_source,
                    torch_dtype=torch.float16,
                    quantization_config=bandb_config,
                    cache_dir=self.models_repository,
                    attn_implementation="flash_attention_2",
                ).eval()
                min_pixels = 1 * 28 * 28
                max_pixels = 2560 * 28 * 28
                self.processor = AutoProcessor.from_pretrained(
                    self.llm_source, min_pixels=min_pixels, max_pixels=max_pixels
                )
                self.llm_type = "qwen2-vl"
            elif "Qwen2.5-VL" in self.llm_source:
                self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.llm_source,
                    torch_dtype=torch.float16,
                    quantization_config=bandb_config,
                    cache_dir=self.models_repository,
                ).eval()
                min_pixels = 1 * 28 * 28
                max_pixels = 2560 * 28 * 28
                self.processor = AutoProcessor.from_pretrained(
                    self.llm_source, min_pixels=min_pixels, max_pixels=max_pixels
                )
                self.llm_type = "qwen2-vl"
            elif "gemma-3" in self.llm_source:
                self.llm = Gemma3ForConditionalGeneration.from_pretrained(
                    self.llm_source,
                    torch_dtype=torch.bfloat16,  # XXX: float16 is not supported
                    quantization_config=bandb_config,
                    cache_dir=self.models_repository,
                    device_map=self.device,
                ).eval()
                self.processor = AutoProcessor.from_pretrained(
                    self.llm_source,
                )
                self.llm_type = "gemma3"
            elif "pixtral" in self.llm_source:
                self.llm = LlavaForConditionalGeneration.from_pretrained(
                    self.llm_source,
                    torch_dtype=torch.float16,
                    quantization_config=bandb_config,
                    device_map=self.device,
                )
                self.processor = AutoProcessor.from_pretrained(
                    self.llm_source,
                )
                self.llm_type = "pixtral"
            elif "SmolVLM" in self.llm_source:
                self.llm = AutoModelForVision2Seq.from_pretrained(
                    self.llm_source,
                    torch_dtype=torch.float16,
                    cache_dir=self.models_repository,
                    attn_implementation="flash_attention_2",
                )
                self.processor = AutoProcessor.from_pretrained(
                    self.llm_source,
                )
                self.llm_type = "smolvlm"
            else:
                msg = "Unknown Multimodal LLM source: " + self.llm_source
                self.logger.error(msg)
                raise LLMModelBadParamException(msg)
        elif self.llm_lib == "vllm":
            self.llm_type = "vllm"
            self.processor = None
            if self.vllm_quantization == "bitsandbytes":
                self.vllm_load_format = "bitsandbytes"
            else:
                self.vllm_load_format = "auto"
            self.llm = VLLM(
                model=self.llm_source,
                download_dir=self.models_repository,
                load_format=self.vllm_load_format,
                quantization=self.vllm_quantization,
                gpu_memory_utilization=self.vllm_memory_utilization,
                max_model_len=4096,
                dtype=torch.bfloat16,
                mm_processor_kwargs={
                    "min_pixels": 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                },
                limit_mm_per_prompt={"image": 10},
                enforce_eager=self.vllm_enforce_eager,
            )
            if "Qwen2-VL" in self.llm_source:
                self.llm_subtype = "qwen2-vl"
            elif "Qwen2.5-VL" in self.llm_source:
                self.llm_subtype = "qwen25-vl"
            elif "SmolVLM" in self.llm_source:
                self.llm_subtype = "smolvlm"
            else:
                msg = "Unknown vllm source: " + self.llm_source
                self.logger.error(msg)
                raise LLMModelBadParamException(msg)
        elif self.llm_lib == "vllm_client":
            self.llm_type = "vllm_client"
            self.processor = None
            self.llm = VllmClient(
                url=self.server_url,
                api_key=self.api_key,
                model=self.llm_source,
                max_model_len=4096,
                min_pixels=28 * 28,
                max_pixels=1280 * 28 * 28,
            )

        else:
            msg = "Unknown VLLM lib: " + self.llm_lib
            self.logger.error(msg)
            raise LLMModelBadParamException(msg)

        if not self.load_in_8bit and self.llm_type != "vllm":  ##TODO: beware with device_map='auto'
            self.llm.to(self.device)  # XXX: unsupported for 8bit loading

        # Cache the model for future use
        if self.shared:
            ModelCache.add(cache_key, self.llm, self.processor, self.llm_type)

    def generate(
        self,
        prompt,
        max_new_tokens,
        docs,
        history=None,
        do_sample=False,
        streaming=False,
    ):
        if history is None:
            history = []
        new_message = []
        imgids = docs["ids"][0]

        # Collect images
        images = []
        docs["images"] = [[]]
        for imgid in imgids:
            # add image to list
            images.append(self.kvstore.retrieve_image(imgid))
            # add image (the last one in the list) to docs
            docs["images"][0].append(images[-1])

        if self.stitch_crops:
            images = [stitch_images_vertically(images)]

        # process inputs
        if self.llm_type == "qwen2-vl":
            content = []
            if not history:
                for image, metadata in zip(docs["images"][0], docs["metadatas"][0], strict=False):
                    # add image metadata to content
                    context_txt = f"document: {metadata['source']}"
                    if "page_number" in metadata:
                        context_txt += f" page: {metadata['page_number']}"
                    if "crop_label" in metadata:
                        context_txt += f" type: {metadata['crop_label']}"
                    contextdict = {
                        "type": "text",
                        "text": context_txt,
                    }
                    content.append(contextdict)
                    # Add image to content
                    cdict = {
                        "type": "image",
                        "image": image,
                    }
                    if self.image_height is not None:
                        cdict["resized_height"] = self.image_height
                    if self.image_width is not None:
                        cdict["resized_width"] = self.image_width
                    content.append(cdict)
            tdict = {"type": "text", "text": prompt}
            content.append(tdict)

            new_message = {
                "role": "user",
                "content": content,
            }
            if not history:
                messages = [
                    new_message,
                ]
            else:
                messages = history.copy()
                messages.append(new_message)

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            model_inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            model_inputs.to(self.device)
        elif self.llm_type == "pixtral":
            if history:
                self.logger.warning("Conversational mode is not available with pixtral")
            p_prompt = "<s>[INST]" + prompt + "\n"
            for image in images:
                if self.image_width and self.image_height:
                    image.thumbnail((self.image_width, self.image_height))
                p_prompt += "[IMG]"
            p_prompt += "[/INST]"
            model_inputs = self.processor(images=images, text=p_prompt, return_tensors="pt").to(
                self.device, torch.float16
            )
        elif self.llm_type == "smolvlm":
            content = []
            for _, metadata in zip(docs["images"][0], docs["metadatas"][0], strict=False):
                # Add image context to content
                context_txt = f"document: {metadata['source']}"
                if "page_number" in metadata:
                    context_txt += f" page: {metadata['page_number']}"
                if "crop_label" in metadata:
                    context_txt += f" type: {metadata['crop_label']}"
                contextdict = {
                    "type": "text",
                    "text": context_txt,
                }
                content.append(contextdict)
                # Add image to content
                cdict = {"type": "image"}
                content.append(cdict)
            # Add query
            tdict = {"type": "text", "text": prompt}
            content.append(tdict)
            new_message = {"role": "user", "content": content}
            messages = [
                new_message,
            ]

            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            model_inputs = self.processor(
                text=[text], images=docs["images"][0], return_tensors="pt"
            )  # size={"longest_edge":2*384}
            model_inputs.to(self.device)
        elif self.llm_type == "gemma3":
            content = []
            for image, metadata in zip(docs["images"][0], docs["metadatas"][0], strict=False):
                # Add image context to content
                context_txt = f"document: {metadata['source']}"
                if "page_number" in metadata:
                    context_txt += f" page: {metadata['page_number']}"
                if "crop_label" in metadata:
                    context_txt += f" type: {metadata['crop_label']}"
                contextdict = {
                    "type": "text",
                    "text": context_txt,
                }
                content.append(contextdict)
                # Add image to content
                imbuffer = BytesIO()
                image.save(imbuffer, format="JPEG")
                b64im = base64.b64encode(imbuffer.getvalue()).decode("utf-8")
                cdict = {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{b64im}",
                }
                if self.image_height is not None:
                    cdict["resized_height"] = self.image_height
                if self.image_width is not None:
                    cdict["resized_width"] = self.image_width
                content.append(cdict)
            # Add query to content
            tdict = {"type": "text", "text": prompt}
            content.append(tdict)
            new_message = {
                "role": "user",
                "content": content,
            }
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                new_message,
            ]

            model_inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs.to(self.device)
            input_len = model_inputs["input_ids"].shape[-1]
            ##TODO: add history
        elif self.llm_type in ["vllm", "vllm_client"]:
            content = []
            if not history:
                for image, metadata in zip(docs["images"][0], docs["metadatas"][0], strict=False):
                    # Add image context to content
                    context_txt = f"document: {metadata['source']}"
                    if "page_number" in metadata:
                        context_txt += f" page: {metadata['page_number']}"
                    if "crop_label" in metadata:
                        context_txt += f" type: {metadata['crop_label']}"
                    contextdict = {
                        "type": "text",
                        "text": context_txt,
                    }
                    content.append(contextdict)
                    # Add image to content
                    imbuffer = BytesIO()
                    image.save(imbuffer, format="JPEG")
                    b64im = base64.b64encode(imbuffer.getvalue()).decode("utf-8")
                    cdict = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64im}"},
                    }
                    # if self.image_height is not None:
                    #     cdict["resized_height"] = self.image_height
                    # if self.image_width is not None:
                    #     cdict["resized_width"] = self.image_width
                    content.append(cdict)
            # Add query to content
            tdict = {"type": "text", "text": prompt}
            content.append(tdict)

            new_message = {
                "role": "user",
                "content": content,
            }
            if not history:
                messages = [
                    new_message,
                ]
            else:
                messages = history.copy()
                messages.append(new_message)

        else:
            msg = "unknown llm type " + self.llm_type
            self.logger.error(msg)
            raise LLMModelBadParamException(msg)

        try:
            # generate output
            if self.shared:
                cache_key = (self.llm_lib, self.llm_source)
                ModelCache.acquire_lock(cache_key)
            with torch.inference_mode():
                if self.llm_type not in ["vllm", "vllm_client"]:
                    generation_kwargs = dict(
                        # **model_inputs, max_new_tokens=512, **self.llm_obj.inference.sampling_params
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        top_k=None,
                        temperature=None,
                        top_p=None,
                    )
                    # return a streamer if streaming was requested
                    if streaming:
                        tokenizer = AutoTokenizer.from_pretrained(
                            self.llm_source,
                        )
                        streamer = TextIteratorStreamer(
                            tokenizer,
                            skip_prompt=True,
                            skip_special_tokens=True,
                        )
                        generation_kwargs["streamer"] = streamer
                        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
                        thread.start()
                        return None, new_message, streamer
                    generation = self.llm.generate(**generation_kwargs)
                    if self.llm_type == "qwen2-vl":
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :]
                            for in_ids, out_ids in zip(model_inputs.input_ids, generation, strict=False)
                        ]
                        decoded = self.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]
                    elif self.llm_type == "pixtral":
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :]
                            for in_ids, out_ids in zip(model_inputs.input_ids, generation, strict=False)
                        ]
                        decoded = self.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]
                    elif self.llm_type == "smolvlm":
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :]
                            for in_ids, out_ids in zip(model_inputs.input_ids, generation, strict=False)
                        ]
                        decoded = self.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]
                    elif self.llm_type == "gemma3":
                        generation = generation[0][input_len:]
                        decoded = self.processor.decode(generation, skip_special_tokens=True)
                    del generation
                    generation = None
                elif self.llm_type == "vllm":  # VLLM case
                    sampling_params = SamplingParams(max_tokens=max_new_tokens)
                    if "qwen2" in self.llm_subtype:
                        outputs = self.llm.chat(
                            messages,
                            mm_processor_kwargs={
                                "min_pixels": 1 * 28 * 28,
                                "max_pixels": 2560 * 28 * 28,
                                "max_model_len": self.vllm_context_size,
                            },
                            sampling_params=sampling_params,
                        )
                    elif "smolvlm" in self.llm_subtype:
                        outputs = self.llm.chat(
                            messages,
                            sampling_params=sampling_params,
                        )
                    decoded = "".join([o.outputs[0].text for o in outputs])
                elif self.llm_type == "vllm_client":
                    decoded = self.llm.chat(messages, max_new_tokens)

            return decoded, new_message, None
        finally:
            if self.shared:
                ModelCache.release_lock(cache_key)

    def delete_model(self):
        if self.shared:
            cache_key = (self.llm_lib, self.llm_source)
        if self.shared:
            ModelCache.acquire_lock(cache_key)
        if not self.shared or not ModelCache.is_in_use(cache_key, 1):
            del self.llm
            del self.processor
            self.llm = None
            self.processor = None

        if self.shared:
            ModelCache.release_lock(cache_key)
            ModelCache.release(cache_key)

        import gc

        gc.collect()
        torch.cuda.empty_cache()
