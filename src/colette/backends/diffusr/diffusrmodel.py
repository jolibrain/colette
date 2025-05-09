import random

import torch
from diffusers import BitsAndBytesConfig, FluxPipeline, HunyuanDiTPipeline

from colette.apidata import LLMModelObj
from colette.llmmodel import LLMModel, LLMModelBadParamException


class DiffusrModel(LLMModel):
    def __init__(self):
        super().__init__()

    def init(self, ad: LLMModelObj, kvstore):
        super().init(ad)

        self.image_width = ad.image_width
        self.image_height = ad.image_height

        if not self.llm_obj.gpu_ids:
            self.cpu = True
        else:
            self.gpu_ids = self.llm_obj.gpu_ids
            self.cpu = False
        self.device = torch.device(self.gpu_ids[0] if not self.cpu else -1)

        self.load_in_8bit = ad.load_in_8bit
        self.initialize_model()

    def initialize_model(self):
        if self.llm_lib == "diffusers":
            bandb_config = None
            if self.load_in_8bit:
                bandb_config = BitsAndBytesConfig(
                    load_in_8bit=True,  # XXX: not applicable to 24Gb GPUs otherwise
                )
            if "FLUX" in self.llm_source:
                self.pipe = FluxPipeline.from_pretrained(
                    self.llm_source,
                    torch_dtype=torch.bfloat16,
                    cache_dir=self.models_repository,
                    quantization_config=bandb_config,
                )
                self.pipe.enable_model_cpu_offload()
                # save some VRAM by offloading the model to CPU.
                # Remove this if you have enough GPU power
                self.model_type = "flux"
            elif "Hunyuan" in self.llm_source:
                self.pipe = HunyuanDiTPipeline.from_pretrained(
                    self.llm_source,
                    torch_dtype=torch.float16,
                    cache_dir=self.models_repository,
                    quantization_config=bandb_config,
                )
                self.pipe.to(self.device)
                self.model_type = "hunyuan"
        else:
            msg = "Unknown image generation lib:" + self.llm_lib
            self.logger.error(msg)
            raise LLMModelBadParamException(msg)

    def generate(self, prompt):
        if self.model_type == "flux":
            image = self.pipe(
                prompt,
                height=self.image_height,  # TODO: per-call configurable ?
                width=self.image_width,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(random.randint(0, 1000000)),
                output_type="pil",
            ).images[0]
        elif self.model_type == "hunyuan":
            image = self.pipe(
                prompt,
                height=self.image_height,  # TODO: per-call configurable ?
                width=self.image_width,
                guidance_scale=3.5,
                num_inference_steps=10,
                generator=torch.Generator("cpu").manual_seed(random.randint(0, 1000000)),
                output_type="pil",
            ).images[0]
        return image
