# Langchain model abstraction
import os

import torch
from huggingface_hub import hf_hub_download
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.llms import VLLM, LlamaCpp
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

from colette.apidata import LLMModelObj
from colette.llmmodel import LLMModel, LLMModelBadParamException


class LangChainModel(LLMModel):
    def __init__(self):
        super().__init__()

    def init(self, ad: LLMModelObj, kvstore):
        super().init(ad)
        self.llm = None
        self.models_dir = self.models_repository
        if not self.llm_obj.gpu_ids:
            self.cpu = True
        else:
            self.gpu_ids = self.llm_obj.gpu_ids
            self.cpu = False
        if self.llm is None:
            if self.llm_lib == "ollama":
                if self.llm_obj.host:
                    base_url = "http://" + self.llm_obj.host + ":" + str(self.llm_obj.port)
                else:
                    base_url = "http://localhost:11434"  # default
                if self.llm_obj.conversational:
                    cls = ChatOllama
                else:
                    cls = OllamaLLM
                self.llm = cls(
                    model=self.llm_source,
                    base_url=base_url,
                    num_ctx=self.llm_obj.context_size,
                    temperature=self.llm_obj.inference.sampling_params["temperature"],
                )
            elif self.llm_lib == "vllm":
                if self.llm_obj.conversational:
                    if self.llm_obj.host:
                        base_url = "http://" + self.llm_obj.host + ":" + str(self.llm_obj.port)
                    else:
                        base_url = "http://localhost:8000/v1"  # default
                    self.llm = ChatOpenAI(
                        model=self.llm_source,
                        openai_api_key="EMPTY",
                        openai_api_base=base_url,
                        temperature=self.llm_obj.inference.sampling_params["temperature"],
                    )
                else:
                    self.llm = VLLM(
                        model=self.llm_source,
                        trust_remote_code=True,
                        download_dir=str(self.models_dir),
                        temperature=self.llm_obj.inference.sampling_params["temperature"],
                        dtype=self.llm_obj.dtype,
                        vllm_kwargs={
                            "quantization": self.llm_obj.vllm_quantization,
                            "max_model_len": self.llm_obj.context_size,
                            "gpu_memory_utilization": self.llm_obj.vllm_memory_utilization,
                            "enforce_eager": self.llm_obj.vllm_enforce_eager,
                        },
                    )
            elif self.llm_lib == "huggingface":
                ##TODO: save model in app directory
                self.llm = HuggingFacePipeline.from_model_id(
                    model_id=self.llm_source,
                    device=self.gpu_ids[0] if not self.cpu else -1,
                    task="text-generation",
                    model_kwargs={
                        "temperature": self.llm_obj.inference.sampling_params["temperature"],
                        "torch_dtype": torch.float16,
                    },
                    pipeline_kwargs={"max_new_tokens": 512, "return_full_text": False},
                )
                if self.llm_obj.conversational:
                    self.llm = ChatHuggingFace(llm=self.llm)
            elif self.llm_lib == "llamacpp":
                # need to download the model (hub deals with local cache and already downloaded models)
                self.model_path = self.models_dir / self.llm_obj.filename
                if not os.path.exists(self.model_path):
                    self.logger.info("Downloading model from hub: " + self.llm_obj.source + "/" + self.llm_obj.filename)
                hf_hub_download(
                    repo_id=self.llm_obj.source,
                    filename=self.llm_obj.filename,
                    local_dir=str(self.models_dir),
                )
                self.logger.info("Using model " + str(self.model_path))
                if self.llm_obj.conversational:
                    cls = ChatLlamaCpp
                else:
                    cls = LlamaCpp
                self.llm = cls(
                    model_path=str(self.model_path),
                    temperature=self.llm_obj.inference.sampling_params["temperature"],
                    top_p=1,
                    # verbose=True,
                    n_ctx=self.llm_obj.context_size,
                    n_gpu_layers=0 if self.cpu else -1,
                    n_batch=512,
                    model_kwargs={
                        "split_mode": 0,
                        "main_gpu": 0 if self.cpu else self.gpu_ids[0],
                        # first gpu id, XXX: no support for list of gpus
                        # (or use split_mode 'layer' (1) or 'row' (2))
                    },
                )
                # print("llamacpp obj client type=", type(self.llm.client)) #-> underlying Llama object
            else:
                msg = "Unknown LLM lib: " + self.llm_lib
                self.logger.error(msg)
                raise LLMModelBadParamException(msg)
            ##TODO: others (vllm, gpt4all) and exceptions on unknown libs

    def delete_model(self):
        if self.llm_lib == "vllm":
            del self.llm.client
        del self.llm
        self.llm = None
        import gc

        gc.collect()
