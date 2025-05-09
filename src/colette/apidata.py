# define APIData
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    NewPath,
    field_serializer,
)


class PreprocessingObj(BaseModel):
    """Preprocessing options for indexing calls"""

    files: list[str] = ["all"]  # extensions
    """ File extensions to preprocess, e.g. all, pdf, png, ... """
    lib: str | None = None  # preprocessing library
    """ Preprocessing lib to use, e.g. unstructured for Text-RAG and leave blank for V-RAG """
    save_output: bool = False  # whether to save the preprocessed output
    """ Text-RAG only: whether to save the raw text output at indexing time """
    strict: bool = False  # do not stop on preprocessing failures
    """ Text-RAG only: whether to fail on preprocessing errors """
    filters: list[str] = Field(default_factory=list)  # list of patterns for filtering data out
    """ Input preprocessing regex filters, e.g. ["\/~[^\/]*$"] """
    strategy: str = "auto"
    """ Text-RAG only: text extraction strategy, e.g. auto, fast, hi_res """
    cleaning: bool = True
    """ Text-RAG only: whether to clean the text output """
    dpi: int = 300
    """ V-RAG only: dpi of the rendered images from PDF """


class RAGMultimodalObj(BaseModel):
    """Multimodal RAG parameters"""

    layout_detection: bool = True
    """ Whether to activate layout detection and chunking """
    layout_detector_gpu_id: int | None = None
    """ GPU to use for layout detection, defaults to embedder GPU otherwise """
    layout_detector_model_path: str = "https://colette.chat/models/layout/layout_detector_publaynet_merged_6000.pt"
    """ Model to use for layout detection, defaults to generic model trained on PubLayNet """
    image_width: int | None = None
    """ V-RAG image width """
    image_height: int | None = None
    """ V-RAG image height """
    auto_scale_for_font: bool = False
    """ Whether to auto scale the images to that fonts are best readable by multimodal LLMs (warning: bypasses image_width and image_height) """
    min_font_size: int = 24  # value empirically obtained for qwen2vl2b, in pixels
    """ Min font size below which multimodal LLMs OCR capabilities degrade, empirically evaluated """
    filter_width: int = -1
    """ Filter width after layout detection, -1 means no filter, removes smaller chunks (e.g. logos, ...) """
    filter_height: int = -1
    """ Filter height after layout detection, -1 means no filter, removes smaller chunks (e.g. logos, ...) """
    index_overview: bool = True  # whether to index the full page as overview
    """ Whether to index the full page image when layout detection chunking is enabled """


class RAGObj(BaseModel):
    """Main RAG parameters"""

    indexdb_lib: Literal["chromadb", "coldb"] = "chromadb"
    """ Indexing database library, chromadb or coldb """
    embedding_lib: Literal["huggingface", "colbert", "vllm"] = "huggingface"
    """ Embedding library, use colbert with coldb, huggingface for V-RAG """
    embedding_model: str | None = None
    """ Embedding model, e.g. MrLight/dse-qwen2-2b-mrl-v1 """
    embedding_model_path: str | None = None
    """ V-RAG and Colbert-only Path to embedding model, e.g. /home/user/models, useful for custom models """
    shared_model: bool = True
    """ Whether to share the embedding model across all RAGs (i.e. if already existing, not reloaded) """
    chunk_size: int = 250
    """ Text-RAG only: default text chunking size """
    chunk_overlap: int = 0
    """ Text-RAG only: default text chunking overlap """
    chunk_num: int = 1
    """ V-RAG only: number of image chunks per image [Not recommended, try using layout_detection instead] """
    gpu_id: int | None = -1
    """ GPU ID to use for RAG embedder """
    search: bool = False
    """ Text-RAG only: whether to add bm25 search engine to similarity-based retrieval """
    reindex: bool = False
    """ Whether to allow full reindexing of an existing RAG """
    index_protection: bool = True
    """ Whether to activate index protection of an existing RAG,
    prevents reindexing from scratch an existing RAG index """
    update_index: bool = False
    """ Whether to allow incremental indexing of an existing RAG in a call
    (diff to the existing index is automatically discovered) """
    top_k: int = 4
    """ Top-k documents retrieved by RAGs. Note: internally more may be retrieved, but top-k are surfaced """
    remove_duplicates: bool = True
    """ Whether to remove duplicates """
    num_partitions: int = -1
    """ Colbert-only: number of clustering partitions, if unset, internally evaluated """
    index_bsize: int = 2
    """ Colbert-only: indexing batch size """
    ragm: RAGMultimodalObj = Field(default_factory=RAGMultimodalObj)
    """ V-RAG specific parameters """
    vllm_rag_quantization: str | None = None
    """ vllm backend LLM weights quantization for vllm, e.g. autoawq, ... """
    vllm_rag_memory_utilization: float = 0.4
    """ VLLM memory utilization ratio """
    vllm_rag_enforce_eager: bool = True
    """ Whether to enforce eager execution """


# class IndexObj(BaseModel):
#   files: List[str] | None = ["pdf"]
#   indexer: str | None = "colpali"


class TemplatePromptObj(BaseModel):
    """Template prompt object"""

    template_prompt: str | None = None
    """ Template prompt with placeholder variables """
    template_prompt_variables: list[str] = Field(default_factory=list)
    """ Template prompt variables, e.g. ['{context}', '{question}'] """


class InputConnectorObj(BaseModel):
    """Input connector, get data into the proper shape for indexing, searching and answering"""

    lib: Literal["coldb", "diffusr", "hf", "langchain"] = "hf"
    """ Colette RAG backend to use, e.g. hf for V-RAG, langchain for txt-only RAG """
    preprocessing: PreprocessingObj = Field(default_factory=PreprocessingObj)
    """ Preprocessing parameters """
    rag: RAGObj | None = None
    """ RAG parameters """
    template: TemplatePromptObj | None = None
    """ Template prompt parameters """
    message: str | None = None
    """ Question to the RAG """
    data: list[DirectoryPath] | None = None
    """ Data folders to be indexed """
    session_id: str | None = None
    """ RAG session id, when using conversational mode """
    summarize: str | None = None
    """ Text-RAG only: whether to summarize conversation as context in conversational mode """
    _rag_question: str | None = None
    """ Internal RAG reformulated question. Not a user parameter """
    # index: IndexObj | None = None
    query_depth_mult: int = 200  # depth multiplier allowing index deep search (multimodal for now)
    """ V-RAG only, depth multiplier with ChromaDB to accomodate lose similarity-based search guarantees """
    streaming: bool = False
    """ Try to stream the output, might be ignored depending on the backend """

    @field_serializer("data", mode="plain")
    def serialize_data(self, data: list[DirectoryPath] | None) -> list[str] | None:
        if data is None:
            return None
        return [str(path) for path in data]


class LLMInferenceObj(BaseModel):
    """LLM and multimodal LLM inference parameters"""

    lib: Literal["diffusers", "huggingface", "llamacpp", "ollama", "vllm", "vllm_client"] = "huggingface"
    """ LLM inference lib, huggingface, ollama, vllm, diffusers """
    sampling_params: dict[str, Any] = {"temperature": 0}
    """ Text-RAG only: inference sampling parameters """


# class LLMIndexObj(BaseModel):
#    name: str = None


class VLLMServerObj(BaseModel):
    url: str = "http://localhost:8000/v1"
    """ base url """
    api_key: str = "token-abc123"
    """ access key """


class LLMModelObj(BaseModel):
    """Main LLM parameters for answering"""

    source: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    """ Source model, e.g. Qwen/Qwen2-VL-2B-Instruct """
    filename: str | None = None
    """ Path to model file, when required (llamacpp), e.g. qwen2.5-0.5b-instruct-q8_0.gguf """
    shared_model: bool = True
    """ Whether to share the model across services, i.e. model is loaded once for multiple applications """
    inference: LLMInferenceObj | None = None
    context_size: int = 2048
    """ LLM context size """
    gpu_ids: list[int] | None = None
    """ vllm backend GPU IDs to use for LLM inference """
    vllm_quantization: str | None = None
    """ vllm backend LLM weights quantization for vllm, e.g. autoawq, ... """
    vllm_memory_utilization: float = 0.9
    """ VLLM memory utilization ratio """
    vllm_enforce_eager: bool = True
    """ Whether to enforce eager execution """
    image_width: int | None = None
    """ Source image width """
    image_height: int | None = None
    """ Source image height """
    dtype: str = "auto"
    host: str | None = None
    """ Ollama: host to connect to, e.g. localhost """
    port: int | None = None
    """ Ollama: port to connect to, e.g. 8000 """
    conversational: bool = False
    """ Whether to use conversational mode: session-based turn-based conversations with one RAG call per session """
    stitch_crops: bool = False
    """ Whether to stitch crops into a single image """
    load_in_8bit: bool = False
    """ Whether to load in 8-bit format """
    query_rephrasing: bool = False
    """ Whether to rephrase queries (V-RAG) only """
    query_rephrasing_num_tok: int = 512
    """ Number of token for rephrasing queries (V-RAG) only """

    external_vllm_server: VLLMServerObj = Field(default_factory=VLLMServerObj)
    """ vllm server parameters """


class OutputConnectorObj(BaseModel):
    """Output connector for post-processing parameters"""

    postprocessing: list[str] = Field(default_factory=list)
    output_format: Literal["json"] = "json"
    base64: bool = True
    num_tokens: int = 512
    # XXX: json_schema


class ParametersObj(BaseModel):
    """Global (input, llm, output) parameters holder"""

    input: InputConnectorObj = Field(default_factory=InputConnectorObj)
    """ Input parameters object """
    llm: LLMModelObj | None = None
    """ LLM parameters object """
    output: OutputConnectorObj = Field(default_factory=OutputConnectorObj)
    """ Output parameters object """


class VerboseEnum(str, Enum):
    """Logging verbosity options"""

    info = "info"
    debug = "debug"
    warning = "warning"
    error = "error"
    critical = "critical"


class AppObj(BaseModel):
    """Application information and settings"""

    repository: DirectoryPath | NewPath | None = None
    """Application repository, containing index, data and models unless configured otherwise"""
    create_repository: bool = True
    """ Whether to create the RAG application repository automatically """
    models_repository: str | None = None
    """ Whether to store models in a dedicated directory, otherwise stored within the RAG application repository """
    # init: AnyUrl | None = None
    verbose: VerboseEnum | None = VerboseEnum.info
    """ Verbosity parameters """
    log_in_app_dir: bool = True
    """ Whether to log in the application directory """

    @field_serializer("repository")
    def serialize_repository(self, value: DirectoryPath | Path | None) -> str | None:
        return str(value) if value is not None else None


class APIData(BaseModel):
    """Main API object"""

    description: str | None = None
    """ Application description string """
    app: AppObj | None = None
    """ Main RAG application object """
    parameters: ParametersObj = Field(default_factory=ParametersObj)
    """ Main parameters object """


class InfoObj(BaseModel):
    """System info response object"""

    services: list[str] | None = None  # List of services or None
    """ List of active services as /info API call output """


class StatusObj(BaseModel):
    """System status response object"""

    code: int | None = None
    """ API status code """
    status: str | None = None
    """ API status description """
    colette_code: int | None = None
    """ Colette status code """
    colette_message: str | None = None
    """ Colette status message """


class APIResponse(BaseModel):
    """Main API Response object to all calls"""

    version: str | None = None
    """ Colette commit version """
    status_code: int | None = None  # Status code can be None
    """ Status code """
    status: StatusObj | None = None  # StatusObj or None
    """ Main status object """
    info: InfoObj | None = None  # InfoObj or None
    """ Main info response object """
    service_name: str | None = None  # service_name can be None
    """ Service name response """
    full_prompt: str | None = None
    """ Full prompt to the LLM when answering a question """
    full_response: dict[str, Any] | None = None
    """ Full RAG response to a question """
    sources: dict[str, list[dict[str, Any]]] | None = None  # sources can be None
    """ RAG sources to the answer """
    message: str | None = None
    """ Original input question as message """
    output: str | None = None
    """ RAG API output, i.e. object to take the answer's from """
