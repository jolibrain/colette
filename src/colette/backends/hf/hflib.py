import base64
from pathlib import Path

from colette.apidata import APIData, APIResponse
from colette.inputconnector import (
    InputConnector,
    InputConnectorInternalException,
)
from colette.llmlib import LLMLib
from colette.llmmodel import LLMModel
from colette.outputconnector import OutputConnector

from .hyde_generator import HyDEGenerator
from .query_rephraser import QueryRephraser
from .session_cache import SessionCache

import time

def load_image_as_base64(path: Path) -> str:
    #
    # Load image as base64
    # @param path: str
    # @return: str
    #
    suffix_to_type = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png"}

    image_type = suffix_to_type[path.suffix]
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/{image_type};base64,{encoded_string}"


class HFLib(LLMLib):
    def __init__(self, inputc: InputConnector, outputc: OutputConnector, llmmodel: LLMModel):
        super().__init__(inputc, outputc, llmmodel)
        self.streamers = {}

    def __del__(self):
        pass

    def init(self, ad: APIData, kvstore):
        self.kvstore = kvstore
        # if rag, do index
        if self.inputc.rag:
            self.inputc.ragobj.kvstore = self.kvstore
            # self.inputc.rag_index(ad.parameters.input)

        if self.llmmodel:
            self.llmmodel.initialize_llm()
            self.llmmodel.kvstore = self.kvstore
            # Wire hindex vlm_fn: lets RAGImg use the inference LLM for hierarchy
            # building (OCR + structure extraction) and tree navigation at query time.
            if self.inputc is not None and getattr(self.inputc, "ragobj", None) is not None:
                self.inputc.ragobj.vlm_fn = self._make_hindex_vlm_fn(self.llmmodel)
                # vllm_client fires concurrent HTTP requests to an external server that
                # batches them natively — use multiple workers.  In-process models (hf,
                # vllm) hold a GPU lock per call so extra workers have no effect.
                self.inputc.ragobj.hindex_max_workers = (
                    8 if self.llmmodel.llm_type == "vllm_client" else 1
                )
                # vllm_client parallelises via workers (one request per page);
                # in-process models batch multiple pages in a single forward pass.
                self.inputc.ragobj.hindex_batch_size = (
                    1 if self.llmmodel.llm_type == "vllm_client" else 4
                )

        # chat history sessions
        self.sessions = SessionCache(max_sessions=100)

        # optional query rephrasing
        if self.llmmodel is not None and self.llmmodel.query_rephrasing:
            self.query_rephraser = QueryRephraser(
                self.llmmodel,
                self.inputc.ragobj,
                3,
                ad.parameters.llm.query_rephrasing_num_tok,
                self.logger,
            )
        else:
            self.query_rephraser = None

        # optional HyDE (Hypothetical Document Embeddings)
        if self.llmmodel is not None and self.llmmodel.use_hyde:
            self.hyde_generator = HyDEGenerator(
                self.llmmodel,
                ad.parameters.llm.hyde_num_tokens,
                self.logger,
            )
        else:
            self.hyde_generator = None

    def predict(self, ad: APIData) -> APIResponse:
        # - input connector transform()
        try:
            time_start = time.time()
            # optional HyDE: replace retrieval query with a hypothetical answer
            retrieval_query = None
            if self.hyde_generator:
                retrieval_query = self.hyde_generator.generate(ad.parameters.input.message)
            message, docs_source, _ = self.inputc.transform(
                ad.parameters.input, self.query_rephraser, retrieval_query=retrieval_query
            )
            time_end = time.time()
            self.logger.info(f"Input transform and document retrieval took {time_end - time_start:.2f} seconds")
            self.logger.debug(f"message: {message}")
        except Exception as e:
            self.logger.error(e, exc_info=True)
            raise InputConnectorInternalException("HFLib input transform error") from e

        # sessions and message
        history = []
        if self.llmmodel.llm_obj.conversational and ad.parameters.input.session_id is not None:
            # create session if does not exist
            if not self.sessions.session_exists(ad.parameters.input.session_id):
                self.sessions.create_session(ad.parameters.input.session_id)
            else:
                # assemble message if session exists
                history = self.sessions.get_session(ad.parameters.input.session_id)

        # - hf call
        time_start = time.time()
        response, new_message, streamer = self.llmmodel.generate(
            message,
            self.outputc.num_tokens,
            docs_source,
            history,
            streaming=ad.parameters.input.streaming,
        )
        self.logger.debug(f"response: {response}")
        time_end = time.time()
        self.logger.info(f"LLM generation took {time_end - time_start:.2f} seconds")

        # if session, update it
        if self.llmmodel.llm_obj.conversational and ad.parameters.input.session_id is not None:
            self.sessions.update_session(ad.parameters.input.session_id, new_message)
            self.sessions.update_session(
                ad.parameters.input.session_id,
                {"role": "user", "content": [{"type": "text", "content": response}]},
            )  # XXX: this is assistant response, but stored as user, otherwise ignored by the chat template

        # create context
        context = []
        for pos, key in enumerate(docs_source["ids"][0]):
            distance = 0
            # if "metadata" in docs_source:
            #    distance = docs_source["metadata"][0][pos].get("distance", 0)
            if "distances" in docs_source:
                distance = docs_source["distances"][0][pos]

            source = {
                "key": key,
                "distance": distance,
                # "content": transform_pil_image_to_base64(docs_source["images"][0][pos]),
            }
            source = self.outputc.add_base64_to_source(source, docs_source["images"][0][pos])
            # Adding metadata if any
            if "metadatas" in docs_source:
                for k, v in docs_source["metadatas"][0][pos].items():
                    source[k] = v

            context.append(source)
        tsources = {"context": context}
        if "text_context" in docs_source:
            tsources["text_context"] = docs_source["text_context"]

        # save streamer object for upcoming streaming call
        if streamer:
            self.streamers[ad.parameters.input.session_id] = streamer

        # - send response back
        response = APIResponse(
            sources=tsources,
            message=message,
            output=response,
        )
        return response

    def streaming(self, ad: APIData):
        streamer = self.streamers[ad.parameters.input.session_id]
        response = ""
        for token in streamer:
            yield token
            response += token
        # if session, update it
        if self.llmmodel.llm_obj.conversational:
            self.sessions.update_streaming(ad.parameters.input.session_id, response)

    def _make_hindex_vlm_fn(self, llmmodel):
        """Return a simple Callable(images, prompt) -> str for use by hindex.

        Wraps the already-initialized inference LLM so HierarchyBuilder and
        HierarchyRetriever can call it without knowing which backend is active.
        Text-only calls (empty images list) are supported for tree navigation.
        """
        logger = self.logger

        def vlm_fn(images, prompt: str, max_new_tokens: int = 512) -> str:
            # 512 tokens is enough for single-page JSON extraction (~250 tokens
            # typical output).  Batch callers pass a larger budget explicitly.
            docs = {
                "ids": [[""] * len(images)],
                "images": [list(images)],
                "metadatas": [
                    [{"source": "hindex", "page_number": i, "crop_label": "full_page"}
                     for i in range(len(images))]
                ],
                # hindex tasks (OCR, extraction, navigation) produce short
                # structured answers — disable thinking to avoid token overflow.
                "_disable_thinking": True,
            }
            try:
                response, _, _ = llmmodel.generate(
                    prompt,
                    max_new_tokens,
                    docs,
                    [],
                    streaming=False,
                )
                return response or ""
            except Exception as exc:
                logger.warning(f"[hindex] vlm_fn call failed: {exc}")
                return ""

        return vlm_fn

    def update_index(self, ad: APIData):
        # kvstore may have been opened read-only (query mode); switch to write mode
        # for the duration of indexing, then flush and revert to read-only.
        self.kvstore.reopen("a")
        try:
            nelt = self.inputc.rag_update_index(ad.parameters.input)
        except Exception as e:
            self.logger.error(e, exc_info=True)
            raise InputConnectorInternalException("HFLib input transform error") from e
        finally:
            self.kvstore.reopen("r")
        response = APIResponse(message=f"{nelt} elements in rag_index")
        return response
