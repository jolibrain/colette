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

from .query_rephraser import QueryRephraser
from .session_cache import SessionCache


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

    def predict(self, ad: APIData) -> APIResponse:
        # - input connector transform()
        try:
            message, docs_source, _ = self.inputc.transform(ad.parameters.input, self.query_rephraser)
            self.logger.info(f"message: {message}")
            # self.logger.debug(f"docs_source: {json.dumps(docs_source, indent=2)}")
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
        response, new_message, streamer = self.llmmodel.generate(
            message,
            self.outputc.num_tokens,
            docs_source,
            history,
            streaming=ad.parameters.input.streaming,
        )
        self.logger.debug(f"response: {response}")

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

    def update_index(self, ad: APIData):
        try:
            nelt = self.inputc.rag_update_index(ad.parameters.input)
        except Exception as e:
            self.logger.error(e, exc_info=True)
            raise InputConnectorInternalException("HFLib input transform error") from e
        response = APIResponse(message=f"{nelt} elements in rag_index")
        return response
