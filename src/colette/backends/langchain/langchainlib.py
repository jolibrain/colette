## Langchain-based backend for Colette

import os

from langchain.globals import set_debug
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from colette.apidata import APIData, APIResponse
from colette.inputconnector import (
    InputConnector,
    InputConnectorInternalException,
)
from colette.llmlib import LLMLib, LLMLibInternalException
from colette.llmmodel import LLMModel
from colette.outputconnector import OutputConnector


class LangChainLib(LLMLib):
    def __init__(self, inputc: InputConnector, outputc: OutputConnector, llmmodel: LLMModel):
        set_debug(os.getenv("LANGCHAIN_DEBUG"))
        super().__init__(inputc, outputc, llmmodel)

    def init(self, ad: APIData, kvstore):
        # if rag, do index
        # if self.inputc.rag:
        #     self.inputc.rag_index(ad.parameters.input)

        # save default prompt
        if self.inputc.template_prompt:
            self.template_prompt = self.inputc.template_prompt
        else:
            self.template_prompt = None

        # chat history sessions
        self.sessions = {}

    def update_index(self, ad: APIData):
        # if rag, do index
        if self.inputc.rag:
            self.inputc.rag_index(ad.parameters.input)
        response = APIResponse(message="rag_index done")
        return response

    def get_session(self, session_id):
        # if no session_id was provided, return a temporary dummy session
        if session_id is None:
            return ChatMessageHistory()
        # retrieve session or create a new one
        session = self.sessions.setdefault(session_id, ChatMessageHistory())
        return session

    def get_rag_question(self, ad: APIData):
        question = ad.parameters.input.message
        # not conversational, rag question is last message
        if not self.llmmodel.llm_obj.conversational:
            return question
        # first interaction, rag question is last (only) message
        history = self.get_session(ad.parameters.input.session_id)
        if not history.messages:
            return question
        # get predict summarize
        summarize = ad.parameters.input.summarize
        # default to service summarize
        if not summarize:
            summarize = self.inputc.ad.summarize
        # no summarize prompt, rag question is last message
        if not summarize:
            return question
        # conversational with multiple interactions, rag question is summary
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", summarize),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        chain = prompt | self.llmmodel.llm
        generated = chain.invoke({"history": history.messages, "question": question})
        return generated.content

    def predict(self, ad: APIData) -> APIResponse:
        if self.inputc.rag:
            ad._rag_question = self.get_rag_question(ad)
            self.logger.debug(f"rag_question: {ad._rag_question}")
        try:
            message, template_prompt, input_dict, docs_source = self.inputc.transform(ad)
        except Exception as e:
            self.logger.error(e)
            raise InputConnectorInternalException("Langchainlib input transform error") from e

        if self.llmmodel.llm_obj.conversational:
            system_template = ""
            if template_prompt:
                system_template = template_prompt.template
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_template),
                    MessagesPlaceholder(variable_name="question"),
                ]
            )

            chain = RunnableWithMessageHistory(
                prompt | self.llmmodel.llm,
                self.get_session,
                input_messages_key="question",
            )
        else:
            if template_prompt and template_prompt.template != "":
                chain = template_prompt | self.llmmodel.llm
            else:
                chain = None
        try:
            self.logger.debug(f"message: {message}")
            self.logger.debug(f"template_prompt: {template_prompt}")
            self.logger.debug(f"input_dict: {input_dict}")
            self.logger.debug(f"docs_source: {docs_source}")
            config = {"configurable": {"session_id": ad.parameters.input.session_id}}
            if chain:
                generated_output = chain.invoke(input_dict, config)
            else:
                generated_output = self.llmmodel.llm.invoke(message)
            self.logger.debug(f"generated_output: {generated_output}")
        except Exception as e:
            self.logger.error(e)
            raise LLMLibInternalException("Langchainlib chain invoke error") from e

        if not isinstance(generated_output, AIMessage):
            generated_output = AIMessage(content=generated_output)

        self.logger.debug(f"docs_source: {docs_source}")
        self.logger.debug(f"message: {message}")
        self.logger.debug(f"output: {generated_output.content}")
        response = APIResponse(
            # full_prompt = str(full_prompt), # TODO: config from output connector
            # full_response = generated_output.dict(), # TODO: config from output connector
            sources=docs_source,
            message=message,
            output=generated_output.content,
        )
        return response

    def train(self, ad: APIData):
        # N/A
        return {}
