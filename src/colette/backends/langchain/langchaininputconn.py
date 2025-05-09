# Langchain as input connector (preprocessing + rag)
from langchain_core.prompts import PromptTemplate

from colette.apidata import InputConnectorObj, TemplatePromptObj
from colette.inputconnector import (
    InputConnector,
    InputConnectorBadParamException,
)

from .rag.rag_txt import RAGTxt


class LangChainInputConn(InputConnector):
    """
    Langchain as input connector (preprocessing + rag)
    """

    def __init__(self):
        super().__init__()

    def init(self, ad: InputConnectorObj, kvstore):
        """
        Initialize the input connector.
        Set up:
         - the cleaning steps for the unstructured data
         - the RAG processing functions for the different file types
        """
        super().init(ad)
        self.ad = ad
        self.template = ad.template  # default value
        if self.template:
            self.template_prompt = self.create_template_prompt(self.template)  # default prompt template
        else:
            self.template_prompt = None

        # select txt or images here
        self.ragobj = None
        self.rag = False
        if ad.rag is not None:
            self.ragobj = RAGTxt()
            self.ragobj.init(
                self.ad,
                self.app_repository,
                self.models_repository,
                self.cpu,
                self.logger,
            )
            self.rag = True

    def create_template_prompt(self, template: TemplatePromptObj):
        """
        Create a prompt template from the template object
        """
        return PromptTemplate(
            template=template.template_prompt,
            input_variables=template.template_prompt_variables,
        )

    def transform(self, ad: InputConnectorObj):
        """
        Transform the input data into a template prompt and

        :param ad: InputConnectorObj
        :return: message, template_prompt, full_prompt, docs_source
        """
        message = ad.parameters.input.message
        template = ad.parameters.input.template
        if template:
            template_prompt = self.create_template_prompt(template)
        else:
            template_prompt = self.template_prompt  # use default prompt in lib

        docs_source = {"context": []}
        if self.ragobj is not None:
            if ad._rag_question is not None:
                # retrieving chunks from retrieved docs
                # docs = self.rag_retriever.invoke(ad._rag_question)
                docs = self.ragobj.retrieve(ad._rag_question)

                # storing the chunks used for the context
                for d in docs:
                    has_start_index = "start_index" in d.metadata
                    self.logger.debug(
                        "source: %s, page: %s, content: %s, start_index: %s",
                        d.metadata["source"],
                        d.metadata["page"] if "page" in d.metadata else None,
                        d.page_content,
                        d.metadata["start_index"] if has_start_index else None,
                    )
                    docs_source["context"].append(
                        {
                            "source": d.metadata.get("source", None),
                            "page": d.metadata.get("page", None),
                            "content": d.page_content if hasattr(d, "page_content") else None,
                            "start_index": d.metadata.get("start_index", None) if has_start_index else None,
                        }
                    )

                self.logger.info("RAG has retrieved %s docs", len(docs))
                context_fill = (
                    "\n\n".join(
                        f"Source {idx + 1}: {doc.metadata['source']}\n{doc.page_content}"
                        for idx, doc in enumerate(docs)
                    )
                    + "\n\n"
                )
                self.logger.debug(
                    "retrieved context has %s chars ~= %s tokens",
                    len(context_fill),
                    len(context_fill) / 4,
                )
                input_dict = {}
                # [TODO]: check if all variables are in the template and possibly retrive their name from the template
                # template_prompt_variables = re.findall(r'\{([a-zA-Z0-9_]+)\}', template_prompt)
                for var in self.template.template_prompt_variables:
                    input_dict[var] = ""
                    if var == "context":
                        input_dict[var] = context_fill
                    elif var == "question":
                        input_dict[var] = message
                return message, template_prompt, input_dict, docs_source
        elif message is not None:
            if self.template is not None:
                if self.template.template_prompt_variables:
                    input_dict = {self.template.template_prompt_variables[0]: message}
            else:
                input_dict = {"question": message}
            return (
                message,
                template_prompt,
                input_dict,
                docs_source,
            )
        else:
            msg = "No data or message as input"
            self.logger.error(msg)
            raise InputConnectorBadParamException(msg)

    def rag_index(self, ad: InputConnectorObj):  ##TODO: index
        """
        Index or Re-Index the documents for RAG
        """
        super().get_data(ad)  # data in self.data and self.sorted_data
        self.rag_indexdb = self.ragobj.index(ad, self.sorted_data)  # returns an indexdb

    def delete_inputc(self):
        pass
