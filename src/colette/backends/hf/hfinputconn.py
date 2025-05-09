from colette.apidata import InputConnectorObj
from colette.inputconnector import InputConnector

from .rag.rag_img import RAGImg


class HFInputConn(InputConnector):
    """
    HF as input connector (img rag for now)
    """

    def __init__(self):
        super().__init__()

    def __del__(self):
        """
        Destroy the input connector.
        """
        if self.ragobj is not None:
            del self.ragobj
            self.ragobj = None

    def init(self, ad: InputConnectorObj, kvstore):
        """
        Initialize the input connector.
        """
        super().init(ad)
        self.ad = ad
        self.kvstore = kvstore
        self.template = ad.template  # default value

        # select images here
        self.ragobj = None
        self.rag = False
        if ad.rag is not None:
            self.ragobj = RAGImg()
            self.ragobj.init(self.ad, self.app_repository, self.models_repository, self.cpu, self.logger, self.kvstore)
            self.rag = True

    def transform(self, ad: InputConnectorObj, query_rephraser=None):
        """
        Transform the input data into a template prompt and

        :param ad: InputConnectorObj
        :return: message, template_prompt, full_prompt, docs_source
        """
        message = ad.message

        # retrieve docs from RAG
        if self.ragobj is not None:
            docs = self.ragobj.retrieve(message, ad.query_depth_mult)

        # optional query rephrasing
        if query_rephraser:
            rqueries, rmerged_docs = query_rephraser.rephrase(message, docs)
            docs = rmerged_docs  ##TODO: prune for top k ?
            # if rqueries:
            #    message = message + "\n".join(rqueries)
        else:
            rqueries = []

        # template prompt (not used for retrieval)
        template = ad.template
        if not template:
            template = self.template

        if template:
            if template.template_prompt and template.template_prompt_variables:
                for var in template.template_prompt_variables:
                    if var == "question":  # no context text var since context is made of images
                        new_message = template.template_prompt.replace("{" + var + "}", message)
                message = new_message

        return message, docs, rqueries

    def rag_index(self, ad: InputConnectorObj):
        super().get_data(ad)
        self.ragobj.index(ad, self.sorted_data)

    def rag_update_index(self, ad: InputConnectorObj):
        if ad.rag.reindex:
            self.rag_index(ad)
            return
        sorted_data_ = self.sorted_data
        self.sorted_data = {}
        super().get_data(ad)
        # self.ragobj.update_index(ad, self.sorted_data)
        self.ragobj.index(ad, self.sorted_data)
        new_sorted_data_ = self.sorted_data
        self.sorted_data = sorted_data_
        for fext in new_sorted_data_:
            if fext in self.sorted_data:
                self.sorted_data[fext].extend(new_sorted_data_[fext])
            else:
                self.sorted_data[fext] = new_sorted_data_[fext]

    def delete_inputc(self):
        if self.ragobj:
            self.ragobj.delete_embedder()
