import json
from copy import copy

from .hfmodel import HFModel
from .rag.rag_img import RAGImg


class QueryRephraser:
    def __init__(self, model: HFModel, ragobj: RAGImg, nq: int, num_tok: int, logger):
        self.model = model
        self.ragobj = ragobj
        self.nq = nq
        self.logger = logger
        self.num_tok = num_tok
        self.prompt = 'You are an expert helping with search for useful and detailed information in technical corpuses. Using an initial question asked by a user, and the retrieved set of relevant documents obtained by a search engine, give a follow-up question that is interesting and relevant to deepen the understanding of the topics and details from the documents. Use informative and precise  words as much as possible. Keep the query in its original language. The initial question is <query>{query}</query>. The new query should be formatted according to the following JSON schema: {"query":{"type": "string", "content": "{query}"}}'
        # self.prompt = 'Tu est un expert chargé d aider à la recherche d information dans des corpus techniques. A la suite d une question posée, et des documents correspondants à une recherche pour cette question, donne une question complémentaire d intérêt pour approfondir l étude des documents proposés. Utilise des mots pertinents de facon à approfondir la compréhension des sujets portés par les documents. La question initiale est <query>{query}</query>. \nLa nouvelle question doit être donnée dans le schema JSON suivant: {"query":{"type": "string", "content": "{query}"}}'

    def rephrase(self, query: str, docs):
        # combine query with prompt
        qprompt = self.prompt.replace("{query}", query).replace("?", "")

        # rephrase from query with docs
        rqueries = []

        for _ in range(self.nq):
            docs_copy = copy(
                docs
            )  # we reuse the same documents for every rephrased query
            rquery = self.model.generate(
                qprompt,
                self.num_tok,
                docs_copy,
                do_sample=True,
            )
            # get query from format
            rquery_content = self.extract_query(rquery[0], query)
            del rquery
            rquery = None
            if rquery_content is None:
                continue
            rqueries.append(rquery_content)
            self.logger.info("\nRephrased query " + query + " as " + rquery_content)

        # apply rag to every rephrased query
        new_docs = [docs]
        for rquery in rqueries:
            rdocs = self.ragobj.retrieve(rquery, 200)
            self.logger.debug(
                "Retrieved "
                + str(len(rdocs["ids"][0]))
                + " documents for rephrased query "
                + rquery
            )
            new_docs.append(rdocs)
            self.logger.debug("New docs: " + str(len(new_docs)))

        # aggregate doc results and return
        rmerged_docs = self.merge_documents(new_docs)
        return rqueries, rmerged_docs

    def extract_query(self, query, query0):
        try:
            clean_query = query.replace("json", "").replace("\n", "")
            jquery = json.loads(clean_query)
            if (
                "query" in jquery
                and isinstance(jquery["query"]["content"], str)
                and jquery["query"]["content"] != ""
            ):
                nquery = jquery["query"]["content"]
                # checks that nquery != initial query
                if nquery.replace(" ", "").replace("?", "") != query0.replace(
                    " ", ""
                ).replace("?", ""):
                    return nquery
                else:
                    return None
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error extracting query from\n-----\n{query}\n-----\n")
            self.logger.error(e)
            return None

    def merge_documents(self, docs):
        # print("docs=", docs)
        merged_docs = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        for doc in docs:
            # check whether the document is already in the merged_docs using every id.
            # if already in merged_docs, remove related indices in embeddings, documents, ... and add the new ids only
            ids = doc["ids"][0]
            for did in ids:
                i = 0
                if did in merged_docs["ids"][0]:
                    index = merged_docs["ids"][0].index(did)
                    merged_docs["ids"][0].pop(index)
                    merged_docs["distances"][0].pop(index)
                    merged_docs["metadatas"][0].pop(index)
                i += 1

            merged_docs["ids"][0].extend(doc["ids"][0])
            merged_docs["distances"][0].extend(doc["distances"][0])
            merged_docs["metadatas"][0].extend(doc["metadatas"][0])

        return merged_docs
