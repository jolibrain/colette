from colette.backends.coldb.searcher import Searcher

RunnableConfig = dict


class Retriever:
    def __init__(
        self,
        embedding_lib,
        embedding_model,
        collection_name,
        index_root,
        checkpoint,
        logger,
        top_k,
        collectiondb,
        gpu_id,
        kvstore,
    ):
        self.embedding_lib = embedding_lib
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.index_root = index_root
        self.checkpoint = checkpoint
        self.logger = logger
        self.gpu_id = gpu_id
        self.kvstore = kvstore

        self.searcher = None
        self.top_k = top_k
        self.collectiondb = collectiondb

    def invoke(self, message: str, config: RunnableConfig = None):
        searcher = Searcher(
            self.embedding_lib,
            self.embedding_model,
            index=self.collection_name,
            index_root=self.index_root,
            checkpoint=self.checkpoint,
            verbose=0,
            logger=self.logger,
        )

        pids, ranks, scores = searcher.search(message, k=self.top_k)
        docs = self.collectiondb[self.collection_name].get(ids=[str(i) for i in pids])
        if self.embedding_lib == "colbert":
            docs = [dict(page_content=docs["documents"][i], metadata=docs["metadatas"][i]) for i in range(len(pids))]
        elif self.embedding_lib in ["huggingface"]:
            ret = {}
            ret["ids"] = [docs["documents"]]
            # ret["metadata"] = [[{**item, "distance": scores}
            # if item is not None else {"distance": score}
            # for item, score in zip(docs["metadatas"], scores)]]
            ret["distances"] = [scores]
            ret["metadatas"] = [docs["metadatas"]]
            # to save time retriever could also load images from kvstore
            # ret["images"] = [[self.kvstore.retrieve_image(key) for key in docs["documents"]]]
            # docs = [
            #     Document(
            #         page_content=docs["documents"][i],
            #         metadata={"path": docs["documents"][i]},
            #     )
            #     for i in range(len(pids))
            # ]
            docs = ret

        del searcher

        return docs
