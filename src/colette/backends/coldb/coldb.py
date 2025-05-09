from pathlib import Path
from typing import Any

import chromadb
import chromadb.config
from chromadb import Documents, EmbeddingFunction, Embeddings

from colette.backends.coldb.embedder import Embedder
from colette.backends.coldb.indexer import Indexer
from colette.backends.coldb.retriever import Retriever

_DEFAULT_COLLECTION_NAME = "mydb"


class DummyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # Embeddings is a List[Embedding]
        # Embedding is a chromadb.Vector
        # chromedb.Vector is Union[Sequence[float], Sequence[int]]
        # Documents is a list of Documents
        return [[1]] * len(input)


class ColDB:
    def __init__(
        self,
        persist_directory: Path = None,
        embedding_model_path: Path = None,
        collection_name: str = _DEFAULT_COLLECTION_NAME,
        embedding_model: str = "colbertv2.0",
        embedding_lib: str = "colbert",
        logger: Any = None,
        recreate=False,
        gpu_id=None,
        num_partitions=-1,
        index_bsize=-1,
        image_width=512,
        image_height=512,
        kvstore=None,
        **kwargs: Any,
    ) -> None:
        persist_directory = Path(persist_directory) if persist_directory else None
        embedding_model_path = (
            Path(embedding_model_path) if embedding_model_path else None
        )
        self.embedding_model_path = str(
            embedding_model_path / embedding_model
            if embedding_model_path is not None
            else persist_directory.parent / "models" / embedding_model
        )
        self.logger = logger
        self.embedding_lib = embedding_lib
        self.embedding_model = embedding_model
        self.gpu_id = gpu_id
        self.num_partitions = num_partitions
        self.index_bsize = index_bsize
        self.kvstore = kvstore
        if self.embedding_lib == "colbert":
            self.typical_doclen = 150
        else:
            self.typical_doclen = 500

        self.image_height = image_height if image_height is not None else 512
        self.image_width = image_width if image_width is not None else 512
        self.current_colname = collection_name
        self.embedder = None
        self.indexer = None
        # will be inited just before indexing

        self.persist_directory = persist_directory
        if self.persist_directory:
            _client_settings = chromadb.config.Settings(
                is_persistent=True, anonymized_telemetry=False
            )
            _client_settings.persist_directory = str(persist_directory)
        else:
            _client_settings = chromadb.config.Settings(anonymized_telemetry=False)
        self._client_settings = _client_settings
        self._client_settings.allow_reset = True
        self._client = chromadb.Client(self._client_settings)
        self._client.clear_system_cache()

        if recreate and self._client.count_collections() != 0:
            self._collection = {}
            self._client.reset()
            self._client.clear_system_cache()

        if not recreate and self._client.count_collections() != 0:
            self._collection = {}
            for c in self._client.list_collections():
                col = self._client.get_collection(name=c)
                self._collection[c] = col
        else:
            self._collection = {}

    def lazy_init(self):
        if self.embedder is not None:
            return
        self.embedder = Embedder(
            str(Path(self.persist_directory).parent),
            self.embedding_lib,
            self.embedding_model,
            self.embedding_model_path,
            logger=self.logger,
            gpu_id=self.gpu_id,
            index_bsize=self.index_bsize,
            image_width=self.image_width,
            image_height=self.image_height,
            kvstore=self.kvstore,
        )
        self.indexer = Indexer(
            self.embedder,
            verbose=0,
            logger=self.logger,
            gpu_id=self.gpu_id,
            persist_directory=str(self.persist_directory),
            typical_doclen=self.typical_doclen,
            num_partitions=self.num_partitions,
        )

    def release_embedder(self):
        del self.indexer
        del self.embedder
        self.indexer = None
        self.embedder = None

    def count(self):
        return self._collection[self.current_colname].count()

    def add_docs(self, texts, metadatas, collection_name):
        if len(texts) == 0:
            return
        self.current_colname = collection_name

        self._collection[collection_name] = self._client.create_collection(
            name=collection_name,
            embedding_function=DummyEmbeddingFunction(),
            get_or_create=True,
        )

        offset = 0
        while offset < len(texts) - 1000:
            myids = [str(i) for i in range(offset, offset + 1000)]
            self._collection[collection_name].add(
                ids=myids,
                documents=texts[offset : offset + 1000],
                metadatas=metadatas[offset : offset + 1000],
            )
            offset += 1000
        myids = [str(i) for i in range(offset, len(texts))]
        self._collection[collection_name].add(
            ids=myids, documents=texts[offset:], metadatas=None
        )

        self.lazy_init()

        self.indexer.index(
            name=collection_name,
            collection=texts,
            dbdir=str(self.persist_directory),
            overwrite=True,
        )
        self.release_embedder()

    def add_imgs(
        self,
        imgs: list[dict],
        metadatas: list[dict] | None = None,
        collection_name: str = "mmdb",
    ):
        if len(imgs) == 0:
            return
        self.current_colname = collection_name

        self._collection[collection_name] = self._client.create_collection(
            name=collection_name,
            embedding_function=DummyEmbeddingFunction(),
            get_or_create=True,
        )

        imgs = [str(name) for name in imgs]

        offset = 0
        while offset < len(imgs) - 1000:
            myids = [str(i) for i in range(offset, offset + 1000)]
            mydocs = [imgs[i] for i in range(offset, offset + 1000)]
            self._collection[collection_name].add(
                ids=myids,
                documents=mydocs,
                metadatas=[] if not metadatas else metadatas[offset : offset + 1000],
            )
            offset += 1000
        myids = [str(i) for i in range(offset, len(imgs))]
        mydocs = [imgs[i] for i in range(offset, len(imgs))]
        self._collection[collection_name].add(
            ids=myids,
            documents=mydocs,
            metadatas=[] if not metadatas else metadatas[offset:],
        )

        self.lazy_init()
        print(type(collection_name), type(imgs), type(self.persist_directory))
        self.indexer.index(
            name=collection_name,
            collection=imgs,
            dbdir=str(self.persist_directory),
            overwrite=True,
        )
        self.release_embedder()

    @classmethod
    def from_documents(
        cls: type["ColDB"],
        documents: list[dict],
        collection_name: str = _DEFAULT_COLLECTION_NAME,
        persist_directory: str | None = None,
        embedding_model_path: str | None = None,
        embedding_lib: str = None,
        embedding_model: str = None,
        logger: Any = None,
        gpu_id: int = None,
        num_partitions: int = -1,
    ) -> "ColDB":
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model_path=embedding_model_path,
            embedding_model=embedding_model,
            embedding_lib=embedding_lib,
            logger=logger,
            gpu_id=gpu_id,
            num_partitions=num_partitions,
        )

    @classmethod
    def from_texts(
        cls: type["ColDB"],
        texts: list[str],
        metadatas: list[dict] | None = None,
        collection_name: str = _DEFAULT_COLLECTION_NAME,
        persist_directory: str | None = None,
        embedding_model_path: str | None = None,
        embedding_model: str = None,
        embedding_lib: str = None,
        logger: Any = None,
        gpu_id: int = None,
        num_partitions: int = -1,
    ) -> "ColDB":
        db = cls(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model_path=embedding_model_path,
            embedding_model=embedding_model,
            embedding_lib=embedding_lib,
            logger=logger,
            recreate=True,
            gpu_id=gpu_id,
            num_partitions=num_partitions,
        )
        db.add_docs(texts, metadatas, collection_name)
        return db

    def as_retriever(self, search_kwargs: dict | None = None, **kwargs: Any):
        top_k = 4
        if search_kwargs is not None:
            if "k" in search_kwargs:
                top_k = search_kwargs["k"]
        return Retriever(
            self.embedding_lib,
            self.embedding_model,
            self.current_colname,
            self.persist_directory,
            self.embedding_model_path,
            self.logger,
            top_k,
            self._collection,
            self.gpu_id,
            self.kvstore,
        )

    def similarity_search(self, query: str, **kwargs: Any) -> list[dict]:
        return self.invoke(query)
