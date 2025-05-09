import concurrent.futures
import logging
import os
import os.path
import pickle
import shutil
import threading
import traceback
from pathlib import Path

from chromadb.config import Settings
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader, PyMuPDFLoader, TextLoader

# from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from unstructured.cleaners.core import clean, replace_unicode_quotes

from colette.apidata import InputConnectorObj
from colette.backends.coldb import ColDB
from colette.inputconnector import (
    InputConnectorBadParamException,
    InputConnectorInternalException,
)


def split_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i : i + chunk_size]


class RAGTxt:
    def __init__(self):
        self.preprocess_lock = threading.Lock()

    def init(
        self,
        ad: InputConnectorObj,
        app_repository: Path,
        models_repository: Path,
        cpu: bool,
        logger: logging.Logger,
    ):
        self.ad = ad
        self.app_repository = app_repository
        self.models_repository = models_repository
        self.cpu = cpu
        self.logger = logger

        self.unstructured_cleaners = [
            clean,
            replace_unicode_quotes,
        ]

        ##TODO: other extractors, as needed
        if ad.rag is not None:
            self.rag = True
            self.rag_chunk_size = ad.rag.chunk_size
            self.rag_chunk_overlap = ad.rag.chunk_overlap
            if self.rag_chunk_size > 0:
                self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    # separators=["\n\n",
                    #           "\n",
                    #           " ",
                    #           ".",
                    #           ",",
                    #           "_"],
                    chunk_size=self.rag_chunk_size,
                    chunk_overlap=self.rag_chunk_overlap,
                    add_start_index=True,
                )
            else:
                self.text_splitter = None
            self.rag_embedding_lib = ad.rag.embedding_lib
            self.rag_search = ad.rag.search
            self.bm25_retriever = None
            # self.rag_reindex = ad.rag.reindex
            # self.rag_index_protection = ad.rag.index_protection
            self.rag_top_k = ad.rag.top_k
            # self.rag_gpu_id = ad.rag.gpu_id
            self.rag_num_partitions = ad.rag.num_partitions

            if ad.rag.embedding_model is None:
                msg = "Missing rag embedding model"
                self.logger.error(msg)
                raise InputConnectorBadParamException(msg)

            self.rag_embedding_model = ad.rag.embedding_model

            device = "cpu" if self.cpu else "cuda:" + str(ad.rag.gpu_id)

            self.logger.info(
                "loading embedding model %s on device %s",
                self.rag_embedding_model,
                device,
            )
            if self.rag_embedding_lib == "huggingface":
                model_kwargs = {"device": device}
                encode_kwargs = {"normalize_embeddings": True}

                self.rag_embedding = HuggingFaceEmbeddings(
                    model_name=self.rag_embedding_model,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    cache_folder=str(self.models_repository),
                )
            elif self.rag_embedding_lib == "colbert":
                self.rag_embedding = None
            else:
                msg = "Unknown embedding lib " + ad.rag.embedding_lib
                self.logger.error(msg)
                raise InputConnectorBadParamException(msg)

            self.rag_indexdb_lib = ad.rag.indexdb_lib
            if self.rag_indexdb_lib == "chromadb":
                self.rag_indexdb_obj = Chroma
                self.chroma_client_settings = Settings(is_persistent=True, anonymized_telemetry=False)
            elif self.rag_indexdb_lib == "coldb":
                self.rag_indexdb_obj = ColDB
            else:
                msg = "Unknown indexdb_lib " + ad.rag.indexdb_lib
                self.logger.error(msg)
                raise InputConnectorBadParamException(msg)

            self.rag_indexdb = None
            self.rag_retriever = None
        else:
            self.rag = False

        self.logger.info(f"init:\n\tRAG: {self.rag}\n")

    def preprocess_doc(self, preproc_func, doc, strat):
        """
        Preprocess a document as text using the given preproc_func and strategy

        :param preproc_func: the preprocessing functions
        :param doc: the document to preprocess
        :param strat: the preprocessing strategy
        """
        self.logger.debug("preprocessing doc %s", doc)
        try:
            if strat:
                fdoc = preproc_func(doc, **strat).load()
            elif preproc_func == JSONLoader:
                fdoc = preproc_func(file_path=doc, jq_schema=".", text_content=False).load()
            else:
                fdoc = preproc_func(doc).load()

            # cleanup / post-processing
            if self.preprocessing_cleaning:
                for pdoc in fdoc:
                    for cleaner in self.unstructured_cleaners:
                        try:
                            if cleaner == clean:
                                pdoc.page_content = clean(
                                    pdoc.page_content,
                                    extra_whitespace=True,
                                    dashes=True,
                                )
                            else:
                                pdoc.page_content = cleaner(pdoc.page_content)
                        except Exception as e:
                            self.logger.warning("error in cleaning doc %s: %s", doc, repr(e))
                            continue

        except Exception as e:
            self.logger.error("error in preprocessing doc %s: %s", doc, repr(e))
            self.logger.error(traceback.format_exc())
            if self.preprocessing_strict:
                raise InputConnectorInternalException(f"error in preprocessing doc {doc}: {repr(e)}") from e
            else:
                return
        word_len = 0
        for d in fdoc:
            word_len += len(d.page_content)
        self.logger.debug("preprocessed doc %s: %s chars", doc, word_len)
        with self.preprocess_lock:
            self.processed_docs.append(fdoc)

    def index(self, ad: InputConnectorObj, sorted_data: dict[str, list[str]]):
        self.preprocessing_strict = ad.preprocessing.strict
        self.preprocessing_lib = ad.preprocessing.lib
        self.preprocessing_cleaning = ad.preprocessing.cleaning
        if self.preprocessing_lib is None or self.preprocessing_lib == "unstructured":
            self.preprocessing_funcs = {
                "csv": UnstructuredLoader,
                "html": UnstructuredLoader,
                "eml": UnstructuredLoader,
                "xls": UnstructuredLoader,
                "xlsx": UnstructuredLoader,
                "png": UnstructuredLoader,
                "jpg": UnstructuredLoader,
                "jpeg": UnstructuredLoader,
                "md": TextLoader,
                "odt": UnstructuredLoader,
                "pdf": UnstructuredLoader,  # uses pdfminer to extract text
                "ppt": UnstructuredLoader,
                "pptx": UnstructuredLoader,
                "rtf": UnstructuredLoader,
                "doc": UnstructuredLoader,
                "docx": UnstructuredLoader,
                "xml": UnstructuredLoader,
                "rst": UnstructuredLoader,
                "yml": TextLoader,
                "txt": UnstructuredLoader,
                "json": JSONLoader,
            }
            ##TODO: urls
            self.others_preprocessing_func = UnstructuredLoader

            # strategy templates for unstructued
            self.unstructured_strategies = {
                "pdf": {
                    "strategy": ad.preprocessing.strategy,
                },
                "doc": {
                    "strategy": ad.preprocessing.strategy,
                    "detect_language_per_element": True,
                },
                "docx": {
                    "strategy": ad.preprocessing.strategy,
                    "detect_language_per_element": True,
                },
                "ppt": {
                    "strategy": ad.preprocessing.strategy,
                    "detect_language_per_element": True,
                },
                "pptx": {
                    "strategy": ad.preprocessing.strategy,
                    "detect_language_per_element": True,
                },
            }

        self.sorted_data = sorted_data
        self.processed_docs = []
        self.indexpath = self.app_repository / "index"
        self.dbpath = self.indexpath
        if self.rag_indexdb_lib == "chromadb":
            self.dbpath = self.indexpath / "chroma_db"
        self.bm25path = self.indexpath / "bm25.pkl"

        if not os.path.isdir(self.indexpath):
            Path(self.indexpath).mkdir(parents=True, exist_ok=True)
            self.logger.debug("created app index dir %s", self.indexpath)

        if self.rag and os.path.isfile(self.dbpath / "chroma.sqlite3") and not ad.rag.reindex:
            # skip preprocessing if rag is activated, index exists and no forcing is requested
            pass
        else:
            for fext in self.sorted_data.keys():
                strat = None
                if fext in self.preprocessing_funcs:
                    if fext in self.unstructured_strategies:
                        strat = self.unstructured_strategies[fext]
                        if fext == "pdf" and strat["strategy"] == "fast":
                            self.preprocessing_funcs[fext] = (
                                PyMuPDFLoader  # override the default unstructured parser (pdfminer)
                            )
                            strat = None
                    preproc_func = self.preprocessing_funcs[fext]
                else:
                    preproc_func = self.others_preprocessing_func

                # parallelized preprocessing
                max_workers = os.cpu_count()
                if max_workers is None:
                    max_workers = 1
                    self.logger.warning(f"could not determine number of CPUs, max_workers set to {max_workers}")

                if fext in ["doc", "ppt", "xls"]:
                    max_workers = 1
                    self.logger.warning(f"setting max_workers to 1 for {fext} files")

                if fext in ["pdf"] and strat is not None and strat["strategy"] != "fast":
                    # unstructured will internally use all cores for non-fast strategy so limit to 1
                    max_workers = 1
                    self.logger.warning(f"setting max_workers to {max_workers} for {fext} files with non-fast strategy")

                self.logger.info("[%s] preprocessing started using %s threads", fext, max_workers)

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []

                    # execute in parallel
                    for doc in self.sorted_data[fext]:
                        futures.append(executor.submit(self.preprocess_doc, preproc_func, doc, strat))

                    # wait for all futures to complete
                    for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc="Processing documents",
                    ):
                        future.result()  # collect potential exceptions

        # save pre processed docs if requested
        if ad.preprocessing.save_output:
            self.save_preprocessed_output(self.processed_docs, self.app_repository / "preprocessed_output")
            self.logger.info(f"preprocessed output saved in {self.app_repository / 'preprocessed_output'}")
        self.logger.info(f"preprocessing completed [{len(self.processed_docs)} files]")

        docs_list = [item for sublist in self.processed_docs for item in sublist]

        if self.rag:
            if self.text_splitter:
                self.logger.info("splitting documents")
                doc_splits = self.text_splitter.split_documents(docs_list)
                self.logger.info(f"splitting completed [{len(doc_splits)} documents]")
            else:
                doc_splits = docs_list

            # filter out complex metadata
            filtered_docs = filter_complex_metadata(doc_splits)

            if self.rag_retriever is None:
                # check whether db exists
                if (
                    (self.rag_indexdb_obj == Chroma and os.path.isfile(self.dbpath / "chroma.sqlite3"))
                    or os.path.isfile(self.indexpath / "chroma.sqlite3")
                ) and not ad.rag.reindex:
                    self.logger.info("existing index found at " + str(self.dbpath))
                    if self.rag_indexdb_obj == Chroma:
                        self.rag_indexdb = self.rag_indexdb_obj(
                            persist_directory=str(self.dbpath),
                            embedding_function=self.rag_embedding,
                            collection_name="mydb",
                            client_settings=self.chroma_client_settings,
                        )
                    else:
                        self.rag_indexdb = self.rag_indexdb_obj(
                            persist_directory=str(self.indexpath),
                            embedding_model_path=str(self.app_repository) + "/models",
                            collection_name="mydb",
                            embedding_model=self.rag_embedding_model,
                            embedding_lib=self.rag_embedding_lib,
                            logger=self.logger,
                            recreate=False,
                            gpu_id=self.rag_gpu_id,
                            num_partitions=self.rag_num_partitions,
                        )
                    self.logger.info("existing index has successfully loaded")
                # otherwise index from scratch
                else:
                    if ad.rag.reindex:
                        if os.path.isfile(self.dbpath / "chroma.sqlite3") and ad.rag.index_protection:
                            msg = """Index already exists and is protected.
                            If you want to reindex, set reindex to True along
                            with index_protection to False"""
                            self.logger.error(msg)
                            raise InputConnectorBadParamException(msg)
                        # remove dbpath file
                        if os.path.exists(self.dbpath):
                            shutil.rmtree(self.dbpath)
                        # remove bm25 file
                        if os.path.exists(self.indexpath):
                            shutil.rmtree(self.indexpath)

                    # XXX: chromadb has a max size set on the number of
                    # chunks that can be indexed at once, around 41,666
                    # so be split the list of chunks and add them
                    # sequentially
                    split_docs_chunked = split_list(filtered_docs, 41000)

                    for split_docs_chunk in tqdm(split_docs_chunked, desc="Indexing documents"):
                        if self.rag_indexdb_obj == Chroma:
                            self.rag_indexdb = self.rag_indexdb_obj.from_documents(
                                documents=split_docs_chunk,
                                collection_name="mydb",
                                embedding=self.rag_embedding,
                                persist_directory=str(self.dbpath),
                                client_settings=self.chroma_client_settings,
                            )
                        else:
                            self.rag_indexdb = self.rag_indexdb_obj.from_documents(
                                documents=split_docs_chunk,
                                collection_name="mydb",
                                persist_directory=str(self.indexpath),
                                embedding_model_path=str(self.app_repository) + "/models",
                                embedding_lib=self.rag_embedding_lib,
                                embedding_model=self.rag_embedding_model,
                                logger=self.logger,
                                gpu_id=self.rag_gpu_id,
                                num_partitions=self.rag_num_partitions,
                            )
                    self.logger.info("New index has completed")
                if self.rag_indexdb_obj == Chroma:
                    self.logger.info("embedding index has " + str(self.rag_indexdb._collection.count()) + " elements")
                else:
                    self.logger.info("embedding index has " + str(self.rag_indexdb.count()) + " elements")

                if self.rag_search:

                    def clean_txt(txt):
                        ctxt = (
                            txt.replace("{", "")
                            .replace("}", "")
                            .replace('"', "")
                            .replace("\n", "")
                            .replace("\\", "")
                            .replace("_", "")
                        )
                        ctxt = word_tokenize(ctxt)
                        return ctxt

                    if doc_splits:
                        self.logger.info("indexing for bm25 search")
                        self.bm25_retriever = BM25Retriever.from_documents(
                            documents=doc_splits,
                            preprocess_func=clean_txt,
                        )
                        with open(self.indexpath / "docs.pkl", "wb") as df:
                            pickle.dump(self.bm25_retriever.docs, df)
                        with open(self.bm25path, "wb") as indf:
                            pickle.dump(self.bm25_retriever.vectorizer, indf)
                            self.logger.info("sucessfully persisted bm25 index %s", self.bm25path)
                    else:
                        # try loading index from disk
                        if os.path.isfile(self.bm25path):
                            with open(self.indexpath / "docs.pkl", "rb") as df:
                                bmdocs = pickle.load(df)
                            with open(self.bm25path, "rb") as indf:
                                vectorizer = pickle.load(indf)
                            self.bm25_retriever = BM25Retriever(vectorizer=vectorizer, docs=bmdocs)
                            self.logger.info(
                                "successfully loaded persisted bm25 index %s",
                                self.bm25path,
                            )
                        else:
                            self.logger.warning(
                                "bm25 has no persistence yet and cannot be setup from previously created apps"
                            )
                            self.bm25_retriever = None

                if not self.bm25_retriever:
                    self.rag_retriever = self.rag_indexdb.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4 * self.rag_top_k},
                    )  # , 'lambda_mult': 0.5})
                else:
                    self.bm25_retriever.k = self.rag_top_k  # number of documents to retrieve
                    self.rag_retriever = EnsembleRetriever(
                        retrievers=[
                            self.bm25_retriever,
                            self.rag_indexdb.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": self.rag_top_k},
                            ),
                        ],
                        weights=[0.5, 0.5],
                    )
        return self.rag_indexdb

    def retrieve(self, rag_question):
        res = self.rag_retriever.invoke(rag_question)
        return res

    def save_preprocessed_output(self, pdocs: list, outdir: Path):
        # output dir
        outdir.mkdir(parents=True, exist_ok=True)

        # save preprocessed output
        self.logger.info("save_preprocessed_output: saving %s files in %s", len(pdocs), outdir)
        for d in pdocs:
            doc_content = ""
            for p in d:
                doc_name = os.path.basename(p.metadata["source"])
                # page = p.metadata.get('page','')
                doc_content += p.page_content
            fpath = outdir / f"{doc_name}.txt"
            fpath.write_text(doc_content)
