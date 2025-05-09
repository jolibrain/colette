import os
import time

import torch.multiprocessing as mp

from .indexing.collection_indexer import encode
from .infra.launcher import Launcher
from .utils.utils import create_directory, print_message


class Indexer:
    def __init__(
        self,
        embedder,
        verbose: int = 3,
        logger=None,
        gpu_id=None,
        persist_directory=None,
        typical_doclen=150,
        num_partitions=-1,
    ):
        """
        Use Run().context() to choose the run's configuration. They are NOT extracted from `config`.
        """

        self.gpu_id = gpu_id
        self.index_path = None
        self.verbose = verbose
        self.embedder = embedder
        self.config = self.embedder.config  # TODO need some cleanup
        self.config.root = persist_directory
        self.config.avoid_fork_if_possible = True
        self.config.nranks = 1
        self.config.doc_maxlen = 512
        self.typical_doclen = typical_doclen
        self.num_partitions = num_partitions

        self.logger = logger

    def get_index(self):
        return self.index_path

    def erase(self, force_silent: bool = False):
        assert self.index_path is not None
        directory = self.index_path
        deleted = []

        for filename in sorted(os.listdir(directory)):
            filename = os.path.join(directory, filename)

            delete = filename.endswith(".json")
            delete = delete and (
                "metadata" in filename or "doclen" in filename or "plan" in filename
            )
            delete = delete or filename.endswith(".pt")

            if delete:
                deleted.append(filename)

        if len(deleted):
            if not force_silent:
                print_message(
                    f"#> Will delete {len(deleted)} files already at {directory} in 2 seconds..."
                )
                time.sleep(2)

            for filename in deleted:
                os.remove(filename)

        return deleted

    def index(self, name, collection, dbdir, overwrite=False):
        assert overwrite in [True, False, "reuse", "resume", "force_silent_overwrite"]

        self.embedder.configure(
            collection=collection, index_name=name, resume=overwrite == "resume"
        )
        # Note: The bsize value set here is ignored internally. Users are encouraged
        # to supply their own batch size for indexing by using the index_bsize parameter in the ColBERTConfig.
        # self.configure(bsize=64, partitions=None)
        self.config.index_path = dbdir + "/" + name
        self.index_path = self.config.index_path_
        index_does_not_exist = not os.path.exists(self.config.index_path_)

        assert (
            overwrite in [True, "reuse", "resume", "force_silent_overwrite"]
        ) or index_does_not_exist, self.config.index_path_
        create_directory(self.config.index_path_)

        if overwrite == "force_silent_overwrite":
            self.erase(force_silent=True)
        elif overwrite is True:
            self.erase()

        if index_does_not_exist or overwrite != "reuse":
            self.__launch(collection)

        return self.config.index_path

    def __launch(self, collection):
        launcher = Launcher(encode)
        launcher.run_config.root = self.config.root
        if self.config.nranks == 1 and self.config.avoid_fork_if_possible:
            self.logger.info("starting indexing / Single Thread")
            shared_queues = []
            shared_lists = []
            launcher.launch_without_fork(
                self.config,
                self.embedder,
                collection,
                shared_lists,
                shared_queues,
                self.verbose,
                self.logger,
                self.gpu_id,
                self.typical_doclen,
                self.num_partitions,
            )
            self.logger.info("end indexing / Single Thread")

            return

        manager = mp.Manager()
        shared_lists = [manager.list() for _ in range(self.config.nranks)]
        shared_queues = [manager.Queue(maxsize=1) for _ in range(self.config.nranks)]

        # Encodes collection into index using the CollectionIndexer class
        self.logger.info("starting indexing / Multi Thread")
        launcher.launch(
            self.config,
            self.embedder,
            collection,
            shared_lists,
            shared_queues,
            self.verbose,
            self.logger,
            self.gpu_id,
            self.typical_doclen,
            self.num_partitions,
        )
        self.logger.info("end indexing / Multi Thread")
