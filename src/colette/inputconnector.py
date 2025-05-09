# input connector abstract class

import os
import re
from abc import abstractmethod
from glob import glob

from .apidata import InputConnectorObj


class InputConnectorBadParamException(Exception):
    pass


class InputConnectorInternalException(Exception):
    pass


class InputConnector:
    def __init__(self):
        self.data = []  ##TODO: dict of files per extension
        self.sorted_data = {}

    @abstractmethod
    def init(self, ad: InputConnectorObj):
        self.data_dirs = ad.data
        if not ad.rag:
            self.cpu = True
        else:
            self.cpu = False if ad.rag.gpu_id >= 0 else True
        pass

    @abstractmethod
    def transform(self, ad: InputConnectorObj):
        pass

    @abstractmethod
    def get_data(self, ad: InputConnectorObj):
        # list files
        filters = []
        if hasattr(ad, "preprocessing") and hasattr(ad.preprocessing, "filters"):
            filters = ad.preprocessing.filters
        filters_re = []
        self.data_dirs = ad.data

        # Load pattern for filtering files out
        if filters:
            for fil in filters:
                filters_re.append(re.compile(fil))

        # iterate data directories
        for dp in self.data_dirs:
            dir_data = glob(str(dp / "**/*"), recursive=True)
            for f in dir_data:
                # check for matching patterns
                if any(fil.search(f) for fil in filters_re):
                    continue

                # only keep files
                if not os.path.isfile(f):
                    continue

                # sort files by extension
                fext = os.path.splitext(f)[1][1:].lower()
                if fext in ad.preprocessing.files or ad.preprocessing.files == ["all"]:
                    if fext not in self.sorted_data:
                        self.sorted_data[fext] = []
                    self.sorted_data[fext].append(f)
                    self.data.append(f)
        self.logger.info("get_data: read %s files", len(self.data))
