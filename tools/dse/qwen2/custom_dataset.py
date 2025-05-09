from torch.utils.data import Dataset
import json
import random
import PIL


def format_query(query: str, prefix: str = "") -> str:
    return f"{prefix} {query.strip()}".strip()


class CustomDataset(Dataset):
    def __init__(self, datafile, imagepath, nneg):
        self.datafile = datafile
        self.imagepath = imagepath
        self.nneg = nneg
        with open(self.datafile, "r") as jf:
            data = json.load(jf)
        self.queries = data["queries"]
        self.imgs = data["images"]
        self.nimgs = len(self.imgs)
        self.no_shuffle = False

    def _getimage(self, imgid):
        return PIL.Image.open(self.imagepath + "/" + self.imgs[imgid])

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, item):
        query = self.queries[item]["query"]
        formated_query = format_query(query, "")
        formated_passages = []
        if self.no_shuffle:
            pos_psg = self.queries[item]["docids"][0]
        else:
            pos_psg = random.choice(self.queries[item]["docids"])

        formated_passages.append(self._getimage(pos_psg))

        negs = set(range(self.nimgs))
        pos = set(self.queries[item]["docids"])
        negs = negs - pos
        random_negs = random.choices(list(negs), k=self.nneg)
        for i in random_negs:
            formated_passages.append(self._getimage(i))

        return formated_query, formated_passages
