import os
import sys
from pathlib import Path

import pytest

col_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src")
sys.path.append(col_dir)

from backends.coldb import ColDB  # noqa
from logger import get_colette_logger  # noqa

pytestmark = pytest.mark.integration


NUM_CHUNKS = 1
CHUNK_OVERLAP = 0


def chunk_image(img, nchunks=10, overlap=20.0, output_dir="output_dir"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    overlap_fraction = overlap / 100.0
    width, height = img.size
    chunk_height = height // nchunks
    overlap_height = int(chunk_height * overlap_fraction)
    img_basename = os.path.splitext(os.path.basename(img.filename))[0]

    chunks = []
    for i in range(nchunks):
        y_start = max(0, i * chunk_height - i * overlap_height)
        y_end = min(height, y_start + chunk_height + overlap_height)

        chunk = img.crop((0, y_start, width, y_end))
        chunk_path = os.path.join(output_dir, f"{img_basename}_chunk_{i}.jpg")
        chunk.save(chunk_path)
        chunks.append((chunk, chunk_path))

    return chunks


def preprocess_image(imgpath):
    pil_image = pytest.importorskip("PIL.Image")
    docimg = pil_image.open(imgpath)
    if NUM_CHUNKS <= 1:
        return [{"imgpath": imgpath, "doc": docimg}]

    chunks = chunk_image(docimg, NUM_CHUNKS, CHUNK_OVERLAP, "./test_coldbimg/chunks")
    chunks.append((docimg, imgpath))
    return [{"imgpath": c[1], "doc": c[0]} for c in chunks]


def test_coldb_retriever_invoke():
    image_path = Path("./RINFANR5L16B2040.jpg-001.jpg")
    if not image_path.exists():
        pytest.skip("coldb test image asset is not available")

    coldb = ColDB(
        persist_directory="./test_coldbimg",
        collection_name="colpali_test",
        embedding_model="vidore/colpali-v1.2-hf",
        embedding_lib="huggingface",
        embedding_model_path="./test_coldbimg",
        logger=get_colette_logger("test_coldb_img"),
    )

    docs = preprocess_image(str(image_path))
    coldb.add_imgs([str(doc["imgpath"]) for doc in docs], "colqwen_test")

    retriever = coldb.as_retriever()
    ret = retriever.invoke("une reponse stp")
    assert ret is not None
