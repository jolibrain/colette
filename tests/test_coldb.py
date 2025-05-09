import os
import sys
import PIL

col_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src")
sys.path.append(col_dir)

from backends.coldb import ColDB  # noqa
from logger import get_colette_logger  # noqa

NUM_CHUNKS = 1
CHUNK_OVERLAP = 0


def chunk_image(img, nchunks=10, overlap=20.0, output_dir="output_dir"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert overlap percentage to a fraction
    overlap_fraction = overlap / 100.0

    # Get image dimensions
    width, height = img.size

    # Calculate chunk size along the height with overlap
    chunk_height = height // nchunks
    overlap_height = int(chunk_height * overlap_fraction)

    # Get the base name of the original image file (without extension)
    img_basename = os.path.splitext(os.path.basename(img.filename))[0]

    chunks = []
    for i in range(nchunks):
        # Calculate the starting and ending y-coordinates of the chunk
        y_start = max(0, i * chunk_height - i * overlap_height)
        y_end = min(height, y_start + chunk_height + overlap_height)

        # Crop the image chunk
        chunk = img.crop((0, y_start, width, y_end))

        # Save chunk to output directory
        chunk_path = os.path.join(output_dir, f"{img_basename}_chunk_{i}.jpg")
        chunk.save(chunk_path)

        # Append the chunk and its path to the list
        chunks.append((chunk, chunk_path))

    return chunks


def preprocess_image(imgpath):
    docimg = PIL.Image.open(imgpath)
    if NUM_CHUNKS <= 1:
        return [{"imgpath": imgpath, "doc": docimg}]
    else:
        chunks = chunk_image(
            docimg,
            NUM_CHUNKS,
            CHUNK_OVERLAP,
            "./test_coldbimg/chunks",
        )
        chunks.append((docimg, imgpath))  # keep a full view of the doc
        docs = []
        for c in chunks:
            docs.append({"imgpath": c[1], "doc": c[0]})
        return docs


coldb = ColDB(
    persist_directory="./test_coldbimg",
    collection_name="colpali_test",
    embedding_model="vidore/colpali-v1.2-hf",
    embedding_lib="huggingface",
    embedding_model_path="./test_coldbimg",
    logger=get_colette_logger("test_coldb_img"),
)

# coldb = ColDB(
#     persist_directory="./test_coldbimg",
#     collection_name="colsmol_test",
#     embedding_model="vidore/colSmol-256M",
#     embedding_lib="huggingface",
#     embedding_model_path="./test_coldbimg",
#     logger=get_colette_logger("test_coldb_img"),
#     num_partitions=5,
#     gpu_id=3,
# )


docs = preprocess_image("./RINFANR5L16B2040.jpg-001.jpg")
coldb.add_imgs([str(doc["imgpath"]) for doc in docs], "colqwen_test")

retriever = coldb.as_retriever()

ret = retriever.invoke("une reponse stp")
print(ret)
