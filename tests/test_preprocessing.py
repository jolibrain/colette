import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

col_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src")
sys.path.append(col_dir)

from backends.hf.layout_detector import LayoutDetector # noqa
from backends.hf.preprocessing import DocumentProcessor, ImageProcessor # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def get_deep_size(obj, seen=None):
    """
    Recursively finds the memory footprint of a Python object.
    Avoids double-counting objects.
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum((get_deep_size(v, seen) for v in obj.values()))
        size += sum((get_deep_size(k, seen) for k in obj.keys()))
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum((get_deep_size(i, seen) for i in obj))
    # Add more types if needed (e.g., custom classes)

    # transform bytes to MB
    return size / 1024 / 1024


def test_docx():
    # docx -- 2.03 sec --> pdf -- 76.25 sec --> 120 images 
    # docx -- 2.52 sec --> pdf -- 76.43 sec --> 120 images (using ~56.83 MB)
    list_of_files = [dict(source=Path("tests/data/RAPPANR5L16B1991.docx"), ext="docx")]

    with tempfile.TemporaryDirectory() as tmpdir:
        dp = DocumentProcessor(Path(tmpdir), logger, dpi=300)

        dp.transform_documents_to_images(list_of_files)

        total_images = sum([len(doc["images"]) for doc in list_of_files])

        assert total_images == 120, f"Expected 120 images, got {total_images}"

        try:
            from pympler import asizeof
            print(f"Total size: {asizeof.asizeof(list_of_files)}")
        except ImportError:
            pass        

def test_html():
    # html -- 8.75 sec --> pdf -- 63.25 sec --> 121 images | using ~48.6 MB
    # html -- 8.69 sec --> pdf -- 63.15 sec --> 121 images
    list_of_files = [dict(source=Path("tests/data/RAPPANR5L16B1991.html"), ext="html")]

    with tempfile.TemporaryDirectory() as tmpdir:
        dp = DocumentProcessor(Path(tmpdir), logger, dpi=300)

        dp.transform_documents_to_images(list_of_files)

        total_images = sum([len(doc["images"]) for doc in list_of_files])

        assert total_images == 121, f"Expected 121 images, got {total_images}"

        try:
            from pympler import asizeof
            print(f"Total size: {asizeof.asizeof(list_of_files)}")
        except ImportError:
            pass    

def test_pdf():
    # pdf -- 63.01 sec --> 118 images | using ~56.28 MB
    # pdf -- 63.15 sec --> 118 images
    list_of_files = [dict(source=Path("tests/data/RINFANR5L16B2040.pdf"), ext="pdf")]

    with tempfile.TemporaryDirectory() as tmpdir:
        dp = DocumentProcessor(Path(tmpdir), logger, dpi=300)

        dp.transform_documents_to_images(list_of_files)

        total_images = sum([len(doc["images"]) for doc in list_of_files])

        assert total_images == 118, f"Expected 118 images, got {total_images}"

        try:
            from pympler import asizeof
            print(f"Total size: {asizeof.asizeof(list_of_files)}")
        except ImportError:
            pass    

def test_jpg():
    list_of_files = [
        dict(source=Path("tests/data_img2/RINFANR5L16B2040.jpg-001.jpg"), ext="jpg"),
        dict(source=Path("tests/data_img2/RINFANR5L16B2040.jpg-016.jpg"), ext="jpg")
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        dp = DocumentProcessor(Path(tmpdir), logger, dpi=300)

        dp.transform_documents_to_images(list_of_files)

        total_images = sum([len(doc["images"]) for doc in list_of_files])

        assert total_images == 2, f"Expected 2 images, got {total_images}"

        try:
            from pympler import asizeof
            print(f"Total size: {asizeof.asizeof(list_of_files)}")
        except ImportError:
            pass    

def test_all():
    # size of list_of_files: 166756584 = 166.75 MB
    list_of_files = [
        dict(source=Path("tests/data/RAPPANR5L16B1991.docx"), ext="docx"),
        dict(source=Path("tests/data/RAPPANR5L16B1991.html"), ext="html"),
        dict(source=Path("tests/data/RINFANR5L16B2040.pdf"), ext="pdf"),
    ]

    # extensions = set(f.suffix.strip(".") for f in Path("tests/data").rglob("*.*"))
    # for ext in extensions:
    #     list_of_files[ext] = [f for f in Path("tests/data").rglob(f"*{ext}")]

    with tempfile.TemporaryDirectory() as tmpdir:
        dp = DocumentProcessor(Path(tmpdir), logger, dpi=300)

        dp.transform_documents_to_images(list_of_files)

        total_images = sum([len(doc["images"]) for doc in list_of_files])

        assert total_images == 359, f"Expected 359 images, got {total_images}"

        try:
            from pympler import asizeof
            print(f"Total size: {asizeof.asizeof(list_of_files)}")
        except ImportError:
            pass

def test_chunking():
    list_of_files = [
        dict(source=Path("tests/data_img2/RINFANR5L16B2040.jpg-001.jpg"), ext="jpg"),
        dict(source=Path("tests/data_img2/RINFANR5L16B2040.jpg-016.jpg"), ext="jpg")
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        dp = DocumentProcessor(Path(tmpdir), logger, dpi=300)

        dp.transform_documents_to_images(list_of_files)

        total_images = sum([len(doc["images"]) for doc in list_of_files])

        assert total_images == 2, f"Expected 2 images, got {total_images}"

        try:
            from pympler import asizeof
            print(f"Total size after image generation: {asizeof.asizeof(list_of_files)}")
        except ImportError:
            pass

        n_chunks = 5
        ip = ImageProcessor(
            None,                       # No layout detector
            n_chunks,                   # rag_chunk_num: int
            0,                          # rag_chunk_overlap: int
            False,                      # rag_index_overview: bool
            False,                      # rag_auto_scale_for_font: bool
            0,                          # rag_min_font_size: int
            0,                          # device: int
            -1,                         # rag_filter_width: int
            -1,                         # rag_filter_height: int
            logger
        )

        ip.preprocess_images(list_of_files)

        assert  sum([len(doc['parts']) for doc in list_of_files]) == len(list_of_files)*n_chunks, f"Expected {n_chunks} chunks"
        try:
            from pympler import asizeof
            print(f"Total size after chunk generation: {asizeof.asizeof(list_of_files)}")
        except ImportError:
            pass

def test_cropping():
    list_of_files = [
        dict(source=Path("tests/data_img2/RINFANR5L16B2040.jpg-001.jpg"), ext="jpg"),
        dict(source=Path("tests/data_img2/RINFANR5L16B2040.jpg-016.jpg"), ext="jpg")
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        models_repository = Path(tmpdir + "/models")
        models_repository.mkdir(parents=True, exist_ok=True)

        layout_detector = LayoutDetector(
            model_path="https://colette.chat/models/layout/layout_detector_publaynet_merged_6000.pt",
            resize_width=768,
            resize_height=1024,
            models_repository=models_repository,
            logger=logger,
            device=0,
        )

        dp = DocumentProcessor(Path(tmpdir), logger, dpi=300)

        dp.transform_documents_to_images(list_of_files)

        total_images = sum([len(doc["images"]) for doc in list_of_files])

        assert total_images == 2, f"Expected 2 images, got {total_images}"

        try:
            from pympler import asizeof
            print(f"Total size after image generation: {asizeof.asizeof(list_of_files)}")
        except ImportError:
            pass

        ip = ImageProcessor(
            layout_detector,            # No layout detector
            5,                          # rag_chunk_num: int
            0,                          # rag_chunk_overlap: int
            False,                      # rag_index_overview: bool
            False,                      # rag_auto_scale_for_font: bool
            0,                          # rag_min_font_size: int
            0,                          # device: int
            -1,                         # rag_filter_width: int
            -1,                         # rag_filter_height: int
            logger
        )

        ip.preprocess_images(list_of_files)

        assert sum([len(doc['parts']) for doc in list_of_files]) == 10, "Expected 10 crops"
        try:
            from pympler import asizeof
            print(f"Total size after chunk generation: {asizeof.asizeof(list_of_files)}")
        except ImportError:
            pass

        logger.info(list_of_files)
