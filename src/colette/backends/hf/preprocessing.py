import logging
import multiprocessing
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import sleep, time
from uuid import NAMESPACE_URL, uuid5

import psutil
import pypandoc
import pypdfium2 as pdfium
from PIL import Image

# from .kvstore import ImageStorageFactory
from .layout_detector import LayoutDetector
from .word_detector import WordDetector


def _is_soffice_running():
    for proc in psutil.process_iter():
        try:
            if "soffice" in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def get_optimal_thread_count(task_type="cpu"):
    """
    Return an optimal thread count based on the task type.
    For CPU-bound tasks, returns number of CPU cores.
    For I/O-bound tasks, returns a higher number (e.g., 2-4 times the CPU cores).
    """
    num_cores = os.cpu_count() or multiprocessing.cpu_count() or 1

    if task_type == "cpu":
        return num_cores
    elif task_type == "io":
        # This is a rough heuristic; adjust factor as needed
        return num_cores * 4
    else:
        raise ValueError("Unknown task type. Use 'cpu' or 'io'.")


class DocumentProcessor:
    """
    class for converting documents to images
    Supports MS Office documents, HTML documents, PDFs, and images.
    """

    MS_EXTENSIONS = {"doc", "docx", "ppt", "pptx", "xls", "xlsx"}
    IM_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff"}
    HTML_EXTENSIONS = {"html", "htm"}

    def __init__(
        self, app_repository: Path, logger: logging.Logger, dpi: int, timeout: int = 60
    ):
        ###
        # This class is responsible for converting documents to images.
        # It uses the pdf2image library to convert PDFs to images.
        # It uses soffice to convert MS documents to PDFs.
        # It uses pandoc to convert HTML documents to PDFs.
        #
        # {"uuid5": "uuid5", "source": "source", "path": "path"}
        #
        # uuid5: is a unique identifier for the document.
        # source: is the path to the original document.
        # path: is the path to the image file
        #
        # Args:
        #     app_repository (Path): Path to the application repository.
        #     logger (logging.Logger): Logger object.
        #     dpi (int): DPI for the images.
        #     timeout (int): Timeout for the conversion process.
        #
        ###
        self.app_repository = app_repository
        self.logger = logger
        self.dpi = dpi
        self.timeout = timeout
        self.pdf_output_dir = self.app_repository / "pdfs"
        self.pdf_output_dir.mkdir(parents=True, exist_ok=True)

    def transform_ms_document_to_pdf(self, doc: Path, pdf_name: Path) -> int:
        """
        Convert MS documents to PDFs using soffice.

        Args:
            doc (Path): Path to the MS document.
            pdf_name (Path): Path to the output PDF.

        Returns:
            int: 0 if successful, 1 if failed.
        """
        self.logger.info(f"Converting {str(doc)} to {str(pdf_name)}.")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf_path = Path(temp_dir) / f"{doc.stem}.pdf"

            command = [
                "soffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(temp_dir),
                str(doc),
            ]

            try:
                wait_time = 0
                sleep_time = 0.5

                if _is_soffice_running():
                    self.logger.info(
                        "An instance of soffice is already running. Wating for it to finish."
                    )
                    while _is_soffice_running() and wait_time < self.timeout:
                        sleep(sleep_time)
                        wait_time += sleep_time

                    if wait_time >= self.timeout:
                        self.logger.debug(
                            f"Timeout reached for {str(doc)}. Skipping to next file."
                        )
                        return 1

                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

                while wait_time < self.timeout:
                    retcode = process.poll()
                    if retcode is not None:
                        stdout, stderr = process.communicate()
                        _out = stdout.decode().strip()
                        if _out:
                            self.logger.info(f"Conversion ouput: {_out}")
                        _err = stderr.decode().strip()
                        if _err:
                            self.logger.debug(f"Conversion error (if any): {_err}")
                        break

                    sleep(sleep_time)
                    wait_time += sleep_time

                if wait_time >= self.timeout:
                    self.logger.debug(
                        f"Timeout reached for {str(doc)}. Terminating the process."
                    )
                    process.terminate()
                    process.wait()
                    return 1

                if temp_pdf_path.exists():
                    shutil.copy(temp_pdf_path, pdf_name)
                    self.logger.info(
                        f"Successfully converted {str(doc)} to {str(pdf_name)}.\n"
                    )
                    return 0
                else:
                    self.logger.error(
                        f"Conversion failed for {str(doc)}. No output file generated.\n"
                    )
                    return 1

            except FileNotFoundError as e:
                self.logger.error(
                    f"An error occurred while converting {str(doc)} to PDF: {str(e)}"
                )
                raise FileNotFoundError(
                    """soffice command was not found. Please install libreoffice
                    on your system and try again.

                    - Install instructions: https://www.libreoffice.org/get-help/install-howto/
                    - Mac: https://formulae.brew.sh/cask/libreoffice
                    - Debian: https://wiki.debian.org/LibreOffice"""
                ) from e

            except Exception as e:
                self.logger.error(f"An unexpected error occurred: {e}")
                return 1

        return 1

    def transform_html_to_pdf(self, doc: Path, pdf_name: Path) -> int:
        """
        Convert HTML documents to PDFs using pandoc.

        Args:
            doc (Path): Path to the HTML document.
            pdf_name (Path): Path to the output PDF.

        Returns:
            int: 0 if successful, 1 if failed.
        """
        try:
            self.logger.info(f"Converting {str(doc)} to {str(pdf_name)}.")
            _ = pypandoc.convert_file(
                str(doc),
                "pdf",
                outputfile=str(pdf_name),
                extra_args=["--pdf-engine=lualatex"],
            )  # XXX: output is "" according to pypandoc docs
        except Exception as e:
            self.logger.error(
                f"An error occurred while converting {str(doc)} to {str(pdf_name)}: {str(e)}"
            )
            return 1
        return 0

    def transform_pdf_to_images(
        self,
        pdf_name: Path,
        doc: dict,
    ):
        self.logger.info(f"Converting {pdf_name} to images.")
        scale = self.dpi / 72.0 # cf https://pypdfium2.readthedocs.io/en/v4/python_api.html#user-unit
        pdf = None
        try:
            # pages = convert_from_path(
            #     pdf_name,
            #     dpi=self.dpi,
            #     fmt="png",
            #     paths_only=True,
            #     thread_count=get_optimal_thread_count("io"),
            # )
            # doc["images"].extend(pages)
            pdf = pdfium.PdfDocument(str(pdf_name))

            # print(pdf.get_metadata_dict())

            for page_index in range(len(pdf)):
                page = pdf.get_page(page_index)
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                doc["images"].append(
                    dict(
                        image=pil_image,
                        metadata=dict(
                            page_number=page_index + 1,
                            source=str(doc["source"])
                        )
                    )
                )
                page.close()

            self.logger.info(f"Converted PDF '{pdf_name}' to {len(pdf)} images.")
        except Exception as e:
            self.logger.error(
                f"An error occurred while converting '{pdf_name}' to images: {e}"
            )
        finally:
            if pdf:
                pdf.close()

    def transform_documents_to_images(self, list_of_documents: list[dict]) -> int:
        """
        Transforms a list of documents to images.
        Args:
            list_of_documents (list[dict]): List of documents.
            {"source": "source", "ext": "ext"}

        Returns:
            the modified list of documents: list[dict]
            {"source": "source", "ext": "ext", "images": [], "uuid5": "uuid5"}
        """
        dt_1 = dt_2 = 0.0

        for doc in list_of_documents:
            doc.update(
                dict(images=[], uuid5=str(uuid5(NAMESPACE_URL, str(doc["source"]))))
            )

            if doc["ext"] in self.IM_EXTENSIONS:
                doc["images"].append(
                    dict(
                        image=Image.open(doc["source"]),
                        metadata=dict(
                            source=str(doc["source"])
                        )
                    )
                )
                continue
            else:
                pdf_name = self.pdf_output_dir / f"{doc['uuid5']}.pdf"
                if doc["ext"] in self.MS_EXTENSIONS:
                    t1 = time()
                    if self.transform_ms_document_to_pdf(doc["source"], pdf_name):
                        self.logger.error(f"Failed to convert {doc['source']} to PDF.")
                        continue
                    dt_1 = time() - t1
                elif doc["ext"] in self.HTML_EXTENSIONS:
                    t1 = time()
                    if self.transform_html_to_pdf(doc["source"], pdf_name):
                        self.logger.error(f"Failed to convert {doc['source']} to PDF.")
                        continue
                    dt_1 = time() - t1
                elif doc["ext"] == "pdf":
                    pdf_name = doc["source"]
                else:
                    self.logger.error(f"Unsupported file type: {doc['source']}")
                    continue

                assert pdf_name.exists(), f"PDF file {pdf_name} does not exist."

                t1 = time()
                err = self.transform_pdf_to_images(pdf_name, doc)
                if err:
                    self.logger.error(f"Failed to convert {doc['source']} to images.")
                dt_2 = time() - t1
                self.logger.info(
                    f"{doc['source']} transformed to images [{dt_1:.2f}/{dt_2:.2f}]."
                )


class ImageProcessor:
    def __init__(
        self,
        rag_layout_detector: LayoutDetector | None,
        rag_chunk_num: int,
        rag_chunk_overlap: int,
        rag_index_overview: bool,
        rag_auto_scale_for_font: bool,
        rag_min_font_size: int,
        device: int,
        rag_filter_width: int,
        rag_filter_height: int,
        logger: logging.Logger,
    ):
        self.rag_layout_detector = rag_layout_detector
        self.rag_chunk_num = rag_chunk_num
        self.rag_chunk_overlap = rag_chunk_overlap
        self.rag_index_overview = rag_index_overview
        self.rag_auto_scale_for_font = rag_auto_scale_for_font
        self.rag_min_font_size = rag_min_font_size
        self.device = device
        self.rag_filter_width = rag_filter_width
        self.rag_filter_height = rag_filter_height
        self.logger = logger

        # Load word detector as needed
        if self.rag_auto_scale_for_font:
            self.word_detector = WordDetector(
                self.rag_min_font_size, self.logger, self.device
            )

        # self.kv_store = ImageStorageFactory.create_storage("hdf5", self.app_repository / "images.hdf5")

    def chunk_image(self, img, nchunks: int = 10, overlap: int = 20):
        # Convert overlap percentage to a fraction
        overlap_fraction = overlap / 100.0

        # Get image dimensions
        width, height = img.size

        # Calculate chunk size along the height with overlap
        chunk_height = height // nchunks
        overlap_height = int(chunk_height * overlap_fraction)

        doc_chunks = []
        for i in range(nchunks):
            # Calculate the starting and ending y-coordinates of the chunk
            y_start = max(0, i * chunk_height - i * overlap_height)
            y_end = min(height, y_start + chunk_height + overlap_height)
            # Crop the image chunk
            chunk = img.crop((0, y_start, width, y_end))
            # Append the chunk and its path to the list
            doc_chunks.append((chunk,))

        return doc_chunks

    def crop_image(self, img):
        # the detector returns the crop, a class name and a confidence score as a tuple
        crops = self.rag_layout_detector.detect(img)

        doc_crops = []
        for c in crops:
            doc_crops.append([c[0].convert("RGB"), c[1]])

        return doc_crops

    def preprocess_images(self, list_of_documents: list[dict]):
        for doc in list_of_documents:
            parts = []
            for img_idx, img_dict in enumerate(doc["images"]):
                # generate a unique basename for the image
                basename = f"{doc['uuid5']}_{img_idx:04d}"
                metadata = img_dict["metadata"]

                if self.rag_layout_detector is not None:
                    # apply detector and store crops in the document
                    crops = self.crop_image(img_dict["image"])
                    for c_idx, c in enumerate(crops):
                        if self.rag_auto_scale_for_font:
                            c[0] = self.word_detector.detect_and_resize(c[0])
                        # metadata is a dictionary containing information about the image
                        # it is initialized with the metadata from the original image

                        metadata_ = metadata.copy()
                        metadata_.update(
                            dict(
                                crop=c_idx,
                                label="crop",
                                crop_label=c[1]
                            )
                        )
                        parts.append(
                            dict(
                                name=f"{basename}_crop_{c_idx:04d}",
                                img=c[0],
                                metadata=metadata_
                            )
                        )
                    # add full view of the doc as overview
                    # avoid blank pages by relying on detected layout

                    if len(crops) > 1 and self.rag_index_overview:
                        img = img_dict["image"]
                        if self.rag_auto_scale_for_font:
                            img = self.word_detector.detect_and_resize(img)
                        metadata_ = metadata.copy()
                        metadata_.update(
                            dict(
                                label="overview"
                            )
                        )
                        parts.append(
                            dict(
                                name=basename,
                                img=img_dict["image"],
                                metadata=metadata_
                            )
                        )
                elif self.rag_chunk_num <= 1:
                    img = img_dict["image"]
                    if self.rag_auto_scale_for_font:
                        img = self.word_detector.detect_and_resize(img)
                    metadata_ = metadata.copy()
                    metadata_.update(
                        dict(
                            label="overview"
                        )
                    )
                    parts.append(
                        dict(
                            name=basename,
                            img=img,
                            metadata=metadata_
                        )
                    )
                else:
                    chunks = self.chunk_image(
                        img_dict["image"],
                        self.rag_chunk_num,
                        self.rag_chunk_overlap,
                    )
                    for c_idx, c in enumerate(chunks):
                        if self.rag_auto_scale_for_font:
                            c[0] = self.word_detector.detect_and_resize(c[0])
                        metadata_ = metadata.copy()
                        metadata_.update(
                            dict(
                                chunk=c_idx,
                                label="chunk"
                            )
                        )
                        parts.append(
                            dict(
                                name=f"{basename}_chunk_{c_idx:04d}",
                                img=c[0],
                                metadata=metadata_
                            )
                        )

                    if self.rag_index_overview:
                        img = img_dict["image"]
                        if self.rag_auto_scale_for_font:
                            img = self.word_detector.detect_and_resize(img)
                        metadata_ = metadata.copy()
                        metadata_.update(
                            dict(
                                label="overview"
                            )
                        )
                        parts.append(
                            dict(
                                name=f"{basename}_overview",
                                img=img,
                                metadata=metadata_
                            )
                        )
            # Filter parts
            doc["parts"] = []
            for p in parts:
                width, height = p["img"].size
                if width > self.rag_filter_width and height > self.rag_filter_height:
                    doc["parts"].append(p)
            if len(doc["parts"]) > 0:
                self.logger.info(f"\tFiltered out {len(parts) - len(doc['parts'])} parts")
