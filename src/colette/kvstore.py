import io
import threading
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from PIL import Image


class ImageStorageInterface(ABC):
    """Interface for image storage backends."""

    @abstractmethod
    def store_image(self, key: str, image: Image.Image) -> None:
        """Store a single image."""
        pass

    @abstractmethod
    def retrieve_image(self, key: str) -> Image.Image:
        """Retrieve a single image."""
        pass

    @abstractmethod
    def has_key(self, key: str) -> bool:
        """Check if a key exists in the storage."""
        pass

    @abstractmethod
    def iter_keys(self) -> iter:
        """Iterate over all keys in the storage."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the storage."""
        pass


class HDF5ImageStorage(ImageStorageInterface):
    """HDF5-based image storage implementation."""

    def __init__(self, file_path: Path, mode: Literal["r", "w", "a"] = "a"):
        self.file_path = file_path
        self.lock = threading.Lock()
        self.hdf5_file = h5py.File(file_path, mode)  # Open in append mode
        self.namespace = uuid.UUID("12344321-1234-4321-1234-456789987654")

    def lock_(self):
        self.lock.acquire()

    def unlock_(self):
        self.lock.release()

    def generate_uuid(self, key: str) -> str:
        """Generate a UUID for the given key."""
        return str(uuid.uuid5(self.namespace, key))

    def store_image(self, key: str, image: Image.Image) -> None:
        """Store an image in HDF5."""
        hashed_key = self.generate_uuid(key)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self.lock_()
        if hashed_key in self.hdf5_file:
            del self.hdf5_file[hashed_key]
        dataset = self.hdf5_file.create_dataset(hashed_key, data=np.void(buffer.getvalue()))
        self.unlock_()
        dataset.attrs["key"] = key

    def retrieve_image(self, key: str) -> Image.Image:
        """Retrieve an image from HDF5."""
        hashed_key = self.generate_uuid(key)
        self.lock_()
        if hashed_key not in self.hdf5_file:
            raise KeyError(f"Key '{key}' not found in HDF5 file.")
        binary_data = self.hdf5_file[hashed_key][()]
        self.unlock_()
        buffer = io.BytesIO(binary_data) if isinstance(binary_data, np.void) else binary_data
        return Image.open(buffer)

    def has_key(self, key: str) -> bool:
        """Check if a key exists in the HDF5 file."""
        hashed_key = self.escape_key(key)
        return hashed_key in self.hdf5_file

    def iter_keys(self) -> iter:
        """Iterate over all keys in the HDF5 file."""
        for hashed_key in self.hdf5_file.keys():
            yield self.hdf5_file[hashed_key].attrs["key"]

    def close(self) -> None:
        """Close the HDF5 file."""
        self.hdf5_file.close()


# class LMDBImageStorage(ImageStorageInterface):
#     """LMDB-based image storage implementation."""

#     def __init__(self, file_path: Path, mode: Literal["r", "w"], map_size: int = 10**9):
#         self.file_path = file_path
#         self.env = lmdb.open(str(file_path), mode, map_size=map_size)

#     def _encode_key(self, key: str) -> bytes:
#         """Helper method to encode keys to bytes."""
#         return key.encode("utf-8")

#     def store_image(self, key: str, image: Image.Image) -> None:
#         """Store an image in LMDB."""
#         encoded_key = self._encode_key(key)
#         with self.env.begin(write=True) as txn:
#             buffer = io.BytesIO()
#             image.save(buffer, format="JPEG")
#             txn.put(encoded_key, buffer.getvalue())

#     def retrieve_image(self, key: str) -> Image.Image:
#         """Retrieve an image from LMDB."""
#         encoded_key = self._encode_key(key)
#         with self.env.begin() as txn:
#             value = txn.get(encoded_key)
#             if value is None:
#                 raise KeyError(f"Key '{key}' not found in LMDB.")
#             buffer = io.BytesIO(value)
#             return Image.open(buffer)

#     def has_key(self, key: str) -> bool:
#         """Check if a key exists in the LMDB file."""
#         encoded_key = self._encode_key(key)
#         with self.env.begin() as txn:
#             return txn.get(encoded_key) is not None

#     def iter_keys(self) -> iter:
#         """Iterate over all keys in the LMDB file."""
#         with self.env.begin() as txn:
#             cursor = txn.cursor()
#             for encoded_key, _ in cursor:
#                 yield encoded_key.decode("utf-8")

#     def close(self) -> None:
#         """Close the LMDB environment."""
#         self.env.close()


class ImageStorageFactory:
    """Factory for creating image storage backends."""

    @staticmethod
    def create_storage(
        backend: str,
        file_path: Path,
        mode: Literal["a", "r", "w"] = "r",  # read only, file must exist
        **kwargs,
    ) -> ImageStorageInterface:
        """
        Create an instance of the desired storage backend.

        Args:
            backend (str): The storage backend to use ('hdf5' or 'lmdb').
            file_path (Path): Path to the storage file.
            kwargs: Additional arguments for backend initialization.

        Returns:
            ImageStorageInterface: An instance of the selected backend.
        """
        if backend == "hdf5":
            return HDF5ImageStorage(file_path, mode)
        # elif backend == "lmdb":
        #     return LMDBImageStorage(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
