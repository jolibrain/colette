import argparse
import io
import json
import os
import re
import shutil
import sys
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import chromadb
import click
import numpy as np
from chromadb.config import Settings
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
        try:
            import h5py
        except ImportError as e:
            raise ImportError("Please install the 'h5py' package to use HDF5 storage.") from e

        self.file_path = file_path
        self.hdf5_file = h5py.File(file_path, mode)  # Open in append mode
        self.namespace = uuid.UUID("12344321-1234-4321-1234-456789987654")

    def generate_uuid(self, key: str) -> str:
        """Generate a UUID for the given key."""
        return str(uuid.uuid5(self.namespace, key))

    def store_image(self, key: str, image: Image.Image) -> None:
        """Store an image in HDF5."""
        hashed_key = self.generate_uuid(key)
        if hashed_key in self.hdf5_file:
            del self.hdf5_file[hashed_key]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        dataset = self.hdf5_file.create_dataset(hashed_key, data=np.void(buffer.getvalue()))
        dataset.attrs["key"] = key

    def retrieve_image(self, key: str) -> Image.Image:
        """Retrieve an image from HDF5."""
        hashed_key = self.generate_uuid(key)
        if hashed_key not in self.hdf5_file:
            raise KeyError(f"Key '{key}' not found in HDF5 file.")
        binary_data = self.hdf5_file[hashed_key][()]
        buffer = io.BytesIO(binary_data) if isinstance(binary_data, np.void) else binary_data
        return Image.open(buffer)

    def has_key(self, key: str) -> bool:
        """Check if a key exists in the HDF5 file."""
        hashed_key = self.generate_uuid(key)
        return hashed_key in self.hdf5_file

    def iter_keys(self) -> iter:
        """Iterate over all keys in the HDF5 file."""
        for hashed_key in self.hdf5_file.keys():
            yield self.hdf5_file[hashed_key].attrs["key"]

    def close(self) -> None:
        """Close the HDF5 file."""
        self.hdf5_file.close()


class LMDBImageStorage(ImageStorageInterface):
    """LMDB-based image storage implementation."""

    def __init__(self, file_path: Path, mode: Literal["r", "w"], map_size: int = 10**9):
        try:
            import lmdb
        except ImportError as e:
            raise ImportError("Please install the 'lmdb' package to use LMDB storage.") from e


        self.file_path = file_path
        self.env = lmdb.open(str(file_path), mode, map_size=map_size)

    def _encode_key(self, key: str) -> bytes:
        """Helper method to encode keys to bytes."""
        return key.encode("utf-8")

    def store_image(self, key: str, image: Image.Image) -> None:
        """Store an image in LMDB."""
        encoded_key = self._encode_key(key)
        with self.env.begin(write=True) as txn:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            txn.put(encoded_key, buffer.getvalue())

    def retrieve_image(self, key: str) -> Image.Image:
        """Retrieve an image from LMDB."""
        encoded_key = self._encode_key(key)
        with self.env.begin() as txn:
            value = txn.get(encoded_key)
            if value is None:
                raise KeyError(f"Key '{key}' not found in LMDB.")
            buffer = io.BytesIO(value)
            return Image.open(buffer)

    def has_key(self, key: str) -> bool:
        """Check if a key exists in the LMDB file."""
        encoded_key = self._encode_key(key)
        with self.env.begin() as txn:
            return txn.get(encoded_key) is not None

    def iter_keys(self) -> iter:
        """Iterate over all keys in the LMDB file."""
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for encoded_key, _ in cursor:
                yield encoded_key.decode("utf-8")

    def close(self) -> None:
        """Close the LMDB environment."""
        self.env.close()


class ImageStorageFactory:
    """Factory for creating image storage backends."""

    @staticmethod
    def create_storage(
        backend: str, 
        file_path: Path,
        mode: Literal["a", "r", "w"] = "r", # read only, file must exist
        **kwargs
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
        elif backend == "lmdb":
            return LMDBImageStorage(file_path, mode, **kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        

def get_colette_collection(app_dir: str) -> chromadb.Collection:
    """Get the Colette collection from the given application directory."""
    config_path = os.path.normpath(f"{app_dir}/config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Config file {config_path} not found in the application directory.")
    
    is_rag = "rag" in config["parameters"]["input"]
    if is_rag and "ragm" in config["parameters"]["input"]["rag"]:
        db_path = os.path.normpath(f"{app_dir}/mm_index")
        cc =  chromadb.PersistentClient(str(db_path), Settings(anonymized_telemetry=False))
    else:
        raise ValueError("Multimodal RAG not found in the application folder.")

    db_path = f"{app_dir}/mm_index"
    cc =  chromadb.PersistentClient(str(db_path), Settings(anonymized_telemetry=False))

    cols = cc.list_collections()
    if cols and not isinstance(cols[0], str):
        cols = [c["name"] for c in cols]

    if "mm_db" not in cols:
        raise ValueError("Multimodal RAG not found in the application folder.")
    
    return cc.get_collection("mm_db")

@click.group()
def cli():
    """A command-line tool with multiple options."""
    pass

@click.command()
@click.option("--app-dir", required=True, type=str, help="Specify the application directory")
def migrate(app_dir):
    """Run the migration process."""
    click.echo(f"Starting migration for application directory: {app_dir}")

    col = get_colette_collection(app_dir)

    if col.count() == 0:
        click.echo("No images found in the collection.")
        return

    pattern = re.compile(
        r'/(crops|chunks|images)'
        r'(?P<path>(?:/[^/]+)+)'
        r'/'
        r'(?P<uuid>[0-9a-fA-F-]{36})-(?P<page>\d+)'
        r'(?:_(?P<kind>crop|chunk)_(?P<number>\d+))?'
        r'\.\w+$'
    )

    if os.path.exists(f"{app_dir}/kvstore.db"):
        click.echo("KVStore already exists in the application directory.")
        return
    
    # Pick first element of the vector store
    first = col.get(offset=0, limit=1)
    if pattern.search(first["ids"][0]) is None:
        click.echo("Invalid vector store.")
        return

    try:    
        col = get_colette_collection(app_dir)

        kvstore = ImageStorageFactory.create_storage("hdf5", f"{app_dir}/kvstore.db", "a")
        
        offset, limit = 0, 1_000
        while len(results := col.get(offset=offset, limit=limit, include=["embeddings", "metadatas"])) > 0 and len(results["ids"]) > 0:
            ids = []
            for pos, key in enumerate(results["ids"]):
                match = pattern.search(key)
                if match:
                    # old format
                    extracted_path = match.group('path').lstrip("/")
                    uuid = match.group('uuid')
                    page = int(match.group('page'))
                    kind = match.group('kind') if match.group('kind') else 'image'
                    number = int(match.group('number')) if match.group('number') else None
                    if not results["metadatas"][pos]:
                        results["metadatas"][pos] = dict()

                    results["metadatas"][pos]["source"] = os.path.normpath(extracted_path)
                    results["metadatas"][pos]["kind"] = kind
                    results["metadatas"][pos]["page"] = page
                    if number is not None:  
                        results["metadatas"][pos]["number"] = number
                else:
                    print(f"Key {key} ignored")
                
                new_key = f"{uuid}_{page:04d}_{kind}"
                if number is not None:
                    new_key += f"_{number:03d}"

                ids.append(new_key)

                if not key.startswith("/"):
                    os.path.abspath(key)
                    key = os.path.join(os.path.abspath(app_dir), "..", key)
                
                img = Image.open(key)
                kvstore.store_image(new_key, img)

            col.delete(ids=results["ids"])
            col.upsert(ids=ids, metadatas=results["metadatas"], embeddings=results["embeddings"])

            offset += len(results["ids"])

        shutil.rmtree(f"{app_dir}/crops", ignore_errors=True)
        shutil.rmtree(f"{app_dir}/chunks", ignore_errors=True)
        shutil.rmtree(f"{app_dir}/images", ignore_errors=True)
        shutil.rmtree(f"{app_dir}/pdfs", ignore_errors=True)

        click.echo(f"Migrated {offset} images.")
    except Exception as e:
        click.echo(f"Migration failed: {e}")
        return
    finally:
        kvstore.close()

    click.echo("Migration completed.")

@click.command()
@click.option("--app-dir", required=True, type=str, help="Specify the application directory")
@click.option("--key", required=True, type=str, help="Specify the extraction key")
def extract(app_dir, key):
    """Extract data with a key."""
    click.echo(f"Extracting data from {app_dir} using key: {key}")
    try:
        kvstore = ImageStorageFactory.create_storage("hdf5", app_dir / "kvstore.db")
        if not kvstore.has_key(key):
            click.echo(f"Key '{key}' not found in the storage.")
            return
        
        img = kvstore.retrieve_image(key)
        img.save(f"{key}.jpg")
    except Exception as e:
        click.echo(f"Extraction failed: {e}")
    finally:
        kvstore.close()
    click.echo("Extraction completed.")

@click.command()
@click.option("--app-dir", required=True, type=str, help="Specify the application directory")
@click.option("--key", type=str, help="Optional key for filtering information")
def info(app_dir, key):
    """Display information about the application."""
    click.echo(f"Displaying info for application directory: {app_dir}")
    try:
        kvstore = ImageStorageFactory.create_storage("hdf5", f"{app_dir}/kvstore.db", "r")

        count = 0
        for _ in kvstore.iter_keys():
            count += 1
        click.echo(f"Total number of images: {count}")

        if key:
            if not kvstore.has_key(key):
                click.echo(f"Key '{key}' not found in the storage.")
                return
            img = kvstore.retrieve_image(key)
            click.echo(f"Image '{key}' has dimensions: {img.size}")

    except Exception as e:
        click.echo(f"Information retrieval failed: {e}")
        return
    finally:
        kvstore.close()

    
    # Add info retrieval logic here
    click.echo("Information displayed.")

@click.command()
@click.option("--app-dir", required=True, type=str, help="Specify the application directory")
def check(app_dir):
    """Run a kvstore check."""
    click.echo(f"Checking application directory: {app_dir}")
    kvstore_path = Path(app_dir) / "kvstore.db"
    if not kvstore_path.exists():
        click.echo("KVStore not found.")
        return
    
    col = get_colette_collection(app_dir)

    kvstore = ImageStorageFactory.create_storage("hdf5", kvstore_path, "r")

    try:
        failed = False
        for key in kvstore.iter_keys():
            if not col.get(ids=[key]):
                failed = True
                click.echo(f"Key '{key}' not found in the collection.")
        if not failed:
            click.echo("All keys found in the collection.")
    except Exception as e:
        click.echo(f"Check failed: {e}")
        return
    finally:
        kvstore.close()

    click.echo("Check completed.")

# Add commands to the CLI group
cli.add_command(migrate)
cli.add_command(extract)
cli.add_command(info)
cli.add_command(check)

if __name__ == "__main__":
    cli()
