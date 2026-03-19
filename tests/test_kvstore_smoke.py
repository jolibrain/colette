from pathlib import Path

import pytest
from PIL import Image

from colette.kvstore import HDF5ImageStorage, ImageStorageFactory


@pytest.mark.smoke
def test_hdf5_storage_roundtrip_and_keys(tmp_path):
    db_path = tmp_path / "kvstore.db"
    storage = HDF5ImageStorage(db_path, mode="a")
    try:
        image = Image.new("RGB", (4, 4), color="red")
        storage.store_image("doc-1", image)

        assert storage.has_key("doc-1") is True

        loaded = storage.retrieve_image("doc-1")
        assert loaded.size == (4, 4)

        keys = list(storage.iter_keys())
        assert "doc-1" in keys

        # Overwrite same key path to cover replacement branch.
        storage.store_image("doc-1", Image.new("RGB", (2, 2), color="blue"))
        loaded_2 = storage.retrieve_image("doc-1")
        assert loaded_2.size == (2, 2)
    finally:
        storage.close()


@pytest.mark.smoke
def test_hdf5_storage_missing_key_raises(tmp_path):
    storage = HDF5ImageStorage(tmp_path / "missing.db", mode="a")
    try:
        with pytest.raises(KeyError):
            storage.retrieve_image("unknown")
    finally:
        storage.close()


@pytest.mark.smoke
def test_image_storage_factory_variants(tmp_path):
    db_path = Path(tmp_path) / "factory.db"

    storage = ImageStorageFactory.create_storage("hdf5", db_path, mode="a")
    assert isinstance(storage, HDF5ImageStorage)
    storage.close()

    with pytest.raises(ValueError):
        ImageStorageFactory.create_storage("unsupported", db_path, mode="a")
