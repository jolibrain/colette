import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from colette.backends.hf.layout_detector import LayoutDetector
from colette.backends.hf.rag.rag_img import resolve_layout_detector_gpu_id


class _DummyJitModel:
    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        return self


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("gpu_id", "expected"),
    [
        (-1, "cpu"),
        (0, "cuda:0"),
        (1, "cuda:1"),
    ],
)
def test_layout_detector_maps_gpu_id_to_device(monkeypatch, gpu_id, expected):
    """A negative gpu id must select the CPU.

    The device arrives as a plain int. torch reads a bare int as a CUDA index, so a
    negative one raises "Device index must not be negative" rather than selecting the
    CPU - meaning a config asking for CPU could never be honoured. Non-negative ids
    must keep their previous cuda:<id> meaning.

    Builds no CUDA context: torch.device() is only a descriptor and the jit model is
    a stub, so this runs on CPU-only machines.
    """
    dummy = _DummyJitModel()
    monkeypatch.setattr(torch.jit, "load", lambda _path: dummy)

    detector = LayoutDetector(
        model_path="/tmp/layout_detector_stub.pt",
        resize_width=768,
        resize_height=1024,
        models_repository=Path("."),
        logger=logging.getLogger(__name__),
        device=gpu_id,
    )

    assert str(detector.device) == expected
    assert str(dummy.moved_to) == expected


@pytest.mark.smoke
def test_layout_detector_accepts_torch_device(monkeypatch):
    """An explicit torch.device must be passed through untouched."""
    dummy = _DummyJitModel()
    monkeypatch.setattr(torch.jit, "load", lambda _path: dummy)

    detector = LayoutDetector(
        model_path="/tmp/layout_detector_stub.pt",
        resize_width=768,
        resize_height=1024,
        models_repository=Path("."),
        logger=logging.getLogger(__name__),
        device=torch.device("cpu"),
    )

    assert detector.device == torch.device("cpu")


def _rag_config(gpu_id=0, ragm=None):
    return SimpleNamespace(gpu_id=gpu_id, ragm=ragm)


@pytest.mark.smoke
def test_index_request_without_ragm_keeps_creation_time_device():
    """The regression: an index request must not silently move the detector to the GPU.

    Index requests rarely carry ragm settings. Overwriting unconditionally with
    rag.gpu_id discarded the CPU placement resolved at service creation, which on a
    memory-tight machine surfaces as a CUDA OOM during indexing.
    """
    assert resolve_layout_detector_gpu_id(_rag_config(gpu_id=0), current=-1) == -1


@pytest.mark.smoke
def test_index_request_ragm_without_layout_id_keeps_creation_time_device():
    """A ragm block that omits layout_detector_gpu_id must not override either."""
    ragm = SimpleNamespace(layout_detector_gpu_id=None)
    assert resolve_layout_detector_gpu_id(_rag_config(gpu_id=0, ragm=ragm), current=-1) == -1


@pytest.mark.smoke
def test_explicit_index_request_value_wins():
    """An explicit layout_detector_gpu_id in the index request takes precedence."""
    ragm = SimpleNamespace(layout_detector_gpu_id=1)
    assert resolve_layout_detector_gpu_id(_rag_config(gpu_id=0, ragm=ragm), current=-1) == 1


@pytest.mark.smoke
def test_falls_back_to_rag_gpu_id_when_nothing_resolved():
    """With no request value and nothing from creation, fall back to rag.gpu_id.

    This is the pre-existing behaviour for every config that does not set
    layout_detector_gpu_id, and it must not change.
    """
    assert resolve_layout_detector_gpu_id(_rag_config(gpu_id=0), current=None) == 0
    ragm = SimpleNamespace(layout_detector_gpu_id=None)
    assert resolve_layout_detector_gpu_id(_rag_config(gpu_id=2, ragm=ragm), current=None) == 2
