import logging
from pathlib import Path

import pytest
import torch

from colette.backends.hf.layout_detector import LayoutDetector


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
