"""Cross-process GPU-to-GPU writes and an all-reduce built on them, for XPU
tensor parallelism on hosts without PCIe peer atomics."""
import torch  # noqa: F401  (loads libtorch_xpu before the extension)

from ._C import PeerBuffer, PeerMapping
from .allreduce import PeerAllReduce

__all__ = ["PeerBuffer", "PeerMapping", "PeerAllReduce"]
