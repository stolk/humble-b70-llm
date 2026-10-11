"""All-reduce by writing into the peers' device memory.

Each rank owns one receive buffer on its GPU, exported to the other ranks
over IPC. For an all-reduce, a rank writes its tensor into a slot of every
peer's buffer, drains its queue, and meets the others at a barrier on the
CPU group; then each rank sums all contributions in rank order on its own
device, so every rank gets identical bits. Data crosses PCIe once per peer,
GPU to GPU, without host staging and without peer atomics.

Receive buffers hold two slots of (world_size - 1) contributions each and
alternate between calls: call n+2 reuses call n's slot, and no rank passes
call n+1's barrier before its queue has finished reading call n's slot, so
one barrier per call is enough.
"""
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ._C import PeerBuffer, PeerMapping

_MIN_CAPACITY = 1 << 20


class PeerAllReduce:
    def __init__(self, cpu_group: ProcessGroup, rank: int, world_size: int):
        self.cpu_group = cpu_group
        self.rank = rank
        self.world_size = world_size
        self.capacity = 0
        self.buffer: PeerBuffer | None = None
        self.peers: dict[int, PeerMapping] = {}
        self.calls = 0

    def _slot_offset(self, slot: int, src: int, dst: int) -> int:
        """Where src's contribution goes in dst's buffer."""
        index = src if src < dst else src - 1
        return (slot * (self.world_size - 1) + index) * self.capacity

    def _gather(self, obj) -> list:
        out: list = [None] * self.world_size
        dist.all_gather_object(out, obj, group=self.cpu_group)
        return out

    def _allocate(self, nbytes: int) -> None:
        # Collective: every rank arrives with the same nbytes, since
        # all-reduce operands have the same shape on every rank. A failure
        # on any rank is shared at each step, so all ranks raise together.
        cap = max(nbytes, 2 * self.capacity, _MIN_CAPACITY)
        self._release()
        error = None
        handle = None
        try:
            self.buffer = PeerBuffer(2 * (self.world_size - 1) * cap)
            handle = self.buffer.ipc_handle()
        except Exception as e:
            error = e
        handles = self._gather(handle)
        if None in handles:
            self._drop()
            raise RuntimeError(
                "peer buffer allocation failed on rank(s) "
                f"{[r for r, h in enumerate(handles) if h is None]}") from error
        try:
            for r, h in enumerate(handles):
                if r != self.rank:
                    self.peers[r] = PeerMapping(h, self.buffer.nbytes)
        except Exception as e:
            error = e
        failed = self._gather(error is not None)
        if any(failed):
            self._drop()
            raise RuntimeError(
                "opening peer buffers failed on rank(s) "
                f"{[r for r, f in enumerate(failed) if f]}") from error
        self.capacity = cap

    def _drop(self) -> None:
        # Local cleanup after a failed setup; no peer has written yet.
        for m in self.peers.values():
            m.close()
        self.peers = {}
        if self.buffer is not None:
            self.buffer.close()
            self.buffer = None
        self.capacity = 0

    def _release(self) -> None:
        # Nothing may still read the old buffer or write through the old
        # mappings: drain this rank's queue, then wait for every rank.
        if self.buffer is None:
            return
        torch.xpu.current_stream().synchronize()
        dist.barrier(group=self.cpu_group)
        for m in self.peers.values():
            m.close()
        self.peers = {}
        self.buffer.close()
        self.buffer = None

    def all_reduce(self, output: torch.Tensor) -> torch.Tensor:
        """Sum output over all ranks, in place; returns output."""
        assert output.is_contiguous()
        nbytes = output.numel() * output.element_size()
        if nbytes > self.capacity:
            self._allocate(nbytes)
        slot = self.calls & 1
        self.calls += 1
        for r, peer in self.peers.items():
            peer.write(output, self._slot_offset(slot, self.rank, r))
        torch.xpu.current_stream().synchronize()
        dist.barrier(group=self.cpu_group)
        local = self.buffer.tensor()
        parts = []
        for r in range(self.world_size):
            if r == self.rank:
                parts.append(output)
            else:
                off = self._slot_offset(slot, r, self.rank)
                parts.append(local[off:off + nbytes].view(output.dtype)
                             .view(output.shape))
        acc = torch.add(parts[0], parts[1])
        for p in parts[2:]:
            acc.add_(p)
        output.copy_(acc)
        return output

    def close(self) -> None:
        self._release()
        self.capacity = 0
