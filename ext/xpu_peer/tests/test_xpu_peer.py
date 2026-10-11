"""Tests for xpu_peer: cross-process GPU-to-GPU writes and the all-reduce
built on them.

Needs two XPU devices. One pair of worker processes (rank r on xpu:r, a gloo
group between them, like vLLM's TP workers) runs every scenario once per
session; each test checks what the scenarios reported.

Run (from ext/xpu_peer):
  LD_LIBRARY_PATH=$VENV/lib $VENV/bin/python -m pytest tests -q
"""
import os
import time
import traceback
import zlib

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MiB = 1 << 20


def _data(seed, shape, dtype, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=g, dtype=torch.float32).to(
            dtype).to(device)
    return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8).to(
        device)


def _exchange(obj):
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


# ---- scenarios: run in each worker, return a value for the tests ---------

def s_handle(rank, dev):
    import xpu_peer
    buf = xpu_peer.PeerBuffer(MiB)
    h = buf.ipc_handle()
    buf.close()
    return type(h).__name__, len(h)


def s_peer_write(rank, dev):
    """Write 8 MiB into the peer's buffer; check what arrived in mine."""
    import xpu_peer
    n = 8 * MiB
    buf = xpu_peer.PeerBuffer(n)
    buf.tensor().zero_()
    torch.xpu.synchronize()
    handles = _exchange(buf.ipc_handle())
    peer = xpu_peer.PeerMapping(handles[1 - rank], n)
    peer.write(_data(rank, (n,), torch.uint8, dev), 0)
    torch.xpu.synchronize()
    dist.barrier()
    got = buf.tensor().clone()
    ok = torch.equal(got, _data(1 - rank, (n,), torch.uint8, dev))
    dist.barrier()
    peer.close()
    buf.close()
    return ok


def s_offset_write(rank, dev):
    """A write at an offset lands there and leaves the rest untouched."""
    import xpu_peer
    n, off, m = 4 * MiB, 1 * MiB + 4096, 2 * MiB
    buf = xpu_peer.PeerBuffer(n)
    buf.tensor().zero_()
    torch.xpu.synchronize()
    handles = _exchange(buf.ipc_handle())
    peer = xpu_peer.PeerMapping(handles[1 - rank], n)
    peer.write(_data(10 + rank, (m,), torch.uint8, dev), off)
    torch.xpu.synchronize()
    dist.barrier()
    t = buf.tensor()
    ok = (torch.equal(t[off:off + m], _data(11 - rank, (m,), torch.uint8, dev))
          and int(t[:off].count_nonzero()) == 0
          and int(t[off + m:].count_nonzero()) == 0)
    dist.barrier()
    peer.close()
    buf.close()
    return ok


def s_bounds(rank, dev):
    """Writes past the peer buffer's end, or from non-contiguous or host
    tensors, are refused before reaching the GPU."""
    import xpu_peer
    n = MiB
    buf = xpu_peer.PeerBuffer(n)
    handles = _exchange(buf.ipc_handle())
    peer = xpu_peer.PeerMapping(handles[1 - rank], n)
    refused = []
    for src, off in [
        (torch.zeros(n, dtype=torch.uint8, device=dev), 1),       # past end
        (torch.zeros(16, dtype=torch.uint8, device=dev), -1),     # negative
        (torch.zeros(64, 64, dtype=torch.uint8, device=dev).t(), 0),  # strided
        (torch.zeros(16, dtype=torch.uint8), 0),                  # host tensor
    ]:
        try:
            peer.write(src, off)
            refused.append(False)
        except (ValueError, RuntimeError):
            refused.append(True)
    torch.xpu.synchronize()
    dist.barrier()
    peer.close()
    buf.close()
    return refused


SHAPES = [(8192, 5120), (4097, 5120), (1, 5120), (3, 7)]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def s_allreduce_exact(rank, dev, ar):
    """Result equals x0 + x1 summed in rank order, bit for bit, on every
    rank, for every shape and dtype."""
    res = {}
    for dtype in DTYPES:
        for shape in SHAPES:
            seeds = [zlib.crc32(f"{dtype}{shape}{r}".encode()) & 0xFFFF
                     for r in (0, 1)]
            xs = [_data(s, shape, dtype, dev) for s in seeds]
            out = ar.all_reduce(xs[rank].clone())
            ref = xs[0] + xs[1]
            res[f"{dtype}-{shape}"] = (torch.equal(out, ref), out.shape == ref.shape)
    return res


def s_allreduce_inplace(rank, dev, ar):
    """all_reduce reduces into the tensor it is given and returns it."""
    x = _data(77 + rank, (64, 5120), torch.float16, dev)
    y = x.clone()
    out = ar.all_reduce(y)
    ref = _data(77, (64, 5120), torch.float16, dev) + _data(78, (64, 5120), torch.float16, dev)
    return out.data_ptr() == y.data_ptr() and torch.equal(y, ref)


def s_allreduce_stress(rank, dev, ar):
    """200 back-to-back all-reduces, the ranks skewed by random host delays
    before calls and right after the barrier (between a peer's write and
    this rank's sum), each result checked: catches a slot being
    overwritten while still in use."""
    import random
    import xpu_peer.allreduce as impl
    rng = random.Random(1000 + rank)
    real = impl.dist

    class SlowBarrier:
        def __getattr__(self, name):
            return getattr(real, name)

        def barrier(self, *args, **kwargs):
            real.barrier(*args, **kwargs)
            if rng.random() < 0.3:
                time.sleep(rng.random() * 0.02)

    # Inputs made up front: generating them takes longer than the delays,
    # which would keep the ranks in lockstep and hide the race.
    shape = (2048, 5120)
    inputs = [[_data(5000 + 2 * k + r, shape, torch.float16, dev) for r in (0, 1)]
              for k in range(10)]
    impl.dist = SlowBarrier()
    try:
        bad = []
        for i in range(200):
            xs = inputs[i % 10]
            if rng.random() < 0.3:
                time.sleep(rng.random() * 0.004)
            out = ar.all_reduce(xs[rank].clone())
            if not torch.equal(out, xs[0] + xs[1]):
                bad.append(i)
    finally:
        impl.dist = real
    return bad


def s_allreduce_growth(rank, dev):
    """Payloads larger than the current capacity grow the buffers (a
    collective step) and stay exact."""
    from xpu_peer import PeerAllReduce
    ar = PeerAllReduce(dist.group.WORLD, rank, 2)
    ok = []
    for rows in (16, 300, 2048, 8192, 100):
        xs = [_data(rows * 10 + r, (rows, 5120), torch.float16, dev) for r in (0, 1)]
        out = ar.all_reduce(xs[rank].clone())
        ok.append(torch.equal(out, xs[0] + xs[1]))
    cap = ar.capacity
    ar.close()
    return ok, cap


def s_failure_is_collective(rank, dev):
    """If setting up fails on one rank (here: rank 1 cannot open the peer
    handle), every rank raises from the same call instead of one rank
    waiting forever for the other, and the operand is left untouched so the
    caller can fall back to another path."""
    import xpu_peer.allreduce as impl
    from xpu_peer import PeerAllReduce
    real = impl.PeerMapping

    def broken(*args, **kwargs):
        raise RuntimeError("zeMemOpenIpcHandle failed: 0x78000004")

    if rank == 1:
        impl.PeerMapping = broken
    ar = PeerAllReduce(dist.group.WORLD, rank, 2)
    x = _data(300 + rank, (512, 5120), torch.float16, dev)
    y = x.clone()
    try:
        ar.all_reduce(y)
        raised = False
    except RuntimeError:
        raised = True
    finally:
        impl.PeerMapping = real
    ar.close()
    dist.barrier()
    return raised, torch.equal(x, y)


def s_close_frees(rank, dev):
    """Opening and closing all-reduces repeatedly does not leak device
    memory (buffers are freed, peer mappings closed)."""
    from xpu_peer import PeerAllReduce
    torch.xpu.synchronize()
    free0 = torch.xpu.mem_get_info()[0]
    for _ in range(8):
        ar = PeerAllReduce(dist.group.WORLD, rank, 2)
        ar.all_reduce(torch.ones(8192, 5120, dtype=torch.float16, device=dev))
        ar.close()
    torch.xpu.synchronize()
    dist.barrier()
    free1 = torch.xpu.mem_get_info()[0]
    return (free0 - free1) / MiB


def s_allreduce_time(rank, dev, ar):
    """Median wall time of a prefill-sized (8192 x 5120 fp16, 80 MiB)
    all-reduce, synchronized, as vLLM would see it."""
    x = _data(rank, (8192, 5120), torch.float16, dev)
    ts = []
    for _ in range(15):
        y = x.clone()
        torch.xpu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        ar.all_reduce(y)
        torch.xpu.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1000


def _worker(rank, port, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    results = {}
    try:
        torch.xpu.set_device(rank)
        dev = torch.device(f"xpu:{rank}")
        dist.init_process_group("gloo", rank=rank, world_size=2)
        for name, fn in [("handle", s_handle), ("peer_write", s_peer_write),
                         ("offset_write", s_offset_write), ("bounds", s_bounds)]:
            try:
                results[name] = ("ok", fn(rank, dev))
            except Exception:
                results[name] = ("error", traceback.format_exc())
        ar = None
        try:
            from xpu_peer import PeerAllReduce
            ar = PeerAllReduce(dist.group.WORLD, rank, 2)
        except Exception:
            results["allreduce_init"] = ("error", traceback.format_exc())
        for name, fn in [("allreduce_exact", s_allreduce_exact),
                         ("allreduce_inplace", s_allreduce_inplace),
                         ("allreduce_stress", s_allreduce_stress),
                         ("allreduce_time", s_allreduce_time)]:
            try:
                if ar is None:
                    raise RuntimeError("PeerAllReduce unavailable")
                results[name] = ("ok", fn(rank, dev, ar))
            except Exception:
                results[name] = ("error", traceback.format_exc())
        if ar is not None:
            ar.close()
        for name, fn in [("allreduce_growth", s_allreduce_growth),
                         ("failure_is_collective", s_failure_is_collective),
                         ("close_frees", s_close_frees)]:
            try:
                results[name] = ("ok", fn(rank, dev))
            except Exception:
                results[name] = ("error", traceback.format_exc())
        dist.barrier()
        dist.destroy_process_group()
    except Exception:
        results["worker"] = ("error", traceback.format_exc())
    q.put((rank, results))


@pytest.fixture(scope="session")
def runs():
    if torch.xpu.device_count() < 2:
        pytest.skip("needs two XPU devices")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29600 + os.getpid() % 300
    procs = [ctx.Process(target=_worker, args=(r, port, q)) for r in (0, 1)]
    for p in procs:
        p.start()
    out = {}
    for _ in procs:
        rank, res = q.get(timeout=900)
        out[rank] = res
    for p in procs:
        p.join(timeout=60)
    return out


def _get(runs, name):
    vals = []
    for rank in (0, 1):
        assert "worker" not in runs[rank], runs[rank]["worker"][1]
        status, val = runs[rank][name]
        assert status == "ok", f"rank {rank} {name}:\n{val}"
        vals.append(val)
    return vals


def test_ipc_handle_is_64_bytes(runs):
    for v in _get(runs, "handle"):
        assert v == ("bytes", 64)


def test_peer_write_arrives(runs):
    assert _get(runs, "peer_write") == [True, True]


def test_offset_write_lands_in_place(runs):
    assert _get(runs, "offset_write") == [True, True]


def test_bad_writes_refused(runs):
    for v in _get(runs, "bounds"):
        assert v == [True, True, True, True]


def test_allreduce_bit_exact_all_shapes_dtypes(runs):
    for res in _get(runs, "allreduce_exact"):
        bad = {k: v for k, v in res.items() if v != (True, True)}
        assert not bad, bad


def test_allreduce_in_place(runs):
    assert _get(runs, "allreduce_inplace") == [True, True]


def test_allreduce_stress_no_slot_race(runs):
    assert _get(runs, "allreduce_stress") == [[], []]


def test_allreduce_grows_capacity(runs):
    for ok, cap in _get(runs, "allreduce_growth"):
        assert ok == [True] * 5
        assert cap >= 8192 * 5120 * 2


def test_setup_failure_raises_on_every_rank(runs):
    assert _get(runs, "failure_is_collective") == [(True, True), (True, True)]


def test_close_frees_device_memory(runs):
    for leaked_mib in _get(runs, "close_frees"):
        assert leaked_mib < 8, f"{leaked_mib:.1f} MiB not returned"


def test_allreduce_faster_than_shm(runs):
    # shm path: 28.3 ms (repro/shm_stage_bench.py, 2026-10-09); target ~9.
    for ms in _get(runs, "allreduce_time"):
        print(f"80 MiB all-reduce: {ms:.2f} ms")
        assert ms < 15
