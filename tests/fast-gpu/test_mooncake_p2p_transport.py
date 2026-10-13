"""P2P weight update's Mooncake transfer must work on this runner: pinned CPU on the trainer side to every
engine GPU, with both ends initialized over "rdma" exactly as miles' sender and SGLang's engine do."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, suite="stage-c-4-gpu-h200", labels=["weight-update"], hardware=["hopper"], num_gpus=4)

import glob
import multiprocessing as mp
import os
import queue
import time

import pytest
import torch

_NUM_GPUS = torch.cuda.device_count()
_NBYTES = 256 << 20
_TIMEOUT_S = 300


def _pattern(gpu: int, device: str) -> torch.Tensor:
    return ((torch.arange(_NBYTES, dtype=torch.int32, device=device) + gpu) % 251).to(torch.uint8)


def _init_engine(local_ip: str):
    from mooncake.engine import TransferEngine

    engine = TransferEngine()
    ret = engine.initialize(local_ip, "P2PHANDSHAKE", "rdma", "")
    if ret != 0:
        raise RuntimeError(f"TransferEngine.initialize returned {ret}")
    return engine


def _engine_rank(gpu: int, out: mp.Queue, verify: mp.Event) -> None:
    # SGLang RemoteInstanceWeightTransporter.init_engine + register_memory_region, on one GPU
    from sglang.srt.utils.network import get_local_ip_auto

    torch.cuda.set_device(gpu)
    local_ip = get_local_ip_auto()
    engine = _init_engine(local_ip)
    buf = torch.zeros(_NBYTES, dtype=torch.uint8, device=f"cuda:{gpu}")
    ret = engine.register_memory(buf.data_ptr(), _NBYTES)
    if ret != 0:
        raise RuntimeError(f"register_memory(cuda:{gpu}) returned {ret}")
    out.put(("ready", gpu, f"{local_ip}:{engine.get_rpc_port()}", buf.data_ptr()))
    verify.wait()
    torch.cuda.synchronize()
    out.put(("data", gpu, bool(torch.equal(buf, _pattern(gpu, buf.device)))))


def _trainer(targets: list[tuple[int, str, int]], out: mp.Queue) -> None:
    # miles create_transfer_engine + register_cpu_memory + batch_transfer_sync_write
    import ray

    engine = _init_engine(ray._private.services.get_node_ip_address())
    src = torch.empty(len(targets) * _NBYTES, dtype=torch.uint8).pin_memory()
    ret = engine.register_memory(src.data_ptr(), src.numel())
    if ret != 0:
        raise RuntimeError(f"register_memory(pinned CPU) returned {ret}")
    for chunk, (gpu, session_id, target_ptr) in zip(src.split(_NBYTES), targets, strict=True):
        chunk.copy_(_pattern(gpu, "cpu"))
        start = time.perf_counter()
        ret = engine.batch_transfer_sync_write(session_id, [chunk.data_ptr()], [target_ptr], [_NBYTES])
        out.put(("write", gpu, ret, time.perf_counter() - start))


def _print_rdma_view() -> None:
    for dev in sorted(glob.glob("/sys/class/infiniband/*")):
        with open(f"{dev}/ports/1/link_layer") as f:
            link_layer = f.read().strip()
        ndevs = set()
        for path in glob.glob(f"{dev}/ports/1/gid_attrs/ndevs/*"):
            try:
                with open(path) as f:
                    ndevs.add(f.read().strip())
            except OSError:
                pass
        print(f"{os.path.basename(dev)}: link_layer={link_layer} gid_ndevs={sorted(ndevs)}", flush=True)
    print(f"netdevs={sorted(os.listdir('/sys/class/net'))}", flush=True)


def _get(out: mp.Queue, kind: str, deadline: float, procs: list) -> tuple:
    while True:
        try:
            msg = out.get(timeout=5)
        except queue.Empty:
            if crashed := [p.name for p in procs if p is not None and p.exitcode not in (None, 0)]:
                raise AssertionError(f"{crashed} exited while waiting for {kind!r}") from None
            if time.monotonic() > deadline:
                raise AssertionError(f"timed out waiting for {kind!r} after {_TIMEOUT_S}s") from None
            continue
        if msg[0] == kind:
            return msg[1:]


def test_p2p_transfer_reaches_every_engine_gpu():
    assert _NUM_GPUS >= 2, _NUM_GPUS
    _print_rdma_view()
    ctx = mp.get_context("spawn")
    out, verify = ctx.Queue(), ctx.Event()
    procs = [
        ctx.Process(target=_engine_rank, args=(gpu, out, verify), name=f"engine-cuda:{gpu}")
        for gpu in range(_NUM_GPUS)
    ]
    deadline = time.monotonic() + _TIMEOUT_S
    try:
        for p in procs:
            p.start()
        targets = sorted(_get(out, "ready", deadline, procs) for _ in range(_NUM_GPUS))
        procs.append(ctx.Process(target=_trainer, args=(targets, out), name="trainer"))
        procs[-1].start()
        writes = {}
        for _ in targets:
            gpu, ret, secs = _get(out, "write", deadline, procs)
            writes[gpu] = (ret, secs)
        verify.set()
        data_ok = dict(_get(out, "data", deadline, procs) for _ in range(_NUM_GPUS))
    finally:
        verify.set()
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.kill()

    for gpu, session_id, _ in targets:
        ret, secs = writes[gpu]
        print(
            f"cuda:{gpu} session={session_id} ret={ret} {secs:.3f}s "
            f"{_NBYTES / secs / 1e9:.2f} GB/s data_ok={data_ok[gpu]}",
            flush=True,
        )
    assert all(ret == 0 for ret, _ in writes.values()), writes
    assert all(data_ok.values()), data_ok


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
