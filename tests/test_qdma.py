"""V80's QDMA host contract without a card: BAR2 isolation, full-width address translation,
short synchronous I/O, partial writes, monitor access, lock identity and failure cleanup."""
import os
from pathlib import Path

import numpy as np
import pytest

from opentpu import target as T
from opentpu.host import board, qdma as Q, regs as R, runstate
from opentpu.host.fake import FakeTransport

DEV = "qdma:0000:41:00.1@4"


@pytest.fixture
def trees(tmp_path, monkeypatch):
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("OTPU_TARGET", raising=False)
    sysfs, dev_root = tmp_path / "pci", tmp_path / "dev"
    root = sysfs / "0000:41:00.1"
    root.mkdir(parents=True)
    dev_root.mkdir()
    (root / "vendor").write_text("0x10ee\n")
    (root / "device").write_text("0x50b5\n")
    (tmp_path / "qdma-pf").mkdir()
    (root / "driver").symlink_to(tmp_path / "qdma-pf")
    with (root / "resource2").open("wb") as f:
        f.truncate(2 << 20)
    (dev_root / "qdma41001-MM-4").touch()
    return dict(sysfs=sysfs, dev_root=dev_root)


@pytest.fixture
def transport(trees):
    t = Q.QdmaTransport(DEV, **trees)
    try:
        yield t
    finally:
        t.close()


class Memory:
    """Synchronous driver's MM address space, with short completions and EINTR."""
    def __init__(self, t, monkeypatch, limit=1 << 20):
        self.data, self.reads, self.writes = {}, [], []
        self.interrupt_read = self.interrupt_write = False
        pread, pwrite = os.pread, os.pwrite

        def read(fd, n, off):
            if fd != t.fd:
                return pread(fd, n, off)
            if self.interrupt_read:
                self.interrupt_read = False
                raise InterruptedError()
            n = min(n, limit)
            self.reads.append((off, n))
            return bytes(self.data.get(off + k, 0) for k in range(n))

        def write(fd, buf, off):
            if fd != t.fd:
                return pwrite(fd, buf, off)
            if self.interrupt_write:
                self.interrupt_write = False
                raise InterruptedError()
            n = min(len(buf), limit)
            self.writes.append((off, n))
            self.data.update((off + k, int(buf[k])) for k in range(n))
            return n

        monkeypatch.setattr(Q.os, "pread", read)
        monkeypatch.setattr(Q.os, "pwrite", write)


def test_mmio_window_and_monitor(trees):
    p = T.load("v80").platform
    bar = trees["sysfs"] / "0000:41:00.1/resource2"
    with bar.open("r+b") as f:
        f.write(b"AMR!")
        f.seek(p["csr_offset"])
        f.write(R.ID_OTPU.to_bytes(4, "little"))
    t = Q.QdmaTransport(DEV, **trees)
    try:
        assert t.reg_read(0) == R.ID_OTPU
        t.reg_write(4, 0x123456789)
        assert t.reg_read_many([0, 4]) == [R.ID_OTPU, 0x23456789]
        assert t.poll(4, 0xFFFFFFFF, 0x23456789, timeout=0) == 0x23456789
        with pytest.raises(TimeoutError):
            t.poll(4, 1, 0, timeout=0)
        for off in [-4, 2, p["csr_bytes"]]:
            with pytest.raises(ValueError):
                t.reg_read(off)
    finally:
        t.close()
    assert bar.read_bytes()[:4] == b"AMR!"
    m = Q.QdmaTransport(DEV, dma=False, **trees)
    try:
        assert m.fd == -1 and m._otpu_lock is None and m.reg_read(0) == R.ID_OTPU
        # Exercise the monitor's real Board.snapshot path, including its SNAP write.
        b = board.Board(m, check=False, lock=False)
        b._info = {"regmap": 5}
        assert b.snapshot()["snaps"] == 1
        with pytest.raises(PermissionError):
            m.reg_write(0, 0)
        with pytest.raises(PermissionError):
            m.reg_write(R.R_CTRL, 1)
        with pytest.raises(RuntimeError, match="not open"):
            m.mem_read(0, 0, 4)
    finally:
        m.close()
    with pytest.raises(RuntimeError, match="closed"):
        m.reg_read(0)


def test_short_io_and_high_addresses(transport, monkeypatch):
    t = transport
    mem = Memory(t, monkeypatch, limit=17)
    monkeypatch.setattr(Q, "DMA_CHUNK", 64)
    mem.interrupt_read = mem.interrupt_write = True
    a = np.arange(256, dtype=np.uint8)
    off = (1 << 32) - len(a)
    t.mem_write(7, off, a)
    base = 0x47FFFFFF00
    assert mem.writes[0][0] == base and mem.writes[-1][0] > base
    assert all(base <= k < 0x4800000000 for k in mem.data)
    out = np.empty_like(a)
    assert t.mem_read(7, off, len(a), out=out) is out
    assert np.array_equal(out, a)
    assert sum(n for _, n in mem.reads) == sum(n for _, n in mem.writes) == len(a)


def test_partial_write_preserves_edge_beats(transport, monkeypatch):
    t = transport
    mem = Memory(t, monkeypatch)
    a = np.arange(256, dtype=np.uint8)
    t.mem_write(2, 0, a)
    t.mem_write(2, 123, np.full(19, 71, np.uint8))
    a[123:142] = 71
    assert np.array_equal(t.mem_read(2, 0, 256), a)
    assert mem.writes == [(0x4200000000, 256)] * 2
    t.mem_write(2, t.channel_bytes, np.zeros(0, np.uint8))


def test_ranges_and_no_progress(transport, monkeypatch):
    t = transport
    for ch, off, n in [(-1, 0, 1), (8, 0, 1), (0, -1, 1), (0, 0, -1),
                       (0, t.channel_bytes - 2, 4)]:
        with pytest.raises(ValueError):
            t.mem_read(ch, off, n)
    with pytest.raises(ValueError, match="output"):
        t.mem_read(0, 0, 4, out=np.zeros(8, np.uint8)[::2])
    with pytest.raises(ValueError, match="channel map"):
        t.bases = (0, 1 << 31)
    monkeypatch.setattr(Q.os, "pread", lambda *args: b"")
    monkeypatch.setattr(Q.os, "pwrite", lambda *args: 0)
    with pytest.raises(OSError, match="no progress"):
        t.mem_read(0, 0, 4)
    with pytest.raises(OSError, match="no progress"):
        t.mem_write(0, 0, np.zeros(128, np.uint8))


def test_selection_identity_and_cleanup(trees, monkeypatch):
    assert runstate.devname(DEV) == runstate.devname("qdma:0000:41:00.1@5")
    for bad in ["qdma:0000:41:00.0@4", "qdma:0001:41:00.1@4", "qdma:41:00.1@4",
                "qdma:0000:41:00.1@-1", "qdma:0000:41:00.1@2048"]:
        with pytest.raises(ValueError):
            Q.device(bad)
    t = Q.QdmaTransport(DEV, **trees)
    try:
        with pytest.raises(runstate.DeviceBusy):
            Q.QdmaTransport("qdma:0000:41:00.1@5", **trees)
    finally:
        t.close()
        t.close()
    before = len(list(Path("/proc/self/fd").iterdir()))
    for _ in range(3):
        with pytest.raises(FileNotFoundError):
            Q.QdmaTransport("qdma:0000:41:00.1@5", **trees)
    assert len(list(Path("/proc/self/fd").iterdir())) == before
    lk = runstate.DeviceLock(Q.lock_name(DEV))
    lk.release()
    called = []
    monkeypatch.setattr(Q, "QdmaTransport", lambda *a, **kw: called.append((a, kw)))
    board.open_transport(DEV, dma=False, target="v80")
    assert called == [((DEV,), dict(dma=False, target="v80"))]


@pytest.mark.parametrize("name", ["v80", "v80:fp16"])
def test_board_map_and_channel_hash(trees, monkeypatch, name):
    """Board's channel hashing and the transport's AMR base addition compose end to end;
    Board.info must not replace the absolute HBM base with zero-based channel addresses."""
    monkeypatch.setenv("OTPU_TARGET", name)
    t = Q.QdmaTransport(DEV, **trees)
    mem = Memory(t, monkeypatch)
    cfg = T.config(name)
    f = FakeTransport(devname=None, regmap=5, pair=False, dstep=True, stream=True,
                      seq_wc=True, chash=True, outer=False, dstep2=True, se_own=True, kt=True)
    f.cfg_words = R.cfg_words(dict(R.cfg(f.cfg_words), WG=cfg.WG, hash=T.arch_hash(cfg)))
    for off in [R.R_ID, R.R_VERSION, R.R_STATUS, R.R_REGMAP, R.R_CAPS, R.R_CAPS2,
                R.R_CORE_KHZ, R.R_BUILD_ID, R.R_TEMP, R.R_DDR_MTS, R.R_ACT_ROWS, *R.CFG_REGS]:
        t.reg_write(off, f.reg_read(off))
    with board.Board(t, calibrate=False) as b:
        assert board.device_config(b.info()) == cfg
        a = np.arange(4096, dtype=np.uint8)
        # Above 4 GiB logically too: tests CHASH's higher bits, not only the HBM base.
        addr = (1 << 32) + 1024
        b.write(addr, a)
        assert np.array_equal(b.read(addr, len(a)), a)
        for ch, off, part in board.split(addr, a, True, 8, 128):
            base = 0x4000000000 + ch * (1 << 32) + off
            assert bytes(mem.data[base + k] for k in range(len(part))) == part.tobytes()


@pytest.mark.parametrize("env, explicit, want", [
    (None, None, "v80"), ("v80:fp16", None, "v80:fp16"),
    ("v80:fp16", "v80", "v80"), ("v80", "v80:fp16", "v80:fp16"),
])
def test_target_selection(trees, monkeypatch, env, explicit, want):
    if env is not None:
        monkeypatch.setenv("OTPU_TARGET", env)
    t = Q.QdmaTransport(DEV, dma=False, target=explicit, **trees)
    try:
        assert t.want_target == want
    finally:
        t.close()
    monkeypatch.setenv("OTPU_TARGET", "ypcb-00338")
    with pytest.raises(ValueError, match="V80 target"):
        Q.QdmaTransport(DEV, dma=False, **trees)


def test_cli_dispatch_and_missing_queue(monkeypatch, capsys):
    from opentpu.host import diag, smi

    calls = []

    def unavailable(dev, **kwargs):
        calls.append((dev, kwargs))
        raise FileNotFoundError("selected queue is missing")

    monkeypatch.setattr(board, "open_transport", unavailable)
    assert diag.main(["--dev", DEV]) == 1
    out = capsys.readouterr().out
    assert "QDMA transport" in out and "bidirectional MM" in out
    assert "XDMA" not in out and "otpu-setup" not in out
    assert calls == [(DEV, {})]
    calls.clear()
    smi.main(["--dev", DEV, "--json"])
    capsys.readouterr()
    assert calls == [(DEV, {"dma": False})]
