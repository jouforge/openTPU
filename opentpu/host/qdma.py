"""AMR V80 PF1: 32-bit MMIO through user BAR2 and synchronous QDMA MM transfers.

Select explicitly with qdma:0000:41:00.1@4 (PCI BDF and an already started bidirectional
MM queue). Queue setup belongs to the platform administrator. The transport does not create
queues, change the driver or program the card. The openTPU CSR overlay must be loaded first;
the unmodified AMR base has no openTPU registers at this window.

Board splits/hashes logical bytes into channels. Its channel bases are relative; this
transport adds the target's AMR HBM base once, preserving offsets above 4 GiB. CSR accesses
use one native 32-bit load/store, as XdmaTransport's (never byte-wise mmap assignments).
"""
from __future__ import annotations

import mmap
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

from opentpu import target as T
from . import regs as R
from .runstate import DeviceLock

DMA_CHUNK = 1 << 20


def device(dev: str) -> tuple[str, int]:
    """Canonical BDF and queue. One lock per PF, shared by all of its queue selections.
    The Linux driver's qdmabbddf names omit the domain; only domain 0000 is supported."""
    m = re.fullmatch(r"qdma:(0000:[0-9a-f]{2}:[0-1][0-9a-f]\.1)@([0-9]+)", dev.lower())
    if not m or not 0 <= int(m[2]) < 2048:
        raise ValueError("QDMA device must be qdma:0000:bb:dd.1@queue (PF1, queue 0..2047)")
    return m[1], int(m[2])


def lock_name(dev: str) -> str:
    bdf, _ = device(dev)
    return f"qdma-{bdf}"


class QdmaTransport:
    """Board transport for the V80 overlay; dma=False allows only SNAP writes for monitors.
    sysfs and dev_root select the OS device trees (temporary trees in the tests)."""
    threaded = True
    streams = False             # host/core concurrency has not yet been qualified on V80
    ecc = False                 # no XDMA/MIG-specific scrub protocol

    def __init__(self, dev: str, dma: bool = True, target: str | None = None, *,
                 sysfs: Path = Path("/sys/bus/pci/devices"), dev_root: Path = Path("/dev")):
        bdf, queue = device(dev)
        if sys.byteorder != "little":
            raise RuntimeError("QDMA MMIO requires a little-endian host")
        self.dev, self.devname = f"qdma:{bdf}@{queue}", lock_name(dev)
        self.want_target = target or os.environ.get("OTPU_TARGET") or "v80"
        t = T.load(self.want_target)
        if t.platform.get("board") != "v80":
            raise ValueError("QDMA transport requires a V80 target")
        p = t.platform
        self.mem_base, self.channel_bytes = int(p["mem_base"]), int(p["channel_bytes"])
        self.nch, self.beat = int(p["channels_per_slice"]), int(p["beat_bytes"])
        self._bases = tuple(i * self.channel_bytes for i in range(t.arch["S"] * self.nch))
        self.csr_bytes = int(p["csr_bytes"])
        self._writable, self._dma = dma, threading.RLock()
        self.fd, self.regs, self.words, self._otpu_lock = -1, None, None, None
        root = Path(sysfs) / bdf
        if (root / "vendor").read_text().strip().lower() != "0x10ee" or \
           (root / "device").read_text().strip().lower() != "0x50b5":
            raise ValueError(f"{bdf}: expected AMD V80 PF1 (10ee:50b5)")
        if (root / "driver").resolve().name != "qdma-pf":
            raise ValueError(f"{bdf}: PF1 must be bound to qdma-pf")
        try:
            # Lock before opening files, as XDMA does. All queues of this PF share the lock.
            self._otpu_lock = DeviceLock(self.devname) if dma else None
            resource = root / f"resource{int(p['user_bar'])}"
            # Monitors latch counters through SNAP, as on XDMA, without opening a DMA queue.
            bar = os.open(resource, os.O_RDWR | os.O_SYNC)
            try:
                prot = mmap.PROT_READ | mmap.PROT_WRITE
                self.regs = mmap.mmap(bar, self.csr_bytes, mmap.MAP_SHARED, prot,
                                      offset=int(p["csr_offset"]))
            finally:
                os.close(bar)
            self.words = memoryview(self.regs).cast("I")
            if dma:
                node = f"qdma{bdf[5:7]}{bdf[8:10]}{bdf[11]}-MM-{queue}"
                self.fd = os.open(Path(dev_root) / node, os.O_RDWR)
        except BaseException:
            self.close()
            raise

    @property
    def bases(self) -> tuple[int, ...]:
        return self._bases

    @bases.setter
    def bases(self, bases) -> None:
        # Board.info installs the device's reported channel map. Refuse a different map
        # rather than translate it into AMR space using this target's geometry.
        if tuple(bases) != self._bases:
            raise ValueError("V80 device channel map differs from the transport's target")

    def close(self) -> None:
        with self._dma:
            if self.words is not None:
                self.words.release()
                self.words = None
            if self.regs is not None:
                self.regs.close()
                self.regs = None
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1
            if self._otpu_lock is not None:
                self._otpu_lock.release()
                self._otpu_lock = None

    def _reg(self, off: int) -> int:
        if self.words is None:
            raise RuntimeError("QDMA transport is closed")
        if off < 0 or off % 4 or off + 4 > self.csr_bytes:
            raise ValueError(f"CSR offset {off:#x} outside the aligned openTPU window")
        return off // 4

    def reg_read(self, off: int) -> int:
        return self.words[self._reg(off)]

    def reg_write(self, off: int, val: int) -> None:
        i = self._reg(off)
        if not self._writable and off != R.R_SNAP:
            raise PermissionError("QDMA monitors may only write SNAP")
        self.words[i] = val & 0xFFFFFFFF

    def reg_read_many(self, offs: list[int]) -> list[int]:
        return [self.reg_read(o) for o in offs]

    def poll(self, off: int, mask: int, val: int, timeout: float = 600.0,
             expect: float = 0.0) -> int:
        """Poll to a wall-clock deadline. expect is accepted for Board compatibility;
        V80 has no calibrated poll timing yet, so poll_exact remains false."""
        deadline = time.monotonic() + timeout
        self.poll_exact = False
        while True:
            r = self.reg_read(off)
            if r & mask == val:
                return r
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"register {off:#x} = {r:#x}, waiting for {val:#x}/{mask:#x}")
            time.sleep(min(0.001, left))

    def _address(self, ch: int, off: int, n: int) -> int:
        if self.fd < 0:
            raise RuntimeError("QDMA MM queue is not open")
        if not 0 <= ch < len(self.bases) or off < 0 or n < 0 or off + n > self.channel_bytes:
            raise ValueError(f"QDMA channel range out of bounds: ch={ch}, off={off}, n={n}")
        return self.mem_base + self.bases[ch] + off

    def _read(self, addr: int, out: np.ndarray) -> None:
        mv, pos = memoryview(out), 0
        while pos < len(out):
            try:
                # Plain synchronous .read; do not route through the driver's async read_iter.
                data = os.pread(self.fd, min(DMA_CHUNK, len(out) - pos), addr + pos)
            except InterruptedError:
                continue
            if not data:
                raise OSError(f"QDMA read made no progress at {addr + pos:#x}")
            mv[pos:pos + len(data)] = data
            pos += len(data)

    def mem_read(self, ch: int, off: int, n: int, out: np.ndarray | None = None) -> np.ndarray:
        addr = self._address(ch, off, n)
        if out is None:
            out = np.empty(n, np.uint8)
        if out.dtype != np.uint8 or out.shape != (n,) or not out.flags.c_contiguous or \
           not out.flags.writeable:
            raise ValueError("QDMA output must be a writable contiguous uint8 vector of n bytes")
        with self._dma:
            self._read(addr, out)
        return out

    def mem_write(self, ch: int, off: int, data: np.ndarray) -> None:
        a = np.ascontiguousarray(data, np.uint8).reshape(-1)
        addr = self._address(ch, off, len(a))
        if not len(a):
            return
        with self._dma:
            # A partial write preserves edge beats, serialized with other transfers and
            # their read/modify/write cycles. Board normally supplies whole channel beats.
            if off % self.beat or len(a) % self.beat:
                start = off // self.beat * self.beat
                end = -(-(off + len(a)) // self.beat) * self.beat
                buf = self.mem_read(ch, start, end - start)
                buf[off - start:off - start + len(a)] = a
                a, addr = buf, self._address(ch, start, len(buf))
            mv, pos = memoryview(a), 0
            while pos < len(a):
                try:
                    n = os.pwrite(self.fd, mv[pos:pos + DMA_CHUNK], addr + pos)
                except InterruptedError:
                    continue
                if n <= 0:
                    raise OSError(f"QDMA write made no progress at {addr + pos:#x}")
                pos += n
