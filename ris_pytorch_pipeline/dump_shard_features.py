# SPDX-License-Identifier: MIT
import json, math, argparse, os, tempfile
import numpy as np
from pathlib import Path
from .dataset import ShardNPZDataset
from .features_autocorr import autocorr_stack_from_y
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# ------------------------- GPU autocorr -------------------------
@torch.no_grad()
def autocorr_stack_torch_batch(Y_cplx_b: torch.Tensor, tau_max: int) -> torch.Tensor:
    """
    Y_cplx_b: [B, L, T] complex64 (CUDA)
    returns:  [B, 2*(tau_max+1), L, L] float32 (CPU)
    """
    assert Y_cplx_b.is_complex() and Y_cplx_b.ndim == 3
    B, L, T = Y_cplx_b.shape
    stacks = []
    for tau in range(tau_max + 1):
        t_eff = max(1, T - tau)
        X0 = Y_cplx_b[:, :, :T - tau]   # [B,L,T-τ]
        X1 = Y_cplx_b[:, :, tau:]       # [B,L,T-τ]
        R = torch.matmul(X0, X1.conj().transpose(-1, -2)) / float(t_eff)  # [B,L,L]
        RI = torch.view_as_real(R).movedim(-1, 1).to(torch.float32)       # [B,2,L,L]
        stacks.append(RI)
    out = torch.cat(stacks, dim=1)  # [B, 2*(τ+1), L, L]
    return out.cpu()

# ------------------------- helpers -------------------------
def _to_cplx(ri):
    ri = np.asarray(ri)
    return ri[..., 0] + 1j*ri[..., 1]

def _atomic_save_npz(path: Path, **arrs):
    """
    Save arrays atomically to avoid half-written chunks on preemption/OOM.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(prefix=f".{path.stem}.", dir=str(path.parent))
    os.close(fd)
    tmppath = Path(tmpname)
    try:
        # Handle object arrays with empty arrays by converting them to a different format
        save_arrs = {}
        for key, value in arrs.items():
            if key in ['ptr', 'phi', 'theta', 'r'] and hasattr(value, 'dtype') and value.dtype == object:
                # Convert object arrays with empty arrays to a format that saves properly
                if len(value) > 0 and hasattr(value[0], 'shape') and value[0].shape == (0,):
                    # All arrays are empty, save as a simple indicator
                    save_arrs[key] = np.array([0], dtype=np.float32)
                else:
                    save_arrs[key] = value
            else:
                save_arrs[key] = value
        
        # Use file handle approach to avoid NumPy's object array issues
        with open(tmppath, 'wb') as f:
            np.savez(f, **save_arrs)
        os.replace(tmppath, path)  # atomic on POSIX
    finally:
        try:
            if tmppath.exists():
                tmppath.unlink()
        except FileNotFoundError:
            pass

def _is_valid_chunk_npz(f: Path) -> bool:
    """
    Robust check that an npz chunk is readable and consistent.
    We only inspect numeric arrays K and Rx. No mmap.
    """
    try:
        if not f.exists():
            return False
        # Check file size - but be more lenient for object arrays
        file_size = f.stat().st_size
        if file_size < 100:  # Reduced from 1024 to 100 bytes
            return False
        # must use allow_pickle=True because file contains object arrays (phi/theta/r/ptr)
        with np.load(f, allow_pickle=True) as z:
            if "K" not in z or "Rx" not in z:
                return False
            K = z["K"]           # numeric
            Rx = z["Rx"]         # numeric
            mK = int(getattr(K, "shape", (0,))[0])
            mR = int(getattr(Rx, "shape", (0,))[0])
            if mK <= 0 or mR != mK:
                return False
        return True
    except Exception as e:
        # Add debugging for verification failures
        print(f"[DEBUG] Verification failed for {f}: {e}")
        return False



def _existing_chunks_npz(out_dir: Path):
    """
    Return two sets: (valid_indices, invalid_indices)
    """
    valid, invalid = set(), set()
    for f in out_dir.glob("rx_chunk_*.npz"):
        try:
            ci = int(f.stem.split("_")[-1])
        except Exception:
            continue
        (valid if _is_valid_chunk_npz(f) else invalid).add(ci)
    return valid, invalid

def _verify_and_retry_save(chunk_path: Path, arrs: dict, compressed: bool, max_retries: int = 1):
    """
    After writing, re-open and verify; if bad, delete and retry.
    Adds fsync + backoff to survive read-after-replace lag on network FS.
    """
    import time

    def _save_once():
        if compressed:
            fd, tmpname = tempfile.mkstemp(prefix=f".{chunk_path.stem}.", dir=str(chunk_path.parent))
            os.close(fd)
            tmp = Path(tmpname)
            try:
                np.savez_compressed(tmp, **arrs)
                os.replace(tmp, chunk_path)
            finally:
                try:
                    if tmp.exists(): tmp.unlink()
                except FileNotFoundError:
                    pass
        else:
            # your atomic uncompressed save
            _atomic_save_npz(chunk_path, **arrs)
        # Best-effort flush on Linux; harmless elsewhere
        try:
            os.sync()
        except Exception:
            pass

    tries = 0
    while True:
        _save_once()
        # Backoff: some FS expose the dir entry before the full file is readable.
        time.sleep(2)
        if _is_valid_chunk_npz(chunk_path):
            return
        tries += 1
        try:
            chunk_path.unlink(missing_ok=True)
        except Exception:
            pass
        if tries > max_retries:
            raise RuntimeError(f"Failed to write verified chunk {chunk_path} after {max_retries} retries")


def _rebuild_index_from_disk(out_dir: Path, chunk_size: int):
    repair = os.environ.get("REPAIR_BAD_CHUNKS", "0") == "1"
    idx, bad = [], []
    for f in sorted(out_dir.glob("rx_chunk_*.npz")):
        if not _is_valid_chunk_npz(f):
            bad.append(f); continue
        ci = int(f.stem.split("_")[-1])
        with np.load(f, allow_pickle=False) as z:  # numeric arrays only
            if "idx" in z.files:
                m = int(z["idx"].shape[0])
                for off in range(m):
                    idx.append({"i": int(z["idx"][off]), "chunk": ci, "offset": off})
            else:
                m = int(z["K"].shape[0])
                for off in range(m):
                    idx.append({"i": ci*chunk_size + off, "chunk": ci, "offset": off})
    if bad:
        print(f"[INDEX] warning: {len(bad)} unreadable chunks: {[b.name for b in bad]}")
        if repair:
            for b in bad:
                try: b.unlink()
                except FileNotFoundError: pass
    (out_dir / "meta.index.json").write_text(json.dumps(idx), encoding="utf-8")
    return len(idx)


# ------------------------- CPU chunk -------------------------
def _process_chunk(split, src_dir, out_dir, tau_max, ci, start, end, compressed, mode="fast"):
    src = Path(src_dir); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dset = ShardNPZDataset(src)

    Rx_list, K_list, snr_list, ptr_list, phi_list, theta_list, r_list, idx_list = [], [], [], [], [], [], [], []

    for i in range(start, end):
        it = dset[i]
        y_ri = it["y"].numpy() if hasattr(it["y"], "numpy") else it["y"]  # [L,M,2] float
        if y_ri.ndim == 3 and y_ri.shape[-1] == 2 and not np.iscomplexobj(y_ri):
            y = y_ri[..., 0].astype(np.float32, copy=False) + 1j*y_ri[..., 1].astype(np.float32, copy=False)
        elif np.iscomplexobj(y_ri) and y_ri.ndim == 2:
            y = y_ri
        else:
            raise ValueError(f"Unexpected y shape/dtype: {y_ri.shape}, complex={np.iscomplexobj(y_ri)}")

        y = np.ascontiguousarray(y.astype(np.complex64, copy=False))
        Rx = autocorr_stack_from_y(y, tau_max).astype(np.float32)  # [C,L,L]
        Rx_list.append(Rx)
        K_list.append(int(it["K"]))
        snr_list.append(float(it["snr"]) if "snr" in it else np.nan)
        ptr_list.append(np.asarray(it["ptr"]) if "ptr" in it else np.array([]))
        phi_list.append(np.asarray(it["phi"]) if "phi" in it else np.array([]))
        theta_list.append(np.asarray(it["theta"]) if "theta" in it else np.array([]))
        r_list.append(np.asarray(it["r"]) if "r" in it else np.array([]))
        idx_list.append(i)

    arrs = dict(
        Rx=np.stack(Rx_list, axis=0),
        K=np.asarray(K_list, dtype=np.int32),
        snr=np.asarray(snr_list, dtype=np.float32),
        ptr=np.array(ptr_list, dtype=object),
        phi=np.array(phi_list, dtype=object),
        theta=np.array(theta_list, dtype=object),
        r=np.array(r_list, dtype=object),
        idx=np.asarray(idx_list, dtype=np.int64),
    )
    chunk_path = Path(out_dir) / f"rx_chunk_{ci:05d}.npz"
    _verify_and_retry_save(chunk_path, arrs, compressed)
    return {"ci": ci, "count": len(idx_list)}

# ------------------------- GPU chunk -------------------------
def _process_chunk_gpu(src_dir, out_dir, tau_max, ci, start, end, batch_size, compressed):
    import torch as _torch
    src = Path(src_dir); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dset = ShardNPZDataset(src)

    Rx_list, K_list, snr_list, ptr_list, phi_list, theta_list, r_list, idx_list = [], [], [], [], [], [], [], []

    b = start
    while b < end:
        e = min(b + batch_size, end)

        # collect & convert RI -> complex64 [L, T]
        Ys, rows = [], []
        for i in range(b, e):
            it = dset[i]
            y_ri = it["y"].numpy() if hasattr(it["y"], "numpy") else it["y"]
            if y_ri.ndim == 3 and y_ri.shape[-1] == 2 and not np.iscomplexobj(y_ri):
                y = y_ri[..., 0].astype(np.float32, copy=False) + 1j * y_ri[..., 1].astype(np.float32, copy=False)
            elif np.iscomplexobj(y_ri) and y_ri.ndim == 2:
                y = y_ri
            else:
                raise ValueError(f"Unexpected y shape/dtype: {y_ri.shape}, complex={np.iscomplexobj(y_ri)}")
            y = np.ascontiguousarray(y.astype(np.complex64, copy=False))  # [L, T]
            Ys.append(y); rows.append(it)

        Yb = _torch.from_numpy(np.stack(Ys, axis=0)).to(_torch.complex64, copy=False).cuda()  # [B, L, T]
        if not (Yb.is_complex() and Yb.ndim == 3):
            raise RuntimeError(f"bad Yb shape={tuple(Yb.shape)} dtype={Yb.dtype}")

        Rx_b = autocorr_stack_torch_batch(Yb, tau_max)  # CPU float32
        del Yb
        if _torch.cuda.is_available():
            _torch.cuda.synchronize()
            _torch.cuda.empty_cache()

        B_now = Rx_b.shape[0]
        for j in range(B_now):
            Rx_list.append(Rx_b[j].numpy())
            it = rows[j]
            K_list.append(int(it["K"]))
            snr_list.append(float(it["snr"]) if "snr" in it else np.nan)
            ptr_list.append(np.asarray(it["ptr"]) if "ptr" in it else np.array([]))
            phi_list.append(np.asarray(it["phi"]) if "phi" in it else np.array([]))
            theta_list.append(np.asarray(it["theta"]) if "theta" in it else np.array([]))
            r_list.append(np.asarray(it["r"]) if "r" in it else np.array([]))
            idx_list.append(b + j)

        del Rx_b, Ys, rows
        b = e

    arrs = dict(
        Rx=np.stack(Rx_list, axis=0),
        K=np.asarray(K_list, dtype=np.int32),
        snr=np.asarray(snr_list, dtype=np.float32),
        ptr=np.array(ptr_list, dtype=object),
        phi=np.array(phi_list, dtype=object),
        theta=np.array(theta_list, dtype=object),
        r=np.array(r_list, dtype=object),
        idx=np.asarray(idx_list, dtype=np.int64),
    )
    chunk_path = Path(out_dir) / f"rx_chunk_{ci:05d}.npz"
    _verify_and_retry_save(chunk_path, arrs, compressed)
    return {"ci": ci, "count": len(idx_list)}

# ------------------------- Orchestrator with RESUME -------------------------
def dump_features_chunked(split="train", tau_max=12, out_dir="results_final/baselines/features_dcd_nf",
                          chunk_size=5000, workers=None, compressed=False,
                          device="gpu", gpu_batch=64,
                          start_chunk: int = 0, end_chunk: int | None = None):
    """
    device: 'gpu' or 'cpu'
    gpu_batch: samples per CUDA batch inside each chunk
    start_chunk/end_chunk: optional range to process (inclusive). Useful with SLURM arrays.
    """
    src = Path(f"results_final/data/shards/{split}")
    out = Path(out_dir) / split
    out.mkdir(parents=True, exist_ok=True)

    dset = ShardNPZDataset(src)
    N = len(dset)
    n_chunks = math.ceil(N / chunk_size)
    if workers is None:
        workers = 1

    # normalize range
    first = max(0, int(start_chunk))
    last  = n_chunks - 1 if end_chunk is None else min(n_chunks - 1, int(end_chunk))

    print(f"[INFO] split={split} N={N} tau={tau_max} chunks={n_chunks} "
          f"chunk_size={chunk_size} workers={workers} device={device} compressed={compressed} "
          f"range=[{first},{last}]")

    # avoid BLAS oversubscription when using CPU
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # --- RESUME: detect already-present chunks; skip valid, re-gen invalid
    valid_existing, invalid_existing = _existing_chunks_npz(out)
    if valid_existing or invalid_existing:
        msg = f"[RESUME] valid={len(valid_existing)} invalid={len(invalid_existing)}"
        if valid_existing:
            msg += f"  min={min(valid_existing)} max={max(valid_existing)}"
        print(msg)

    # ---- SINGLE-PROCESS PATH (CUDA-safe) ----
    if int(workers) <= 1:
        for ci in range(first, last + 1):
            s = ci * chunk_size
            e = min((ci + 1) * chunk_size, N)
            if ci in valid_existing:
                print(f"[SKIP] chunk {ci:05d} already valid")
                continue
            # if invalid (truncated) or missing, (re)generate
            if device == "gpu":
                _ = _process_chunk_gpu(str(src), str(out), int(tau_max), ci, s, e, int(gpu_batch), bool(compressed))
            else:
                _ = _process_chunk(split, str(src), str(out), int(tau_max), ci, s, e, bool(compressed), "fast")
            print(f"[OK] [{split}] wrote chunk {ci:05d} ({e - s} samples)")
        # always rebuild index from disk view
        total = _rebuild_index_from_disk(out, int(chunk_size))
        print(f"[INDEX] {split}: {total} rows → {out/'meta.index.json'}")
        return

    # ---- MULTI-PROCESS PATH (not recommended with CUDA) ----
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        futs = []
        for ci in range(first, last + 1):
            if ci in valid_existing:
                print(f"[SKIP] chunk {ci:05d} already valid")
                continue
            s = ci * chunk_size
            e = min((ci + 1) * chunk_size, N)
            if device == "gpu":
                fn = partial(_process_chunk_gpu, str(src), str(out), int(tau_max), ci, s, e, int(gpu_batch), bool(compressed))
            else:
                fn = partial(_process_chunk, split, str(src), str(out), int(tau_max), ci, s, e, bool(compressed), "fast")
            futs.append(ex.submit(fn))
        for f in as_completed(futs):
            res = f.result()
            print(f"[OK] [{split}] wrote chunk {res['ci']:05d} ({res['count']} samples)")

    total = _rebuild_index_from_disk(out, int(chunk_size))
    print(f"[INDEX] {split}: {total} rows → {out/'meta.index.json'}")

# ------------------------- CLI -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("dump_shard_features (chunked, resume-safe, CUDA-ready)")
    ap.add_argument("--splits", nargs="+", default=["train","val"])
    ap.add_argument("--tau", type=int, default=12)
    ap.add_argument("--out", default="results_final/baselines/features_dcd_nf")
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--compressed", action="store_true")
    ap.add_argument("--device", choices=["cpu","gpu"], default="gpu")
    ap.add_argument("--gpu-batch", type=int, default=64)
    ap.add_argument("--start-chunk", type=int, default=0, help="inclusive start chunk index")
    ap.add_argument("--end-chunk", type=int, default=None, help="inclusive end chunk index")
    args = ap.parse_args()
    for split in args.splits:
        dump_features_chunked(split=split, tau_max=args.tau, out_dir=args.out,
                              chunk_size=args.chunk_size, workers=args.workers,
                              compressed=args.compressed, device=args.device, gpu_batch=args.gpu_batch,
                              start_chunk=args.start_chunk, end_chunk=args.end_chunk)
