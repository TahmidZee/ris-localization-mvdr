# SPDX-License-Identifier: MIT
from pathlib import Path
import json, numpy as np, torch
from torch.utils.data import Dataset
from .configs import cfg

def _unpack_ptr(ptr_vec, K, KMAX):
    a = np.asarray(ptr_vec, dtype=np.float32).reshape(-1)
    if a.size < 3*KMAX:
        pad = np.full(3*KMAX, np.nan, np.float32)
        pad[:a.size] = a
        a = pad
    phi   = a[0:KMAX][:K]
    theta = a[KMAX:2*KMAX][:K]
    r     = a[2*KMAX:3*KMAX][:K]
    return phi, theta, r

# pad to fixed length so default collate works
KMAX = int(cfg.K_MAX)
def _pad_to_kmax(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size >= KMAX:
        return x[:KMAX]
    out = np.full((KMAX,), np.nan, np.float32)
    out[:x.size] = x
    return out

class RxFeatureDataset(Dataset):
    """
    Loads Rx stacks and labels from results_final/baselines/features_dcd_nf/{split}/
    Supports both:
      - chunked: rx_chunk_*.npz + meta.index.json
      - files:   rx_*.npy + meta.json
    Returns:
      Rx:   float32 [C, L, L]
      K:    int64   []
      i:    int64   []
      phi/theta/r: float32 [K_MAX] (padded with NaN) if use_labels=True
    """
    def __init__(self, split="train", root="results_final/baselines/features_dcd_nf", use_labels=True):
        self.root = Path(root) / split
        self.use_labels = use_labels
        # Prefer chunked if present
        self.chunks = sorted(self.root.glob("rx_chunk_*.npz"))
        if self.chunks:
            # Build flat index from meta.index.json
            meta_idx = json.loads((self.root / "meta.index.json").read_text())
            self.flat = [{"chunk": x["chunk"], "offset": x["offset"]} for x in meta_idx]
            self.mode = "chunked"
        else:
            # fallback to per-sample npy
            self.meta = json.loads((self.root / "meta.json").read_text())
            self.ids = [m["i"] for m in self.meta]
            self.mode = "files"

    def __len__(self):
        return len(self.flat) if self.mode == "chunked" else len(self.ids)

    def __getitem__(self, ix):
        if self.mode == "chunked":
            ci = self.flat[ix]["chunk"]
            off = self.flat[ix]["offset"]
            pack = np.load(self.chunks[ci], allow_pickle=True)
            Rx = pack["Rx"][off]  # [C, L, L], float32 expected
            K  = int(pack["K"][off])

            rec = {
                "Rx": torch.from_numpy(np.asarray(Rx, dtype=np.float32)),
                "K": torch.tensor(K, dtype=torch.int64),
                "i": torch.tensor(int(pack["idx"][off]), dtype=torch.int64),
            }

            if self.use_labels:
                # prefer explicit phi/theta/r; else decode from ptr
                has_phi = ("phi" in pack) and (pack["phi"].dtype == object) and (len(pack["phi"][off]) > 0)
                if has_phi:
                    phi = np.asarray(pack["phi"][off], dtype=np.float32)
                    th  = np.asarray(pack["theta"][off], dtype=np.float32)
                    rr  = np.asarray(pack["r"][off], dtype=np.float32)
                else:
                    has_ptr = ("ptr" in pack) and (pack["ptr"].dtype == object) and (len(pack["ptr"][off]) > 0)
                    if has_ptr:
                        phi, th, rr = _unpack_ptr(pack["ptr"][off], K, KMAX)
                    else:
                        phi = th = rr = np.empty((0,), np.float32)

                # pad to K_MAX for batch collation
                phi_p = _pad_to_kmax(phi)
                th_p  = _pad_to_kmax(th)
                rr_p  = _pad_to_kmax(rr)

                rec["phi"]   = torch.from_numpy(phi_p)
                rec["theta"] = torch.from_numpy(th_p)
                rec["r"]     = torch.from_numpy(rr_p)

            return rec

        # files mode (per-sample npy + meta.json)
        i = self.ids[ix]
        Rx = np.load(self.root / f"rx_{i:07d}.npy")
        m  = self.meta[ix]
        K  = int(m["K"])

        rec = {
            "Rx": torch.from_numpy(np.asarray(Rx, dtype=np.float32)),
            "K": torch.tensor(K, dtype=torch.int64),
            "i": torch.tensor(int(i), dtype=torch.int64),
        }

        if self.use_labels:
            if m.get("phi") is not None:
                phi = np.asarray(m["phi"], dtype=np.float32)
                th  = np.asarray(m["theta"], dtype=np.float32)
                rr  = np.asarray(m["r"], dtype=np.float32)
            else:
                phi, th, rr = _unpack_ptr(m["ptr"], K, KMAX)

            phi_p = _pad_to_kmax(phi)
            th_p  = _pad_to_kmax(th)
            rr_p  = _pad_to_kmax(rr)

            rec["phi"]   = torch.from_numpy(phi_p)
            rec["theta"] = torch.from_numpy(th_p)
            rec["r"]     = torch.from_numpy(rr_p)

        return rec
