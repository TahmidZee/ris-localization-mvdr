# SPDX-License-Identifier: MIT
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BENCH_DIR = Path("results_final/benches")
FIG_DIR   = Path("results_final/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["Hybrid", "Ramezani-MOD-MUSIC", "DCD-MUSIC", "NF-SubspaceNet", "Decoupled-MOD-MUSIC"]
COLORS = {m:None for m in METHOD_ORDER}  # let mpl pick defaults


def _load(csvname):
    p = BENCH_DIR / csvname
    if not p.exists():
        print(f"Missing {p}")
        return None
    return pd.read_csv(p)


def _median_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan, np.nan
    med = np.median(x)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return med, iqr


def plot_overall_box(tag="B1_all_blind", metric="phi"):
    df = _load(f"{tag}.csv"); 
    if df is None: return
    plt.figure()
    data = [df[df["who"]==m][metric].values for m in METHOD_ORDER if m in df["who"].unique()]
    labels = [m for m in METHOD_ORDER if m in df["who"].unique()]
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel(f"RMSE {metric}"); plt.title(f"{tag}: overall")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG_DIR/f"{tag}_{metric}_box.png"); plt.close()


def plot_snr_sweep_k2(edges=(-10,-5,0,5,10,15,20,25), metric="phi"):
    # expects per-bin CSVs created by B4K2
    plt.figure()
    centers=[]
    for lo,hi in zip(edges[:-1], edges[1:]):
        tag=f"B4K2_SNR_{int(lo)}_{int(hi)}_blind"
        df=_load(f"{tag}.csv")
        if df is None: continue
        centers.append((lo+hi)/2)
        # one point per method (median with errorbar=IQR)
        for m in [mm for mm in METHOD_ORDER if mm in df["who"].unique()]:
            med,iqr=_median_iqr(df[df["who"]==m][metric])
            plt.errorbar([centers[-1]], [med], yerr=[[iqr/2],[iqr/2]], fmt="o", label=m if lo==edges[0] else None)
    plt.xlabel("SNR (dB)"); plt.ylabel(f"RMSE {metric}"); plt.title("K=2: RMSE vs SNR (median±IQR)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR/"B4K2_snr_sweep.png"); plt.close()


def plot_rmse_vs_K_at_snr(snr=15, metric="phi"):
    # expects B8_SNR{snr}_K{k}_blind.csv
    plt.figure()
    Ks=[]; lines={}
    for k in range(1, 16):  # up to K=15; adjust if needed
        tag=f"B8_SNR{int(round(snr))}_K{k}_blind"
        df=_load(f"{tag}.csv")
        if df is None: continue
        Ks.append(k)
        for m in [mm for mm in METHOD_ORDER if mm in df["who"].unique()]:
            med,iqr=_median_iqr(df[df["who"]==m][metric])
            lines.setdefault(m, ([],[]))
            lines[m][0].append(k); lines[m][1].append(med)
    for m,(kx,vy) in lines.items():
        plt.plot(kx, vy, marker="o", label=m, color=COLORS.get(m))
    plt.xlabel("K"); plt.ylabel(f"RMSE {metric}"); plt.title(f"RMSE vs K @ SNR≈{snr} dB (median)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR/f"B8_rmse_vs_K_{int(round(snr))}.png"); plt.close()


def plot_heatmap(metric="phi"):
    df=_load("B9_heatmap.csv")
    if df is None: return
    # pivot median RMSE by (K_bin, SNR_bin)
    df2=(df.groupby(["who","K_bin","SNR_bin"])[metric]
           .median().reset_index())
    for m in [mm for mm in METHOD_ORDER if mm in df2["who"].unique()]:
        x_sorted=sorted(df2["SNR_bin"].unique(), key=lambda s: int(s.split(":")[0]))
        y_sorted=sorted(df2["K_bin"].unique())
        P=np.full((len(y_sorted), len(x_sorted)), np.nan, dtype=float)
        for yi,k in enumerate(y_sorted):
            for xi,sb in enumerate(x_sorted):
                rows=df2[(df2["who"]==m)&(df2["K_bin"]==k)&(df2["SNR_bin"]==sb)]
                if len(rows)==0: continue
                P[yi,xi]=rows.iloc[0][metric]
        plt.figure()
        im=plt.imshow(P, aspect="auto", origin="lower")
        plt.xticks(range(len(x_sorted)), x_sorted, rotation=45)
        plt.yticks(range(len(y_sorted)), y_sorted)
        plt.xlabel("SNR bin (dB)"); plt.ylabel("K")
        plt.title(f"Median RMSE {metric} — {m}")
        plt.colorbar(im)
        plt.tight_layout(); plt.savefig(FIG_DIR/f"B9_heatmap_{metric}_{m}.png"); plt.close()


def make_all_plots():
    # headline boxplots
    for metric in ("phi","theta","rng"):
        plot_overall_box("B1_all_blind", metric)
        plot_overall_box("B2_all_oracle", metric)
    # K=2 SNR sweep
    plot_snr_sweep_k2(metric="phi")
    # RMSE vs K @ 15 dB
    plot_rmse_vs_K_at_snr(15, metric="phi")
    # Heatmaps
    for metric in ("phi","theta","rng"):
        plot_heatmap(metric)
    print("Saved figures to", FIG_DIR)
