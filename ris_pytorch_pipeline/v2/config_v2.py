"""
V2 configuration — extends v1 SysConfig / ModelConfig with v2-specific knobs.

Usage:
    from ris_pytorch_pipeline.v2.config_v2 import v2_cfg, v2_mdl
"""

from __future__ import annotations
from ..configs import cfg as _v1_cfg, mdl_cfg as _v1_mdl


class V2SysConfig:
    """System-level config for v2 physics-first pipeline."""

    def __init__(self, base: object = None):
        base = base or _v1_cfg

        # ── inherit geometry from v1 ──
        self.M = base.M                  # BS antennas (16)
        self.M_BS = base.M_BS
        self.N_H = base.N_H              # RIS horizontal (12)
        self.N_V = base.N_V              # RIS vertical (12)
        self.N = base.N                  # RIS elements (144)
        self.L = base.L                  # Temporal snapshots (16)
        self.K_MAX = base.K_MAX          # Max sources (5)
        self.WAVEL = base.WAVEL
        self.k0 = base.k0
        self.d_H = base.d_H
        self.d_V = base.d_V
        self.ANGLE_RANGE_PHI = base.ANGLE_RANGE_PHI
        self.ANGLE_RANGE_THETA = base.ANGLE_RANGE_THETA
        self.RANGE_R = base.RANGE_R
        self.SNR_DB_RANGE = base.SNR_DB_RANGE

        # ── data paths (reuse v1 narrowband for Phase 1) ──
        self.DATA_SHARDS_DIR = base.DATA_SHARDS_DIR
        self.DATA_SHARDS_TRAIN = base.DATA_SHARDS_TRAIN
        self.DATA_SHARDS_VAL = base.DATA_SHARDS_VAL
        self.DATA_SHARDS_TEST = base.DATA_SHARDS_TEST
        self.NUM_WORKERS = 0
        self.PIN_MEMORY = True

        # ── v2-specific results ──
        self.RESULTS_DIR = "results_v2"
        self.LOGS_DIR = f"{self.RESULTS_DIR}/logs"
        self.CKPT_DIR = f"{self.RESULTS_DIR}/checkpoints"

        # ── wideband knobs (Phase 2+) ──
        self.F_SUBCARRIERS = 1          # Start narrowband; set > 1 for wideband
        self.USE_WIDEBAND_INPUT = False
        self.CARRIER_HZ = 3.5e9         # 3.5 GHz for wideband phases
        self.SUBCARRIER_SPACING_HZ = 30e3  # 30 kHz (3GPP NR)
        self.D_TAPS = 4                 # Tap-domain channel taps (indoor)
        self.OFDM_FFT_SIZE = 256
        self.OFDM_ACTIVE_SC = 200

        # ── covariance / inference (inherited) ──
        self.C_EPS = getattr(base, "C_EPS", 1.0)
        self.HYBRID_COV_BLEND = False     # v2: no hybrid blend needed (direct NMSE)
        self.HYBRID_COV_BETA = 0.0

        # ── MUSIC / MVDR (inherited as-is) ──
        for attr in [
            "MUSIC_COARSE", "MUSIC_USE_FBA", "MUSIC_USE_ADAPTIVE_SHRINK",
            "MUSIC_PEAK_REFINE", "USE_NEWTON_REFINE", "MUSIC_GRID_PHI",
            "MUSIC_GRID_THETA", "NEWTON_ITERS", "NEWTON_STEP",
            "NEWTON_MIN_SEP", "NEWTON_NEARFIELD", "RANGE_MUSIC_NF",
            "RANGE_GRID_STEPS", "RANGE_PRIOR_SPAN", "R_MIN", "R_MAX",
            "USE_JOINT_NEWTON", "JOINT_NEWTON_ITERS", "JOINT_NEWTON_STEP",
            "MVDR_GRID_PHI", "MVDR_GRID_THETA", "MVDR_THRESH_MODE",
            "MVDR_THRESH_DB", "MVDR_CFAR_Z",
        ]:
            setattr(self, attr, getattr(base, attr, None))

        # ── codebook (inherited) ──
        self.M_beams = getattr(base, "M_beams", getattr(base, "M_BEAMS_TARGET", 64))
        self.RIS_2D_DFT_COLS = getattr(base, "RIS_2D_DFT_COLS", None)

        # ── diagnostics ──
        self.COV_SANITY_CHECK = True


class V2ModelConfig:
    """Model-level config for v2 CovariancePredictor."""

    def __init__(self, base: object = None):
        base = base or _v1_mdl

        # ── architecture ──
        self.D_MODEL = 512
        self.NUM_HEADS = 8
        self.FF_DIM = 512 * 4           # 4× D_MODEL
        self.N_LAYERS = 4
        self.NUM_LAYERS = 4             # alias for consistency
        self.DROPOUT = 0.1              # lighter than v1 (0.20)
        self.FACTOR_RANK = 10           # 2 × K_MAX

        # ── training ──
        self.BATCH_SIZE = 64
        self.EPOCHS = 60
        self.LR_BACKBONE = 3e-4
        self.HEAD_LR_MULT = 4.0        # factor head gets 4× backbone LR
        self.LR_MIN = 1e-6
        self.CLIP_NORM = 5.0
        self.WEIGHT_DECAY = 1e-4
        self.WARMUP_FRAC = 0.10
        self.USE_AMP = True
        self.SEED = 42
        self.LOG_EVERY = 20

        # ── loss weights ──
        self.LAM_SUBSPACE = 0.0         # off by default (Phase 1a)
        self.LAM_PEAK = 0.0             # off by default (Phase 1a)

        # ── PSD factor construction ──
        self.EPS_PSD = 1e-4             # diagonal loading in A Aᴴ + ε I

        # ── overfit test ──
        self.OVERFIT_N = 200
        self.OVERFIT_EPOCHS = 200
        self.OVERFIT_LR = 1e-3
        self.OVERFIT_HEAD_LR_MULT = 4.0


# ── singletons ──
v2_cfg = V2SysConfig()
v2_mdl = V2ModelConfig()
