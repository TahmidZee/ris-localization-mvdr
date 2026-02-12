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

        # ── data paths ──
        # v2 is OFDM-wideband first. Narrowband paths are fallback only.
        self.DATA_SHARDS_DIR = base.DATA_SHARDS_DIR
        self.DATA_SHARDS_TRAIN = base.DATA_SHARDS_TRAIN
        self.DATA_SHARDS_VAL = base.DATA_SHARDS_VAL
        self.DATA_SHARDS_TEST = base.DATA_SHARDS_TEST
        self.DATA_SHARDS_WIDEBAND_DIR = "data_shards_ofdm_tr38901"
        self.DATA_SHARDS_WIDEBAND_TRAIN = f"{self.DATA_SHARDS_WIDEBAND_DIR}/train"
        self.DATA_SHARDS_WIDEBAND_VAL = f"{self.DATA_SHARDS_WIDEBAND_DIR}/val"
        self.DATA_SHARDS_WIDEBAND_TEST = f"{self.DATA_SHARDS_WIDEBAND_DIR}/test"
        self.REQUIRE_WIDEBAND_DATA = True
        self.ALLOW_NARROWBAND_FALLBACK = False
        self.NUM_WORKERS = 0
        self.PIN_MEMORY = True

        # ── v2-specific results ──
        self.RESULTS_DIR = "results_v2"
        self.LOGS_DIR = f"{self.RESULTS_DIR}/logs"
        self.CKPT_DIR = f"{self.RESULTS_DIR}/checkpoints"

        # ── wideband knobs (primary path) ──
        self.F_SUBCARRIERS = 16         # Start at F=16 (wideband minimal), scale to 64+
        self.USE_WIDEBAND_INPUT = True
        self.CARRIER_HZ = 3.5e9         # 3.5 GHz FR1
        self.SUBCARRIER_SPACING_HZ = 30e3  # 30 kHz (3GPP NR)
        self.D_TAPS = 8                 # Tap-domain channel taps (indoor TR 38.901 style)
        self.OFDM_FFT_SIZE = 2048
        self.OFDM_ACTIVE_SC = 1596      # 50 MHz at 30 kHz SCS (NRB~133)
        self.PILOT_SUBCARRIERS = 256
        self.BW_HZ = float(self.OFDM_ACTIVE_SC) * float(self.SUBCARRIER_SPACING_HZ)
        self.WIDEBAND_Y_KEY = "y"
        self.WIDEBAND_H_TAPS_KEY = "H_taps_ri"
        self.WIDEBAND_R_F_KEY = "R_f"
        self.WIDEBAND_R_F_ALT_KEYS = ("R_f_true", "R_f", "Rf")
        self.REQUIRE_R_F_SUPERVISION = False

        # ── covariance / inference (inherited) ──
        self.C_EPS = getattr(base, "C_EPS", 1.0)
        self.HYBRID_COV_BLEND = False     # v2: no hybrid blend needed (direct NMSE)
        self.HYBRID_COV_BETA = 0.0
        self.APPLY_EFFECTIVE_COV_IN_LOSS = False

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
        self.PRIMARY_PLAN_DOC = "OFDM_TR38901_INDOOR_PLAN.md"
        self.V2_EXEC_PLAN_DOC = "V2_PHYSICS_FIRST_WIDEBAND_EXECUTION_PLAN_20260212.md"


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
        self.USE_TONE_FACTOR_HEAD = True
        self.USE_OPERATOR_TAP_CONDITIONING = True

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
        self.WIDEBAND_START_F = 16
        self.WIDEBAND_TARGET_F = 64

        # ── loss weights ──
        self.LAM_COV_MAIN = 1.0
        self.LAM_COV_F = 0.5
        self.LAM_COV_CONSIST = 0.1
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
