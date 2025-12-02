"""
Fixed configs with B8: 2D separable DFT codebook
"""
import math, numpy as np, torch, random
import os

class SysConfig:
    def __init__(self):
        # --- Array / geometry ---
        self.M, self.N_H, self.N_V = 16, 12, 12  # 12x12 UPA
        self.N = self.N_H * self.N_V  # N = 144 elements
        self.L = 16  # L=16 snapshots

        self.WAVEL = 0.3  # 1 GHz frequency 
        self.k0 = 2 * math.pi / self.WAVEL
        self.d_H = self.d_V = 0.5 * self.WAVEL  # λ/2 spacing
        
        # FOV: φ ±60°, θ ±30°
        self.ANGLE_RANGE_PHI = math.pi / 3.0    # ±60° azimuth  
        self.ANGLE_RANGE_THETA = math.pi / 6.0  # ±30° elevation
        
        self.ANGLE_RANGE = self.ANGLE_RANGE_PHI  # Backward compat
        self.RANGE_R = (0.5, 10.0)

        self.P_DB_RANGE = (-30, 20)
        self.NOISE_STD_RANGE = (1e-5, 1e-2)
        self.KFAC_RANGE = (0.5, 10.0)
        self.K_MAX = 5

        # B8: 2D separable DFT codebook (4×4 Kronecker for L=16)
        self._build_2d_dft_codebook()
        
        # Also keep 1D for backward compatibility
        L_MAX_CODEBOOK = 200
        theta = 2*np.pi * np.arange(L_MAX_CODEBOOK)[:, None] * np.arange(self.N)[None, :] / L_MAX_CODEBOOK
        self.RIS_CONFIG_DFT_COLS = np.exp(1j * theta).astype(np.complex64)

        # --- Results / data / HPO paths ---
        self.RESULTS_DIR = "results_final_L16_12x12_fixed"
        self.LOGS_DIR = f"{self.RESULTS_DIR}/logs"
        self.CKPT_DIR = f"{self.RESULTS_DIR}/checkpoints"

        self.DATA_SHARDS_DIR  = f"{self.RESULTS_DIR}/data/shards"
        self.DATA_SHARDS_TRAIN = f"{self.DATA_SHARDS_DIR}/train"
        self.DATA_SHARDS_VAL   = f"{self.DATA_SHARDS_DIR}/val"
        self.DATA_SHARDS_TEST  = f"{self.DATA_SHARDS_DIR}/test"

        self.HPO_DIR = f"{self.RESULTS_DIR}/hpo"
        self.HPO_DB  = f"{self.HPO_DIR}/hpo.db"
        self.HPO_BEST_JSON = f"{self.HPO_DIR}/best.json"

        self.USE_MDL_K_FOR_BLIND = True
        self.DEFAULT_NEWTON_FROM_HPO = True
        self.DEFAULT_RANGE_GRID_FROM_HPO = True

    def _build_2d_dft_codebook(self):
        """
        B8: Build 2D separable DFT codebook using Kronecker product.
        For L=16, use 4×4 spatial frequency grid.
        This balances φ/θ sensitivity unlike 1D flattened DFT.
        """
        # For L=16, use 4×4 grid (16 beams total)
        kH, kV = 4, 4  # Adjust if L changes
        
        # Element positions (centered)
        i_h = np.arange(self.N_H) - (self.N_H - 1) / 2.0  # [N_H]
        i_v = np.arange(self.N_V) - (self.N_V - 1) / 2.0  # [N_V]
        
        # Spatial frequencies
        u = 2 * np.pi * np.arange(kH) / self.N_H  # [kH]
        v = 2 * np.pi * np.arange(kV) / self.N_V  # [kV]
        
        # Build 1D DFT matrices
        Fh = np.exp(1j * np.outer(i_h, u))  # [N_H, kH]
        Fv = np.exp(1j * np.outer(i_v, v))  # [N_V, kV]
        
        # Kronecker product: each of kH×kV beams covers full UPA
        codes_2d = []
        for j in range(kV):
            for i in range(kH):
                # Kronecker product of single columns
                beam = np.kron(Fv[:, j], Fh[:, i])  # [N]
                codes_2d.append(beam)
        
        self.RIS_2D_DFT_COLS = np.stack(codes_2d, axis=0).astype(np.complex64)  # [kH*kV, N]
        
        print(f"✓ Built 2D DFT codebook: {kH}×{kV} = {len(codes_2d)} beams for {self.N_H}×{self.N_V} UPA")


class ModelConfig:
    # global misc
    WARMUP_FRAC = 0.10
    EMA_DECAY = 0.999
    CLIP_NORM = 1.0
    USE_SWA = True
    SWA_START_FRAC = 0.8
    SWA_LR_FACTOR = 0.1

    DH, DV = 3, 3

    def __init__(self):
        # --- model ---
        self.D_MODEL = 384
        self.CNN_FILTERS = 48
        self.NUM_HEADS = 6
        self.FF_DIM = 512
        self.NUM_LAYERS = 4
        self.DROPOUT = 0.10

        # --- training ---
        self.BATCH_SIZE = 64
        self.EPOCHS = 60
        self.LR_INIT = 3e-4
        self.PATIENCE = 15
        self.AMP = True
        self.OPT = "adamw"
        self.WEIGHT_DECAY = 1e-4
        self.SEED = 42

        # --- inference / loss knobs ---
        self.PHASE_BITS = 3
        self.ETA_PERTURB = 0.05
        self.INFERENCE_GRID_SIZE_COARSE = 61
        self.INFERENCE_GRID_SIZE_RANGE = 201  # Dense range grid
        self.NEWTON_ITER = 3
        self.NEWTON_LR = 0.075
        self.SHRINK_BASE_ALPHA = 1e-3
        self.DELTA_GAP = 0.10
        
        # --- robust K estimation ---
        self.K_CONF_THRESH = 0.75
        self.EPS_PSD = 1e-4
        
        # --- softmax temperature annealing ---
        self.SOFTMAX_TAU = 0.15
        
        # --- subspace alignment loss ---
        self.LAM_ALIGN = 0.002
        self.ALIGN_ON_PRED = True
        
        # --- AntiDiagPool feature extraction ---
        self.USE_ANTIDIAG_POOL = True
        
        # --- CUDNN policy ---
        self.DETERMINISTIC_TRAINING = True

        # --- DataLoader ---
        self.NUM_WORKERS = 0
        self.PREFETCH = 2
        self.PIN_MEMORY = True
        self.PERSISTENT_WORKERS = False

        # --- GPU-caching mode ---
        self.TRAIN_USE_GPU_CACHE = True
        self.GPU_CACHE_BUILD_WORKERS = 4
        self.GPU_CACHE_BUILD_BATCH   = 512

        # --- robust-track sampling defaults ---
        self.TRAIN_PHI_FOV_DEG   = 120.0
        self.TRAIN_THETA_FOV_DEG = 60.0
        self.TRAIN_R_MIN_MAX     = (0.5, 10.0)
        self.P_DB_RANGE          = (-10.0, 10.0)
        self.NOISE_STD_RANGE     = (1.5e-3, 7e-3)
        self.GRID_OFFSET_ENABLED = False
        self.GRID_OFFSET_FRAC    = 0.5
        self.DR_ENABLED          = False
        self.DR_PHASE_SIGMA      = 0.04
        self.DR_AMP_JITTER       = 0.05
        self.DR_DROPOUT_P        = 0.03
        self.DR_WAVELEN_JITTER   = 0.002
        self.DR_DSPACING_JITTER  = 0.005


# singletons
cfg = SysConfig()
mdl_cfg = ModelConfig()


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

