import math, numpy as np, torch, random
import os

class SysConfig:
    def __init__(self):
        # --- Array / geometry ---
        # TERMINOLOGY (critical for paper!):
        # - M or M_BS: Base station antennas (hardware, measurements per snapshot)
        # - M_beams: RIS codebook size (spatial beams, set via M_BEAMS_TARGET below)
        # - M_cov or N: Covariance dimension for MDL/AIC (= RIS elements = 144)
        # - L: Temporal snapshots (time budget)
        
        self.M, self.N_H, self.N_V = 16, 12, 12  # M=M_BS (BS antennas), 12x12 UPA
        self.M_BS = self.M  # Alias for clarity in paper
        self.N = self.N_H * self.N_V  # N = M_cov = 144 elements
        self.L = 16  # temporal snapshots (L≤16 for "few-snapshot" story)
        
        # M_beams: RIS codebook size (spatial beams in 2D DFT)
        # For excellent θ: set to 64 (8×8 balanced)
        # For sub-1° θ: set to 32 (4×8 vertical-tilted)
        # For baseline: set to 16 (4×4 balanced)
        self.M_BEAMS_TARGET = 64  # Options: 16, 24, 30, 32, 36, 48, 64

        # For 1 GHz: wavelength = c/f = 3e8/1e9 = 0.3 m
        self.WAVEL = 0.3  # 1 GHz frequency 
        self.k0 = 2 * math.pi / self.WAVEL
        self.d_H = self.d_V = 0.5 * self.WAVEL  # λ/2 spacing
        # FOV: φ ±60°, θ ±30° for realistic wall-mounted panel (train=inference)
        self.ANGLE_RANGE_PHI = math.pi / 3.0    # ±60° azimuth  
        self.ANGLE_RANGE_THETA = math.pi / 6.0  # ±30° elevation
        
        # FOV clamps (degrees) for evaluation & MUSIC grid
        self.PHI_MIN_DEG = -60.0
        self.PHI_MAX_DEG = 60.0
        self.THETA_MIN_DEG = -30.0
        self.THETA_MAX_DEG = 30.0
        
        # Backward compatibility fallback (deprecated)
        self.ANGLE_RANGE = self.ANGLE_RANGE_PHI  # Use azimuth range as default
        self.RANGE_R = (0.5, 10.0)  # Extended range 0.5-10m as requested

        # --- Simulation ranges (pristine defaults; robust via CLI overrides in pregen) ---
        # Legacy knobs (still used if you disable targeted SNR)
        self.P_DB_RANGE = (-30, 20)
        self.NOISE_STD_RANGE = (1e-5, 1e-2)
        self.KFAC_RANGE = (0.5, 10.0)
        self.K_MAX = 5
        
        # New: target SNR policy (fixes SNR distribution issue)
        self.SNR_DB_RANGE = (-5.0, 20.0)
        
        # --- Low-SNR Rescue Kit Configuration ---
        # SNR-adaptive shrinkage + loading
        self.SNR_THRESHOLD = 3.0  # dB: threshold for SNR gating
        self.SNR_GATE_WIDTH = 3.0  # dB: sigmoid gate width
        self.ALPHA_MIN = 0.02  # Minimum shrinkage
        self.ALPHA_MAX = 0.15  # Maximum shrinkage
        self.K_ALPHA = 0.3  # Shrinkage scaling factor
        self.C_EPS = 1.0  # Diagonal loading scaling
        
        # Soft projector (eigen-weighted MUSIC)
        self.USE_SOFT_PROJECTOR = True  # Enable soft projector at low SNR
        self.SOFT_PROJECTOR_SNR_THR = 0.0  # dB: threshold for soft projector
        self.SOFT_PROJECTOR_GATE_WIDTH = 3.0  # dB: gate width
        self.SOFT_PROJECTOR_P = 0.7  # Power in eigen weights
        self.SOFT_PROJECTOR_TAU = 1e-6  # Regularization in eigen weights
        
        # Training-inference alignment (ENABLED with schedule)
        # Loss weight schedule: warm-up → main → final
        # Warm-up (first 2-3 epochs): lower structure terms to prevent early peaky gradients
        self.LAM_SUBSPACE_ALIGN_WARMUP = 0.02
        self.LAM_PEAK_CONTRAST_WARMUP = 0.05
        # Main phase (most of training)
        self.LAM_SUBSPACE_ALIGN = 0.05  # ENABLED: Subspace alignment loss weight
        self.LAM_PEAK_CONTRAST = 0.1   # ENABLED: Peak contrast loss weight
        # Final phase (last 10-20% of epochs): bump structure slightly
        self.LAM_SUBSPACE_ALIGN_FINAL = 0.07
        self.LAM_PEAK_CONTRAST_FINAL = 0.12
        
        self.LAM_COV_PRED = 0.05       # Auxiliary NMSE on R_pred (5% of lam_cov=1.0, prevents hiding)
        
        # Beta jitter: tighter during HPO, wider during full training
        self.BETA_JITTER_HPO = 0.02    # Tighter for HPO (less variance in objective)
        self.BETA_JITTER_FULL = 0.05   # Wider for full training (more robustness)
        
        # NF-MLE polish (short MLE refinement with monotonic ML acceptance)
        self.USE_NF_MLE_POLISH = True  # Enable NF-MLE polish after Newton
        self.NF_MLE_SNR_THRESHOLD = 8.0  # dB: only run MLE at low/medium SNR (HPO: 4-10 dB)
        self.NF_MLE_ITERS = 3  # Number of MLE iterations (HPO: 2,3,4)
        self.NF_MLE_STEP_SIZE = 0.7  # Step size scaling for MLE gradient descent
        self.NF_MLE_TIKHONOV_LAMBDA = 1e-6  # Tikhonov regularization for multi-source amplitude solve
        self.SNR_TARGETED = True  # <- default ON for pregen

        # --- canonical DFT codebook (L first rows) ---
        # Generate codebook for max L=200 to support baseline comparison at L=100
        L_MAX_CODEBOOK = 200
        theta = 2*np.pi * np.arange(L_MAX_CODEBOOK)[:, None] * np.arange(self.N)[None, :] / L_MAX_CODEBOOK
        self.RIS_CONFIG_DFT_COLS = np.exp(1j * theta).astype(np.complex64)
        
        # --- 2D separable DFT codebook (RECOMMENDED for training) ---
        self._build_2d_dft_codebook()

        # --- Results / data / HPO paths (L=16 12x12 system) ---
        self.RESULTS_DIR = "results_final_L16_12x12"
        self.LOGS_DIR = f"{self.RESULTS_DIR}/logs"
        self.CKPT_DIR = f"{self.RESULTS_DIR}/checkpoints"

        # shards root for L=16 M_beams=64 data; split folders live here
        self.DATA_SHARDS_DIR = "data_shards_M64_L16"
        self.DATA_SHARDS_TRAIN = f"{self.DATA_SHARDS_DIR}/train"
        self.DATA_SHARDS_VAL   = f"{self.DATA_SHARDS_DIR}/val"
        self.DATA_SHARDS_TEST  = f"{self.DATA_SHARDS_DIR}/test"

        # HPO artifacts
        self.HPO_DIR = f"{self.RESULTS_DIR}/hpo"
        self.HPO_DB  = f"{self.HPO_DIR}/hpo.db"
        self.HPO_BEST_JSON = f"{self.HPO_DIR}/best.json"

        # Inference policy (use HPO knobs for Newton+range grid)
        self.USE_MDL_K_FOR_BLIND = True
        self.DEFAULT_NEWTON_FROM_HPO = True
        self.DEFAULT_RANGE_GRID_FROM_HPO = True
    
    def _build_2d_dft_codebook(self):
        """
        Build 2D separable DFT codebook using Kronecker product.
        
        TERMINOLOGY:
        - M_BS: Base station antennas (hardware, fixed at 16)
        - M_beams: Number of spatial beams in RIS codebook (this value)
        - M_cov or N: Covariance dimension (RIS elements = 144)
        - L: Temporal snapshots (16)
        
        For excellent θ performance, use balanced grid:
        - kH=8, kV=8 → M_beams=64 (recommended for θ<0.5°)
        - kH=4, kV=8 → M_beams=32 (sub-1° θ, θ≈0.7-1.1°)
        - kH=4, kV=4 → M_beams=16 (baseline, θ≈2-3°)
        
        Balanced grid (kH = kV) provides equal spatial frequency resolution
        for both azimuth and elevation, optimal for 3D near-field localization.
        """
        # CONFIGURATION: Set M_beams target
        M_beams_target = getattr(self, 'M_BEAMS_TARGET', 32)  # Default to 32 for sub-1°
        
        if M_beams_target == 16:
            kH, kV = 4, 4  # Baseline
        elif M_beams_target == 24:
            kH, kV = 4, 6  # Moderate
        elif M_beams_target == 32:
            kH, kV = 4, 8  # Recommended (vertical-tilted)
        elif M_beams_target == 30:
            kH, kV = 5, 6  # Balanced
        elif M_beams_target == 36:
            kH, kV = 6, 6  # High-res
        elif M_beams_target == 48:
            kH, kV = 6, 8  # Moderate elevation bias
        elif M_beams_target == 64:
            kH, kV = 8, 8  # Balanced (excellent θ)
        else:
            # Default: approximate square root
            kH = int(np.sqrt(M_beams_target))
            kV = (M_beams_target + kH - 1) // kH  # Ceiling division
        
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
        self.M_beams = len(codes_2d)  # CRITICAL: Set M_beams for model access
        
        # ===  ICC FIX: MUSIC pipeline defaults (SOTA-ready) - MOVED TO SysConfig ===
        # Enable unified angle pipeline: MUSIC → Parabolic → Newton
        self.MUSIC_COARSE = True              # Enable MUSIC coarse scan
        self.MUSIC_USE_FBA = True             # Forward-Backward Averaging
        self.MUSIC_USE_ADAPTIVE_SHRINK = True # Adaptive shrinkage (vs fixed)
        self.MUSIC_PEAK_REFINE = True         # Parabolic sub-grid refinement
        self.USE_NEWTON_REFINE = True         # Newton refinement after MUSIC
        self.MUSIC_GRID_PHI = 361             # ~0.33° azimuth grid (HPO: denser for better signal)
        self.MUSIC_GRID_THETA = 181           # ~0.33° elevation grid (HPO: denser for better signal)
        self.MUSIC_SHRINK = None              # None = adaptive shrinkage (recommended)
        self.NEWTON_ITERS = 10                # Newton iterations
        self.NEWTON_STEP = 0.5                # Newton step size (safe default)
        self.NEWTON_MIN_SEP = 0.5             # Minimum source separation (degrees)
        self.NEWTON_NEARFIELD = True          # ICC SUB-DEGREE UNLOCK: Enabled for sub-degree accuracy
        self.MUSIC_DEBUG = False              # Set True for MUSIC diagnostics
        
        # === PRODUCTION: 1-D Range MUSIC (2× range improvement) ===
        self.RANGE_MUSIC_NF = True            # Enable 1-D near-field range MUSIC
        self.RANGE_GRID_STEPS = 121           # Range grid steps (~0.02-0.05m spacing)
        self.RANGE_PRIOR_SPAN = 0.25          # ±25% window around prior estimate
        self.RANGE_MUSIC_REFINE = True        # Parabolic sub-grid refinement for range
        self.R_MIN = 0.5                      # Minimum range (meters)
        self.R_MAX = 10.0                     # Maximum range (meters)
        
        # === L=16 CRITICAL FIX: Hybrid Covariance Blending ===
        self.HYBRID_COV_BLEND = True          # Enable hybrid covariance blending (R_pred + R_samp)
        self.HYBRID_COV_BETA = 0.30           # Blend weight for sample covariance (0.30 for HPO, 0.40 for final)
        self.HYBRID_COV_DEBUG = True          # Set True for hybrid covariance diagnostics (ONE-TIME PRINT)

        # === MVDR (K-free) inference defaults ===
        # Used by ris_pytorch_pipeline.infer.hybrid_estimate_final (MVDR-first).
        self.MVDR_GRID_PHI = 361
        self.MVDR_GRID_THETA = 181
        self.MVDR_THRESH_DB = 12.0
        self.MVDR_DELTA_SCALE = 1e-2
        self.MVDR_DO_REFINEMENT = True
        # If True and a SpectrumRefiner is loaded, inference will run:
        # MVDR spectrum -> SpectrumRefiner -> peak detection (instead of peak-picking on raw MVDR).
        self.USE_SPECTRUM_REFINER_IN_INFER = False
        # Refiner peak selection threshold (probability map). If None, fallback to top-K peaks.
        self.REFINER_PEAK_THRESH = 0.5
        # Minimum NMS separation in grid cells for refined heatmap peak picking.
        self.REFINER_NMS_MIN_SEP = 2

        # === SpectrumRefiner Stage-2 defaults (Option B) ===
        # Use much coarser grids during training for speed.
        self.REFINER_GRID_PHI = 61
        self.REFINER_GRID_THETA = 41
        # If None/empty, use estimator defaults (log-spaced near-field planes)
        self.REFINER_R_PLANES = None
        
        # === Validation & Checkpointing Strategy ===
        # VAL_PRIMARY options:
        # - "loss": legacy, NMSE-driven (not recommended for production)
        # - "k_loc": composite K̂ + AoA/Range via MUSIC (SLOW, for final eval only)
        # - "surrogate": K̂ accuracy + aux angle/range RMSE (FAST, for training/HPO)
        self.VAL_PRIMARY = "surrogate"        # DEFAULT: Use surrogate metrics for training/HPO
        
        # Whether to run full MUSIC-based metrics inside validation
        # This should be False for training/HPO, True only for offline eval scripts.
        self.USE_MUSIC_METRICS_IN_VAL = False
        
        # Weights for surrogate validation score (when VAL_PRIMARY="surrogate")
        # NOTE: K-head removed. Surrogate score is now based on loss + aux errors only.
        self.SURROGATE_METRIC_WEIGHTS = {
            "w_loss": 1.0,        # Weight for validation loss (minimize)
            "w_aux_ang": 0.01,    # Penalty for aux angle RMSE (deg)
            "w_aux_r": 0.01,      # Penalty for aux range RMSE (m)
        }

        # --- Phase-aware training knobs (3-phase plan) ---
        # TRAIN_PHASE options:
        #   - "geom"   : geometry/covariance warm-up (short)
        #   - "k_only" : K-centric fine-tune (backbone frozen)
        #   - "joint"  : final joint refinement (full model)
        # Default to joint training so the backbone can learn K-correlated features immediately.
        # (k_only is only meaningful as a warm-started fine-tune.)
        self.TRAIN_PHASE = "joint"
        # Freezing behaviour during K-only phase
        # Safer default: if you do run k_only, don’t accidentally freeze the backbone “from scratch”.
        self.FREEZE_BACKBONE_FOR_K_PHASE = False
        self.FREEZE_AUX_IN_K_PHASE = False
        # Optional warm-start checkpoint path (weights-only). If empty, no warm-start is applied.
        self.INIT_CKPT = ""
        # Recommended loss weights per phase (used by Trainer when present)
        self.PHASE_LOSS = {
            # NOTE: K-head removed - using MVDR peak detection instead
            "geom": {
                "lam_cov": 0.5,
                "lam_subspace_align": 0.5,
                "lam_aux": 1.0,
                "lam_peak_contrast": 0.0,
            },
            "joint": {
                "lam_cov": 0.1,
                "lam_subspace_align": 0.5,
                "lam_aux": 1.0,
                "lam_peak_contrast": 0.1,
            },
            # SpectrumRefiner-only stage (Option B): freeze backbone, train heatmap head only
            "refiner": {
                "lam_cov": 0.0,
                "lam_subspace_align": 0.0,
                "lam_aux": 0.0,
                "lam_peak_contrast": 0.0,
            },
        }
        
        # === PHASE 2 CRITICAL FIXES: Joint Newton & Hungarian ===
        self.USE_JOINT_NEWTON = True          # Enable joint {φ,θ,r} Newton refinement (sub-degree polish!)
        self.JOINT_NEWTON_ITERS = 2           # Joint Newton iterations (1-3, already close from MUSIC)
        self.JOINT_NEWTON_STEP = 0.3          # Joint Newton step size (conservative for stability)


class ModelConfig:
    # global misc
    WARMUP_FRAC = 0.10
    EMA_DECAY = 0.999
    CLIP_NORM = 1.0
    USE_SWA = True  # Enable SWA for last 20% epochs to improve generalization
    SWA_START_FRAC = 0.8  # Start SWA at 80% through training (last 20% epochs)
    SWA_LR_FACTOR = 0.1  # SWA learning rate = initial_lr * SWA_LR_FACTOR
    USE_3_PHASE_CURRICULUM = True  # Enable 3-phase curriculum (disable for HPO)

    # geometry helpers if needed
    DH, DV = 3, 3

    def __init__(self):
        # --- model ---
        self.D_MODEL = 512  # Increased for M_beams=64 (more spatial information)
        self.CNN_FILTERS = 48
        self.NUM_HEADS = 8  # Increased for M_beams=64 (more attention capacity)
        self.FF_DIM = 512
        self.NUM_LAYERS = 4
        self.DROPOUT = 0.20  # Increased regularization for M_beams=64

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
        self.INFERENCE_GRID_SIZE_COARSE = 61  # For L=16: stable angle resolution
        self.INFERENCE_GRID_SIZE_RANGE = 201  # Dense range grid (was 101)
        self.NEWTON_ITER = 3              # Reduced for efficiency with 144 elements
        self.NEWTON_LR = 0.075            # Slightly reduced for stability
        self.SHRINK_BASE_ALPHA = 1e-3
        self.DELTA_GAP = 0.10
        
        # --- PSD parameterization ---
        self.EPS_PSD = 1e-4  # diagonal loading for PSD parameterization
        
        # --- head LR multiplier ---
        self.HEAD_LR_MULTIPLIER = 4.0  # 4× LR for heads vs backbone (3-5× range)
        
        # --- softmax temperature annealing ---
        self.SOFTMAX_TAU = 0.15  # initial temperature, will be annealed during training
        
        # --- subspace alignment loss ---
        self.LAM_ALIGN = 0.002  # subspace alignment penalty - keep modest for 12x12 (larger eigengap)
        self.ALIGN_ON_PRED = True  # if True: use shrink(R̂), if False: use R_true (train-test coupling)

        # --- covariance factor blending (match loss/infer) ---
        # R_pred = A_angle A_angle^H + LAM_RANGE_FACTOR * A_range A_range^H
        self.LAM_RANGE_FACTOR = 0.3
        
        # --- AntiDiagPool feature extraction ---
        self.USE_ANTIDIAG_POOL = True  # enable anti-diagonal pooling for covariance features
        
        # --- θ loss emphasis (C12) ---
        self.THETA_LOSS_SCALE = 1.5  # Multiply θ loss by this factor for better elevation

        # --- SpectrumRefiner / heatmap supervision (optional) ---
        # Enable by setting LAM_HEATMAP > 0 and making the model output 'refined_spectrum'.
        self.LAM_HEATMAP = 0.0
        self.HEATMAP_SIGMA_PHI = 2.0
        self.HEATMAP_SIGMA_THETA = 2.0

        # --- Debugging / assertions ---
        # If True, UltimateHybridLoss will raise when covariance outputs do not require grads in TRAIN.
        self.STRICT_GRAD_ASSERTS = False
        
        # --- CUDNN policy ---
        self.DETERMINISTIC_TRAINING = True  # True: reproducible but slower, False: faster but non-deterministic

        # --- DataLoader (CPU path); GPU cache ignores these ---
        self.NUM_WORKERS = 0
        self.PREFETCH = 2
        self.PIN_MEMORY = True
        self.PERSISTENT_WORKERS = False

        # --- GPU-caching mode (recommended) ---
        self.TRAIN_USE_GPU_CACHE = True
        self.GPU_CACHE_BUILD_WORKERS = 4
        self.GPU_CACHE_BUILD_BATCH   = 512

        # --- robust-track sampling defaults (used only during pregen via ris_pipeline) ---
        self.TRAIN_PHI_FOV_DEG   = 120.0  # ±60° to match inference FOV
        self.TRAIN_THETA_FOV_DEG = 60.0   # ±30° to match inference FOV  
        self.TRAIN_R_MIN_MAX     = (0.5, 10.0)  # Match inference range
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
        
        # NOTE: MUSIC pipeline defaults moved to SysConfig (lines 140-162)
        # They belong to cfg, not mdl_cfg, since they control inference behavior


# singletons
cfg = SysConfig()
mdl_cfg = ModelConfig()


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        # CUDNN settings moved to Trainer.__init__ to avoid global conflicts


