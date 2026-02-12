"""
V2 physics-first wideband pipeline package.

Main entrypoints:
- config: `v2_cfg`, `v2_mdl`
- model: `CovariancePredictor`
- loss: `V2CovarianceLoss`
- trainer: `V2Trainer`
"""

from .config_v2 import v2_cfg, v2_mdl
from .model_v2 import CovariancePredictor
from .loss_v2 import V2CovarianceLoss

__all__ = [
    "v2_cfg",
    "v2_mdl",
    "CovariancePredictor",
    "V2CovarianceLoss",
]
