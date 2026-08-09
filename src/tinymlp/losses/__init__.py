"""
Loss functions module.
"""

from tinymlp.losses.mse import mse_loss
from tinymlp.losses.cross_entropy import binary_cross_entropy

__all__ = ["mse_loss", "binary_cross_entropy"]
