"""Evaluation helpers for regime detection models.

Functions here compute regime-level statistics, transition matrices and
create plots that help compare and interpret different models.
"""

from .metrics import compute_regime_stats, compute_transition_matrix
from .plots import plot_conditional_stats, plot_regime_timeline

__all__ = [
    "compute_regime_stats",
    "compute_transition_matrix",
    "plot_regime_timeline",
    "plot_conditional_stats",
]
