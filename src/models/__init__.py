"""Regime detection models used in the project.

This subpackage implements three alternative approaches to detecting
risk-on / risk-off regimes: HMM, GMM, and Autoencoder + KMeans.
"""

from .autoencoder import fit_autoencoder, predict_autoencoder
from .gmm import fit_gmm, predict_gmm
from .hmm import decode_hmm, fit_hmm

__all__ = [
    "fit_hmm",
    "decode_hmm",
    "fit_gmm",
    "predict_gmm",
    "fit_autoencoder",
    "predict_autoencoder",
]
