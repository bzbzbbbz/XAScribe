"""CDF neural-network regressors, predictions and gradient attribution (NumPy only).

The released weights (models/xascribe_nik_nmc_v1.npz) contain two single-hidden-layer
tanh networks (index 0: Ni oxidation state, index 1: mean Ni-O distance in Angstrom)
acting on the 100-point cumulative distribution (CDF) of a spectrum on the model grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "xascribe_nik_nmc_v1.npz"
TARGETS = ("Ni oxidation state", "Ni-O bond length (A)")

# Spectral regions on the experimental energy axis (model grid + 8.8 eV registration).
REGIONS_EXP = {
    "edge onset": (8338.8, 8344.0),
    "rising edge": (8344.0, 8352.0),
    "white line": (8352.0, 8358.0),
    "post-edge": (8358.0, 8378.8),
}


def load(path=DEFAULT_WEIGHTS) -> dict:
    return dict(np.load(path))


def cdf(X):
    C = np.cumsum(np.maximum(np.atleast_2d(X), 0), axis=1)
    return C / C[:, -1:]


def predict(pars, X):
    """Predictions (n, 2) for spectra X (n, 100) on the model grid."""
    C = cdf(X)
    out = []
    for j in range(2):
        h = np.tanh(((C - pars[f"mean{j}"]) / pars[f"scale{j}"]) @ pars[f"W0_{j}"] + pars[f"b0_{j}"])
        out.append((h @ pars[f"W1_{j}"] + pars[f"b1_{j}"]).ravel() * pars["yscale"][j] + pars["ymean"][j])
    return np.array(out).T


def gradient(pars, X):
    """d prediction / d intensity through the CDF, shape (n, 100, 2)."""
    X = np.atleast_2d(X)
    total = X.sum(1, keepdims=True)
    C = np.cumsum(X, axis=1) / total
    out = []
    for j in range(2):
        H = np.tanh(((C - pars[f"mean{j}"]) / pars[f"scale{j}"]) @ pars[f"W0_{j}"] + pars[f"b0_{j}"])
        g = ((1 - H ** 2) * pars[f"W1_{j}"].ravel()) @ pars[f"W0_{j}"].T / pars[f"scale{j}"] * pars["yscale"][j]
        out.append((np.cumsum(g[:, ::-1], axis=1)[:, ::-1] - (g * C).sum(1, keepdims=True)) / total)
    return np.stack(out, axis=-1)


def integrated_gradients(pars, X, B, nodes: int = 128):
    """Integrated gradients of X relative to reference B (same shape), (n, 100, 2).
    The attributions sum to predict(X) - predict(B)."""
    X = np.atleast_2d(X); B = np.atleast_2d(B)
    D = X - B
    t, w = np.polynomial.legendre.leggauss(nodes)
    A = np.zeros((*X.shape, 2))
    for a, ww in zip((t + 1) / 2, w / 2):
        A += ww * gradient(pars, B + a * D) * D[:, :, None]
    return A


def expected_gradients(pars, X, background, nodes: int = 64):
    """Integrated gradients averaged over background spectra (SHAP-style attribution)."""
    X = np.atleast_2d(X)
    A = np.zeros((*X.shape, 2))
    for b in background:
        A += integrated_gradients(pars, X, np.repeat(b[None, :], len(X), 0), nodes)
    return A / len(background)


def region_sums(attr, energy_exp, regions=REGIONS_EXP):
    """Sum a (100, 2) attribution over named regions of the experimental energy axis."""
    out = {}
    names = list(regions)
    for k, name in enumerate(names):
        lo, hi = regions[name]
        m = (energy_exp >= lo) & ((energy_exp < hi) if k < len(names) - 1 else (energy_exp <= hi + 1e-6))
        out[name] = attr[m].sum(0)
    return out
