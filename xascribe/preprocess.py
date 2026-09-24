"""Spectrum preprocessing: model grid, energy registration, missing-point reconstruction,
and single-reference calibration of the oxidation-state scale."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIOR = ROOT / "models" / "nik_missing_point_prior_v1.npz"
E_MODEL = np.linspace(8330.0, 8370.0, 100)   # computed (DFT) energy frame
REGISTRATION_EV = 8.8                        # E_exp = E_model + 8.8 eV for Ni K-edge data calibrated to the Ni foil


def prepare_simulated(I, sigma_ev: float = 2.0):
    """Preprocessing applied to computed spectra before training/prediction."""
    I = np.atleast_2d(I).astype(float)
    I = np.maximum(I - I[:, :3].mean(1, keepdims=True), 0)
    I = I / I.max(1, keepdims=True)
    I = gaussian_filter1d(I, sigma_ev / (E_MODEL[1] - E_MODEL[0]), axis=1)
    return I / I.max(1, keepdims=True)


def read_spectrum(path_or_df) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column spectrum (energy in eV, normalised absorption)."""
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)
    df = df.iloc[:, :2].copy(); df.columns = ["energy", "mu"]
    df = df.dropna().sort_values("energy").groupby("energy", as_index=False).mu.mean()
    return df.energy.to_numpy(float), df.mu.to_numpy(float)


def to_model_grid(energy, mu, registration_ev: float = REGISTRATION_EV):
    """Interpolate a measured spectrum onto the model grid (inside its support only).
    Returns (values with NaN where unobserved, experimental energy axis)."""
    e_exp = E_MODEL + registration_ev
    obs = (e_exp >= energy.min()) & (e_exp <= energy.max())
    y = np.interp(e_exp, energy, mu)
    y[~obs] = np.nan
    return y, e_exp


class MissingPointPrior:
    """Conditional Gaussian for unobserved low-energy points, estimated on simulated spectra."""

    def __init__(self, path=DEFAULT_PRIOR):
        p = dict(np.load(path))
        self.mean, self.cov = p["mean"], p["cov"]
        self.noise, self.context = float(p["noise"]), int(p["context_points"])

    def complete(self, y, draws: int = 2048, seed: int = 0):
        """Return (draws, 100) completed spectra; observed values are never changed.
        Only a missing low-energy prefix is supported (the common case for digitised data)."""
        y = np.asarray(y, float)
        miss = np.isnan(y)
        n = int(miss.sum())
        if n == 0:
            return np.repeat(y[None, :], draws, 0)
        if not np.all(miss[:n]):
            raise ValueError("only a contiguous missing prefix can be reconstructed")
        o = np.arange(n, min(100, n + self.context)); m = np.arange(n)
        K = self.cov[np.ix_(o, o)] + np.eye(len(o)) * self.noise ** 2
        W = cho_solve(cho_factor(K, lower=True), self.cov[np.ix_(o, m)]).T
        mu = self.mean[:n] + W @ (y[o] - self.mean[o])
        V = self.cov[np.ix_(m, m)] - W @ self.cov[np.ix_(o, m)]
        ev, U = np.linalg.eigh((V + V.T) / 2); V = (U * np.maximum(ev, 1e-13)) @ U.T
        Z = np.repeat(y[None, :], draws, 0)
        Z[:, :n] = np.maximum(np.random.default_rng(seed).multivariate_normal(mu, V, size=draws, method="eigh"), 0)
        return Z


def calibrate_valence(pred_ox, reference_index: int, reference_valence: float):
    """Single-reference calibration: shift all oxidation-state predictions by the constant that
    makes the reference spectrum (e.g. the pristine electrode) equal its known valence."""
    pred_ox = np.asarray(pred_ox, float)
    delta = reference_valence - pred_ox[reference_index]
    return pred_ox + delta, delta
