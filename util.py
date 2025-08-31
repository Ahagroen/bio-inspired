from dataclasses import dataclass
import numpy as np


@dataclass
class State:
    azimuth: float
    elevation: float


def weighted_quadratic_regression(x, z, w):
    w = np.array(w)
    targets = z
    X = np.column_stack([x**2, x, np.ones_like(x)])
    W = np.diag(1.0 / (np.sqrt(w) + 1e-6))
    coeffs = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ targets)
    return coeffs


def predict(x_new, coeffs):
    x_new = np.atleast_1d(x_new)
    X_new = np.column_stack([x_new**2, x_new, np.ones_like(x_new)])
    return X_new @ coeffs


def position_error(pred: State, true):
    return (pred.elevation - true[1]) ** 2


def generate_parabola(x1, E_max: float = 90.0):
    if not (0 <= x1 <= 180):
        raise ValueError("x1 must be in range [0, 360]")
    if not (0 < E_max <= 90):
        raise ValueError("E_max must be in (0, 90]")
    denom = (x1 / 2) * (x1 / 2 - x1)
    if denom == 0:
        raise ValueError("Invalid x1: parabola degenerates")

    # Choose 'a' so that the peak elevation = E_max
    a = -E_max / abs(denom)

    def parabola(x):
        return a * (x) * (x - x1)

    return parabola
