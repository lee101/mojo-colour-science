"""CIE colour-difference functions."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib

_METHODS = {
    "cie 1976": 0,
    "cie1976": 0,
    "cie 1994": 1,
    "cie1994": 1,
    "cie 2000": 2,
    "cie2000": 2,
}


def delta_E(a, b, method="CIE 2000", **kwargs):
    """Compute CIE 1976, CIE 1994, or CIE 2000 colour difference."""
    key = str(method).strip().lower()
    if key not in _METHODS:
        raise ValueError(
            "covered methods are 'CIE 1976', 'CIE 1994', and 'CIE 2000'"
        )
    left, right = np.broadcast_arrays(np.asarray(a), np.asarray(b))
    if left.ndim == 0 or left.shape[-1] != 3:
        raise ValueError("a and b must have a final dimension of size 3")
    left = f64(left)
    right = f64(right)
    destination = np.empty(left.shape[:-1], dtype=np.float64)
    textiles = bool(kwargs.pop("textiles", False))
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {', '.join(kwargs)}")
    if destination.size:
        lib().mcs_delta_e(
            addr(left),
            addr(right),
            addr(destination),
            destination.size,
            _METHODS[key],
            int(textiles),
        )
    return destination[()] if destination.ndim == 0 else destination


def delta_E_CIE1976(Lab_1, Lab_2):
    return delta_E(Lab_1, Lab_2, method="CIE 1976")


def delta_E_CIE1994(Lab_1, Lab_2, textiles=False):
    return delta_E(Lab_1, Lab_2, method="CIE 1994", textiles=textiles)


def delta_E_CIE2000(Lab_1, Lab_2, textiles=False):
    return delta_E(Lab_1, Lab_2, method="CIE 2000", textiles=textiles)
