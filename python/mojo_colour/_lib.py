"""ctypes bridge to the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "dist" / "libmojo-colour-science.so"
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mcs_srgb_to_xyz": ([I, I, I, I], None),
    "mcs_xyz_to_srgb": ([I, I, I, I], None),
    "mcs_srgb_transfer": ([I, I, I, I], None),
    "mcs_xyz_to_lab": ([I, I, I, F, F, F], None),
    "mcs_lab_to_xyz": ([I, I, I, F, F, F], None),
    "mcs_xyz_to_luv": ([I, I, I, F, F, F], None),
    "mcs_luv_to_xyz": ([I, I, I, F, F, F], None),
    "mcs_xyz_xyy": ([I, I, I, I, F, F], None),
    "mcs_perceptual": ([I, I, I, I], None),
    "mcs_jzazbz": ([I, I, I, I], None),
    "mcs_jzazbz_gpu": ([I, I, I, I], I),
    "mcs_delta_e": ([I, I, I, I, I, I], None),
    "mcs_appearance_forward": ([I, I, I, I] + [F] * 7, None),
    "mcs_appearance_inverse": ([I, I, I, I] + [F] * 7, None),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not LIBRARY.exists():
            raise RuntimeError(
                f"{LIBRARY} does not exist; run `pixi run build` first"
            )
        _library = ctypes.CDLL(str(LIBRARY))
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(value: Any) -> np.ndarray:
    """Return an owned-or-borrowed C-contiguous float64 array for the FFI."""
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError("complex inputs cannot be converted to real float64 safely")
    return np.ascontiguousarray(array, dtype=np.float64)


def addr(value: np.ndarray) -> int:
    if value.dtype != np.float64 or not value.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous float64 arrays")
    address = int(value.ctypes.data)
    if value.size and address == 0:
        raise RuntimeError("NumPy returned a null address for a non-empty buffer")
    return address
