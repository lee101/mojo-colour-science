"""Upstream-compatible colour model conversion functions."""

from __future__ import annotations

import warnings

import numpy as np

from ._lib import addr, f64, lib

D65 = np.array([0.3127, 0.3290])


def _triples(value, name: str) -> tuple[np.ndarray, tuple[int, ...], int]:
    array = f64(value)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have a final dimension of size 3")
    return array, array.shape, array.size // 3


def _white_xyz(illuminant) -> tuple[float, float, float]:
    value = f64(illuminant).reshape(-1)
    if value.size not in (2, 3):
        raise ValueError("illuminant must be CIE xy or xyY coordinates")
    x, y = float(value[0]), float(value[1])
    luminance = float(value[2]) if value.size == 3 else 1.0
    if y == 0:
        raise ValueError("illuminant y chromaticity cannot be zero")
    return x * luminance / y, luminance, (1 - x - y) * luminance / y


def _same_shape_kernel(value, name: str, function: str, *arguments):
    source, shape, count = _triples(value, name)
    destination = np.empty_like(source)
    if count == 0:
        return destination.reshape(shape)
    getattr(lib(), function)(addr(source), addr(destination), count, *arguments)
    return destination.reshape(shape)


def sRGB_to_XYZ(
    RGB,
    illuminant=D65,
    chromatic_adaptation_transform="CAT02",
    apply_cctf_decoding=True,
):
    """Convert sRGB values to D65 CIE XYZ."""
    if not np.allclose(illuminant, D65):
        raise NotImplementedError("sRGB chromatic adaptation is not covered")
    if chromatic_adaptation_transform not in ("CAT02", None):
        raise NotImplementedError("only CAT02 or no adaptation is covered")
    return _same_shape_kernel(
        RGB, "RGB", "mcs_srgb_to_xyz", int(apply_cctf_decoding)
    )


def XYZ_to_sRGB(
    XYZ,
    illuminant=D65,
    chromatic_adaptation_transform="CAT02",
    apply_cctf_encoding=True,
):
    """Convert D65 CIE XYZ values to sRGB."""
    if not np.allclose(illuminant, D65):
        raise NotImplementedError("sRGB chromatic adaptation is not covered")
    if chromatic_adaptation_transform not in ("CAT02", None):
        raise NotImplementedError("only CAT02 or no adaptation is covered")
    return _same_shape_kernel(
        XYZ, "XYZ", "mcs_xyz_to_srgb", int(apply_cctf_encoding)
    )


def cctf_decoding_sRGB(value):
    source = f64(value)
    destination = np.empty_like(source)
    if source.size:
        lib().mcs_srgb_transfer(addr(source), addr(destination), source.size, 0)
    return destination


def cctf_encoding_sRGB(value):
    source = f64(value)
    destination = np.empty_like(source)
    if source.size:
        lib().mcs_srgb_transfer(addr(source), addr(destination), source.size, 1)
    return destination


def cctf_decoding(value, function="sRGB", **kwargs):
    if str(function).lower() != "srgb":
        raise NotImplementedError("only the sRGB decoding function is covered")
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {', '.join(kwargs)}")
    return cctf_decoding_sRGB(value)


def cctf_encoding(value, function="sRGB", **kwargs):
    if str(function).lower() != "srgb":
        raise NotImplementedError("only the sRGB encoding function is covered")
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {', '.join(kwargs)}")
    return cctf_encoding_sRGB(value)


def XYZ_to_Lab(XYZ, illuminant=D65):
    return _same_shape_kernel(
        XYZ, "XYZ", "mcs_xyz_to_lab", *_white_xyz(illuminant)
    )


def Lab_to_XYZ(Lab, illuminant=D65):
    return _same_shape_kernel(
        Lab, "Lab", "mcs_lab_to_xyz", *_white_xyz(illuminant)
    )


def XYZ_to_Luv(XYZ, illuminant=D65):
    return _same_shape_kernel(
        XYZ, "XYZ", "mcs_xyz_to_luv", *_white_xyz(illuminant)
    )


def Luv_to_XYZ(Luv, illuminant=D65):
    return _same_shape_kernel(
        Luv, "Luv", "mcs_luv_to_xyz", *_white_xyz(illuminant)
    )


def XYZ_to_xyY(XYZ):
    return _same_shape_kernel(XYZ, "XYZ", "mcs_xyz_xyy", 0, 0.0, 0.0)


def xyY_to_XYZ(xyY):
    return _same_shape_kernel(xyY, "xyY", "mcs_xyz_xyy", 1, 0.0, 0.0)


def XYZ_to_xy(XYZ):
    return XYZ_to_xyY(XYZ)[..., :2]


def xy_to_XYZ(xy):
    value = f64(xy)
    if value.ndim == 0 or value.shape[-1] != 2:
        raise ValueError("xy must have a final dimension of size 2")
    return xyY_to_XYZ(np.concatenate([value, np.ones(value.shape[:-1] + (1,))], -1))


def XYZ_to_Oklab(XYZ):
    return _same_shape_kernel(XYZ, "XYZ", "mcs_perceptual", 0)


def Oklab_to_XYZ(Lab):
    return _same_shape_kernel(Lab, "Lab", "mcs_perceptual", 1)


def XYZ_to_IPT(XYZ):
    return _same_shape_kernel(XYZ, "XYZ", "mcs_perceptual", 2)


def IPT_to_XYZ(IPT):
    return _same_shape_kernel(IPT, "IPT", "mcs_perceptual", 3)


def _validate_jz_constants(constants) -> None:
    if constants is None:
        return
    expected = {
        "b": 1.15,
        "g": 0.66,
        "d": -0.56,
        "d_0": 1.6295499532821565e-11,
        "m_1": 0.1593017578125,
        "m_2": 134.034375,
        "c_1": 0.8359375,
        "c_2": 18.8515625,
        "c_3": 18.6875,
    }
    for key, value in expected.items():
        actual = constants[key] if hasattr(constants, "__getitem__") else getattr(constants, key)
        if not np.isclose(actual, value):
            raise NotImplementedError("only Safdar 2017 Jzazbz constants are covered")


def _jzazbz(value, name, inverse, device):
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    source, shape, count = _triples(value, name)
    destination = np.empty_like(source)
    used_gpu = False
    if device == "gpu":
        used_gpu = bool(
            lib().mcs_jzazbz_gpu(
                addr(source), addr(destination), count, inverse
            )
        ) if count else True
        if not used_gpu:
            warnings.warn(
                "Mojo GPU execution was unavailable or failed; using the CPU kernel",
                RuntimeWarning,
                stacklevel=2,
            )
    if not used_gpu:
        if count:
            lib().mcs_jzazbz(addr(source), addr(destination), count, inverse)
    return destination.reshape(shape)


def XYZ_to_Jzazbz(XYZ_D65, constants=None, *, device="cpu"):
    _validate_jz_constants(constants)
    return _jzazbz(XYZ_D65, "XYZ_D65", 0, device)


def Jzazbz_to_XYZ(Jzazbz, constants=None, *, device="cpu"):
    _validate_jz_constants(constants)
    return _jzazbz(Jzazbz, "Jzazbz", 1, device)
