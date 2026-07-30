"""CAM16 and CIECAM02 appearance model correlates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._lib import addr, f64, lib


@dataclass
class InductionFactors_CIECAM02:
    F: float
    c: float
    N_c: float


InductionFactors_CAM16 = InductionFactors_CIECAM02
_AVERAGE = InductionFactors_CIECAM02(1.0, 0.69, 1.0)
VIEWING_CONDITIONS_CAM16 = {
    "Average": _AVERAGE,
    "Dim": InductionFactors_CIECAM02(0.9, 0.59, 0.9),
    "Dark": InductionFactors_CIECAM02(0.8, 0.525, 0.8),
}
VIEWING_CONDITIONS_CIECAM02 = VIEWING_CONDITIONS_CAM16.copy()


@dataclass
class CAM_Specification_CAM16:
    J: object = None
    C: object = None
    h: object = None
    s: object = None
    Q: object = None
    M: object = None
    H: object = None
    HC: object = None


@dataclass
class CAM_Specification_CIECAM02(CAM_Specification_CAM16):
    pass


_M16 = np.array(
    [
        [0.401288, 0.650173, -0.051461],
        [-0.250268, 1.204414, 0.045854],
        [-0.002079, 0.048952, 0.953127],
    ]
)
_CAT02 = np.array(
    [
        [0.7328, 0.4296, -0.1624],
        [-0.7036, 1.6975, 0.0061],
        [0.0030, 0.0136, 0.9834],
    ]
)
_HPE = np.array(
    [
        [0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340, 0.04641],
        [0.0, 0.0, 1.0],
    ]
)


def _viewing_setup(XYZ_w, L_A, Y_b, surround, discount_illuminant, model):
    white = f64(XYZ_w)
    if white.shape != (3,):
        raise ValueError("XYZ_w must be one CIE XYZ triple")
    if np.ndim(L_A) or np.ndim(Y_b):
        raise NotImplementedError("array-valued L_A and Y_b are not covered")
    la, yb, yw = float(L_A), float(Y_b), float(white[1])
    if not np.all(np.isfinite(white)) or not np.isfinite([la, yb]).all():
        raise ValueError("viewing conditions must be finite")
    if la < 0 or yb <= 0 or yw <= 0:
        raise ValueError("L_A must be non-negative and Y_b and XYZ_w[1] positive")
    k = 1.0 / (5.0 * la + 1.0)
    k4 = k**4
    fl = 0.2 * k4 * (5.0 * la) + 0.1 * (1.0 - k4) ** 2 * (5.0 * la) ** (1 / 3)
    background_n = yb / yw
    nbb = 0.725 * background_n ** -0.2
    zbase = 1.48 + np.sqrt(background_n)
    degree = 1.0 if discount_illuminant else surround.F * (
        1.0 - np.exp((-la - 42.0) / 92.0) / 3.6
    )
    base = _M16 if model == "cam16" else _CAT02
    if model == "cam16":
        degree = np.clip(degree, 0.0, 1.0)
    white_rgb = base @ white
    diagonal = degree * yw / white_rgb + 1.0 - degree
    if model == "cam16":
        matrix = np.diag(diagonal) @ _M16
    else:
        matrix = _HPE @ np.linalg.inv(_CAT02) @ np.diag(diagonal) @ _CAT02
    rgb_aw = matrix @ white
    compressed = _compress(rgb_aw, fl)
    aw = (2 * compressed[0] + compressed[1] + compressed[2] / 20 - 0.305) * nbb
    return f64(matrix), fl, background_n, nbb, zbase, float(aw)


def _compress(rgb, fl):
    magnitude = (fl * np.abs(rgb) / 100.0) ** 0.42
    return 400.0 * np.sign(rgb) * magnitude / (27.13 + magnitude) + 0.1


def _hue_quadrature(h):
    values = np.asarray(h)
    hi = np.array([20.14, 90.0, 164.25, 237.53, 380.14])
    ei = np.array([0.8, 0.7, 1.0, 1.2, 0.8])
    index = np.searchsorted(hi, np.nan_to_num(values), side="left") - 1
    index = np.clip(index, 0, 3)
    h0, h1 = hi[index], hi[index + 1]
    e0, e1 = ei[index], ei[index + 1]
    result = index * 100 + 100 * ((values - h0) / e0) / (
        (values - h0) / e0 + (h1 - values) / e1
    )
    result = np.where(
        values < 20.14,
        385.9
        + (14.1 * values / 0.856)
        / (values / 0.856 + (20.14 - values) / 0.8),
        result,
    )
    return np.where(
        values >= 237.53,
        300
        + (85.9 * (values - 237.53) / 1.2)
        / ((values - 237.53) / 1.2 + (360 - values) / 0.856),
        result,
    )


def _appearance_forward(
    XYZ, XYZ_w, L_A, Y_b, surround, discount_illuminant, compute_H, model, cls
):
    source = f64(XYZ)
    if source.ndim == 0 or source.shape[-1] != 3:
        raise ValueError("XYZ must have a final dimension of size 3")
    shape = source.shape[:-1]
    matrix, fl, background_n, nbb, zbase, aw = _viewing_setup(
        XYZ_w, L_A, Y_b, surround, discount_illuminant, model
    )
    destination = np.empty((source.size // 3, 6))
    if destination.shape[0]:
        lib().mcs_appearance_forward(
            addr(source),
            addr(destination),
            addr(matrix),
            destination.shape[0],
            fl,
            background_n,
            nbb,
            zbase,
            float(surround.c),
            float(surround.N_c),
            aw,
        )
    columns = [
        destination[:, i].reshape(shape) for i in range(destination.shape[1])
    ]
    columns = [value[()] if value.ndim == 0 else value for value in columns]
    H = _hue_quadrature(columns[2]) if compute_H else np.full(shape, np.nan)
    if np.ndim(H) == 0:
        H = float(H)
    return cls(
        J=columns[0],
        C=columns[1],
        h=columns[2],
        s=columns[3],
        Q=columns[4],
        M=columns[5],
        H=H,
        HC=None,
    )


def XYZ_to_CAM16(
    XYZ,
    XYZ_w,
    L_A,
    Y_b,
    surround=_AVERAGE,
    discount_illuminant=False,
    compute_H=True,
):
    return _appearance_forward(
        XYZ,
        XYZ_w,
        L_A,
        Y_b,
        surround,
        discount_illuminant,
        compute_H,
        "cam16",
        CAM_Specification_CAM16,
    )


def XYZ_to_CIECAM02(
    XYZ,
    XYZ_w,
    L_A,
    Y_b,
    surround=_AVERAGE,
    discount_illuminant=False,
    compute_H=True,
):
    return _appearance_forward(
        XYZ,
        XYZ_w,
        L_A,
        Y_b,
        surround,
        discount_illuminant,
        compute_H,
        "ciecam02",
        CAM_Specification_CIECAM02,
    )


def _appearance_inverse(
    specification, XYZ_w, L_A, Y_b, surround, discount_illuminant, model
):
    J, h = np.asarray(specification.J), np.asarray(specification.h)
    C = specification.C
    if C is None or np.all(np.isnan(C)):
        if specification.M is None:
            raise ValueError('Either "C" or "M" correlate must be defined')
        setup = _viewing_setup(
            XYZ_w, L_A, Y_b, surround, discount_illuminant, model
        )
        C = np.asarray(specification.M) / setup[1] ** 0.25
    J, C, h = np.broadcast_arrays(J, np.asarray(C), h)
    source = f64(np.stack([J, C, h], axis=-1))
    matrix, fl, background_n, nbb, zbase, aw = _viewing_setup(
        XYZ_w, L_A, Y_b, surround, discount_illuminant, model
    )
    inverse = f64(np.linalg.inv(matrix))
    destination = np.empty_like(source)
    if source.size:
        lib().mcs_appearance_inverse(
            addr(source),
            addr(destination),
            addr(inverse),
            source.size // 3,
            fl,
            background_n,
            nbb,
            zbase,
            float(surround.c),
            float(surround.N_c),
            aw,
        )
    return destination


def CAM16_to_XYZ(
    specification,
    XYZ_w,
    L_A,
    Y_b,
    surround=_AVERAGE,
    discount_illuminant=False,
):
    return _appearance_inverse(
        specification, XYZ_w, L_A, Y_b, surround, discount_illuminant, "cam16"
    )


def CIECAM02_to_XYZ(
    specification,
    XYZ_w,
    L_A,
    Y_b,
    surround=_AVERAGE,
    discount_illuminant=False,
):
    return _appearance_inverse(
        specification,
        XYZ_w,
        L_A,
        Y_b,
        surround,
        discount_illuminant,
        "ciecam02",
    )
