"""Numerical parity with colour-science on identical inputs."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

colour = pytest.importorskip("colour")
colour_difference = pytest.importorskip("colour.difference")

import mojo_colour as mcolour
from mojo_colour._lib import addr, lib


def gpu_memory_free_mib():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.splitlines()[0].strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


@pytest.fixture(scope="module")
def samples():
    rng = np.random.default_rng(2026)
    rgb = np.ascontiguousarray(rng.random((257, 3)))
    xyz = np.ascontiguousarray(colour.sRGB_to_XYZ(rgb))
    lab = np.ascontiguousarray(colour.XYZ_to_Lab(xyz))
    return rgb, xyz, lab


def test_srgb_to_xyz(samples):
    rgb, _, _ = samples
    assert np.allclose(mcolour.sRGB_to_XYZ(rgb), colour.sRGB_to_XYZ(rgb), atol=2e-9)


def test_xyz_to_srgb(samples):
    rgb, xyz, _ = samples
    # colour-science's rounded sRGB matrices are not exact inverses.
    assert np.allclose(mcolour.XYZ_to_sRGB(xyz), rgb, atol=3e-4)
    assert np.allclose(
        mcolour.XYZ_to_sRGB(xyz), colour.XYZ_to_sRGB(xyz), atol=2e-9
    )


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (mcolour.cctf_decoding, colour.cctf_decoding),
        (mcolour.cctf_encoding, colour.cctf_encoding),
    ],
)
def test_srgb_transfer_functions(samples, ours, theirs):
    rgb, _, _ = samples
    assert np.allclose(ours(rgb), theirs(rgb), atol=2e-12)


@pytest.mark.parametrize("illuminant", [np.array([0.3127, 0.3290]), np.array([0.3457, 0.3585])])
def test_xyz_lab(samples, illuminant):
    _, xyz, _ = samples
    assert np.allclose(
        mcolour.XYZ_to_Lab(xyz, illuminant),
        colour.XYZ_to_Lab(xyz, illuminant),
        atol=3e-7,
    )


def test_lab_xyz(samples):
    _, xyz, lab = samples
    assert np.allclose(mcolour.Lab_to_XYZ(lab), colour.Lab_to_XYZ(lab), atol=2e-12)
    assert np.allclose(mcolour.Lab_to_XYZ(lab), xyz, atol=2e-12)


@pytest.mark.parametrize(
    "illuminant", [np.array([0.3127, 0.3290]), np.array([0.3457, 0.3585])]
)
def test_lab_xyz_custom_illuminant(samples, illuminant):
    _, xyz, _ = samples
    lab = colour.XYZ_to_Lab(xyz, illuminant)
    assert np.allclose(
        mcolour.Lab_to_XYZ(lab, illuminant),
        colour.Lab_to_XYZ(lab, illuminant),
        atol=2e-12,
    )


@pytest.mark.parametrize(
    "illuminant", [np.array([0.3127, 0.3290]), np.array([0.3457, 0.3585])]
)
def test_xyz_luv(samples, illuminant):
    _, xyz, _ = samples
    assert np.allclose(
        mcolour.XYZ_to_Luv(xyz, illuminant),
        colour.XYZ_to_Luv(xyz, illuminant),
        atol=2e-7,
    )


@pytest.mark.parametrize(
    "illuminant", [np.array([0.3127, 0.3290]), np.array([0.3457, 0.3585])]
)
def test_luv_xyz(samples, illuminant):
    _, xyz, _ = samples
    luv = colour.XYZ_to_Luv(xyz, illuminant)
    assert np.allclose(
        mcolour.Luv_to_XYZ(luv, illuminant),
        colour.Luv_to_XYZ(luv, illuminant),
        atol=2e-12,
    )
    assert np.allclose(mcolour.Luv_to_XYZ(luv, illuminant), xyz, atol=2e-12)


def test_black_luv_round_trip():
    black = np.zeros(3)
    assert np.array_equal(mcolour.Luv_to_XYZ(mcolour.XYZ_to_Luv(black)), black)


def test_xyy_conversions(samples):
    _, xyz, _ = samples
    xyy = mcolour.XYZ_to_xyY(xyz)
    assert np.allclose(xyy, colour.XYZ_to_xyY(xyz))
    assert np.allclose(mcolour.xyY_to_XYZ(xyy), xyz)
    assert np.array_equal(mcolour.XYZ_to_xyY(np.zeros(3)), np.zeros(3))


def test_xy_conversions(samples):
    _, xyz, _ = samples
    assert np.allclose(mcolour.XYZ_to_xy(xyz), colour.XYZ_to_xy(xyz))
    xy = np.array([[0.3127, 0.3290], [0.3457, 0.3585]])
    assert np.allclose(mcolour.xy_to_XYZ(xy), colour.xy_to_XYZ(xy))


@pytest.mark.parametrize(
    ("forward", "inverse", "reference_forward", "reference_inverse"),
    [
        (
            mcolour.XYZ_to_Oklab,
            mcolour.Oklab_to_XYZ,
            colour.XYZ_to_Oklab,
            colour.Oklab_to_XYZ,
        ),
        (
            mcolour.XYZ_to_IPT,
            mcolour.IPT_to_XYZ,
            colour.XYZ_to_IPT,
            colour.IPT_to_XYZ,
        ),
    ],
)
def test_perceptual_spaces(
    samples, forward, inverse, reference_forward, reference_inverse
):
    _, xyz, _ = samples
    encoded = forward(xyz)
    assert np.allclose(encoded, reference_forward(xyz), atol=3e-9)
    assert np.allclose(inverse(encoded), reference_inverse(encoded), atol=3e-7)
    assert np.allclose(inverse(encoded), xyz, atol=3e-7)


def test_jzazbz(samples):
    _, xyz, _ = samples
    encoded = mcolour.XYZ_to_Jzazbz(xyz)
    assert np.allclose(encoded, colour.XYZ_to_Jzazbz(xyz), rtol=2e-7, atol=2e-10)
    assert np.allclose(
        mcolour.Jzazbz_to_XYZ(encoded),
        colour.Jzazbz_to_XYZ(encoded),
        rtol=2e-6,
        atol=2e-7,
    )
    assert np.allclose(mcolour.Jzazbz_to_XYZ(encoded), xyz, atol=3e-7)


def test_simd_tail_and_parallel_threshold():
    rng = np.random.default_rng(91)
    xyz = np.ascontiguousarray(
        colour.sRGB_to_XYZ(rng.random((65_539, 3)))
    )
    assert np.allclose(
        mcolour.XYZ_to_Lab(xyz),
        colour.XYZ_to_Lab(xyz),
        atol=3e-7,
    )
    assert np.allclose(
        mcolour.XYZ_to_Jzazbz(xyz),
        colour.XYZ_to_Jzazbz(xyz),
        rtol=2e-7,
        atol=2e-10,
    )
    white = np.array([95.05, 100.0, 108.88])
    ours = mcolour.XYZ_to_CAM16(
        xyz * 100, white, 64.0, 20.0, compute_H=False
    )
    theirs = colour.XYZ_to_CAM16(
        xyz * 100, white, 64.0, 20.0, compute_H=False
    )
    for correlate in ("J", "C", "h", "s", "Q", "M"):
        assert np.allclose(
            getattr(ours, correlate),
            getattr(theirs, correlate),
            atol=2e-6,
        )


def test_jzazbz_gpu_path():
    free_mib = gpu_memory_free_mib()
    if free_mib is None or free_mib < 4000:
        pytest.skip("GPU has less than 4000 MiB free or is unavailable")
    xyz = np.ascontiguousarray(
        colour.sRGB_to_XYZ(
            np.random.default_rng(92).random((4_099, 3))
        )
    )
    destination = np.empty_like(xyz)
    assert lib().mcs_jzazbz_gpu(
        addr(xyz), addr(destination), len(xyz), 0
    )
    assert np.allclose(
        destination,
        colour.XYZ_to_Jzazbz(xyz),
        rtol=2e-7,
        atol=2e-10,
    )
    assert np.allclose(
        mcolour.XYZ_to_Jzazbz(xyz, device="gpu"),
        destination,
        rtol=2e-7,
        atol=2e-10,
    )
    assert np.allclose(
        mcolour.Jzazbz_to_XYZ(destination, device="gpu"),
        xyz,
        atol=3e-7,
    )


def test_jzazbz_device_validation():
    with pytest.raises(ValueError, match="device"):
        mcolour.XYZ_to_Jzazbz([[0.1, 0.2, 0.3]], device="tpu")


@pytest.mark.parametrize("method", ["CIE 1976", "CIE 1994", "CIE 2000"])
@pytest.mark.parametrize("textiles", [False, True])
def test_delta_e_against_colour(samples, method, textiles):
    _, _, lab = samples
    other = np.ascontiguousarray(lab + np.array([1.0, -0.4, 0.7]))
    kwargs = {"textiles": textiles} if method != "CIE 1976" else {}
    assert np.allclose(
        mcolour.delta_E(lab, other, method=method, **kwargs),
        colour.delta_E(lab, other, method=method, **kwargs),
        atol=2e-12,
    )


@pytest.mark.parametrize(
    ("ours", "theirs", "kwargs"),
    [
        (mcolour.delta_E_CIE1976, colour_difference.delta_E_CIE1976, {}),
        (
            mcolour.delta_E_CIE1994,
            colour_difference.delta_E_CIE1994,
            {"textiles": True},
        ),
        (
            mcolour.delta_E_CIE2000,
            colour_difference.delta_E_CIE2000,
            {"textiles": True},
        ),
    ],
)
def test_method_specific_delta_e_names(samples, ours, theirs, kwargs):
    _, _, lab = samples
    other = lab + np.array([1.0, -0.4, 0.7])
    assert np.allclose(ours(lab, other, **kwargs), theirs(lab, other, **kwargs))


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ([50, 2.6772, -79.7751], [50, 0, -82.7485], 2.0425),
        ([50, 3.1571, -77.2803], [50, 0, -82.7485], 2.8615),
        ([50, 2.8361, -74.0200], [50, 0, -82.7485], 3.4412),
        ([50, -1.3802, -84.2814], [50, 0, -82.7485], 1.0000),
    ],
)
def test_ciede2000_published_sharma_vectors(first, second, expected):
    assert mcolour.delta_E(first, second) == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize(
    ("ours_name", "theirs_name"),
    [
        ("XYZ_to_CAM16", "XYZ_to_CAM16"),
        ("XYZ_to_CIECAM02", "XYZ_to_CIECAM02"),
    ],
)
def test_appearance_forward(ours_name, theirs_name, samples):
    rgb, _, _ = samples
    xyz100 = np.ascontiguousarray(colour.sRGB_to_XYZ(rgb[:100]) * 100)
    white = np.array([95.05, 100.0, 108.88])
    ours = getattr(mcolour, ours_name)(xyz100, white, 318.31, 20.0)
    theirs = getattr(colour, theirs_name)(xyz100, white, 318.31, 20.0)
    for correlate in ("J", "C", "h", "s", "Q", "M", "H"):
        assert np.allclose(
            getattr(ours, correlate), getattr(theirs, correlate), atol=2e-6
        )


@pytest.mark.parametrize(
    ("ours", "theirs", "conditions"),
    [
        (
            mcolour.XYZ_to_CAM16,
            colour.XYZ_to_CAM16,
            mcolour.VIEWING_CONDITIONS_CAM16["Dim"],
        ),
        (
            mcolour.XYZ_to_CIECAM02,
            colour.XYZ_to_CIECAM02,
            mcolour.VIEWING_CONDITIONS_CIECAM02["Dark"],
        ),
    ],
)
def test_appearance_custom_scalar_conditions(ours, theirs, conditions):
    xyz = np.array([[19.01, 20.0, 21.78], [57.06, 43.06, 31.96]])
    white = np.array([95.05, 100.0, 108.88])
    candidate = ours(
        xyz, white, 64.0, 10.0, conditions, discount_illuminant=True
    )
    reference = theirs(
        xyz, white, 64.0, 10.0, conditions, discount_illuminant=True
    )
    for correlate in ("J", "C", "h", "s", "Q", "M", "H"):
        assert np.allclose(
            getattr(candidate, correlate), getattr(reference, correlate), atol=2e-6
        )


@pytest.mark.parametrize(
    ("forward", "inverse"),
    [
        (mcolour.XYZ_to_CAM16, mcolour.CAM16_to_XYZ),
        (mcolour.XYZ_to_CIECAM02, mcolour.CIECAM02_to_XYZ),
    ],
)
def test_appearance_inverse(forward, inverse):
    xyz = np.array(
        [[19.01, 20.0, 21.78], [57.06, 43.06, 31.96], [3.53, 6.56, 2.14]]
    )
    white = np.array([95.05, 100.0, 108.88])
    specification = forward(xyz, white, 318.31, 20.0)
    assert np.allclose(
        inverse(specification, white, 318.31, 20.0), xyz, atol=2e-7
    )


def test_appearance_published_example():
    specification = mcolour.XYZ_to_CAM16(
        np.array([19.01, 20.0, 21.78]),
        np.array([95.05, 100.0, 108.88]),
        318.31,
        20.0,
    )
    assert specification.J == pytest.approx(41.7312079, abs=1e-7)
    assert specification.h == pytest.approx(217.0679598, abs=1e-7)


def test_shapes_are_preserved(samples):
    _, xyz, _ = samples
    cube = xyz[:24].reshape(2, 3, 4, 3)
    assert mcolour.XYZ_to_Lab(cube).shape == cube.shape
    assert mcolour.delta_E(
        mcolour.XYZ_to_Lab(cube), mcolour.XYZ_to_Lab(cube)
    ).shape == cube.shape[:-1]


def test_empty_inputs_do_not_cross_the_ffi():
    triples = np.empty((2, 0, 3))
    assert mcolour.XYZ_to_Lab(triples).shape == triples.shape
    assert mcolour.XYZ_to_Jzazbz(triples, device="gpu").shape == triples.shape
    assert mcolour.delta_E(triples, triples).shape == triples.shape[:-1]
    result = mcolour.XYZ_to_CAM16(
        triples, np.array([95.05, 100.0, 108.88]), 64.0, 20.0
    )
    assert result.J.shape == triples.shape[:-1]


def test_unsafe_dtype_narrowing_is_rejected():
    with pytest.raises(TypeError, match="complex"):
        mcolour.XYZ_to_Lab(np.ones((1, 3), dtype=np.complex128))


@pytest.mark.parametrize(("L_A", "Y_b"), [(-1.0, 20.0), (64.0, 0.0)])
def test_invalid_viewing_conditions_are_rejected(L_A, Y_b):
    with pytest.raises(ValueError):
        mcolour.XYZ_to_CAM16(
            [19.01, 20.0, 21.78],
            [95.05, 100.0, 108.88],
            L_A,
            Y_b,
        )


def test_unsupported_srgb_adaptation_is_explicit():
    with pytest.raises(NotImplementedError):
        mcolour.sRGB_to_XYZ([0.1, 0.2, 0.3], illuminant=[0.3457, 0.3585])
