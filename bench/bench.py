"""Benchmark Mojo kernels against colour-science on identical arrays."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import colour  # noqa: E402
import mojo_colour as mcolour  # noqa: E402


def time_best(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


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


def gpu_name():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.splitlines()[0].strip()
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        return None


def main():
    rng = np.random.default_rng(7)
    count = 1_000_000
    rgb = np.ascontiguousarray(rng.random((count, 3)))
    xyz = np.ascontiguousarray(colour.sRGB_to_XYZ(rgb))
    lab = np.ascontiguousarray(colour.XYZ_to_Lab(xyz))
    other_lab = np.ascontiguousarray(lab + rng.normal(0, 1, lab.shape))
    xyz100 = np.ascontiguousarray(xyz * 100)
    white = np.array([95.05, 100.0, 108.88])

    cases = [
        (
            "sRGB_to_XYZ (1M)",
            lambda: mcolour.sRGB_to_XYZ(rgb),
            lambda: colour.sRGB_to_XYZ(rgb),
        ),
        (
            "XYZ_to_Lab (1M)",
            lambda: mcolour.XYZ_to_Lab(xyz),
            lambda: colour.XYZ_to_Lab(xyz),
        ),
        (
            "Lab_to_XYZ (1M)",
            lambda: mcolour.Lab_to_XYZ(lab),
            lambda: colour.Lab_to_XYZ(lab),
        ),
        (
            "XYZ_to_Oklab (1M)",
            lambda: mcolour.XYZ_to_Oklab(xyz),
            lambda: colour.XYZ_to_Oklab(xyz),
        ),
        (
            "XYZ_to_Jzazbz (1M)",
            lambda: mcolour.XYZ_to_Jzazbz(xyz),
            lambda: colour.XYZ_to_Jzazbz(xyz),
        ),
        (
            "delta_E CIE 2000 (1M)",
            lambda: mcolour.delta_E(lab, other_lab),
            lambda: colour.delta_E(lab, other_lab),
        ),
        (
            "XYZ_to_CAM16 (1M, H off)",
            lambda: mcolour.XYZ_to_CAM16(
                xyz100, white, 64.0, 20.0, compute_H=False
            ),
            lambda: colour.XYZ_to_CAM16(
                xyz100, white, 64.0, 20.0, compute_H=False
            ),
        ),
    ]
    free_mib = gpu_memory_free_mib()
    if free_mib is not None and free_mib >= 4000:
        cases.insert(
            5,
            (
                "XYZ_to_Jzazbz GPU (1M)",
                lambda: mcolour.XYZ_to_Jzazbz(xyz, device="gpu"),
                lambda: colour.XYZ_to_Jzazbz(xyz),
            ),
        )

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    if name := gpu_name():
        print(f"GPU: {name}")
    if free_mib is None:
        print("GPU benchmark skipped: device memory could not be queried.")
    elif free_mib < 4000:
        print(
            f"GPU benchmark skipped: {free_mib} MiB free is below the "
            "4000 MiB safety threshold."
        )
    print()
    print("| Kernel | Mojo | colour-science | Speedup |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs in cases:
        ours()
        reference = theirs()
        candidate = ours()
        if hasattr(candidate, "J"):
            assert np.allclose(candidate.J, reference.J, atol=2e-6)
        else:
            assert np.allclose(candidate, reference, rtol=2e-6, atol=3e-7)
        mojo_seconds = time_best(ours)
        colour_seconds = time_best(theirs)
        ratio = colour_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{colour_seconds * 1000:.2f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
