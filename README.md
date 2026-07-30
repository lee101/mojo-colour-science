# mojo-colour-science

`mojo-colour-science` is a standalone Mojo port of the compute-heavy core of
[`colour-science`](https://pypi.org/project/colour-science/). It provides fused,
native kernels for image-sized arrays and a NumPy-facing Python API with the
same function names and compatible parameters for the covered subset.

The compiled library has no runtime dependency on `colour-science`; that
package is present in the development environment so every covered algorithm
can be parity-tested against the real upstream implementation.

## Coverage

The package exposes 28 upstream-named entry points:

- sRGB transfer functions and `sRGB_to_XYZ` / `XYZ_to_sRGB`
- CIE XYZ, xy, xyY, Lab, and Luv conversions, including custom Lab/Luv
  illuminants
- Oklab, IPT, and Jzazbz forward and inverse conversions
- CIE 1976, CIE 1994, and CIE 2000 colour differences through both
  `delta_E` and the method-specific function names
- Forward and inverse CAM16 and CIECAM02, with custom scalar viewing
  conditions, surround factors, illuminant discounting, and hue quadrature

This is deliberately not a port of the entire upstream project. Spectral
distributions, colour matching functions, illuminant datasets, plotting,
camera/display characterisation, arbitrary RGB colourspaces, non-D65 sRGB
chromatic adaptation, alternate Jzazbz constants, hue composition, and
array-valued CAM viewing parameters are not covered. Unsupported sRGB
adaptation and Jzazbz constant choices raise `NotImplementedError`.

## Install

Install the pinned Mojo nightly and Python dependencies, then build the shared
library:

```bash
pixi install
pixi run build
```

The Pixi environment places `python/` on `PYTHONPATH`. A development checkout
can also be installed as a normal Python package with `pip install -e .` after
the shared library has been built.

## Usage

The covered names can be used as a drop-in by changing the import:

```python
import numpy as np
import mojo_colour as colour

rgb = np.array([[0.70573936, 0.19248266, 0.22354169]])
xyz = colour.sRGB_to_XYZ(rgb)
lab = colour.XYZ_to_Lab(xyz)
jz_gpu = colour.XYZ_to_Jzazbz(xyz, device="gpu")

print(xyz)
# [[0.2065429  0.12197943 0.0513714 ]]
print(lab)
# [[41.52900347 52.6350894  26.92424054]]
```

Inputs may have any leading shape as long as the final dimension contains
three components. Outputs preserve those leading dimensions.

Jzazbz forward and inverse conversion accept `device="gpu"` as an explicit
opt-in. CPU remains the default. The GPU path checks that at least 4000 MiB is
free, caps its two device buffers at 2 GiB total, and falls back to the CPU
implementation with a `RuntimeWarning` if the device is unavailable or any
GPU operation fails.

## Validation

Run the complete parity suite with:

```bash
pixi run build
pixi run test
```

The 48 tests compare the Mojo results against `colour-science` 0.4.7 on scalar,
matrix, and higher-dimensional inputs. They also exercise forward/inverse
round trips, custom illuminants and scalar viewing conditions, empty arrays,
dtype rejection, SIMD remainders, the parallel threshold, the optional GPU
path, and published Sharma CIEDE2000 vectors.

## Benchmarks

Measured on an Intel Xeon E5-2697 v4 at 2.30 GHz and an NVIDIA GeForce RTX 5090,
Linux x86-64. Each row uses the same contiguous float64 input, warms the
shared library before timing, and reports the best of three runs. The GPU row
includes context, allocation, transfer, kernel, and synchronization time and
uses two device buffers. These are the real results from `pixi run bench` on
this machine; a speedup below 1 means Mojo is slower.

| Kernel | Mojo | colour-science | Speedup |
|---|---:|---:|---:|
| sRGB_to_XYZ (1M) | 103.65 ms | 375.34 ms | 3.62x |
| XYZ_to_Lab (1M) | 25.03 ms | 264.29 ms | 10.56x |
| Lab_to_XYZ (1M) | 8.46 ms | 157.35 ms | 18.61x |
| XYZ_to_Oklab (1M) | 135.08 ms | 347.57 ms | 2.57x |
| XYZ_to_Jzazbz (1M) | 37.50 ms | 593.11 ms | 15.82x |
| XYZ_to_Jzazbz GPU (1M) | 17.30 ms | 744.59 ms | 43.04x |
| delta_E CIE 2000 (1M) | 273.76 ms | 725.05 ms | 2.65x |
| XYZ_to_CAM16 (1M, H off) | 69.08 ms | 757.64 ms | 10.97x |

All measured kernels were faster in this run. `XYZ_to_Lab` processes
hardware-width float64 SIMD batches with a scalar tail. Lab, Jzazbz, and CAM16
switch to at most 16 CPU workers only for arrays of at least 65,536 colours;
smaller inputs remain serial to avoid launch overhead. The arithmetic-heavy
Jzazbz transform also benefits from an explicit GPU path.

## How it works

All numerical kernels live in one Mojo compilation unit and build into
`dist/libmojo-colour-science.so`. Python loads it with `ctypes`. NumPy owns all
input and output memory; contiguous float64 buffer addresses cross the C ABI as
64-bit integers and are reconstructed as mutable Mojo pointers inside exported
functions. No array memory is allocated across the FFI boundary.

Colour triples use ordinary C-order `(..., 3)` layout. Conversion kernels fuse
matrix multiplication, transfer functions, and nonlinear response steps into
one pass. The XYZ-to-Lab SIMD loop uses strided loads and stores directly on
that interleaved layout, including an explicit remainder loop. Large,
independent ranges are split into 32,768-colour chunks. CAM16 and CIECAM02
reduce the scalar viewing-condition setup to an effective 3-by-3 adaptation
matrix in Python, then run all per-colour response compression and correlate
calculations in Mojo.

## License

MIT
