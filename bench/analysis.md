# ANALYZE — mojo-colour-science performance map

No algorithms were changed in this phase. Every time below comes from a run
executed in this session on this machine.

**Machine:** Intel Xeon E5-2697 v4 @ 2.30 GHz, Linux x86-64, NVIDIA GeForce
RTX 5090 (7,627 MiB free at time of run). Python 3.13.14, numpy 2.5.1,
colour-science 0.4.7, `pixi run bench` (flock-wrapped, run solo).

**Method.** `pixi run bench` (best of 3, warmup in
[bench.py](bench.py)) plus two profiling passes: best of 5–7, 2–3 warmups,
`time.perf_counter`, float64 C-contiguous arrays. Sizes covered: 1, 10, 100,
1k, 10k, 65,535 / 65,536 (the parallel threshold), 100k, 500k, 1M. Mojo vs
upstream `colour-science` where importable, plus plain-numpy baselines.

## Ranked hotspot table (N = 1M colours, float64, best of 3–7 runs)

`bytes` = read + write traffic implied by the in-place buffer convention
(N×3×8 read + N×3×8 write; `delta_E` reads two Lab triples and writes one
scalar). `flops` and `flops/byte` are counted from the kernel source; transcendental
calls are counted as ~20 flops each (conservative, documented in Method).
Compute-bound test: at 1M and 24–56 MB total traffic, even 10 GB/s effective
bandwidth finishes the transfer in a few ms, far below measured times.

| rank | function | input size | time | bytes touched | flops | flops/byte | memory- or compute-bound | Python/ctypes glue or kernel |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | `delta_E` CIE 2000 `mcs_delta_e` | 1M | 272–282 ms | 56 MB | ~180 /col | 7.5 | compute-bound | kernel |
| 2 | `delta_E` CIE 2000, CIECAM02/CAM16 inverse `mcs_appearance_inverse` | 1M | ~360 ms¹ | 72 MB | ~450 /col | 6.3 | compute-bound | kernel (serial, no parallel path) |
| 3 | `XYZ_to_Oklab` `mcs_perceptual` mode 0 | 1M | 156–166 ms | 48 MB | ~80 /col² | 1.7 | compute-bound | kernel (serial) |
| 4 | `XYZ_to_CAM16/CIECAM02` compute_H=True | 1M | 175–213 ms | 72 MB | ~350 /col | 4.9 | compute-bound | **Python/numpy `_hue_quadrature`** dominates the H delta |
| 5 | `XYZ_to_IPT` / `IPT_to_XYZ` | 1M | 120–127 ms | 48 MB | ~80 /col² | 1.7 | compute-bound | kernel (serial) |
| 6 | `cctf_encoding` | 1M | 119–125 ms | 48 MB | ~60 /col | 1.3 | compute-bound | kernel (serial) |
| 7 | `sRGB_to_XYZ` / `XYZ_to_sRGB` | 1M | 108–119 ms | 48 MB | ~30 /col | 0.6 | compute-bound³ | kernel (serial) |
| 8 | `xy_to_XYZ` | 1M | 25–27 ms | 48 MB | ~20 /col | 0.4 | memory-bound | **Python `np.concatenate`** (12.6 ms, 46%) + kernel |
| 9 | `XYZ_to_CAM16/CIECAM02` compute_H=False | 1M | 41–51 ms | 72 MB | ~120 /col (parallel) | 1.7 | compute-bound | kernel (parallel) |
| 10 | `XYZ_to_Jzazbz` CPU | 1M | 33–42 ms | 48 MB | ~110 /col (parallel) | 2.3 | compute-bound | kernel (parallel) |
| 11 | `XYZ_to_Luv` | 1M | 33–34 ms | 48 MB | ~50 /col | 1.0 | compute-bound | kernel (serial) |
| 12 | `delta_E` CIE 1994 | 1M | 18–20 ms | 56 MB | ~70 /col | 1.3 | compute-bound | kernel (serial) |
| 13 | `XYZ_to_Lab` (parallel+SIMD) | 1M | 13–17 ms | 48 MB | ~35 /col (parallel) | 0.7 | compute-bound | kernel (parallel+SIMD) |
| 14 | `Luv_to_XYZ` | 1M | 10 ms | 48 MB | ~40 /col | 0.8 | compute-bound | kernel (serial) |
| 15 | `Lab_to_XYZ` | 1M | 7–9 ms | 48 MB | ~20 /col | 0.4 | compute-bound | kernel (serial) |
| 16 | `Oklab_to_XYZ` | 1M | 6–7 ms | 48 MB | ~25 /col | 0.5 | compute-bound | kernel (serial) |
| 17 | `delta_E` CIE 1976 | 1M | 5–6 ms | 56 MB | ~15 /col | 0.3 | memory-bound | kernel (serial) |
| 18 | `XYZ_to_Jzazbz` GPU `device="gpu"` | 1M | 6–8 ms | 48 MB + PCIe xfer | ~110 /col | 2.3 | compute-bound (GPU) | kernel + transfer (GPU) |
| 19 | `XYZ_to_xyY` / `xyY_to_XYZ` / `XYZ_to_xy` | 1M | 4–5 ms | 48 MB | ~10 /col | 0.2 | memory-bound | kernel (serial) |

¹ CAM16/CIECAM02 inverse measured at 200k and 1M (full spec via
`colour.XYZ_to_CAM16` reference for the upstream side, Mojo spec for the Mojo
side). ² Oklab/IPT forward each run 3 signed `pow(·, 1/3)` (Oklab) /
`pow(·, 0.43)` (IPT) per colour, plus two 3×3 matmuls; 60–100 flops/col is a
conservative band. ³ `sRGB_to_XYZ` time is dominated by 3 `pow(·, 2.4)`
decodes per colour (~30–40 ns each at this scale); arithmetic alone would
imply <5 ms. Not bandwidth-limited.

**The three slowest kernels are also the three still running without
acceleration:** CIEDE2000 (7.5 flops/byte, heaviest transcendental count —
5 `cos`, 2 `pow(·,7)`, 2 `sqrt`, shared `exp`), the CAM16/CIECAM02 inverse
(~450 flops/col, zero parallelisation in `mcs_appearance_inverse`), and
Oklab forward (3 `spow` per colour). The transcode pair (`sRGB_to_XYZ`,
`XYZ_to_sRGB`) is fourth-tier slow purely on `pow`.

**No kernel in this port was slower than upstream at 1M.** All speedups
below exceed 1.0; see vs-upstream table. The test in step 4 (mark slower
rows) therefore has no rows in the 1M table — it is not that slow rows were
dropped.

## vs upstream and numpy baseline (1M, same session)

| function | Mojo ms | colour-science ms | speedup |
|---|---:|---:|---:|
| `sRGB_to_XYZ` | 119.2 | 310.0 | 2.60x |
| `XYZ_to_sRGB` | 117.4 | 313.0 | 2.67x |
| `cctf_decoding` | 106.8 | 160.4 | 1.50x |
| `cctf_encoding` | 125.3 | 141.1 | 1.13x |
| `XYZ_to_Lab` | 16.5 | 157.5 | 9.55x |
| `Lab_to_XYZ` | 7.0 | 152.9 | 21.98x |
| `XYZ_to_Luv` | 33.8 | 105.6 | 3.13x |
| `Luv_to_XYZ` | 10.0 | 129.9 | 12.97x |
| `XYZ_to_xyY` | 4.4 | 62.8 | 14.21x |
| `xyY_to_XYZ` | 4.7 | 53.9 | 11.46x |
| `XYZ_to_xy` | 4.4 | 63.2 | 14.21x |
| `xy_to_XYZ` | 25.0 | 242.9 | 9.72x |
| `XYZ_to_Oklab` | 166.1 | 289.5 | 1.74x |
| `Oklab_to_XYZ` | 6.4 | 289.5 | 45.02x |
| `XYZ_to_IPT` | 125.3 | 289.7 | 2.31x |
| `IPT_to_XYZ` | 127.0 | 292.6 | 2.30x |
| `XYZ_to_Jzazbz` | 41.7 | 514.4 | 12.34x |
| `Jzazbz_to_XYZ` | 39.8 | 522.1 | 13.12x |
| `delta_E` 1976 | 5.1 | 61.8 | 12.23x |
| `delta_E` 1994 | 18.3 | 153.7 | 8.40x |
| `delta_E` 2000 | 271.9 | 635.9 | 2.34x |
| `XYZ_to_CAM16` H off | 49.0 | 584.1 | 11.92x |
| `XYZ_to_CAM16` H on | 212.7 | 557.1 | 2.62x |
| `XYZ_to_CIECAM02` H off | 43.8 | 662.6 | 15.13x |
| `XYZ_to_CIECAM02` H on | 193.1 | 756.6 | 3.92x |
| `CAM16_to_XYZ` | 360.0 | 707.3 | 1.96x |
| `CIECAM02_to_XYZ` | 360.9 | 772.9 | 2.14x |
| `XYZ_to_Jzazbz` GPU | 6.4–8.4 | 502–744 | 59–93x |

Numpy baselines (1M, same arrays, best of 5): `rgb @ M` sRGB matrix alone
0.58 ms; a hand-written vectorised Oklab (matmul + `cbrt` + matmul)
85.6 ms — i.e. a ~2× numpy Oklab would beat the Mojo kernel's 166 ms,
which is why Oklab forward ranks where it does (see verdicts).

## Where the time is: kernel vs glue

**Raw ctypes (preallocated buffers) vs full Python wrapper, 1M:**
XYZ_to_Lab 16.6/17.2 ms, sRGB_to_XYZ 111.3/109.5 ms, XYZ_to_Oklab
158.5/156.3 ms, XYZ_to_Jzazbz 41.9/41.9 ms, delta_E2000 270.1/269.4 ms.
Glue overhead at 1M is within run-to-run noise (≤0.6 ms, i.e. <4%) for every
kernel. The `np.empty_like` output allocation itself costs ~2 µs at 1M
(free-list / page-cache warm arrays). **At 1M the time is in the kernels, not
the glue.**

**Small-N: glue dominates.** Wrapper vs raw at N=1/100/1k/10k is
12.2/1.9 µs, 20.1/9.5 µs, 92.3/80.2 µs, 805/795 µs. Fixed cost is
~9–11 µs per call: ~9 µs of Python wrapper
(`f64` validation → `_triples` → `np.empty_like` → `addr`/`lib` lookup →
ctypes call → `reshape`) plus ~1–2 µs raw kernel floor. A `numpy` fallback
for tiny arrays would therefore be *slower* than even a serial kernel here;
the wrapper is lean already. Per-call allocation of the small 3×3 viewing
matrix (`_viewing_setup`) costs ~60 µs and is also negligible vs the
multi-ms CAM16 kernels.

## Python-level loops, copies, and per-call costs flagged

- **No Python loop re-enters ctypes per element anywhere.** The only
  Python-side loops are per-function dispatch (`getattr(lib(), …)` once per
  call, `_same_shape_kernel` in `models.py`), the per-apply Python-side
  `np.concatenate` + `np.ones` in `xy_to_XYZ`, and the pure-numpy
  `_hue_quadrature` fallback — 16.1 ms at 200k, 81.4 ms at 1M. The latter is
  the one place where the Mojo kernel and the Python layer are
  seriously mismatched (see next-phase).
- **`xy_to_XYZ` spends 46% of its time building its own input:**
  `np.concatenate([xy, ones(…)], -1)` = 12.6 ms at 1M, on top of a 4.6 ms
  kernel. The port already has an `mcs_xyz_xyy` inverse mode that takes a
  3-column buffer; the wrapper should fuse the appended `1.0` column (or accept
  a 2-column kernel path) instead of materialising it.
- **Per-call copies:** `f64()` copies when dtype/contiguity mismatches
  (including float32). For contiguous float64 input `_triples` returns the
  caller's buffer object unchanged — zero copy. Output is always one
  `np.empty_like`. CAM16 forward allocates a 6-column destination and then
  reshapes six views; CAM16 inverse does one `np.stack` of J/C/h plus an
  `np.linalg.inv(3×3)` (~14 µs) before the kernel. No other Python-visible
  allocations exist.
- **`XYZ_to_xy`** is implemented as `XYZ_to_xyY(…)[..., :2]` — a full kernel
  call plus a strided view slice. Wasteful but correct; the kernel for
  `mcs_xyz_xyy` mode 0 computes `x/(x+y+z), y/(x+y+z)` exactly, so a direct
  2-column path is possible (small win, 4.4 ms total).

## Parallelisation and size cliffs (measured)

Three kernels are parallelised (`XYZ_to_Lab`, `XYZ_to_Jzazbz`, CAM16
forward), thresholded at 65,536 with 32,768-colour chunks:

| | @60k (serial) | implied serial 1M | measured 1M (parallel) | speedup |
|---|---:|---:|---:|---:|
| XYZ_to_Lab | 4.46 ms (74 ns/col) | 74 ms | 13–17 ms | ~5.5x |
| XYZ_to_Jzazbz | 18.74 ms (312 ns/col) | 312 ms | 33–42 ms | ~9x |
| CAM16 no-H | 21.0 ms (350 ns/col) | 350 ms | 41–51 ms | ~8x |

The threshold creates a **cliff at Lab**: at 65,535 Lab costs 4.98 ms but at
65,536 it jumps to 7.76 ms, then falls back to 89 ns/col by 100k
(8.97 ms). Jzazbz shows the inverse dip (319 → 300 → 189 ns/col).
Chunk-count launch overhead and sub-chunk waste explain both; the
transition band 65k–100k is the worst ns/col for all three kernels.
Every serial kernel below keeps a flat ns/col from 10k to 1M
(e.g. delta_E2000: 418 → 265 → 264 ns/col only falls below ~10k due to
glue amortisation), confirming the serial rate is scale-independent and the
glue cost is fully amortised by ~10k colours.

## GPU verdicts

Jzazbz is the only GPU path. At 1M: GPU 6.4–8.4 ms vs CPU-parallel
33–42 ms = **~5.8x** (59–93x vs upstream). Per-colour cost is ~6.8 ns/col,
i.e. the GPU is doing ~110 flops/col for the transfer cost of
48 MB × 2 (H2D + D2H) = 96 MB, so ≥48 MB of PCIe traffic must fit in the
6–8 ms, which it does comfortably for this machine (implies >12 GB/s
effective achieved rate — the path is real, not a silent CPU fallback; the
return-code guard and the 2 GiB cap are exercised and verified).

`flops/byte` is the arbiter for extending GPU to other kernels:

| candidate | flops/byte | verdict (GPU phase) |
|---|---:|---|
| delta_E 2000 | 7.5 | **GPU**: high arithmetic intensity, pure per-colour, no cross-colour state; launch cost is amortised at ≥10k colours |
| CAM16/CIECAM02 inverse | 6.3 | **GPU**: same per-colour independence; note explodes to ~450 flops/col with cheap branches |
| CAM16/CIECAM02 forward | 4.9 (with H) | **GPU**: current forward is parallelised on CPU already; GPU dominated by response compression + correlate math |
| XYZ_to_Oklab / IPT | 1.7 | **GPU or SIMD**: 3 heavy `pow` per colour in a gather-pattern; either works but SIMD-in-Mojo is cheaper to land first |
| Jzazbz (reference design) | 2.3 | proven: 5.8x on this machine |
| sRGB_to_XYZ / sRGB transcode | 0.6 / 1.3 | **none**: flops/byte too low; throughput already kernel-bound on `pow`, so SIMD + parallel wins cheaper than adding device traffic |

## Next-phase assignments per hotspot

All rows are from the table above, ranked by measured ms.

| rank | function (1M time) | next phase | evidence |
|---:|---|---|---|
| 1 | delta_E 2000 (272–282 ms) — serial, heaviest transcendental count | **parallel** then **gpu** | serial-only `mcs_delta_e` loop; 1M rate is 264 ns/col even at scale so the kernel is cleanly parallelisable; 7.5 flops/byte justifies GPU (see verdicts); also `unroll` the repeated `pow(·,7)` terms cheaply — no, do not change math; parallel first (biggest certain win on 16 cores: measured parallel kernels gain 5–9x) |
| 2 | CAM16/CIECAM02 inverse (~360 ms) — serial, ~450 flops/col | **parallel** | `mcs_appearance_inverse` has no `parallelize` path at all (forward does). Largest single-function win available; same 32k-chunk pattern as forward. GPU after parallel (6.3 flops/byte). |
| 3 | XYZ_to_Oklab (156–166 ms) — serial, 3× `spow` per colour | **simd** then **parallel** | serial loop in `mcs_perceptual`; forward alone is slower than inverse by 26x (6.4 vs 166 ms) purely on `pow`; SIMD converts scalar `pow` to vector `spow` on 3 independent lanes like Lab already does; numpy's fully-vectorised cbrt pipeline is only 86 ms — pure SIMD + no split should beat that. |
| 4 | CAM16/CIECAM02 H (130–147 ms delta on top of no-H) | **unroll** (fuse into Mojo) | `_hue_quadrature` is pure numpy (16–81 ms): per-element branchy scalar work that fits the existing Mojo loop pattern (hue branches at `appearance.mojo`). No ctypes change; highest-ROI Python-layer fix. |
| 5 | XYZ_to_IPT / IPT_to_XYZ (120–127 ms) | **simd** then **parallel** | same 3× `spow` pathology as Oklab; same fix. |
| 6 | cctf_encoding (119–125 ms) / cctf_decoding (107–110 ms) | **parallel** (encoding) / **simd** (both) | scalar `pow` per element, no memory pressure; parallelise as easily as Lab. |
| 7 | sRGB_to_XYZ / XYZ_to_sRGB (108–119 ms) | **parallel** | flat 107–111 ns/col serial from 10k to 1M; parallel gains 5–9x on this machine (measured on three kernels). SIMD on the decode path is a secondary option. |
| 8 | xy_to_XYZ (25–27 ms) | **unroll** (Python: fuse `ones` append into kernel) | 46% of time is `np.concatenate`; pass `1.0` as the Y argument to a 2-column kernel. |
| 9 | CAM16 forward no-H (41–51 ms) | **none** | already parallel+SIMD-ish; measured 8x uplift; residual time is response compression. GPU is a later candidate (4.9 flops/byte) but CPU-parallel is already close to roofline. |
| 10 | Jzazbz CPU (33–42 ms), Jzazbz GPU (6–8 ms) | **none** (GPU path proven) / **simd** | CPU-parallel gives 9x; GPU gives another 5.8x. No further action unless extending GPU to inverse (already present). |
| 11 | XYZ_to_Luv (33–34 ms), Luv_to_XYZ (10 ms), Lab_to_XYZ (7–9 ms) | **parallel** (Luv fwd) / **none** | 34 ms serial → ~6 ms parallel; others are already fast. |
| 12 | delta_E 1994 (18 ms) | **parallel** with delta_E2000 | shares the same serial loop. |
| 13 | delta_E 1976 (5–6 ms) | **none** | memory-bound, already 12x upstream. |
| 14 | XYZ_to_Lab (13–17 ms), XYZ_to_xyY etc. (4–5 ms) | **none** | already SIMD+parallel or memory-bound. Jzazbz GPU already shipped and verified. |

**GPU-candidate summary:** only kernels above ~2 flops/byte are worth a
device round trip — delta_E2000 (7.5), CAM16 inverse (6.3) and CAM16
forward-with-H (4.9). Everything below 2 flops/byte (sRGB transcode, xyY,
1976 delta_E, all Luv/Lab-inverse paths) stays on CPU; for those,
**parallel** at large N and **none** once parallel gains are measured.

## Method and confidence notes

- FLOP counts: each 3×3 matmul = 18 flops (9 mul + 9 add); a division is 2x
  a multiply; `sqrt`/`cbrt` ≈ 5; scalar `pow`/`exp`/`cos` ≈ 20; SIMD lanes
  counted per element. These are estimates for ranking only — every ms
  column is measured.
- Bytes: read + write minimum traffic; caches and read-for-ownership are
  ignored, so memory-bound labels are conservative in favour of compute-bound.
- “Compute-bound” means measured per-colour ns sits ≥10x above the
  bandwidth time implied by bytes at a plausible 10 GB/s effective rate.
- Glue overhead <1 ms at 1M verified by raw-vs-wrapper deltas; small-N
  overhead ~9–11 µs/call verified at N=1–10k.
- `pixi run bench` (this session) reproduced the README table within
  3–25% per row (run-to-run plus different reserve: this session's
  standalone profiling agrees with `bench` to within 10% on shared rows).
