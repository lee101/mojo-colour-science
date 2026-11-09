# VERIFY — mojo-colour-science benchmark and parity verification

Independent re-measurement of the README claims, performed in the VERIFY phase
on the same machine named in the README: Intel Xeon E5-2697 v4 @ 2.30 GHz,
Linux x86-64, NVIDIA GeForce RTX 5090 (8,419 MiB free). Stack: Python 3.13.14,
numpy 2.5.1, colour-science 0.4.7, Mojo 1.2.0.dev2026092605.

## What was run, in order

```
pixi run build && pixi run test   # green: build OK (deprecation warnings only), 48 passed
pixi run bench                    # 4 full passes in this session (all exit 0)
```

Each `pixi run bench` is `bench/bench.py` (best of 3, warmup inside, same
contiguous float64 1M-colour inputs for both sides, per-row output-parity
assert at full size before timing). The README table was then replaced with
one of those runs verbatim (see below); no figure in the README comes from
any other source.

## Reproduced / not reproduced (README figures vs two independent passes)

"Within 20%" = Mojo time, upstream time, and speedup each within ±20% of the
old README figure. Independent passes: an interleaved A/B script (two
best-of-3 passes, order swapped) and repeated `pixi run bench` passes.

| Kernel | Reproduced within 20%? | Evidence |
|---|---|---|
| sRGB_to_XYZ (1M) | Yes | 3 of 4 session runs within 20% on all three figures |
| XYZ_to_Lab (1M) | No | Mojo ≈ 17 ms (old: 25.03, −32%); upstream ≈152 ms (old: 264.29, −43%) |
| Lab_to_XYZ (1M) | Yes (borderline) | Mojo −11%, upstream −14%, speedup −3% in independent pass |
| XYZ_to_Oklab (1M) | No | Mojo ≈157–167 ms (old: 135.08); upstream ≈266–303 ms (old: 347.57); speedup ≈1.6–1.9x (old: 2.57x) |
| XYZ_to_Jzazbz (1M) | No | Upstream ≈460–490 ms (old: 593.11); speedup ≈10–13x (old: 15.82x) |
| XYZ_to_Jzazbz GPU (1M) | No | Mojo ≈6.4–7.2 ms (old: 17.30); upstream ≈460–510 ms (old: 744.59); speedup ≈67–79x (old: 43.04x) |
| delta_E CIE 2000 (1M) | Yes | Mojo −3%, upstream −18%, speedup −15% in independent pass |
| XYZ_to_CAM16 (1M, H off) | No (times) | Both sides ≈25–27% faster than old figures; speedup itself reproduces (10.5–11.3x vs 10.97x) |

For every row that did not reproduce on all three figures, the README was
corrected to the figures from a single `pixi run bench` pass executed in
this session:

```
| sRGB_to_XYZ (1M) | 108.84 ms | 367.20 ms | 3.37x |
| XYZ_to_Lab (1M) | 21.49 ms | 142.70 ms | 6.64x |
| Lab_to_XYZ (1M) | 6.98 ms | 124.02 ms | 17.78x |
| XYZ_to_Oklab (1M) | 157.53 ms | 261.49 ms | 1.66x |
| XYZ_to_Jzazbz (1M) | 41.93 ms | 493.28 ms | 11.76x |
| XYZ_to_Jzazbz GPU (1M) | 7.06 ms | 487.32 ms | 69.05x |
| delta_E CIE 2000 (1M) | 277.55 ms | 632.06 ms | 2.28x |
| XYZ_to_CAM16 (1M, H off) | 53.23 ms | 559.36 ms | 10.51x |
```

All eight rows remain speedups in this run; no row is slower than upstream,
so no "slower" row had to be kept. The accompanying README prose ("All
measured kernels were faster in this run") is unchanged and still accurate
for the corrected table.

Note on spread: upstream `colour-science` times are the noisy side (e.g.
XYZ_to_Lab upstream 137–154 ms across session runs). Mojo times are tighter.
The old README's upstream figures for Lab/Jzazbz/GPU/CAM16 look like they
came from a more loaded machine moment; none of the four session runs got
near them, so they are not defensible as current-machine numbers.

## Same input / no skipped work — audited

- **Same input, both sides:** every case in `bench/bench.py` closes over one
  shared array (`rgb`, `xyz`, `lab`, `other_lab`, `xyz100`) created with
  `default_rng(7)`; both lambdas receive that identical object. Inputs are
  `ascontiguousarray` float64. Verified by reading the harness and by
  reproducing its setup in the independent script (results agree).
- **No skipped work on the Mojo side:** each wrapper allocates a fresh
  `np.empty_like` output and calls the full kernel over all N colours —
  two successive calls return distinct objects with equal values (checked).
  No caching or early exit (empty-input guard is `count == 0` only).
- **No skipped work on the upstream side:** `colour-science` vectorises over
  the full arrays (1M unique output rows, float64, C-contiguous — checked);
  its runtime scales ~1000x from 10 to 1M colours for `XYZ_to_Lab`
  (0.21 ms → 177 ms), i.e. no sampling/shortcut.
- **CAM16 `compute_H=False` is symmetric:** both sides return all six
  correlates plus `H = NaN`; parity holds on J/C/h/s/Q/M at 1k and in the
  65,539-colour parity test. The H quadrature work is off on *both* sides,
  so the comparison is fair; with H on, the Python-side `_hue_quadrature`
  (numpy) would be the bottleneck for Mojo (see `bench/analysis.md`).
- **GPU row symmetry:** the GPU case times Mojo's `device="gpu"` path —
  which *includes* device-context creation, buffer allocation, H2D/D2H
  transfers, kernel launch, and host synchronisation (Mojo
  `mcs_jzazbz_gpu`, called on every bench iteration; falls back to CPU with
  a warning only on failure) — against upstream CPU `XYZ_to_Jzazbz`. The
  ~70x speedup is therefore an honest end-to-end (transfer-inclusive)
  GPU-vs-CPU figure, not kernel-only.
- **Parity gate inside the bench:** every row asserts
  `allclose` (rtol 2e-6 / atol 3e-7, CAM `J` atol 2e-6) between Mojo and
  upstream outputs on the full 1M input before timing. All rows passed in
  every session run.

## Tests — `git log -p -- tests/` audit

Tests were added once, in `85c1d26` (initial port), and never modified or
deleted afterwards (no later commit touches `tests/`; `git log --all -p --
tests/` shows zero removed assertion/tolerance lines; working tree clean
before this phase). So there were no loosened tolerances or removed
assertions from earlier phases to restore.

Current tolerances, all asserting parity against real `colour-science`
0.4.7 (not just round trips): sRGB 2e-9, Lab 3e-7, Luv 2e-7, Oklab/IPT 3e-9
fwd / 3e-7 inv, Jzazbz rtol 2e-7 / atol 2e-10 fwd, delta_E 2e-12, CAM16/
CIECAM02 correlates 2e-6, Sharma CIEDE2000 vectors abs 5e-5, published
CAM16 example J/h abs 1e-7. These are tight enough that the parity
assertion in the bench itself (rtol 2e-6) is the *loosest* check in the
repo, and it still gates the benchmark.

## What changed in this phase

- `README.md`: benchmark table replaced with the session-measured
  `pixi run bench` figures above (5 rows corrected outright, the other 3
  re-anchored to the same run for consistency).
- `bench/verify.md`: this file.
- **No code changed.** No bug reproducible by a failing test was found:
  `pixi run build && pixi run test` was green before (48 passed) and after
  (48 passed), and every bench row's parity assertion held at full 1M size.

## Final state

```
pixi run build && pixi run test   # exit 0; build OK, 48 passed
```

Reproduction note: exact millisecond figures move run to run (see spread
above); the README numbers are one genuine `pixi run bench` pass from this
machine and session, quoted verbatim.
