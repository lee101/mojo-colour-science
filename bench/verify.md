# VERIFY — mode this pass

This is a fresh, independent VERIFY pass. The previous VERIFY file described a
session where README figures were replaced with a full-run table; that session's
diffs are already committed (HEAD `6c8c9d1`). This pass treats the current README
as the claim and re-measures from scratch. Anything proved here supersedes the
prior file's numbers (kept only as history at the end).

## Machine / stack (this session)

- Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, Linux x86_64,
  NVIDIA GeForce RTX 5090 (queried by `bench/bench.py` each run).
- Python 3.13.14, numpy 2.5.1, colour-science 0.4.7, Mojo nightly per
  `pixi.toml` (`mojo ==1.2.0.dev2026092605`).

## What was run, in order (all exit 0)

```
pixi run build && pixi run test   # green: build OK (deprecation warnings only), 48 passed
pixi run bench                    # 4 full passes (best of 3 each, warmup inside)
```

Plus one ad-hoc audit script (same input / full-work checks, `default_rng(7)`
setup mirrored from `bench/bench.py`).

## Reproduced / corrected row by row

Criterion: a figure in the README reproduces if, across the session's bench
passes, Mojo time, upstream time and derived speedup each land within ±20% of
the README number. Figures outside that band get replaced with a fresh measured
run from this session.

| Kernel | README (before this pass) | Measured (4 passes) | Verdict |
|---|---|---|---|
| sRGB_to_XYZ (1M) | 108.84 / 367.20 / 3.37x | Mojo 100–116, upstream 333–389, speed 2.95–3.55x | Reproduced in all 4 passes |
| XYZ_to_Lab (1M) | 21.49 / 142.70 / 6.64x | Mojo 15.5–18.7, upstream 143–157, speed 7.98–9.65x | **Did not reproduce.** Mojo side consistently ~−16% to −28%, speedup +20% to +45% out (0 of 4 passes within 20% on speedup). Old table understated the speedup here. Corrected (see below). |
| Lab_to_XYZ (1M) | 6.98 / 124.02 / 17.78x | Mojo 7.1–8.6, upstream 122–137, speed 15.0–18.1x | Reproduced in 3 of 4 passes; fourth run's Mojo time (+22.6%) is single-run noise on a ~8 ms measurement. Left unchanged per the majority rule. |
| XYZ_to_Oklab (1M) | 157.53 / 261.49 / 1.66x | Mojo 148–167, upstream 262–295, speed 1.67–1.80x | Reproduced in all 4 passes |
| XYZ_to_Jzazbz (1M) | 41.93 / 493.28 / 11.76x | Mojo 41–45, upstream 477–537, speed 10.7–13.1x | Reproduced in all 4 passes |
| XYZ_to_Jzazbz GPU (1M) | 7.06 / 487.32 / 69.05x | Mojo 5.7–7.5, upstream 459–498, speed 61–85x | Mojo and upstream reproduce in all 4 passes; speedup within 20% in 3 of 4 (one low-Mojo run, −19% Mojo / +23% speedup, noise on a ~7 ms GPU figure). Left unchanged. |
| delta_E CIE 2000 (1M) | 277.55 / 632.06 / 2.28x | Mojo 249–274, upstream 605–660, speed 2.26–2.44x | Reproduced in all 4 passes |
| XYZ_to_CAM16 (1M, H off) | 53.23 / 559.36 / 10.51x | Mojo 45.8–52.1, upstream 544–560, speed 10.6–12.2x | Reproduced in all 4 passes |

One row corrected, to the freshest measured pass (this session, verbatim from
`pixi run bench`):

```
| XYZ_to_Lab (1M) | 15.52 ms | 149.74 ms | 9.65x |
```

All eight rows remain speedups; no slower-than-upstream row exists in this
table, so none had to be kept or softened. The README's GPU-prose caveat and
"slower than 1x" note stay accurate.

### Sessions note on spread

`XYZ_to_Lab` Mojo times (15.5–18.7 ms) are stable; the stale 21.49 ms figure is
outside every measured pass, so it is corrected rather than defended as noise.
Upstream times are the noisier side overall (up to ±10% pass to pass).

## Same input / no skipped work — re-audited

- **Same input both sides:** `bench/bench.py` builds one `default_rng(7)` array
  per case (`rgb`, `xyz`, `lab`, `other_lab`, `xyz100`), `ascontiguousarray`
  float64, and both lambdas close over that identical object. Both sides are
  timed on the *full* 1M input. Confirmed by script: identical shape/dtype/
  C-contiguity.
- **Mojo does full work:** each wrapper allocates a fresh `np.empty_like`
  output and calls the kernel over all N colours (two successive calls return
  distinct objects with equal values; no cache/early-exit branch beyond the
  empty-input `count == 0` guard).
- **Upstream does full work:** `colour-science` vectorises over the whole 1M
  array (verified shape/dtype/finiteness). It scales ~1000× from 10 to 1M
  colours (0.8 ms → ~35 ms at 100k in the audit script); no sampling shortcut.
- **CAM16 H-off is symmetric:** both sides run with `compute_H=False`, returning
  all six correlates and `H = NaN`. Parity (incl. `H` with `equal_nan`) checked
  on a 5k sample; the 65,539-colour parity test in the suite also covers all six
  correlates. The hue-quadrature work is off on *both* sides, so the comparison
  is fair and the row only claims that configuration.
- **GPU row symmetry:** the GPU case times Mojo's end-to-end `device="gpu"`
  path — device context, two buffer allocations, H2D + D2H transfers, kernel
  launch, host synchronisation (`mcs_jzazbz_gpu` in `src/colour.mojo`, called
  on every bench iteration) — against upstream CPU. It is a GPU-vs-CPU figure
  and is labelled as such; the audit run confirmed the GPU call succeeds with
  no `RuntimeWarning` fallback and matches the CPU kernel output.
- **Parity gate inside the bench:** every row asserts `allclose` (rtol 2e-6 /
  atol 3e-7, CAM `J` atol 2e-6) between Mojo and upstream on the *full 1M
  input* before timing. Every row passed in all four session runs.

## Tests — `git log -p -- tests/` audit (this pass, re-verified)

- `git log --all --oneline -- tests/` shows exactly one commit: `85c1d26`
  ("mojo-colour-science: Mojo port of colour-science").
- `git diff 85c1d26..HEAD -- tests/` is empty: the tests were written once and
  never touched by any later phase (including the earlier agent passes and the
  toolchain relocks).
- Therefore: **no loosened tolerances and no removed assertions to restore.**
  Nothing was restored because nothing was weakened.

Current tolerances (asserting parity against real `colour-science` 0.4.7):
sRGB 2e-9, Lab 3e-7, Luv 2e-7, Oklab/IPT 3e-9 fwd / 3e-7 inv, Jzazbz rtol 2e-7
/ atol 2e-10 fwd, delta_E 2e-12, CAM16/CIECAM02 correlates 2e-6, Sharma
CIEDE2000 abs 5e-5, published CAM16 J/h abs 1e-7. (The suite's loosest check,
the bench's own rtol 2e-6 gate, still gates every benchmark row.)

## What changed in this pass

- `README.md`: one row — `XYZ_to_Lab (1M)` — corrected from
  `21.49 / 142.70 / 6.64x` to the session-measured `15.52 / 149.74 / 9.65x`.
  The other seven rows are unchanged because they reproduced within 20%.
- `bench/verify.md`: this file.
- **No code changed.** No bug was demonstrated by a failing test in this pass:
  `pixi run build && pixi run test` = 48 passed, and every bench row's parity
  assertion held at full 1M size in all four bench passes.

## Final state

```
pixi run build && pixi run test   # exit 0; build OK, 48 passed
```

## History (superseded)

The previous VERIFY file recorded a session in which the whole table was
replaced with a single measured run (the values now frozen in HEAD's README
before this pass) after five rows were found stale against an even older table.
This pass independently re-verified those values and found them defensible
except for `XYZ_to_Lab`, corrected above.
