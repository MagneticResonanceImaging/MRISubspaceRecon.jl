# Changelog

All notable changes to MRISubspaceRecon are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-1.0: the minor version is bumped for breaking changes, the patch version
otherwise).

## [0.10.1] - unreleased

### Fixed

- **GPU reconstruction with a complex-valued basis was entirely broken.**
  `kernel_mul!` (the `Complex`-`U` variant in the CUDA extension) referenced
  the undefined names `k` and `j` when choosing between `Λ` and `conj(Λ)`,
  which aborted kernel compilation with
  `InvalidIRError: unsupported use of an undefined name`. The intended
  comparison is between the two coefficient loop indices, and since the
  `ind_lookup` table is symmetric only the condition needed correcting
  (`if ic2 >= ic1`). Cross-validated against the CPU implementation, which
  stores the full `Λ[ic1, ic2, it]` rather than the packed upper triangle:
  relative difference 1.5e-6, i.e. the same tolerance as the previously
  working real-`U` paths.

- **`test/runtests.jl` did not include the GPU test files.**
  `reconstruct_radial_gpu.jl`, `reconstruct_radial_gpu_real.jl`,
  `reconstruct_cart_trj_gpu.jl` and `wrapper_gpu.jl` existed in `test/` but
  were never run, which is why the bug above shipped in 0.10.0. They are now
  included behind a `CUDA.functional()` guard (skipped with an `@info`
  message on CPU-only machines). This adds ~16 400 assertions.

### Added

- GPU entry points now accept `AnyCuArray` instead of the concrete `CuArray`,
  so callers may pass **views and reshapes** of device arrays directly:
  `calculate_backprojection`, `NFFTNormalOp`, `calculate_kmask_indcs`,
  `calculate_kernel_noncartesian`, `calculate_coil_maps` and
  `reconstruct_coilwise`. This is backward compatible, because
  `CuArray <: AnyCuArray`.

  The motivating case is per-subsequence or per-cycle processing of a large
  dataset: slicing a contiguous range out of a multi-GiB device array
  previously forced a full copy, and the copy stayed GC-rooted for the
  lifetime of the enclosing function frame. Passing an `@view` avoids the
  copy entirely (`reshape(@view trj[:, :, :, r], 3, :)` is zero-copy — the
  result is a plain `CuArray` with an offset pointer). Previously such views
  failed with `MethodError: no method matching _prepend!(...)`.

### Changed

- **Lower peak GPU memory in the CUDA extension.** The sample mask is now
  resolved once via the new `MaskUtils.jl` helpers rather than repeatedly
  inside loops:

  - `findall` is hoisted out of the coil × coefficient loop in
    `calculate_backprojection` (previously re-run on every iteration).
  - A fully-sampled mask (`all(sample_mask)`) now takes a `nothing` fast
    path that skips `findall` altogether and reshapes instead of gathering.
  - Where a gather is still needed, linear `Int` indices are used rather
    than the `CartesianIndex{2}` that GPU logical indexing produces for a
    2-D mask — 8 instead of 16 bytes per selected sample, and no `cumsum`
    temporary of `8 * length(mask)` bytes.
  - The trajectory is gathered and converted to NUFFT point convention
    **once** and shared between `calculate_kmask_indcs` and `set_points!`
    (new `points` keyword and the new `_kmask_and_plan` helper), instead of
    twice. Doing this in a separate function also means the temporary is
    unrooted on return, before `Λ`/`λ`/`S` are allocated.

  Note that GPU logical indexing is *eager*: unlike the CPU, where
  `A[mask]` builds a lazy `Base.LogicalIndex`, `Base.to_index` on a device
  array calls `findall`, so every masked index materialises a device array.
  This is the reason the above matters.

  Measured with a ballast experiment that pins the amount of free device
  memory (`img_shape = (96, 96, 64)`, 2240 readout points, 20 cycles,
  600 time points, 20 coils, 3 coefficients): the copy-based caller runs out
  of memory at 11.3 GiB of ballast while the view-based caller still
  succeeds, and views leave ~2.1 GiB more headroom at the `NFFTNormalOp`
  call — exactly the size of the avoided copy. All successful runs produce
  an identical result.

### Validation

- `runtests.jl` (including the newly wired GPU tests) passes in full.
- Backprojection and `NFFTNormalOp` outputs were compared against the
  0.10.0 release with **dependency versions held fixed** (only the manifest
  `path` entry differing), for real and complex bases, masked and
  fully-sampled. Relative differences of 1e-7 … 2e-7 were observed — but a
  determinism control (the *same* code run twice) gives differences of the
  same magnitude, so this is GPU atomic-ordering nondeterminism rather than
  a change in behaviour. Nothing here is bitwise reproducible.
- GPU results were additionally cross-checked against the independent CPU
  implementation: 1.5e-6 … 2.2e-6 relative difference throughout.
