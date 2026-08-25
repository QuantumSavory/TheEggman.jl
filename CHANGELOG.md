# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-25

### Added

- `hafnian` and `hafnian_repeated` functions using three backend implementations: unrolled expression, dynamic programming, and finite-difference sieve.
- Cost estimator to automatically choose between unrolled, DP, and sieve based on prior benchmark calibration.
- GPU support through `KernelAbstractions.jl`.
- Benchmark suite comparing against `thewalrus`, `piquasso`, and `perceval` (permanents only).
