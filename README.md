# MRISubspaceRecon.jl


| **Documentation**         | **Paper**                   | **Build Status**                      |
|:------------------------- |:--------------------------- |:------------------------------------- |
| [![][docs-img]][docs-url] | [![][paper-img]][paper-url] | [![][gh-actions-img]][gh-actions-url] |
|                           | [![][arXiv-img]][arXiv-url] | [![][codecov-img]][codecov-url]       |


MRISubspaceRecon.jl is a Julia package that implements sub-space reconstructions as described by [Jon Tamir et al.](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.26102) and by us in the paper [Low-rank alternating direction method of multipliers reconstruction for MR fingerprinting](https://doi.org/10.1002/mrm.26639) as *low rank inversion*.

Currently, the package supports Cartesian and non-Cartesian trajectories on single- or multithreaded CPUs, as well as NVIDIA GPUs. The package is still under active development. Its ultimate goal is to provide a comprehensive Julia implementation for MRI subspace reconstructions, analagous to our [Matlab package](https://bitbucket.org/asslaender/nyu_mrf_recon).


[docs-img]: https://img.shields.io/badge/docs-latest%20release-blue.svg
[docs-url]: https://MagneticResonanceImaging.github.io/MRISubspaceRecon.jl/stable

[gh-actions-img]: https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl/workflows/CI/badge.svg
[gh-actions-url]: https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl/actions

[codecov-img]: https://codecov.io/gh/MagneticResonanceImaging/MRISubspaceRecon.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/MagneticResonanceImaging/MRISubspaceRecon.jl

[arXiv-img]: https://img.shields.io/badge/arXiv-1608.06974-blue.svg
[arXiv-url]: https://arxiv.org/pdf/1608.06974.pdf

[paper-img]: https://img.shields.io/badge/doi-10.1002/mrm.26639-blue.svg
[paper-url]: https://doi.org/10.1002/mrm.26639
