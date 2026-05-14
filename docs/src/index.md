```@meta
CurrentModule = MRISubspaceRecon
```

# MRISubspaceRecon.jl

The [MRISubspaceRecon.jl](https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl) package aims to enable rapid iterative reconstruction 
of Cartesian and non-Cartesian MRI data using subspace modeling [1,2]. Particular care is given to enable the reconstruction of large numbers of subspace coefficients along with large image grid sizes.

For compatibility with other Julia packages, such as [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl) and [RegularizedLeastSquares.jl](https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl), operations are defined in terms of linear operators and their effects on the data vectors. [MRISubspaceRecon.jl](https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl) is designed to compute these objects with multi-threaded CPUs and on NVIDIA GPUs. The package further contains functions to perform GRAPPA operator gridding (GROG) [3] or to generate radial trajectories. However, all methods that require an explicit k-space trajectory as input generalize to arbitrary trajectories.

The documentation of all exported functions can be found in the [API](@ref) Section.

# Background

A general (non-Cartesian) subspace reconstruction of coefficient
images $\hat{\bm{x}_c}$ from k-space data $\bm{y}$ can be formulated as:
```math
\hat{\bm{x}}_c = \underset{\bm{x}_c}{\text{arg min}} \lVert \mathbf{GFSU}_R\,\bm{x}_c-\bm{y} \rVert_2^2 
\equiv \underset{\bm{x}_c}{\text{arg min}} \lVert \mathbf{A}\bm{x}_c-\bm{y} \rVert_2^2
```
where $\mathbf{F}$ and $\mathbf{G}$ are block-diagonal Fourier and gridding matrices. $\mathbf{A}$ and $\mathbf{S}$ represent the forward operator (or system matrix) and coil sensitivity profiles, respectively. $\mathbf{U}_R$ is a block matrix composed of $R$ temporal
basis functions; its adjoint $\mathbf{U}_R^H$ projects the time series of images onto the basis: $\bm{x}_c = \mathbf{U}_R^H\bm{x}_t$, where $\bm{x}_t$ denotes the complete time series of images. 
    
To decrease computation time, iterative reconstructions employ Toeplitz embedding to avoid repeated gridding operations of non-Cartesian data [4,5,6]. This means that the normal operator $\mathbf{A}^H\mathbf{A}$ 
is implemented as a pointwise multiplication at the cost of an oversampling factor of 2 across each spatial dimension. The resulting operator is an $R\times R$ matrix per k-space sample and thus has a large total size of $8N_xN_yN_zR^2$ for a 3D image. We exploit several methods throughout to handle the large memory requirement, especially when using a GPU for reconstruction. A main point for users is that a real-valued basis $\mathbf{U}_R$ reduces the memory requirement for storing the normal operator by a factor of 2.

# CPU or GPU
Reconstructions of non-Cartesian MRI in [MRISubspaceRecon.jl](https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl) are implemented for CPU and NVIDIA GPUs. The GPU code is included as a Julia extension and is loaded after importing CUDA.jl via:
```
using MRISubspaceRecon
using CUDA
```
We recommend using the GPU code which can be faster by a factor of 10--20 than CPU multi-threading (for typical solvers like conjugate gradient or FISTA [7]).

# References
1. Assländer J, et al. “Low rank alternating direction method of multipliers reconstruction for MR fingerprinting”. Magn Reson Med 79.1 (2018), pp. 83–96. https://doi.org/10.1002/mrm.26639
2. Tamir JI, et al. “T2 shuffling: Sharp, multicontrast, volumetric fast spin-echo imaging”. Magn Reson Med. 77.1 (2017), pp. 180–195. https://doi.org/10.1002/mrm.26102
3. Seiberlich N, Breuer F, Blaimer M, Jakob P, and Griswold M. "Self-calibrating GRAPPA operator gridding for radial and spiral trajectories". Magn. Reson. Med. 59 (2008), pp. 930-935. https://doi.org/10.1002/mrm.21565
4. FTAW Wajer and KP Pruessmann. “Major Speedup of Reconstruction for Sensitivity Encoding with Arbitrary Trajectories”. In: Proc. Intl. Soc. Mag. Reson. Med 9 (2001)
5. M Uecker, S Zhang, and J Frahm. “Nonlinear inverse reconstruction for real-time MRI of the human heart using undersampled radial FLASH”. In: Magnetic Resonance in Medicine 63.6 (2010), pp. 1456–1462. doi: 10.1002/mrm.22453.
6. M Mani et al. “Fast iterative algorithm for the reconstruction of multishot non-cartesian diffusion data” In: Magnetic Resonance in Medicine 74.4 (2015), pp. 1086–1094. doi: 10.1002/mrm.25486.
7. A Beck and M Teboulle. “A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems”. In: SIAM Journal on Imaging Sciences 2.1 (2009), pp. 183–202. doi: 10.1137/080716542.