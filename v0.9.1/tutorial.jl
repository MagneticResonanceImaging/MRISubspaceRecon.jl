#md # [![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/USER/REPO/main?filepath=docs/src/build_literate/tutorial.ipynb)

# # Non-Cartesian MRI

# This example illustrates how to perform an iterative multi-coil MRI subspace reconstruction from non-Cartesian k-space data
# with the conjugate gradient algorithm. To perform such a reconstruction, we need the following packages:
using MRISubspaceRecon
using IterativeSolvers # for conjugate gradient reconstruction
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide #!nb
nothing #hide #!nb
# # Data simulation

# We first simulate some data from a Shepp-Logan phantom and generate some coil maps using
# various phase modulations. The below packages are only used for the data simulation:
using FFTW
using ImagePhantoms
using NonuniformFFTs
using Random
using LinearAlgebra

Nx = 32
Nc = 2 # nr of coefficients in the temporal subspace
Nt = 10 # nr of acquired time frames per cycle
Ncyc = 20 # nr of cycles (i.e., repeats of flip angle pattern)
Ncoil = 2
img_shape = (Nx, Nx) # 2D image in this example

## create test coefficient image
x = zeros(ComplexF32, Nx, Nx, Nc)
x[:, :, 1] = transpose(shepp_logan(Nx)) .* exp(1im * π / 3)
x[:, :, 2] = shepp_logan(Nx)

p = heatmap(abs.(x[:,:,1]), layout=(1,2), subplot=1, ticks=[], colorbar=false, size=(700,350), title="coeff. 1")
heatmap!(p, abs.(x[:,:,2]), subplot=2, ticks=[], colorbar=false, title="coeff. 2")
#md Main.HTMLPlot(p) #hide
# Next, we set up a set of coil maps and a trajectory for data acquisition. We then generate a set of basis functions. The non-Cartesian methods
# use float trajectories in range $k \in [-0.5, 0.5)$, as opposed to integer trajectories for Cartesian methods.
## coil maps as vector of complex arrays
cmaps = [ones(ComplexF32, Nx, Nx); ones(ComplexF32, Nx, Nx) .* ComplexF32.(exp(1im * π / 2))]
println("typeof(cmaps) = $(typeof(cmaps))")
println("size(cmaps) = $(size(cmaps))")

## set up a 2D radial trajectory
trj = traj_2d_radial_goldenratio(2Nx, Ncyc, Nt; adc_dim=false) # 2Nx for oversampling

## set up basis functions
U = randn(ComplexF32, Nt, Nc)
U, _, _ = svd(U)
println("typeof(trj) = $(typeof(trj))")
println("typeof(U) = $(typeof(U))")
println("size(U) = $(size(U))")
# Finally, we use the phantom image `x`, the coil maps `cmaps`, the trajectory `trj`, and the basis functions `U` to simulate k-space data:
## simulate data as (2Nx*Ncyc, Nt, Ncoil)-shaped array
data = Array{ComplexF32,3}(undef, 2Nx * Ncyc, Nt, Ncoil)
nfftplan = PlanNUFFT(ComplexF32, img_shape; fftshift=true)
for icoil ∈ axes(data, 3)
    xcoil = copy(x) 
    xcoil .*= cmaps[icoil] # scale image by coil map
    for it ∈ axes(data, 2)
        set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(reshape(trj[:, :, it], 2, :))) # prep NUFFT
        xt = reshape(reshape(xcoil, :, Nc) * U[it, :], Nx, Nx)
        ## simulate data from image using type-2 (uniform to non-uniform) NUFFT
        @views NonuniformFFTs.exec_type2!(data[:, it, icoil], nfftplan, xt)
    end
end
println("size(data)   = $(size(data))") # array shape of data
println("typeof(data) = $(typeof(data))") # type of input data

# By default, the data format uses either 3D arrays. Alternatively, 4D arrays can be used as inputs, where the 4D format is used to place ADC points within a separate array axis from the total number of samples.
# Internally, all code relies on 3D arrays and the 4D arrays are handled by wrappers.
data = reshape(data, 2Nx, Ncyc, Nt, Ncoil)
trj = reshape(trj, 2, 2Nx, Ncyc, Nt)
println("size(data) = $(size(data))")
println("size(trj)  = $(size(trj))")
# # Sensitivity profiles
# Coil maps may be auto-calibrated from k-space measurements using ESPIRiT:
cmaps = calculate_coil_maps(data, trj, img_shape; U)
println("size(cmaps) = $(size(cmaps))")
# # Sample mask
# Furthermore, reconstructions can make use of a binary mask to exclude specific samples from being included in the reconstruction. 
# To illustrate the data removal, we create a mask that removes one time frame from one cycle:
## create sampling mask
it_rm = 1
icyc_rm = 5
sample_mask = trues(2Nx, Ncyc, Nt)
sample_mask[:, icyc_rm, it_rm] .= false
println("typeof(sample_mask) = $(typeof(sample_mask))")
println("size(sample_mask) = $(size(sample_mask))")
# # Normal operator and adjoint 
# Now, we can compute the normal operator using the sample mask:
AᴴA = NFFTNormalOp(img_shape, trj, U; cmaps, sample_mask)
println(AᴴA)
# We can also compute the adjoint NUFFT (backprojection) with the specified sampling mask:
b = calculate_backprojection(data, trj, cmaps; U, sample_mask)
println("size(b) = $(size(b))")

p = heatmap(abs.(b[:,:,1]), layout=(1,2), subplot=1, ticks=[], colorbar=false, title="coeff. 1", size=(700,350))
heatmap!(p, abs.(b[:,:,2]), subplot=2, ticks=[], colorbar=false, title="coeff. 2")
#md Main.HTMLPlot(p) #hide
# # Iterative solvers
# The normal operator `A` and the backprojection `b` are compatible with the iterative solvers from [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl) and 
# [RegularizedLeastSquares.jl](https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl). This enables solving the inverse problem with various algorithms, including conjugate gradient (CG). In this way we can
# recover the original image from the backprojection:
## solve inverse problem with CG. GPU-methods are called through multiple dispatch, i.e., when objects of type `CuArray` are passed as arguments. 
xr = cg(AᴴA, vec(b), maxiter=20)
xr = reshape(xr, Nx, Nx, Nc) # reshape vector back to 2D image with Nc coefficients

p = heatmap(abs.(xr[:,:,1]), layout=(1,2), subplot=1, ticks=[], colorbar=false, size=(700,350), title="coeff. 1")
heatmap!(p, abs.(xr[:,:,2]), subplot=2, ticks=[], colorbar=false, title="coeff. 2")
#md Main.HTMLPlot(p) #hide