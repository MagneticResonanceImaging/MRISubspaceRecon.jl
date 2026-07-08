# Test that the low-memory (decomposed Toeplitz) operator produces
# the same results as the standard NFFTNormalOp.

using MRISubspaceRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using NonuniformFFTs
using Test
using Random
Random.seed!(42)

T = Float32
Nx = 32
Nc = 4
Nt = 20
Ncyc = 10

img_shape = (Nx, Nx)

## create test image
x = zeros(Complex{T}, Nx, Nx, Nc)
x[:, :, 1] = transpose(shepp_logan(Nx))
x[1:end÷2, :, 1] .*= exp(1im * π / 3)
x[:, :, 2] = shepp_logan(Nx)

## coil maps
Ncoil = 9
cmaps = ones(Complex{T}, Nx, Nx, Ncoil)
[cmaps[i, :, 2] .*= exp(1im * π * i / Nx) for i ∈ axes(cmaps, 1)]
[cmaps[i, :, 3] .*= exp(-1im * π * i / Nx) for i ∈ axes(cmaps, 1)]
[cmaps[:, i, 4] .*= exp(1im * π * i / Nx) for i ∈ axes(cmaps, 2)]
[cmaps[:, i, 5] .*= exp(-1im * π * i / Nx) for i ∈ axes(cmaps, 2)]
[cmaps[i, :, 6] .*= exp(2im * π * i / Nx) for i ∈ axes(cmaps, 1)]
[cmaps[i, :, 7] .*= exp(-2im * π * i / Nx) for i ∈ axes(cmaps, 1)]
[cmaps[:, i, 8] .*= exp(2im * π * i / Nx) for i ∈ axes(cmaps, 2)]
[cmaps[:, i, 9] .*= exp(-2im * π * i / Nx) for i ∈ axes(cmaps, 2)]

for i ∈ CartesianIndices(@view cmaps[:, :, 1])
    cmaps[i, :] ./= norm(cmaps[i, :])
end
cmaps = [cmaps[:, :, ic] for ic = 1:Ncoil]

## set up trajectory
α_g = 2π / (1 + √5)
phi = Float32.(α_g * (1:Nt*Ncyc))
theta = Float32.(0 * (1:Nt*Ncyc) .+ pi / 2)
phi = reshape(phi, Ncyc, Nt)
theta = reshape(theta, Ncyc, Nt)

trj = traj_kooshball(2Nx, theta, phi; adc_dim=false)
trj = trj[1:2, :, :]

## set up complex basis functions
U = randn(Complex{T}, Nt, Nc)
U, _, _ = svd(U)

## =========================================================================
## Test 1: Operator output matches between standard and lowmem
## =========================================================================
A_std    = NFFTNormalOp(img_shape, trj, U; cmaps)
A_lowmem = NFFTNormalOp(img_shape, trj, U; cmaps, lowmem=true)

# random input vector
b_test = randn(Complex{T}, prod(img_shape) * Nc)

res_std    = A_std  * b_test
res_lowmem = A_lowmem * b_test

@test res_lowmem ≈ res_std rtol = 1e-5

## =========================================================================
## Test 2: CG reconstruction gives the same result
## =========================================================================

# simulate data
data = Array{Complex{T},3}(undef, 2Nx * Ncyc, Nt, Ncoil)
nfftplan = PlanNUFFT(Complex{T}, img_shape; fftshift=true)
xcoil = copy(x)

for icoil ∈ axes(data, 3)
    xcoil .= x
    xcoil .*= cmaps[icoil]
    for it ∈ axes(data, 2)
        set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(reshape(trj[:, :, it], 2, :)))
        xt = reshape(reshape(xcoil, :, Nc) * U[it, :], Nx, Nx)
        @views NonuniformFFTs.exec_type2!(data[:, it, icoil], nfftplan, xt)
    end
end

b = calculate_backprojection(data, trj, cmaps; U)

xr_std    = cg(A_std, vec(b), maxiter=20)
xr_lowmem = cg(A_lowmem, vec(b), maxiter=20)

@test xr_lowmem ≈ xr_std rtol = 1e-4

## =========================================================================
## Test 3: Real-valued basis (triggers real kernel path)
## =========================================================================
U_real = randn(T, Nt, Nc)
U_real, _, _ = svd(U_real)

A_std_r    = NFFTNormalOp(img_shape, trj, U_real; cmaps)
A_lowmem_r = NFFTNormalOp(img_shape, trj, U_real; cmaps, lowmem=true)

res_std_r    = A_std_r  * b_test
res_lowmem_r = A_lowmem_r * b_test

@test res_lowmem_r ≈ res_std_r rtol = 1e-5

## =========================================================================
## Test 4: Without coil maps (single-coil case)
## =========================================================================
A_std_nc    = NFFTNormalOp(img_shape, trj, U)
A_lowmem_nc = NFFTNormalOp(img_shape, trj, U; lowmem=true)

res_std_nc    = A_std_nc  * b_test
res_lowmem_nc = A_lowmem_nc * b_test

@test res_lowmem_nc ≈ res_std_nc rtol = 1e-5