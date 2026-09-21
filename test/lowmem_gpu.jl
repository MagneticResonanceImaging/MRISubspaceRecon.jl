# Test that the GPU low-memory (decomposed Toeplitz) operator produces
# the same results as the standard GPU NFFTNormalOp.

using CUDA
using MRISubspaceRecon
using LinearAlgebra
using Test
using FFTW
using IterativeSolvers
using ImagePhantoms
using Random
using NonuniformFFTs

Random.seed!(42)

## set parameters
T = Float32
Nx = 32
Nc = 4
Nt = 20
Ncyc = 10

img_shape = (Nx, Nx)

## create test image
x = zeros(Complex{T}, img_shape..., Nc)
x[:, :, 1] = transpose(shepp_logan(Nx))
x[1:end÷2, :, 1] .*= exp(1im * π / 3)
x[:, :, 2] = shepp_logan(Nx)

## coil maps
Ncoil = 9
cmaps = ones(Complex{T}, img_shape..., Ncoil)
[cmaps[i, :, 2] .*= exp(+1im * π * i / Nx) for i ∈ axes(cmaps, 1)]
[cmaps[i, :, 3] .*= exp(-1im * π * i / Nx) for i ∈ axes(cmaps, 1)]
[cmaps[:, i, 4] .*= exp(+1im * π * i / Nx) for i ∈ axes(cmaps, 2)]
[cmaps[:, i, 5] .*= exp(-1im * π * i / Nx) for i ∈ axes(cmaps, 2)]
[cmaps[i, :, 6] .*= exp(+2im * π * i / Nx) for i ∈ axes(cmaps, 1)]
[cmaps[i, :, 7] .*= exp(-2im * π * i / Nx) for i ∈ axes(cmaps, 1)]
[cmaps[:, i, 8] .*= exp(+2im * π * i / Nx) for i ∈ axes(cmaps, 2)]
[cmaps[:, i, 9] .*= exp(-2im * π * i / Nx) for i ∈ axes(cmaps, 2)]

for i ∈ CartesianIndices(@view cmaps[:, :, 1])
    cmaps[i, :] ./= norm(cmaps[i, :])
end
cmaps_d = [cu(cmaps[:, :, ic]) for ic = 1:Ncoil]

## set up trajectory
α_g = 2π / (1 + √5)
phi = Float32.(α_g * (1:Nt*Ncyc))
theta = Float32.(0 * (1:Nt*Ncyc) .+ pi / 2)
phi = reshape(phi, Ncyc, Nt)
theta = reshape(theta, Ncyc, Nt)

trj = traj_kooshball(2Nx, theta, phi; adc_dim=false)
trj = trj[1:2, :, :]
trj_d = cu(trj)

## set up complex basis functions
U = randn(Complex{T}, Nt, Nc)
U, _, _ = svd(U)
U_d = cu(U)

## =========================================================================
## Test 1: Operator output matches between standard GPU and lowmem GPU
## =========================================================================
println("Building standard GPU operator...")
A_std = NFFTNormalOp(img_shape, trj_d, U_d; cmaps=cmaps_d)
println("Building lowmem GPU operator...")
A_low = NFFTNormalOp(img_shape, trj_d, U_d; cmaps=cmaps_d, lowmem=true)

# random input vector
b_test = CUDA.randn(Complex{T}, prod(img_shape) * Nc)

println("Applying standard GPU operator...")
res_std = A_std * b_test
println("Applying lowmem GPU operator...")
res_low = A_low * b_test

err1 = norm(res_low - res_std) / norm(res_std)
println("Test 1 - operator output relative error: $err1")
@test err1 < 1e-5

## =========================================================================
## Test 2: CG reconstruction gives the same result
## =========================================================================
println("\nSimulating data...")
data = Array{Complex{T},3}(undef, 2Nx * Ncyc, Nt, Ncoil)
cmaps_cpu = [cmaps[:, :, ic] for ic = 1:Ncoil]
nfftplan = PlanNUFFT(Complex{T}, img_shape; fftshift=true)
xcoil = copy(x)

for icoil ∈ axes(data, 3)
    xcoil .= x
    xcoil .*= cmaps_cpu[icoil]
    for it ∈ axes(data, 2)
        set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(reshape(trj[:, :, it], 2, :)))
        xt = reshape(reshape(xcoil, :, Nc) * U[it, :], Nx, Nx)
        @views NonuniformFFTs.exec_type2!(data[:, it, icoil], nfftplan, xt)
    end
end
data_d = cu(data)

b_d = calculate_backprojection(data_d, trj_d, cmaps_d; U=U_d)

println("CG with standard GPU operator...")
xr_std = cg(A_std, vec(b_d), maxiter=20)
println("CG with lowmem GPU operator...")
xr_low = cg(A_low, vec(b_d), maxiter=20)

err2 = norm(xr_low - xr_std) / norm(xr_std)
println("Test 2 - CG reconstruction relative error: $err2")
@test err2 < 1e-5

## =========================================================================
## Test 3: Real-valued basis (triggers real kernel path)
## =========================================================================
println("\nTest 3: Real basis...")
U_real = randn(T, Nt, Nc)
U_real, _, _ = svd(U_real)
U_real_d = cu(U_real)

A_std_r = NFFTNormalOp(img_shape, trj_d, U_real_d; cmaps=cmaps_d)
A_low_r = NFFTNormalOp(img_shape, trj_d, U_real_d; cmaps=cmaps_d, lowmem=true)

res_std_r = A_std_r * b_test
res_low_r = A_low_r * b_test

err3 = norm(res_low_r - res_std_r) / norm(res_std_r)
println("Test 3 - real basis relative error: $err3")
@test err3 < 1e-5

## =========================================================================
## Test 4: Without coil maps (single-coil case)
## =========================================================================
println("\nTest 4: No coil maps...")
A_std_nc = NFFTNormalOp(img_shape, trj_d, U_d)
A_low_nc = NFFTNormalOp(img_shape, trj_d, U_d; lowmem=true)

res_std_nc = A_std_nc * b_test
res_low_nc = A_low_nc * b_test

err4 = norm(res_low_nc - res_std_nc) / norm(res_std_nc)
println("Test 4 - no coils relative error: $err4")
@test err4 < 1e-5

## =========================================================================
## Test 5: Partially sampled `sample_mask` (exercises the gather path)
## =========================================================================
println("\nTest 5: Partial sample_mask...")
sample_mask = trues(2Nx, Ncyc, Nt)
sample_mask[:, 1:2, 1:3] .= false # remove a few cycles in a few time frames
sample_mask = reshape(sample_mask, :, Nt)
@test !all(sample_mask) # make sure we are not silently testing the fully-sampled path
sample_mask_d = cu(sample_mask)

A_std_m = NFFTNormalOp(img_shape, trj_d, U_d; cmaps=cmaps_d, sample_mask=sample_mask_d)
A_low_m = NFFTNormalOp(img_shape, trj_d, U_d; cmaps=cmaps_d, sample_mask=sample_mask_d, lowmem=true)

res_std_m = A_std_m * b_test
res_low_m = A_low_m * b_test

err5 = norm(res_low_m - res_std_m) / norm(res_std_m)
println("Test 5 - masked relative error: $err5")
@test err5 < 1e-5

# real basis with a mask, too
A_std_mr = NFFTNormalOp(img_shape, trj_d, U_real_d; cmaps=cmaps_d, sample_mask=sample_mask_d)
A_low_mr = NFFTNormalOp(img_shape, trj_d, U_real_d; cmaps=cmaps_d, sample_mask=sample_mask_d, lowmem=true)

err5r = norm(A_low_mr * b_test - A_std_mr * b_test) / norm(A_std_mr * b_test)
println("Test 5 - masked real-basis relative error: $err5r")
@test err5r < 1e-5

## =========================================================================
## Test 6: Views and reshapes of device trajectories are accepted
## =========================================================================
println("\nTest 6: AnyCuArray trajectory...")
trj_view = @view trj_d[:, :, :]
A_low_v = NFFTNormalOp(img_shape, trj_view, U_d; cmaps=cmaps_d, lowmem=true)

err6 = norm(A_low_v * b_test - res_low) / norm(res_low)
println("Test 6 - view trajectory relative error: $err6")
@test err6 < 1e-5

println("\n✓ All GPU lowmem tests passed!")