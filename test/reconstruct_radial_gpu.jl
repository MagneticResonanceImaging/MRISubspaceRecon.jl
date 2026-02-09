using MRISubspaceRecon
using CUDA
using LinearAlgebra
using BenchmarkTools
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
[x[i, :, 1] .*= exp(1im * π * i / Nx) for i ∈ axes(x, 1)]
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
cmaps = [cmaps[:, :, ic] for ic = 1:Ncoil]

## set up trajectory
α_g = 2π / (1 + √5)
phi = Float32.(α_g * (1:Nt*Ncyc))
theta = Float32.(0 * (1:Nt*Ncyc) .+ pi / 2)
phi = reshape(phi, Ncyc, Nt)
theta = reshape(theta, Ncyc, Nt)

trj = traj_kooshball(2Nx, theta, phi; adc_dim=false)
trj = trj[1:2, :, :]

## set up basis functions
U = randn(Complex{T}, Nt, Nc) # diff methods for complex and real basis!
U, _, _ = svd(U)

## simulate data
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

# Create sampling mask
it_rm = 1
icyc_rm = 5
sample_mask = trues(2Nx, Ncyc, Nt)
sample_mask[:, icyc_rm, it_rm] .= false
sample_mask = reshape(sample_mask, :, Nt)

## CPU
A = NFFTNormalOp(img_shape, trj, U; cmaps, sample_mask)
b = calculate_backprojection(data, trj, cmaps; U, sample_mask)
xr = cg(A, vec(b), maxiter=50)
xr = reshape(xr, img_shape..., Nc)

## Write to CUDA arrays
trj_d = cu(trj)
data_d = cu(data)
U_d = cu(U)
cmaps_d = [cu(cmaps[i]) for i ∈ eachindex(cmaps)]
sample_mask_d = cu(sample_mask)

## GPU
A_d = NFFTNormalOp(img_shape, trj_d, U_d; cmaps=cmaps_d, sample_mask=sample_mask_d) # kernels are expected to slightly differ between CPU/GPU
b_d = calculate_backprojection(data_d, trj_d, cmaps_d; U=U_d, sample_mask=sample_mask_d)
xr_d = cg(A_d, vec(b_d), maxiter=50)
xr_d = reshape(Array(xr_d), img_shape..., Nc) # end results should be equivalent

## Test equivalence CPU & GPU CG reconstruction with complex basis U
@test xr ≈ xr_d rtol = 1e-2
@test !isreal(A_d.prod!.A.Λ)