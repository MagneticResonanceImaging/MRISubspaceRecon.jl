using BenchmarkTools
using MRISubspaceRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using NonuniformFFTs
using Test

using Random
Random.seed!(42)

## set parameters
T  = Float32
Nx = 32
Nc = 4
Nt = 20
Ncyc = 10
img_shape = (Nx, Nx)

## create test image
x = zeros(Complex{T}, Nx, Nx, Nc)
x[:,:,1] = transpose(shepp_logan(Nx))
x[1:end÷2,:,1] .*= exp(1im * π/3)
x[:,:,2] = shepp_logan(Nx)

## coil maps
Ncoil = 9
cmaps = ones(Complex{T}, Nx, Nx, Ncoil)
[cmaps[i,:,2] .*= exp( 1im * π * i/Nx) for i ∈ axes(cmaps,1)]
[cmaps[i,:,3] .*= exp(-1im * π * i/Nx) for i ∈ axes(cmaps,1)]
[cmaps[:,i,4] .*= exp( 1im * π * i/Nx) for i ∈ axes(cmaps,2)]
[cmaps[:,i,5] .*= exp(-1im * π * i/Nx) for i ∈ axes(cmaps,2)]
[cmaps[i,:,6] .*= exp( 2im * π * i/Nx) for i ∈ axes(cmaps,1)]
[cmaps[i,:,7] .*= exp(-2im * π * i/Nx) for i ∈ axes(cmaps,1)]
[cmaps[:,i,8] .*= exp( 2im * π * i/Nx) for i ∈ axes(cmaps,2)]
[cmaps[:,i,9] .*= exp(-2im * π * i/Nx) for i ∈ axes(cmaps,2)]

for i ∈ CartesianIndices(@view cmaps[:,:,1])
    cmaps[i,:] ./= norm(cmaps[i,:])
end
cmaps = [cmaps[:,:,ic] for ic=1:Ncoil]

## set up trajectory
α_g = 2π / (1+√5)
phi = Float32.(α_g * (1:Nt*Ncyc))
theta = Float32.(0 * (1:Nt*Ncyc) .+ pi/2)
phi = reshape(phi, Ncyc, Nt)
theta = reshape(theta, Ncyc, Nt)
trj = traj_kooshball(2Nx, theta, phi; adc_dim=false)
trj = trj[1:2, :, :]

## set up basis functions
U = randn(Complex{T}, Nt, Nc)
U,_,_ = svd(U)

## simulate data
data = Array{Complex{T}, 3}(undef, 2Nx*Ncyc, Nt, Ncoil);
nfftplan = PlanNUFFT(Complex{T}, img_shape; fftshift=true);
xcoil = copy(x);

for icoil ∈ axes(data, 3)
    xcoil .= x
    xcoil .*= cmaps[icoil]
    for it ∈ axes(data, 2)
        set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(reshape(trj[:,:,it], 2, :)))
        xt = reshape(reshape(xcoil, :, Nc) * U[it,:], Nx, Nx)
        @views NonuniformFFTs.exec_type2!(data[:,it,icoil], nfftplan, xt)
    end
end

## create sampling mask
it_rm = 1
icyc_rm = 5
sample_mask = trues(2Nx, Ncyc, Nt)
sample_mask[:, icyc_rm, it_rm] .= false
sample_mask = reshape(sample_mask, 2Nx*Ncyc, Nt)

## Test BackProjection
b = calculate_backprojection(data, trj, cmaps; U, sample_mask)

## construct forward operator
A = NFFTNormalOp((Nx,Nx), trj, U; cmaps, sample_mask)

## reconstruct
xr = cg(A, vec(b), maxiter=20)
xr = reshape(xr, Nx, Nx, Nc)

## crop x
xc = fftshift(fft(x, 1:2), 1:2)
for i ∈ CartesianIndices(xc)
    if (i[1] - Nx/2)^2 + (i[2] - Nx/2)^2 > (Nx/2)^2
        xc[i] = 0
    end
end
xc = ifft(ifftshift(xc, 1:2), 1:2)

##
@test xr ≈ xc rtol = 1e-1