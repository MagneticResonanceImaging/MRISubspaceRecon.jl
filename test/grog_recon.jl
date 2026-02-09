using BenchmarkTools
using MRISubspaceRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using NonuniformFFTs
using Test

## set parameters
T  = Float32
Nx = 64
Nr = 2Nx
Nc = 4
Nt = 100
Ncyc = 200
Ncoil = 9
img_shape=(Nx, Nx)

## create test image
x = zeros(Complex{T}, Nx, Nx, Nc)
x[:,:,1] = transpose(shepp_logan(Nx))
x[1:end÷2,:,1] .*= exp(1im * π/3)
x[:,:,2] = shepp_logan(Nx)

## coil maps
cmaps = ones(Complex{T}, Nx, Nx, Ncoil)
cmaps[:,:,1] .= phantom(1:Nx, 1:Nx, [gauss2((Nx÷8,  Nx÷8),  (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,2] .= phantom(1:Nx, 1:Nx, [gauss2((Nx÷8,  Nx÷2),  (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,3] .= phantom(1:Nx, 1:Nx, [gauss2((Nx÷8,  7Nx÷8), (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,4] .= phantom(1:Nx, 1:Nx, [gauss2((Nx÷2,  Nx÷8),  (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,5] .= phantom(1:Nx, 1:Nx, [gauss2((Nx÷2,  Nx÷2),  (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,6] .= phantom(1:Nx, 1:Nx, [gauss2((Nx÷2,  7Nx÷8), (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,7] .= phantom(1:Nx, 1:Nx, [gauss2((7Nx÷8, Nx÷8),  (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,8] .= phantom(1:Nx, 1:Nx, [gauss2((7Nx÷8, Nx÷2),  (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,9] .= phantom(1:Nx, 1:Nx, [gauss2((7Nx÷8, 7Nx÷8), (Nx÷1.5,Nx÷1.5))], 2)

for i ∈ CartesianIndices(@view cmaps[:,:,1])
    cmaps[i,:] ./= norm(cmaps[i,:])
end
cmaps = [cmaps[:,:,ic] for ic=1:Ncoil]

## set up trajectory
α_g = 2π / (1+√5)
phi = Float32.(α_g * (0:Nt*Ncyc-1))
theta = Float32.(0 * (1:Nt*Ncyc) .+ pi/2)
phi = reshape(phi, Ncyc, Nt)
theta = reshape(theta, Ncyc, Nt)

trj = traj_kooshball(Nr, theta, phi; adc_dim=false)
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

## Ground truth reconstruction by cropping k-space
xc = fftshift(fft(x, 1:2), 1:2)
for i ∈ CartesianIndices(xc)
    if (i[1] - Nx/2)^2 + (i[2] - Nx/2)^2 > (Nx/2)^2
        xc[i] = 0
    end
end
xc = ifft(ifftshift(xc, 1:2), 1:2)

## NFFT Reconstruction
xbp_rad = calculate_backprojection(data, trj, cmaps; U=U)
A_rad = NFFTNormalOp((Nx,Nx), trj, U; cmaps=cmaps)
xr = cg(A_rad, vec(xbp_rad), maxiter=20)
xr = reshape(xr, Nx, Nx, Nc)

## GROG Reconstruction
trj_cart = radial_grog!(data, trj, Nr, (Nx,Nx))
xbp_grog = calculate_backprojection(data, trj_cart, cmaps; U)
A_cart = FFTNormalOp((Nx,Nx), trj_cart, U; cmaps)
xg = cg(A_cart, vec(xbp_grog), maxiter=20)
xg = reshape(xg, Nx, Nx, Nc)

## Fix irrelevant phase slope
[xg[i,j,:] .*= -exp(1im * π * (i + j - 2)/Nx) for i = 1:Nx, j = 1:Nx]

## test recon equivalence
@test xc ≈ xr  rtol = 5e-2
@test xc ≈ xg  rtol = 5e-2