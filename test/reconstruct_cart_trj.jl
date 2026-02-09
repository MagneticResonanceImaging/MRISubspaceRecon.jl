using BenchmarkTools
using MRISubspaceRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using Test

## set parameters
T = Float32
Tint = Int32
Nx = 128
Nc = 4
Nt = 40
Ncoil = 9
img_shape = (Nx, Nx)

## create test image
x = zeros(Complex{T}, Nx, Nx, Nc)
x[:,:,1] = transpose(shepp_logan(Nx))
x[1:end÷2,:,1] .*= exp(1im * π/3)
x[:,:,2] = shepp_logan(Nx)

## coil maps
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

## set up basis functions
U = randn(Complex{T}, Nt, Nc)
U,_,_ = svd(U)

## simulate data
data = Array{Complex{T}}(undef, Nx, Nx, Nt, Ncoil)
for icoil = 1:Ncoil
    Threads.@threads for i ∈ CartesianIndices(@view x[:,:,1])
        data[i,:,icoil] .= U * x[i,:] .* cmaps[icoil][i]
    end
end

data .= ifftshift(data, (1, 2))
data = reshape(data, size(data,1), size(data,2), size(data,3)*size(data,4)) # reshape prevents MKL backend error
fft!(data, [1, 2])
data = fftshift(data, (1, 2))
data = reshape(data, Nx*Nx, Nt, Ncoil)

sample_mask = rand(Nx, Nx, Nt) .< 0.8 # sampling mask
trj = collect(Iterators.product(1:Nx, 1:Nx, 1:Nt))
kx = reshape(getindex.(trj, 1), (1, Nx*Nx, Nt))
ky = reshape(getindex.(trj, 2), (1, Nx*Nx, Nt))
trj = Tint.(cat(kx, ky; dims=1))
sample_mask = reshape(sample_mask, Nx*Nx, Nt)

##
A = FFTNormalOp((Nx,Nx), trj, U; cmaps, sample_mask)

## test that forward operator is symmetric
Λ = zeros(Complex{T}, Nc, Nc, Nx^2)
Λ[:,:,A.prod!.A.kmask_indcs] .= A.prod!.A.Λ
Λ = reshape(Λ, Nc, Nc, Nx, Nx)
for i ∈ CartesianIndices((Nx, Nx))
    @test Λ[:,:,i]' ≈ Λ[:,:,i]
end

## test cg recon
xbp = calculate_backprojection(data, trj, cmaps; U, sample_mask)
xr = cg(A, vec(xbp), maxiter=20)
xr = reshape(xr, Nx, Nx, Nc)

## test equivalence
@test xr ≈ x rtol=1e-3