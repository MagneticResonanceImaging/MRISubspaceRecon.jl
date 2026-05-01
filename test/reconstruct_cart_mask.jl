using MRISubspaceRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using Test

## set parameters
T  = Float32
Nx = 64
Nc = 4
Nt = 100
Ncoil = 9

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

## simulate data and backprojection
D = rand(Nx, Nx, Nt) .< 0.1
data = Array{Complex{T}}(undef, Nx, Nx, Nt)
xbp = similar(x) .= 0
for icoil = 1:Ncoil
    Threads.@threads for i ∈ CartesianIndices(@view x[:,:,1])
        data[i,:] .= U * x[i,:] .* cmaps[icoil][i]
    end
    data .= ifftshift(data, (1,2))
    fft!(data, (1,2))
    data .*= ifftshift(D, (1,2))
    ifft!(data, (1,2))
    data .= fftshift(data, (1,2))
    Threads.@threads for i ∈ CartesianIndices(@view x[:,:,1])
        xbp[i,:] .+= U' * data[i,:] * conj(cmaps[icoil][i])
    end
end

## construct forward operator
A = FFTNormalOp(D, U; cmaps)

## test forward operator is symmetric
Λ = zeros(Complex{T}, Nc, Nc, Nx^2)
Λ[:,:,A.prod!.A.kmask_indcs] .= A.prod!.A.Λ
Λ = reshape(Λ, Nc, Nc, Nx, Nx)
for i ∈ CartesianIndices((Nx, Nx))
    @test Λ[:,:,i]' ≈ Λ[:,:,i]
end

## reconstruct
xr = zeros(Complex{T}, size(vec(x)))
cg!(xr, A, vec(xbp), maxiter=20)
xr = reshape(xr, Nx, Nx, Nc)

## test equivalence
mask = abs.(x[:,:,1]) .> 0
@test xr[mask,:] ≈ x[mask,:] rtol = 1e-3