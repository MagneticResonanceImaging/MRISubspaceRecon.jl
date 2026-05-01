using MRISubspaceRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using NonuniformFFTs
using Test

##
T  = Float32
Nx = 32
Nr = 2Nx
Nt = 1
Ncoil = 9
img_shape = (Nx, Nx)

## Create trajectory
trj = traj_cartesian(Nx, Nx, 1, Nt; T=Float32) # float for plan_nfft()
trj = trj[1:2, :, :]

## Create phantom geometry
x = shepp_logan(Nx)

## Simulate coil sensitivity maps
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

## Simulate data
data = Array{Complex{T}}(undef, size(trj)[2:end]..., Ncoil)
nfftplan = NonuniformFFTs.NFFTPlan(trj[:,:,1], (Nx,Nx))

xcoil = similar(x, Complex{T})
for icoil ∈ 1:Ncoil
    xcoil .= x
    xcoil .*= cmaps[icoil]
    for it ∈ axes(data, 2)
        set_points!(nfftplan.p, trj[:,:,it])
        @views mul!(data[:,it,icoil], nfftplan, xcoil)
    end
end

## Reconstruction and test
U = ones(ComplexF32, Nt, 1)

## Create the equivalent integer trajectory
trj = traj_cartesian(Nx, Nx, 1, Nt; T=Int32)
trj = trj[1:2, :, :]
reco = calculate_backprojection(data, trj, cmaps; U)
reco = dropdims(reco, dims=3)

@test x ≈ reco atol = 4e-5