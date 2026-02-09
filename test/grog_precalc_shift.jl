using BenchmarkTools
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
Nt = 100
Ncoil = 9
Nrep = 3
Nd = 2
Ncyc = 3
img_shape = (Nx,Nx)

## Create trajectory
trj = traj_2d_radial_goldenratio(Nr, Ncyc, Nt; adc_dim=false)

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

##
data = Array{Complex{T}, 3}(undef, 2Nx*Ncyc, Nt, Ncoil);
nfftplan = PlanNUFFT(Complex{T}, img_shape; fftshift=true);
xcoil = similar(x, Complex{T})
for icoil ∈ 1:Ncoil
    xcoil .= x
    xcoil .*= cmaps[icoil]
    for it ∈ axes(data, 2)
        set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(reshape(trj[:,:,it], 2, :)))
        @views NonuniformFFTs.exec_type2!(data[:,it,icoil], nfftplan, xcoil)
    end
end

# Create repeating pattern
data2 = repeat(data, outer = [1, 1, 1, Nrep])

## #####################################
# Test Calibration of GROG kernel
########################################
lnG = MRISubspaceRecon.grog_calib(data, trj, Nr)
lnG2 = MRISubspaceRecon.grog_calib(data2, trj, Nr)

@test lnG ≈ lnG2 rtol = 1e-6

## #####################################
# Test Gridding with GROG kernel
########################################
trj2 = deepcopy(trj)

# Gridding of each sample with non repeating trajectory (Reference)
trj = MRISubspaceRecon.grog_gridding!(data, trj, lnG, Nr, (Nx,Nx))

# Exploit Precalculated Shifts
trj2 = MRISubspaceRecon.grog_gridding!(data2, trj2, lnG2, Nr, (Nx,Nx))

# Compare gridding with and without repeating pattern
@test data ≈ data2[:,:,:,1] rtol = 1e-5
@test data ≈ data2[:,:,:,3] rtol = 1e-5

## #####################################
# Test Gridded Reconstruction with and without Repeating Pattern
########################################
U = ones(ComplexF32, Nt, 1)

# Reconstruction without repeating pattern
A_grog = FFTNormalOp((Nx,Nx), trj, U; cmaps)
x1 = calculate_backprojection(data, trj, cmaps; U)
xg1 = cg(A_grog, vec(x1), maxiter=20)
xg1 = reshape(xg1, Nx, Nx)

# Reconstruction with repeating pattern
U2 = repeat(U, 1, 1, Nrep) # For joint subspace reconstruction
A_grog = FFTNormalOp((Nx,Nx), trj2, U2; cmaps)
x2 = calculate_backprojection(data2, trj2, cmaps; U=U2)
xg2 = cg(A_grog, vec(x2), maxiter=20)
xg2 = reshape(xg2, Nx, Nx)

@test xg1 ≈ xg2 rtol = 5e-3