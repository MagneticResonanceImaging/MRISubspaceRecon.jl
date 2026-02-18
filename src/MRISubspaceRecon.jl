module MRISubspaceRecon

using OhMyThreads
using LinearAlgebra
using FFTW
using NonuniformFFTs
using MRICoilSensitivities
using LinearOperators
using ExponentialUtilities
using IterativeSolvers

export NFFTNormalOp, calculate_coil_maps, calculate_backprojection, traj_kooshball, traj_kooshball_goldenratio, traj_2d_radial_goldenratio, traj_cartesian
export FFTNormalOp, radial_grog!

function __init__()
    if Threads.nthreads() > 1
      BLAS.set_num_threads(1)
    end
    FFTW.set_num_threads(Threads.nthreads())
end

include("GROG.jl")
include("FFTNormalOp.jl")
include("NFFTNormalOp.jl")
include("CoilMaps.jl")
include("BackProjection.jl")
include("Trajectories.jl")

end # module