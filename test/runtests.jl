using MRISubspaceRecon
using Test

@testset "Recon Radial" begin
    include("cmaps.jl")
    include("reconstruct_radial.jl")
    include("reconstruct_radial_asym.jl")
    include("data_removal.jl")
end

@testset "Recon Cartesian" begin
    include("backprojection_cart.jl")
    include("reconstruct_cart_mask.jl")
    include("reconstruct_cart_trj.jl")
end

@testset "GROG" begin
    include("grog_spoke_shift.jl")
    include("grog_precalc_shift.jl")
    include("grog_recon.jl")
end

@testset "Wrapper" begin
    include("wrapper.jl")
end

# GPU tests are run whenever a functional GPU is available; skip cleanly otherwise.
gpu_available = try
    using CUDA
    CUDA.functional()
catch err
    @info "Skipping GPU tests: CUDA is unavailable ($(sprint(showerror, err) |> x -> first(x, 120)))"
    false
end

if gpu_available
    @testset "GPU" begin
        include("reconstruct_radial_gpu.jl")
        include("reconstruct_radial_gpu_real.jl")
        include("reconstruct_cart_trj_gpu.jl")
        include("wrapper_gpu.jl")
    end
else
    @info "Skipping GPU tests: no functional CUDA device."
end
