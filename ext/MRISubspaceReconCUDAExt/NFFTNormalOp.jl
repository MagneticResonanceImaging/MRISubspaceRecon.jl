function MRISubspaceRecon.NFFTNormalOp(
    img_shape,
    trj::CuArray{T,3},
    U::CuArray{Tc};
    cmaps=(1,),
    sample_mask=CUDA.ones(Bool, size(trj)[2:end]),
    verbose=false,
    lowmem=false,
    ) where {T <: Real, Tc <: Union{T, Complex{T}}}

    Λ, kmask_indcs = calculate_kernel_noncartesian(2 .* img_shape, trj, U; sample_mask, verbose)

    if lowmem
        Ncoeff = (isqrt(8 * size(Λ, 1) + 1) - 1) ÷ 2
        Λ_d, kmask_d = decompose_kernel_gpu(img_shape, Λ, kmask_indcs, Ncoeff)
        return MRISubspaceRecon.NFFTNormalOpLowmem(img_shape, Λ_d, kmask_d; cmaps)
    else
        return MRISubspaceRecon.NFFTNormalOp(img_shape, Λ, kmask_indcs; cmaps=cmaps)
    end
end

# Wrapper for 4D data arrays
function MRISubspaceRecon.NFFTNormalOp(img_shape, trj::CuArray{T,4}, U::CuArray{Tc}; sample_mask=CUDA.ones(Bool, size(trj)[2:end]), kwargs...) where {T, Tc <: Union{T, Complex{T}}}
    trj = reshape(trj, size(trj,1), :, size(trj,4))
    sample_mask = reshape(sample_mask, :, size(sample_mask,3))
    return MRISubspaceRecon.NFFTNormalOp(img_shape, trj, U; kwargs..., sample_mask)
end

function MRISubspaceRecon.NFFTNormalOp(
    img_shape,
    Λ::CuArray{Tc},
    kmask_indcs;
    cmaps=(1,)
    ) where {T <: Real, Tc <: Union{T, Complex{T}}}
    @assert length(kmask_indcs) == size(Λ, length(size(Λ))) # ensure that kmask is not out of bound as we use `@inbounds` in `mul!`
    @assert all(kmask_indcs .> 0)
    @assert all(kmask_indcs .<= prod(2 .* img_shape))

    # derive Ncoeff from length of packed axis using quadratic eqn
    Ncoeff = (isqrt(8 * size(Λ, 1) + 1) - 1) ÷ 2

    img_shape_os = 2 .* img_shape
    kL1 = CuArray{Complex{T}}(undef, img_shape_os..., Ncoeff)
    kL2 = CuArray{Complex{T}}(undef, img_shape_os..., Ncoeff)

    fftplan  = plan_fft!( kL1, 1:length(img_shape_os))
    ifftplan = plan_ifft!(kL2, 1:length(img_shape_os))

    # indexing into the upper triangular matrix
    ind_lookup = CuArray([j<k ? j+k*(k-1)÷2 : k+j*(j-1)÷2 for j ∈ 1:Ncoeff, k ∈ 1:Ncoeff])

    # set up the threading for the GPU
    kL1_rs = reshape(kL1, :, Ncoeff)
    kL2_rs = reshape(kL2, :, Ncoeff)
    kernel = @cuda launch=false kernel_mul!(kL2_rs, Λ, kL1_rs, kmask_indcs, ind_lookup)
    config = launch_configuration(kernel.fun)

    threads_x = min(config.threads ÷ Ncoeff, length(kL2_rs))
    threads_y = min(config.threads, Ncoeff)
    threads = (threads_x, threads_y)
    blocks = cld.((length(kmask_indcs), Ncoeff), threads)

    # Set up the actual object
    A = MRISubspaceRecon._NFFTNormalOp(img_shape, Ncoeff, fftplan, ifftplan, Λ, kmask_indcs, kL1, kL2, cmaps, ind_lookup, threads, blocks)

    return LinearOperator(
        Complex{T},
        prod(A.shape) * A.Ncoeff,
        prod(A.shape) * A.Ncoeff,
        true,
        true,
        (res, x, α, β) -> mul!(res, A, x, α, β),
        nothing,
        (res, x, α, β) -> mul!(res, A, x, α, β);
        S = typeof(similar(Λ, Complex{T}, 0))
    )
end

## ##########################################################################
# Internal use
#############################################################################
function calculate_kmask_indcs(img_shape_os, trj::CuArray{T,3}; sample_mask=CUDA.ones(Bool, size(trj)[2:end])) where {T}
    @assert all([i .== nextprod((2, 3, 5), i) for i ∈ img_shape_os]) "img_shape_os has to be composed of the prime factors 2, 3, and 5 (cf. NonuniformFFTs.jl documentation)."

    backend = CUDABackend()
    p = PlanNUFFT(Complex{T}, img_shape_os; σ=1, kernel=GaussianKernel(), backend=backend) # default is without fftshift
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj[:, sample_mask]))

    S = CUDA.ones(Complex{T}, size(p.points[1]))
    NonuniformFFTs.spread_from_points!(p.backend, NUFFTCallbacks().nonuniform, p.point_transform_fold, p.blocks, p.kernels, p.kernel_evalmode, p.data.us, p.points, (S,))
    kmask_indcs = findall(vec(p.data.us[1] .!= 0))
    return kmask_indcs
end

# Kernel is complex-valued (case of complex basis matrix U)
function calculate_kernel_noncartesian(img_shape_os, trj::CuArray{T,3}, U::CuArray{Tc}; sample_mask=CUDA.ones(Bool, size(trj)[2:end]), verbose=false) where {T <: Real, Tc <: Complex{T}}
    kmask_indcs = calculate_kmask_indcs(img_shape_os, trj; sample_mask)

    @assert all(kmask_indcs .> 0) # ensure that kmask is not out of bound
    @assert all(kmask_indcs .<= prod(img_shape_os))

    nsamp_t = cu(sum(sample_mask, dims=1)) # number of samples per time frame
    cumsum_nsamp = CUDA.zeros(eltype(nsamp_t), size(nsamp_t)) # the cumulative sum indicates in which time frame each sample is contained
    cumsum_nsamp[2:end] = cumsum(nsamp_t[1:end-1])

    # Allocate kernel arrays, write Λ as packed storage array (https://www.netlib.org/lapack/lug/node123.html)
    λ  = CuArray{Complex{T}}(undef, img_shape_os)
    λ2 = similar(λ)

    Ncoeff = size(U, 2)
    Λ = CuArray{Complex{T}}(undef, Int(Ncoeff*(Ncoeff+1)/2), length(kmask_indcs)) # allow complex U
    S = CuArray{Complex{T}}(undef, sum(nsamp_t))

    # Prep FFT and NUFFT plans
    fftplan  = plan_fft(λ)
    nfftplan = PlanNUFFT(Complex{T}, img_shape_os; backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size = Val(200)) # use plan specific to real inputs
    set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(trj[:, sample_mask]))

   # Configure threads and blocks for each kernel within the coefficient loop
    threads_multiply, blocks_multiply, threads_store, blocks_store = launch_config_kernel(nsamp_t, kmask_indcs)

    verbose && println("calculating non-Cartesian kernel...")
    t = @elapsed CUDA.@sync for ic2 ∈ 1:Ncoeff, ic1 ∈ 1:Ncoeff
        if ic2 >= ic1 # eval. only upper triangular matrix
            @cuda threads=threads_multiply blocks=blocks_multiply multiply_basis_vectors!(S, U, nsamp_t, cumsum_nsamp, ic1, ic2)

            exec_type1!(λ2, nfftplan, vec(S)) # type 1: non-uniform points to uniform grid
            mul!(λ, fftplan, λ2)

            @cuda threads=threads_store blocks=blocks_store store_packed_kernel!(Λ, λ, kmask_indcs, ic1, ic2)
        end
    end
    verbose && println("time to compute kernel: t = $t s")
    return Λ, kmask_indcs
end

# Kernel is assumed to be real-valued to reduce storage by half (method only works with real basis U)
function calculate_kernel_noncartesian(img_shape_os, trj::CuArray{T,3}, U::CuArray{T}; sample_mask=CUDA.ones(Bool, size(trj)[2:end]), verbose=false) where {T <: Real}
    kmask_indcs = calculate_kmask_indcs(img_shape_os, trj; sample_mask)
    @assert all(kmask_indcs .> 0) # ensure that kmask is not out of bound
    @assert all(kmask_indcs .<= prod(img_shape_os))

    nsamp_t = cu(sum(sample_mask, dims=1)) # number of samples per time frame
    @assert sum(nsamp_t) > 0 "Sample_mask removes all samples, cannot compute kernel."

    cumsum_nsamp = CUDA.zeros(eltype(nsamp_t), size(nsamp_t))
    cumsum_nsamp[2:end] = cumsum(nsamp_t[1:end-1])

    # Allocate kernel arrays, write Λ as packed storage array (https://www.netlib.org/lapack/lug/node123.html)
    λ  = CuArray{T}(undef, img_shape_os)
    λ2 = CuArray{Complex{T}}(undef, img_shape_os[1] ÷ 2 + 1, Base.tail(img_shape_os)...)

    Ncoeff = size(U, 2)
    Λ = CuArray{T}(undef, Int(Ncoeff*(Ncoeff+1)/2), length(kmask_indcs)) # requires basis U to be real
    S = CuArray{T}(undef, sum(nsamp_t))

    # Prep FFT and NUFFT plans
    # Use brfft (and conjugate λ2 in loop below) because a forward rfft with complex input does not exist in FFTW package
    brfftplan = plan_brfft(λ2, img_shape_os[1])
    nfftplan = PlanNUFFT(T, img_shape_os; backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size = Val(200)) # use plan specific to real inputs
    set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(trj[:, sample_mask]))

    # Configure threads and blocks for each kernel within the coefficient loop
    threads_multiply, blocks_multiply, threads_store, blocks_store = launch_config_kernel(nsamp_t, kmask_indcs)

    verbose && println("calculating non-Cartesian kernel...")
    t = @elapsed CUDA.@sync for ic2 ∈ 1:Ncoeff, ic1 ∈ 1:Ncoeff
        if ic2 >= ic1 # eval. only upper triangular matrix
            t = @elapsed begin
                @cuda threads=threads_multiply blocks=blocks_multiply multiply_basis_vectors!(S, U, nsamp_t, cumsum_nsamp, ic1, ic2)

                exec_type1!(λ2, nfftplan, vec(S)) # type 1: non-uniform points to uniform grid
                λ2 .= conj.(λ2) # conjugate to flip the sign of the exponential in brfft
                mul!(λ, brfftplan, λ2)

                @cuda threads=threads_store blocks=blocks_store store_packed_kernel!(Λ, λ, kmask_indcs, ic1, ic2)
            end
        end
    end
    verbose && println("time to compute kernel: t = $t s")
    return Λ, kmask_indcs
end

function LinearAlgebra.mul!(x::CuArray, S::MRISubspaceRecon._NFFTNormalOp, b, α, β)
    b = reshape(b, S.shape..., S.Ncoeff)
    if β == 0
        x .= 0
    else
        x .*= β
    end
    xr = reshape(x, S.shape..., S.Ncoeff)

    idx = CartesianIndices(S.shape)

    for cmap ∈ S.cmaps
        fill!(S.kL1, 0)
        fill!(S.kL2, 0)
        S.kL1[idx, :] .= cmap .* b
        S.fftplan * S.kL1

        kL1_rs = reshape(S.kL1, :, S.Ncoeff)
        kL2_rs = reshape(S.kL2, :, S.Ncoeff)
        @cuda threads=S.threads blocks=S.blocks kernel_mul!(kL2_rs, S.Λ, kL1_rs, S.kmask_indcs, S.ind_lookup)

        S.ifftplan * S.kL2
        @views xr .+= α .* conj.(cmap) .* S.kL2[idx, :]
    end
    return x
end

function multiply_basis_vectors!(S, U, nsamp_t, cumsum_nsamp, ic1, ic2)
    ik = (blockIdx().x-1) * blockDim().x + threadIdx().x
    it = (blockIdx().y-1) * blockDim().y + threadIdx().y

    # Grid-stride loop over time frames to handle cases where the number of
    # time frames exceeds the maximum number of blocks in the y-dimension
    stride_y = gridDim().y * blockDim().y
    while it <= length(nsamp_t)
        Uprod = conj(U[it, ic1]) * U[it, ic2]
        if ik <= nsamp_t[it]
            S[cumsum_nsamp[it] + ik] = Uprod
        end
        it += stride_y
    end
    return
end

# Place elements of kernel in packed Λ
function store_packed_kernel!(Λ, λ, kmask_indcs, ic1, ic2)
    ik = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # Packed storage of Λ by columns
    ind_packed = ic1 + ic2 * (ic2-1) ÷ 2
    if ik <= length(kmask_indcs)
        Λ[ind_packed, ik] = λ[kmask_indcs[ik]]
    end
    return
end

# For complex basis U
function kernel_mul!(kL2_rs, Λ::CuDeviceMatrix{Tc}, kL1_rs, kmask_indcs, ind_lookup) where {Tc <: Complex}
    ik  = (blockIdx().x-1) * blockDim().x + threadIdx().x
    ic1 = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if ik <= length(kmask_indcs) && ic1 <= size(kL2_rs, 2)
        ind = kmask_indcs[ik]
        acc = zero(eltype(kL2_rs))

        @inbounds for ic2 ∈ axes(ind_lookup, 2)
            if ic1 <= ic2
                ind_packed = ind_lookup[ic1, ic2]
                acc += Λ[ind_packed, ik] * kL1_rs[ind, ic2]
            else
                ind_packed = ind_lookup[ic2, ic1]
                acc += conj(Λ[ind_packed, ik]) * kL1_rs[ind, ic2]
            end
        end
        kL2_rs[ind, ic1] = acc
    end
    return
end

# For real basis U
function kernel_mul!(kL2_rs, Λ::CuDeviceMatrix{T}, kL1_rs, kmask_indcs, ind_lookup) where {T <: Real}
    ik  = (blockIdx().x-1) * blockDim().x + threadIdx().x
    ic1 = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if ik <= length(kmask_indcs) && ic1 <= size(kL2_rs, 2)
        ind = kmask_indcs[ik]
        acc = zero(eltype(kL2_rs))

        @inbounds for ic2 ∈ axes(ind_lookup, 2)
            ind_packed = ind_lookup[ic1, ic2]
            acc += Λ[ind_packed, ik] * kL1_rs[ind, ic2]
        end

        kL2_rs[ind, ic1] = acc
    end
    return
end

function launch_config_kernel(nsamp_t, kmask_indcs)
    n_timeframes = length(nsamp_t)
    max_frame_samples = maximum(nsamp_t)
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    max_blocks_y = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)

    # kernel threads/blocks settings for multiply_basis_vectors!
    threads_x = min(max_threads, max_frame_samples)
    threads_y = min(max_threads ÷ threads_x, n_timeframes)
    threads_multiply = (threads_x, threads_y)

    blocks_x = ceil(Int, max_frame_samples / threads_x)
    blocks_y = min(ceil(Int, n_timeframes / threads_y), max_blocks_y)
    blocks_multiply = (blocks_x, blocks_y)

    # kernel threads/blocks settings for store_packed_kernel!
    threads_store = min(max_threads, length(kmask_indcs))
    blocks_store = ceil.(Int, length(kmask_indcs) / threads_store)

    return threads_multiply, blocks_multiply, threads_store, blocks_store
end