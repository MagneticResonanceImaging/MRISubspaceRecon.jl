## ##########################################################################
# NFFTNormalOpLowMem - GPU implementation of decomposed Toeplitz trick
# Kernel stored as (Ncoeff*(Ncoeff+1)÷2, n_kmask_1x, 2^D) — packed upper triangular
#############################################################################

function MRISubspaceRecon.NFFTNormalOpLowmem(
    img_shape,
    Λ_decomp::CuArray{Tc,3},
    kmask_indcs::CuArray;
    cmaps=(1,)
    ) where {T <: Real, Tc <: Union{T, Complex{T}}}

    D = length(img_shape)
    Nshift = 2^D
    Ncoeff = (isqrt(8 * size(Λ_decomp, 1) + 1) - 1) ÷ 2
    @assert size(Λ_decomp, 3) == Nshift
    @assert length(kmask_indcs) == size(Λ_decomp, 2)

    # Buffers on NON-oversampled grid
    kL1 = CuArray{Complex{T}}(undef, img_shape..., Ncoeff)
    kL2 = CuArray{Complex{T}}(undef, img_shape..., Ncoeff)

    fftplan  = plan_fft!( kL1, 1:D)
    ifftplan = plan_ifft!(kL2, 1:D)

    # Precompute phase ramps on GPU
    phases = CuArray(MRISubspaceRecon._compute_linphases(img_shape, T))

    # Indexing into the upper triangular matrix
    ind_lookup = CuArray([j<=k ? j+k*(k-1)÷2 : k+j*(j-1)÷2 for j ∈ 1:Ncoeff, k ∈ 1:Ncoeff])

    # Configure thread/block layout for the kernel_mul_lowmem! kernel
    kL1_rs = reshape(kL1, :, Ncoeff)
    kernel = @cuda launch=false kernel_mul_lowmem!(kL1_rs, Λ_decomp, kL1_rs, kmask_indcs, ind_lookup, Int32(1))
    config = launch_configuration(kernel.fun)

    threads_x = min(config.threads ÷ Ncoeff, length(kmask_indcs))
    threads_y = min(config.threads ÷ threads_x, Ncoeff)
    threads = (threads_x, threads_y)
    blocks = cld.((length(kmask_indcs), Ncoeff), threads)

    A = MRISubspaceRecon._NFFTNormalOpLowmem(img_shape, Ncoeff, fftplan, ifftplan,
        Λ_decomp, kmask_indcs, kL1, kL2, cmaps, phases, ind_lookup, threads, blocks)

    return LinearOperator(
        Complex{T},
        prod(img_shape) * Ncoeff,
        prod(img_shape) * Ncoeff,
        true, true,
        (res, x, α, β) -> mul!(res, A, x, α, β),
        nothing,
        (res, x, α, β) -> mul!(res, A, x, α, β);
        S = typeof(similar(Λ_decomp, Complex{T}, 0))
    )
end

#############################################################################
# Direct kernel computation into decomposed packed shape (GPU)
#############################################################################

# Complex basis U → complex kernel
function MRISubspaceRecon.calculate_kernel_lowmem(img_shape, trj::CuArray{T,3}, U::CuArray{Tc};
    sample_mask=CUDA.ones(Bool, size(trj)[2:end]), verbose=false) where {T <: Real, Tc <: Complex{T}}

    img_shape_os = 2 .* img_shape
    D = length(img_shape)
    Nshift = 2^D

    kmask_indcs_os, kmask_indcs_1x, map_1x_cpu, map_shift_cpu = _compute_lowmem_mask_gpu(img_shape, img_shape_os, trj; sample_mask)

    n1x = length(kmask_indcs_1x)
    Ncoeff = size(U, 2)
    Npacked = Ncoeff * (Ncoeff + 1) ÷ 2

    Λ_decomp = CUDA.zeros(Complex{T}, Npacked, n1x, Nshift)

    map_1x_gpu = CuArray(Int32.(map_1x_cpu))
    map_shift_gpu = CuArray(Int32.(map_shift_cpu))
    kmask_indcs_os_gpu = CuArray(kmask_indcs_os)

    nsamp_t = cu(sum(sample_mask, dims=1))
    cumsum_nsamp = CUDA.zeros(eltype(nsamp_t), size(nsamp_t))
    cumsum_nsamp[2:end] = cumsum(nsamp_t[1:end-1])

    λ  = CuArray{Complex{T}}(undef, img_shape_os)
    λ2 = similar(λ)
    S = CuArray{Complex{T}}(undef, sum(nsamp_t))

    fftplan  = plan_fft(λ)
    nfftplan = PlanNUFFT(Complex{T}, img_shape_os; backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size=Val(200))
    set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(trj[:, sample_mask]))

    threads_multiply, blocks_multiply, _, _ = launch_config_kernel(nsamp_t, kmask_indcs_os_gpu)

    n_os = length(kmask_indcs_os)
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    scatter_threads = min(max_threads, n_os)
    scatter_blocks = cld(n_os, scatter_threads)

    verbose && println("calculating decomposed non-Cartesian kernel (complex, GPU)...")
    t = @elapsed CUDA.@sync for ic2 ∈ 1:Ncoeff, ic1 ∈ 1:Ncoeff
        if ic2 >= ic1
            @cuda threads=threads_multiply blocks=blocks_multiply multiply_basis_vectors!(S, U, nsamp_t, cumsum_nsamp, ic1, ic2)

            exec_type1!(λ2, nfftplan, vec(S))
            mul!(λ, fftplan, λ2)

            @cuda threads=scatter_threads blocks=scatter_blocks store_decomposed_kernel!(
                Λ_decomp, λ, kmask_indcs_os_gpu, map_1x_gpu, map_shift_gpu, Int32(ic1), Int32(ic2))
        end
    end
    verbose && println("time to compute kernel: t = $t s")

    return Λ_decomp, CuArray(kmask_indcs_1x)
end

# Real basis U → real kernel (half memory)
function MRISubspaceRecon.calculate_kernel_lowmem(img_shape, trj::CuArray{T,3}, U::CuArray{T};
    sample_mask=CUDA.ones(Bool, size(trj)[2:end]), verbose=false) where {T <: Real}

    img_shape_os = 2 .* img_shape
    D = length(img_shape)
    Nshift = 2^D

    kmask_indcs_os, kmask_indcs_1x, map_1x_cpu, map_shift_cpu = _compute_lowmem_mask_gpu(img_shape, img_shape_os, trj; sample_mask)

    n1x = length(kmask_indcs_1x)
    Ncoeff = size(U, 2)
    Npacked = Ncoeff * (Ncoeff + 1) ÷ 2

    Λ_decomp = CUDA.zeros(T, Npacked, n1x, Nshift)

    map_1x_gpu = CuArray(Int32.(map_1x_cpu))
    map_shift_gpu = CuArray(Int32.(map_shift_cpu))
    kmask_indcs_os_gpu = CuArray(kmask_indcs_os)

    nsamp_t = cu(sum(sample_mask, dims=1))
    @assert sum(nsamp_t) > 0 "Sample_mask removes all samples, cannot compute kernel."
    cumsum_nsamp = CUDA.zeros(eltype(nsamp_t), size(nsamp_t))
    cumsum_nsamp[2:end] = cumsum(nsamp_t[1:end-1])

    λ  = CuArray{T}(undef, img_shape_os)
    λ2 = CuArray{Complex{T}}(undef, img_shape_os[1] ÷ 2 + 1, Base.tail(img_shape_os)...)
    S = CuArray{T}(undef, sum(nsamp_t))

    brfftplan = plan_brfft(λ2, img_shape_os[1])
    nfftplan = PlanNUFFT(T, img_shape_os; backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size=Val(200))
    set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(trj[:, sample_mask]))

    threads_multiply, blocks_multiply, _, _ = launch_config_kernel(nsamp_t, kmask_indcs_os_gpu)

    n_os = length(kmask_indcs_os)
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    scatter_threads = min(max_threads, n_os)
    scatter_blocks = cld(n_os, scatter_threads)

    verbose && println("calculating decomposed non-Cartesian kernel (real, GPU)...")
    t = @elapsed CUDA.@sync for ic2 ∈ 1:Ncoeff, ic1 ∈ 1:Ncoeff
        if ic2 >= ic1
            @cuda threads=threads_multiply blocks=blocks_multiply multiply_basis_vectors!(S, U, nsamp_t, cumsum_nsamp, ic1, ic2)

            exec_type1!(λ2, nfftplan, vec(S))
            λ2 .= conj.(λ2)
            mul!(λ, brfftplan, λ2)

            @cuda threads=scatter_threads blocks=scatter_blocks store_decomposed_kernel!(
                Λ_decomp, λ, kmask_indcs_os_gpu, map_1x_gpu, map_shift_gpu, Int32(ic1), Int32(ic2))
        end
    end
    verbose && println("time to compute kernel: t = $t s")

    return Λ_decomp, CuArray(kmask_indcs_1x)
end

#############################################################################
# Mask computation helper
#############################################################################

function _compute_lowmem_mask_gpu(img_shape, img_shape_os, trj::CuArray{T,3}; sample_mask) where {T}
    D = length(img_shape)
    kmask_indcs_os = calculate_kmask_indcs(img_shape_os, trj; sample_mask)
    @assert all(kmask_indcs_os .> 0)
    @assert all(kmask_indcs_os .<= prod(img_shape_os))

    ci_os_all = CartesianIndices(img_shape_os)
    li_1x_all = LinearIndices(img_shape)

    kmask_1x_set = Set{Int}()
    for ki in kmask_indcs_os
        ci = ci_os_all[ki]
        ci_1x = CartesianIndex(ntuple(d -> (ci[d] - 1) >> 1 + 1, D))
        push!(kmask_1x_set, li_1x_all[ci_1x])
    end
    kmask_indcs_1x = sort!(collect(kmask_1x_set))

    pos_lookup = Dict{Int,Int}()
    for (i, ki) in enumerate(kmask_indcs_1x)
        pos_lookup[ki] = i
    end

    n_os = length(kmask_indcs_os)
    map_1x = Vector{Int}(undef, n_os)
    map_shift = Vector{Int}(undef, n_os)

    for (ios, ki_os) in enumerate(kmask_indcs_os)
        ci = ci_os_all[ki_os]
        shift_bits = 0
        for d in 1:D
            shift_bits |= ((ci[d] - 1) & 1) << (d - 1)
        end
        ci_1x = CartesianIndex(ntuple(d -> (ci[d] - 1) >> 1 + 1, D))
        map_1x[ios] = pos_lookup[li_1x_all[ci_1x]]
        map_shift[ios] = shift_bits + 1
    end

    return kmask_indcs_os, kmask_indcs_1x, map_1x, map_shift
end

#############################################################################
# CUDA scatter kernel — store into packed upper triangular
#############################################################################

function store_decomposed_kernel!(Λ_decomp, λ, kmask_indcs_os, map_1x, map_shift, ic1::Int32, ic2::Int32)
    ios = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if ios > length(kmask_indcs_os)
        return
    end

    i_1x = map_1x[ios]
    s = map_shift[ios]
    ind_packed = ic1 + ic2 * (ic2 - Int32(1)) ÷ Int32(2)
    @inbounds Λ_decomp[ind_packed, i_1x, s] = λ[kmask_indcs_os[ios]]
    return
end

#############################################################################
# mul! for GPU low-memory operator
#############################################################################

function LinearAlgebra.mul!(x::CuArray, S::MRISubspaceRecon._NFFTNormalOpLowmem, b, α, β)
    b = reshape(b, S.shape..., S.Ncoeff)
    if β == 0
        x .= 0
    else
        x .*= β
    end
    xr = reshape(x, S.shape..., S.Ncoeff)

    idx = CartesianIndices(S.shape)
    D = length(S.shape)
    Nshift = 2^D

    for cmap in S.cmaps
        for s in 1:Nshift
            phase_s = @view S.phases[idx, s]

            # 1) Multiply by phase and coil map
            S.kL1[idx, :] .= phase_s .* cmap .* b

            # 2) FFT on non-oversampled grid
            S.fftplan * S.kL1

            # 3) Apply packed kernel
            kL1_rs = reshape(S.kL1, :, S.Ncoeff)
            kL2_rs = reshape(S.kL2, :, S.Ncoeff)
            fill!(S.kL2, 0)
            @cuda threads=S.threads blocks=S.blocks kernel_mul_lowmem!(
                kL2_rs, S.Λ_decomp, kL1_rs, S.kmask_indcs, S.ind_lookup, Int32(s))

            # 4) IFFT on non-oversampled grid
            S.ifftplan * S.kL2

            # 5) Accumulate with conjugate phase and coil map
            @views xr .+= α .* conj.(cmap) .* conj.(phase_s) .* S.kL2[idx, :]
        end
    end
    return x
end

#############################################################################
# CUDA kernel for packed decomposed kernel multiply
#############################################################################

# Complex kernel: Hermitian unpacking
function kernel_mul_lowmem!(kL2_rs, Λ_decomp::CuDeviceArray{Tc,3}, kL1_rs, kmask_indcs, ind_lookup, s::Int32) where {Tc <: Complex}
    ik  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ic1 = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ik <= length(kmask_indcs) && ic1 <= size(kL2_rs, 2)
        ind = kmask_indcs[ik]
        acc = zero(eltype(kL2_rs))

        @inbounds for ic2 in 1:size(ind_lookup, 2)
            if ic1 <= ic2
                ip = ind_lookup[ic1, ic2]
                acc += Λ_decomp[ip, ik, s] * kL1_rs[ind, ic2]
            else
                ip = ind_lookup[ic2, ic1]
                acc += conj(Λ_decomp[ip, ik, s]) * kL1_rs[ind, ic2]
            end
        end

        kL2_rs[ind, ic1] = acc
    end
    return
end

# Real kernel: symmetric unpacking
function kernel_mul_lowmem!(kL2_rs, Λ_decomp::CuDeviceArray{T,3}, kL1_rs, kmask_indcs, ind_lookup, s::Int32) where {T <: Real}
    ik  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ic1 = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ik <= length(kmask_indcs) && ic1 <= size(kL2_rs, 2)
        ind = kmask_indcs[ik]
        acc = zero(eltype(kL2_rs))

        @inbounds for ic2 in 1:size(ind_lookup, 2)
            ip = ind_lookup[ic1, ic2]
            acc += Λ_decomp[ip, ik, s] * kL1_rs[ind, ic2]
        end

        kL2_rs[ind, ic1] = acc
    end
    return
end