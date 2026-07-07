## ##########################################################################
# NFFTNormalOpLowMem - GPU implementation of decomposed Toeplitz trick
#############################################################################

function MRISubspaceRecon.NFFTNormalOpLowmem(
    img_shape,
    Λ_decomp::CuArray{Tc,4},
    kmask_indcs::CuArray;
    cmaps=(1,)
    ) where {T <: Real, Tc <: Union{T, Complex{T}}}

    Ncoeff = size(Λ_decomp, 1)
    D = length(img_shape)
    Nshift = 2^D
    @assert size(Λ_decomp, 4) == Nshift
    @assert length(kmask_indcs) == size(Λ_decomp, 3)

    # Buffers on NON-oversampled grid (2^D times smaller!)
    kL1 = CuArray{Complex{T}}(undef, img_shape..., Ncoeff)
    kL2 = CuArray{Complex{T}}(undef, img_shape..., Ncoeff)

    fftplan  = plan_fft!( kL1, 1:D)
    ifftplan = plan_ifft!(kL2, 1:D)

    # Precompute phase ramps on GPU
    phases = CuArray(MRISubspaceRecon._compute_linphases(img_shape, T))

    # Configure thread/block layout for the kernel_mul_lowmem! kernel
    kL1_rs = reshape(kL1, :, Ncoeff)
    kernel = @cuda launch=false kernel_mul_lowmem!(kL1_rs, Λ_decomp, kL1_rs, kmask_indcs, Int32(1))
    config = launch_configuration(kernel.fun)

    threads_x = min(config.threads ÷ Ncoeff, length(kmask_indcs))
    threads_y = min(config.threads ÷ threads_x, Ncoeff)
    threads = (threads_x, threads_y)
    blocks = cld.((length(kmask_indcs), Ncoeff), threads)

    A = MRISubspaceRecon._NFFTNormalOpLowmem(img_shape, Ncoeff, fftplan, ifftplan,
        Λ_decomp, kmask_indcs, kL1, kL2, cmaps, phases, threads, blocks)

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
# Kernel decomposition on GPU
#############################################################################

"""
    decompose_kernel_gpu(img_shape, Λ_packed, kmask_indcs_os) -> (Λ_decomp, kmask_indcs_1x)

GPU version of kernel decomposition. Takes the packed upper-triangular kernel
on the 2× grid and produces the full (Ncoeff, Ncoeff, n_kmask_1x, 2^D) decomposed
kernel on the 1× grid.
"""
function decompose_kernel_gpu(img_shape, Λ_packed::CuArray, kmask_indcs_os, Ncoeff::Int)
    T = real(eltype(Λ_packed))
    img_shape_os = 2 .* img_shape
    D = length(img_shape)
    Nshift = 2^D

    # Ensure kmask is on CPU for index mapping (one-time cost)
    kmask_os_cpu = kmask_indcs_os isa CuArray ? Array(kmask_indcs_os) : Vector(kmask_indcs_os)
    ci_os_all = CartesianIndices(img_shape_os)
    li_1x_all = LinearIndices(img_shape)

    # Find 1x grid positions and build mapping
    kmask_1x_set = Set{Int}()
    for ki in kmask_os_cpu
        ci = ci_os_all[ki]
        ci_1x = CartesianIndex(ntuple(d -> (ci[d] - 1) >> 1 + 1, D))
        push!(kmask_1x_set, li_1x_all[ci_1x])
    end
    kmask_indcs_1x = sort!(collect(kmask_1x_set))

    pos_lookup = Dict{Int,Int}()
    for (i, ki) in enumerate(kmask_indcs_1x)
        pos_lookup[ki] = i
    end

    # Build the mapping: for each OS index, what is the (1x_position, shift_index)?
    n_os = length(kmask_os_cpu)
    n_1x = length(kmask_indcs_1x)
    # map_1x[ios] = position in kmask_indcs_1x
    # map_shift[ios] = shift index (1-based)
    map_1x = Vector{Int32}(undef, n_os)
    map_shift = Vector{Int32}(undef, n_os)

    for (ios, ki_os) in enumerate(kmask_os_cpu)
        ci = ci_os_all[ki_os]
        shift_bits = 0
        for d in 1:D
            shift_bits |= ((ci[d] - 1) & 1) << (d - 1)
        end
        ci_1x = CartesianIndex(ntuple(d -> (ci[d] - 1) >> 1 + 1, D))
        map_1x[ios] = pos_lookup[li_1x_all[ci_1x]]
        map_shift[ios] = shift_bits + 1
    end

    map_1x_gpu = CuArray(map_1x)
    map_shift_gpu = CuArray(map_shift)

    # Allocate output
    Λ_decomp = CUDA.zeros(Complex{T}, Ncoeff, Ncoeff, n_1x, Nshift)

    # Launch kernel to scatter and unpack
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    nthreads = min(max_threads, n_os)
    nblocks = cld(n_os, nthreads)

    if eltype(Λ_packed) <: Complex
        @cuda threads=nthreads blocks=nblocks scatter_kernel_complex!(
            Λ_decomp, Λ_packed, map_1x_gpu, map_shift_gpu, Int32(Ncoeff))
    else
        @cuda threads=nthreads blocks=nblocks scatter_kernel_real!(
            Λ_decomp, Λ_packed, map_1x_gpu, map_shift_gpu, Int32(Ncoeff))
    end

    return Λ_decomp, CuArray(kmask_indcs_1x)
end

# CUDA kernel: scatter packed complex OS kernel entries into decomposed 4D array
function scatter_kernel_complex!(Λ_decomp, Λ_packed, map_1x, map_shift, Ncoeff::Int32)
    ios = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if ios > length(map_1x)
        return
    end

    i_1x = map_1x[ios]
    s = map_shift[ios]

    @inbounds for ic2 in Int32(1):Ncoeff
        for ic1 in Int32(1):Ncoeff
            if ic1 <= ic2
                ind_packed = ic1 + ic2 * (ic2 - Int32(1)) ÷ Int32(2)
                Λ_decomp[ic1, ic2, i_1x, s] = Λ_packed[ind_packed, ios]
            else
                ind_packed = ic2 + ic1 * (ic1 - Int32(1)) ÷ Int32(2)
                Λ_decomp[ic1, ic2, i_1x, s] = conj(Λ_packed[ind_packed, ios])
            end
        end
    end
    return
end

# CUDA kernel: scatter packed real OS kernel entries into decomposed 4D array
# Real packed kernel is symmetric, so Λ[ic1,ic2] == Λ[ic2,ic1]
function scatter_kernel_real!(Λ_decomp, Λ_packed, map_1x, map_shift, Ncoeff::Int32)
    ios = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if ios > length(map_1x)
        return
    end

    i_1x = map_1x[ios]
    s = map_shift[ios]

    @inbounds for ic2 in Int32(1):Ncoeff
        for ic1 in Int32(1):Ncoeff
            if ic1 <= ic2
                ind_packed = ic1 + ic2 * (ic2 - Int32(1)) ÷ Int32(2)
            else
                ind_packed = ic2 + ic1 * (ic1 - Int32(1)) ÷ Int32(2)
            end
            val = Λ_packed[ind_packed, ios]
            Λ_decomp[ic1, ic2, i_1x, s] = val
        end
    end
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

            # 3) Apply decomposed kernel
            kL1_rs = reshape(S.kL1, :, S.Ncoeff)
            kL2_rs = reshape(S.kL2, :, S.Ncoeff)
            fill!(S.kL2, 0)
            @cuda threads=S.threads blocks=S.blocks kernel_mul_lowmem!(
                kL2_rs, S.Λ_decomp, kL1_rs, S.kmask_indcs, Int32(s))

            # 4) IFFT on non-oversampled grid
            S.ifftplan * S.kL2

            # 5) Accumulate with conjugate phase and coil map
            @views xr .+= α .* conj.(cmap) .* conj.(phase_s) .* S.kL2[idx, :]
        end
    end
    return x
end

#############################################################################
# CUDA kernel for decomposed kernel multiply
#############################################################################

# Full matrix multiply at each masked k-space position for a given shift
function kernel_mul_lowmem!(kL2_rs, Λ_decomp, kL1_rs, kmask_indcs, s::Int32)
    ik  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ic1 = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ik <= length(kmask_indcs) && ic1 <= size(kL2_rs, 2)
        ind = kmask_indcs[ik]
        acc = zero(eltype(kL2_rs))

        @inbounds for ic2 in 1:size(Λ_decomp, 2)
            acc += Λ_decomp[ic1, ic2, ik, s] * kL1_rs[ind, ic2]
        end

        kL2_rs[ind, ic1] = acc
    end
    return
end