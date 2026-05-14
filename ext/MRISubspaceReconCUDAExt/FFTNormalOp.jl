function MRISubspaceRecon.FFTNormalOp(img_shape, trj::CuArray{<:Integer,3}, U; cmaps=(1,), sample_mask=CUDA.ones(Bool, size(trj)[2:end]))
    Λ = calculate_kernel_cartesian(img_shape, trj, U; sample_mask)
    return MRISubspaceRecon.FFTNormalOp(Λ; cmaps)
end

# Wrapper for 4D data arrays
function MRISubspaceRecon.FFTNormalOp(img_shape, trj::CuArray{<:Integer,4}, U; sample_mask=CUDA.ones(Bool, size(trj)[2:end]), kwargs...)
    trj = reshape(trj, size(trj, 1), :, size(trj, 4))
    sample_mask = reshape(sample_mask, :, size(sample_mask, 3))
    return MRISubspaceRecon.FFTNormalOp(img_shape, trj, U; sample_mask, kwargs...)
end

# GPU FFTNormalOp constructor from pre-computed Λ
function MRISubspaceRecon.FFTNormalOp(Λ::CuArray{Tc}; cmaps=(1,), eltype_x=eltype(Λ)) where {Tc <: Complex}

    Ncoeff = size(Λ, 1)
    img_shape = size(Λ)[3:end]

    kL1 = CuArray{Tc}(undef, img_shape..., Ncoeff)
    kL2 = similar(kL1)

    @views kmask = (Λ[1, 1, CartesianIndices(img_shape)] .!= 0)
    kmask_indcs = findall(vec(kmask))
    Λ = reshape(Λ, Ncoeff, Ncoeff, :)
    Λ = Λ[:, :, kmask_indcs]

    fftplan  = plan_fft!(kL1, 1:length(img_shape))
    ifftplan = plan_ifft!(kL2, 1:length(img_shape))

    # Pre-compute kernel launch configuration
    Nk = size(Λ, 3)
    kL1_rs = reshape(kL1, :, Ncoeff)
    kL2_rs = reshape(kL2, :, Ncoeff)
    kernel = @cuda launch=false kernel_mul_cartesian!(kL2_rs, Λ, kL1_rs, kmask_indcs)
    config = launch_configuration(kernel.fun)

    threads_x = min(config.threads ÷ Ncoeff, Nk)
    threads_y = min(config.threads ÷ threads_x, Ncoeff)
    threads = (threads_x, threads_y)
    blocks = cld.((Nk, Ncoeff), threads)

    A = MRISubspaceRecon._FFTNormalOp(img_shape, Ncoeff, fftplan, ifftplan, Λ, kmask_indcs, kL1, kL2, cmaps, threads, blocks)

    return LinearOperator(
        eltype_x,
        prod(A.shape) * A.Ncoeff,
        prod(A.shape) * A.Ncoeff,
        true,
        true,
        (res, x, α, β) -> mul!(res, A, x, α, β),
        nothing,
        (res, x, α, β) -> mul!(res, A, x, α, β);
        S = typeof(similar(Λ, eltype_x, 0))
    )
end

## ##########################################################################
# Internal use
#############################################################################

function calculate_kernel_cartesian(img_shape, trj::CuArray{<:Integer,3}, U; sample_mask=CUDA.ones(Bool, size(trj)[2:end]), verbose=false)
    trj_cpu = Array(trj)
    mask_cpu = Array(sample_mask)
    @assert all(d -> all((@view(trj_cpu[d, :, :])[mask_cpu]) .>= 1) && all((@view(trj_cpu[d, :, :])[mask_cpu]) .<= img_shape[d]), 1:size(trj, 1)) "Cartesian trajectory values must be in the range 1:img_shape[d] for each dimension d."
    Ncoeff = size(U, 2)
    Nrep = size(U, 3) # number of repetitions (defaults to 1)

    # For complex U, we need to accumulate real and imaginary parts separately via atomics        
    Λ_real = CUDA.zeros(real(eltype(U)), 2, Ncoeff, Ncoeff, img_shape...) # Use a real-valued array with 2x the leading dimension to handle complex atomics

    Nsamp = size(trj, 2)
    Nt = size(trj, 3)

    verbose && println("calculating Cartesian kernel on GPU...")
    t = @elapsed CUDA.@sync begin
        # Configure kernel launch
        kernel = @cuda launch=false kernel_cartesian_complex!(Λ_real, trj, U, sample_mask, img_shape, Ncoeff, Nsamp, Nt, Nrep)
        config = launch_configuration(kernel.fun)
        threads = min(config.threads, Nsamp)
        blocks = cld(Nsamp, threads)

        @cuda threads=threads blocks=blocks kernel_cartesian_complex!(Λ_real, trj, U, sample_mask, img_shape, Ncoeff, Nsamp, Nt, Nrep)
    end
    verbose && println("time to compute kernel: t = $t s")

    # Reinterpret the 2-element real array as complex
    Λ = reinterpret(eltype(U), reshape(Λ_real, 2, Ncoeff * Ncoeff * prod(img_shape)))
    Λ = reshape(Λ, Ncoeff, Ncoeff, img_shape...)
    return Λ
end

function kernel_cartesian_complex!(Λ_real, trj, U, sample_mask, img_shape::NTuple{N,Int}, Ncoeff, Nsamp, Nt, Nrep) where N
    is = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if is <= Nsamp
        for it in 1:Nt
            if sample_mask[is, it]
                # Compute k-space index with ifftshift incorporated
                k_idx = ntuple(Val(N)) do j
                    mod1(Int(trj[j, is, it]) - img_shape[j] ÷ 2, img_shape[j])
                end

                # Accumulate into kernel using atomic adds on real/imag parts
                for ic2 in 1:Ncoeff, ic1 in 1:Ncoeff
                    val = zero(eltype(U))
                    for irep in 1:Nrep
                        val += conj(U[it, ic1, irep]) * U[it, ic2, irep]
                    end
                    # CUDA atomics does not support complex directly, so split into real/imag
                    CUDA.@atomic Λ_real[1, ic1, ic2, k_idx...] += real(val)
                    CUDA.@atomic Λ_real[2, ic1, ic2, k_idx...] += imag(val)
                end
            end
        end
    end
    return
end

## ##########################################################################
# mul! for GPU _FFTNormalOp
#############################################################################

function LinearAlgebra.mul!(x::CuArray, S::MRISubspaceRecon._FFTNormalOp, b, α, β)
    idx = CartesianIndices(S.shape)

    b = reshape(b, S.shape..., S.Ncoeff)
    if β == 0
        x .= 0
    else
        x .*= β
    end
    xr = reshape(x, S.shape..., S.Ncoeff)

    for cmap ∈ S.cmaps
        # Forward FFT: multiply by coil map and transform
        S.kL1[idx, :] .= cmap .* b
        S.fftplan * S.kL1

        # Multiply by Toeplitz kernel
        kL1_rs = reshape(S.kL1, :, S.Ncoeff)
        kL2_rs = reshape(S.kL2, :, S.Ncoeff)
        fill!(S.kL2, 0)

        @cuda threads=S.threads blocks=S.blocks kernel_mul_cartesian!(kL2_rs, S.Λ, kL1_rs, S.kmask_indcs)

        # Inverse FFT and accumulate
        S.ifftplan * S.kL2
        @views xr .+= α .* conj.(cmap) .* S.kL2[idx, :]
    end
    return x
end

function kernel_mul_cartesian!(kL2_rs, Λ, kL1_rs, kmask_indcs)
    ik  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ic1 = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ik <= length(kmask_indcs) && ic1 <= size(kL2_rs, 2)
        ind = kmask_indcs[ik]
        acc = zero(eltype(kL2_rs))

        @inbounds for ic2 in axes(Λ, 2)
            acc += Λ[ic1, ic2, ik] * kL1_rs[ind, ic2]
        end
        kL2_rs[ind, ic1] = acc
    end
    return
end