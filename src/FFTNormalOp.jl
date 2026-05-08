## ##########################################################################
# FFTNormalOp
#############################################################################

"""
    FFTNormalOp(img_shape, trj, U; cmaps)
    FFTNormalOp(M, U; cmaps)
    FFTNormalOp(Λ; cmaps, eltype_x)

Create normal operator of FFT operator.
Differentiate between functions exploiting a pre-calculated kernel basis `Λ` and the functions which calculate Λ based on a passed trajectory `trj` or mask `M`.

# Arguments
- `img_shape::Tuple{Int}`: Image dimensions
- `traj::Vector{Matrix}`: Trajectory
- `U::Matrix`: Basis coefficients of subspace
- `cmaps::Matrix=(1,)`: Coil sensitivities
- `sample_mask::AbstractArray{Bool}`=`trues(size(trj)[2:end])`: Mask indicating which acquired k-space samples are included in the reconstruction
- `M::Vector{Matrix}`: Mask specific to `FFTNormalOp(M, U; cmaps)` method. Here, `M` represents a Cartesian binary mask that indicates in every time frame which phase-encoding lines were acquired. This means, for example, `M` is sized `(Nx, Ny, Nt)` for a 2D image. Explicit use of a trajectory with `FFTNormalOp(img_shape, trj, U; cmaps)` is generally recommended.
- `Λ::Array{Complex{T},3}`: Toeplitz kernel basis resulting from internal `calculate_kernel_cartesian` method. 
- `num_fft_threads::Int`=`round(Int, Threads.nthreads()/size(U, 2))` or `round(Int, Threads.nthreads()/size(Λ, 1)): Number of Threads for FFT
- `eltype_x`=`eltype(Λ)` define the type of `x` (in the product `FFTNormalOp(Λ) * x`). The default is the same eltype as `Λ`

# References
1. Tamir JI, et al. “T2 shuffling: Sharp, multicontrast, volumetric fast spin-echo imaging”. Magn Reson Med. 77.1 (2017), pp. 180–195. https://doi.org/10.1002/mrm.26102
2. Assländer J, et al. “Low rank alternating direction method of multipliers reconstruction for MR fingerprinting”. Magn Reson Med 79.1 (2018), pp. 83–96. https://doi.org/10.1002/mrm.26639
"""
function FFTNormalOp(img_shape, trj::AbstractArray{<:Integer, 3}, U; cmaps=(1,), sample_mask=trues(size(trj)[2:end]), num_fft_threads=round(Int, Threads.nthreads()/size(U, 2)))
    Λ = calculate_kernel_cartesian(img_shape, trj, U; sample_mask)
    return FFTNormalOp(Λ; cmaps, num_fft_threads)
end

# Wrapper for 4D data arrays
function FFTNormalOp(img_shape, trj::AbstractArray{<:Integer,4}, U; sample_mask=trues(size(trj)[2:end]), kwargs...)
    trj = reshape(trj, size(trj, 1), :, size(trj,4))
    sample_mask = reshape(sample_mask, :, size(sample_mask,3))
    return FFTNormalOp(img_shape, trj, U; sample_mask, kwargs...)
end

function FFTNormalOp(M, U; cmaps=(1,), num_fft_threads=round(Int, Threads.nthreads()/size(U, 2)))
    Λ = calculate_kernel_cartesian(M, U)
    return FFTNormalOp(Λ; cmaps, num_fft_threads )
end

function FFTNormalOp(Λ; cmaps=(1,), num_fft_threads=round(Int, Threads.nthreads()/size(Λ, 1)), eltype_x=eltype(Λ))
    Ncoeff = size(Λ, 1)
    img_shape = size(Λ)[3:end]
    kL1 = Array{eltype_x}(undef, img_shape..., Ncoeff)
    kL2 = similar(kL1)

    @views kmask = (Λ[1, 1, CartesianIndices(img_shape)] .!= 0)
    kmask_indcs = findall(vec(kmask))
    Λ = reshape(Λ, Ncoeff, Ncoeff, :)
    Λ = Λ[:, :, kmask_indcs]

    ktmp = @view kL1[CartesianIndices(img_shape), 1]
    fftplan = plan_fft!(ktmp; flags=FFTW.MEASURE, num_threads=num_fft_threads)
    ifftplan = plan_ifft!(ktmp; flags=FFTW.MEASURE, num_threads=num_fft_threads)
    A = _FFTNormalOp(img_shape, Ncoeff, fftplan, ifftplan, Λ, kmask_indcs, kL1, kL2, cmaps, nothing, nothing)

    return LinearOperator(
        eltype_x,
        prod(A.shape) * A.Ncoeff,
        prod(A.shape) * A.Ncoeff,
        true,
        true,
        (res, x, α, β) -> mul!(res, A, x, α, β),
        nothing,
        (res, x, α, β) -> mul!(res, A, x, α, β),
    )
end

## ##########################################################################
# Internal use
#############################################################################
struct _FFTNormalOp{S,E,F,G,H,I,J,K,L,M}
    shape::S
    Ncoeff::Int
    fftplan::E
    ifftplan::F
    Λ::G
    kmask_indcs::H
    kL1::I
    kL2::J
    cmaps::K
    threads::L
    blocks::M
end

function calculate_kernel_cartesian(img_shape, trj::AbstractArray{<:Integer,3}, U; sample_mask=trues(size(trj)[2:end]), verbose=false)
    Ncoeff = size(U, 2)
    Λ = zeros(eltype(U), Ncoeff, Ncoeff, img_shape...)

    verbose && println("calculating Cartesian kernel...")
    t = @elapsed Threads.@threads for ic ∈ CartesianIndices((Ncoeff, Ncoeff))
        for it ∈ axes(U, 1), is ∈ axes(trj, 2)
            if sample_mask[is, it]
                k_idx = ntuple(j -> mod1(Int(trj[j, is, it]) - img_shape[j] ÷ 2, img_shape[j]), length(img_shape)) # incorporates ifftshift
                k_idx = CartesianIndex(k_idx)
                for irep ∈ axes(U, 3)
                    Λ[ic[1], ic[2], k_idx] += conj(U[it, ic[1], irep]) * U[it, ic[2], irep]
                end
            end
        end
    end
    verbose && println("time to compute kernel: t = $t s")
    return Λ
end

function calculate_kernel_cartesian(M, U)
    Ncoeff = size(U, 2)
    img_shape = size(M)[1:end-1]
    Λ = Array{eltype(U)}(undef, Ncoeff, Ncoeff, img_shape...)

    M .= ifftshift(M, 1:length(img_shape))
    Threads.@threads for i ∈ CartesianIndices(img_shape)
        Λ[:, :, i] .= U' * (M[i, :] .* U) #U' * diagm(D) * U
    end
    return Λ
end

function LinearAlgebra.mul!(x::AbstractVector{T}, S::_FFTNormalOp, b, α, β) where {T}
    idx = CartesianIndices(S.shape)

    b = reshape(b, S.shape..., S.Ncoeff)
    if β == 0
        fill!(x, zero(T)) # to avoid 0 * NaN == NaN
    else
        x .*= β
    end
    xr = reshape(x, S.shape..., S.Ncoeff)

    bthreads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        for cmap ∈ S.cmaps
            Threads.@threads for i ∈ 1:S.Ncoeff # multiply by C and F
                @views S.kL1[idx, i] .= cmap .* b[idx, i]
                @views S.fftplan * S.kL1[idx, i]
            end

            kL1_rs = reshape(S.kL1, :, S.Ncoeff)
            kL2_rs = reshape(S.kL2, :, S.Ncoeff)
            Threads.@threads for i in eachindex(kL2_rs)
                kL2_rs[i] = 0
            end
            Threads.@threads for i ∈ axes(S.Λ, 3)
                @views @inbounds mul!(kL2_rs[S.kmask_indcs[i], :], S.Λ[:, :, i], kL1_rs[S.kmask_indcs[i], :])
            end

            Threads.@threads for i ∈ 1:S.Ncoeff # multiply by C' and F'
                @views S.ifftplan * S.kL2[idx, i]
                @views xr[idx, i] .+= α .* conj.(cmap) .* S.kL2[idx, i]
            end
        end
    finally
        BLAS.set_num_threads(bthreads)
    end
    return x
end