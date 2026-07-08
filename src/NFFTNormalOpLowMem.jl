## ##########################################################################
# NFFTNormalOpLowMem - Decomposed Toeplitz trick for reduced memory usage
#
# Memory layout: Λ_decomp has shape (Ncoeff*(Ncoeff+1)÷2, n_kmask_1x, 2^D)
#   - First axis: packed upper-triangular storage (same as standard operator)
#   - Second axis: non-zero k-space positions on the 1× grid
#   - Third axis: shift index (even/odd frequency sub-grids)
#
# The kernel is real when U is real, complex when U is complex.
#############################################################################

"""
    NFFTNormalOpLowmem(img_shape, Λ_decomp, kmask_indcs; cmaps, num_fft_threads)

Low-memory Toeplitz normal operator. Buffers `kL1`, `kL2` are allocated on the
non-oversampled grid (2^D times smaller than the standard operator).

The decomposed kernel `Λ_decomp` has shape `(Ncoeff*(Ncoeff+1)÷2, n_kmask_1x, 2^D)`
using packed upper-triangular storage.
"""
function NFFTNormalOpLowmem(
    img_shape,
    Λ_decomp::AbstractArray{Tc,3},
    kmask_indcs::Vector{<:Integer};
    cmaps=(1,),
    num_fft_threads=1
    ) where {T, Tc <: Union{T, Complex{T}}}

    D = length(img_shape)
    Nshift = 2^D
    @assert size(Λ_decomp, 3) == Nshift
    @assert length(kmask_indcs) == size(Λ_decomp, 2)
    @assert all(kmask_indcs .> 0)
    @assert all(kmask_indcs .<= prod(img_shape))

    # Derive Ncoeff from packed size
    Ncoeff = (isqrt(8 * size(Λ_decomp, 1) + 1) - 1) ÷ 2

    kL1 = Array{Complex{T}}(undef, img_shape..., Ncoeff)
    kL2 = similar(kL1)

    ktmp = @view kL1[CartesianIndices(img_shape), 1]
    fftplan  = plan_fft!(ktmp; flags=FFTW.MEASURE, num_threads=num_fft_threads)
    ifftplan = plan_ifft!(ktmp; flags=FFTW.MEASURE, num_threads=num_fft_threads)

    phases = _compute_linphases(img_shape, T)

    ind_lookup = [j<=k ? j+k*(k-1)÷2 : k+j*(j-1)÷2 for j ∈ 1:Ncoeff, k ∈ 1:Ncoeff]

    A = _NFFTNormalOpLowmem(img_shape, Ncoeff, fftplan, ifftplan, Λ_decomp, kmask_indcs, kL1, kL2, cmaps, phases, ind_lookup, nothing, nothing)

    return LinearOperator(
        Complex{T},
        prod(img_shape) * Ncoeff,
        prod(img_shape) * Ncoeff,
        true, true,
        (res, x, α, β) -> mul!(res, A, x, α, β),
        nothing,
        (res, x, α, β) -> mul!(res, A, x, α, β),
    )
end

#############################################################################
# Internal struct
#############################################################################
struct _NFFTNormalOpLowmem{S,E,F,G,H,I,J,K,P,L,M,N}
    shape::S
    Ncoeff::Int
    fftplan::E
    ifftplan::F
    Λ_decomp::G    # (Ncoeff*(Ncoeff+1)÷2, n_kmask_1x, 2^D) packed upper triangular
    kmask_indcs::H  # indices into the 1x grid
    kL1::I
    kL2::J
    cmaps::K
    phases::P       # (img_shape..., 2^D)
    ind_lookup::L   # (Ncoeff, Ncoeff) packed index lookup
    threads::M      # GPU thread config (nothing on CPU)
    blocks::N       # GPU block config (nothing on CPU)
end

#############################################################################
# Direct kernel computation into decomposed packed shape
#############################################################################

# Compute the 1x kmask and the mapping from 2x mask to (1x position, shift)
function _compute_lowmem_mask(img_shape, img_shape_os, trj; sample_mask)
    D = length(img_shape)
    kmask_indcs_os = calculate_kmask_indcs(img_shape_os, trj; sample_mask)
    @assert all(kmask_indcs_os .> 0)
    @assert all(kmask_indcs_os .<= prod(img_shape_os))

    ci_os_all = CartesianIndices(img_shape_os)
    li_1x_all = LinearIndices(img_shape)

    # Find 1x positions
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

    # Build per-os-entry mapping: (1x_index_in_mask, shift_index)
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

# Complex basis U → complex kernel (packed upper triangular)
function calculate_kernel_lowmem(img_shape, trj::AbstractArray{T,3}, U::AbstractArray{Tc}; sample_mask=trues(size(trj)[2:end]), verbose=false) where {T, Tc <: Complex{T}}
    img_shape_os = 2 .* img_shape
    D = length(img_shape)
    Nshift = 2^D

    kmask_indcs_os, kmask_indcs_1x, map_1x, map_shift = _compute_lowmem_mask(img_shape, img_shape_os, trj; sample_mask)

    nsamp_t = vec(sum(sample_mask; dims=1))
    @assert sum(nsamp_t) > 0 "Mask removes all samples, cannot compute kernel."
    cumsum_nsamp = cumsum(nsamp_t) .+ 1
    prepend!(cumsum_nsamp, 1)

    λ  = Array{Complex{T}}(undef, img_shape_os)
    λ2 = similar(λ)

    Ncoeff = size(U, 2)
    Npacked = Ncoeff * (Ncoeff + 1) ÷ 2
    n1x = length(kmask_indcs_1x)
    Λ_decomp = zeros(Complex{T}, Npacked, n1x, Nshift)
    S = Vector{Complex{T}}(undef, sum(nsamp_t))

    fftplan  = plan_fft(λ; flags=FFTW.MEASURE, num_threads=Threads.nthreads())
    nfftplan = PlanNUFFT(Complex{T}, img_shape_os)
    set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(trj[:, sample_mask]))

    verbose && println("calculating decomposed non-Cartesian kernel (complex)...")
    t = @elapsed for ic2 ∈ 1:Ncoeff, ic1 ∈ 1:Ncoeff
        if ic2 >= ic1
            @simd for it ∈ axes(U, 1)
                idx1 = cumsum_nsamp[it]
                idx2 = cumsum_nsamp[it + 1] - 1
                @inbounds S[idx1:idx2] .= conj(U[it, ic1]) * U[it, ic2]
            end

            NonuniformFFTs.exec_type1!(λ2, nfftplan, vec(S))
            mul!(λ, fftplan, λ2)

            ind_packed = ic1 + ic2 * (ic2 - 1) ÷ 2
            Threads.@threads for ios ∈ eachindex(kmask_indcs_os)
                @inbounds Λ_decomp[ind_packed, map_1x[ios], map_shift[ios]] = λ[kmask_indcs_os[ios]]
            end
        end
    end
    verbose && println("time to compute kernel: t = $t s")
    return Λ_decomp, kmask_indcs_1x
end

# Real basis U → real kernel (packed upper triangular, half memory)
function calculate_kernel_lowmem(img_shape, trj::AbstractArray, U::AbstractArray{T}; sample_mask=trues(size(trj)[2:end]), verbose=false) where {T <: Real}
    img_shape_os = 2 .* img_shape
    D = length(img_shape)
    Nshift = 2^D

    kmask_indcs_os, kmask_indcs_1x, map_1x, map_shift = _compute_lowmem_mask(img_shape, img_shape_os, trj; sample_mask)

    nsamp_t = vec(sum(sample_mask; dims=1))
    @assert sum(nsamp_t) > 0 "Mask removes all samples, cannot compute kernel."
    cumsum_nsamp = cumsum(nsamp_t) .+ 1
    prepend!(cumsum_nsamp, 1)

    λ  = Array{T}(undef, img_shape_os)
    λ2 = Array{Complex{T}}(undef, img_shape_os[1] ÷ 2 + 1, Base.tail(img_shape_os)...)

    Ncoeff = size(U, 2)
    Npacked = Ncoeff * (Ncoeff + 1) ÷ 2
    n1x = length(kmask_indcs_1x)
    Λ_decomp = zeros(T, Npacked, n1x, Nshift)
    S = Array{T}(undef, sum(nsamp_t))

    brfftplan = plan_brfft(λ2, img_shape_os[1]; flags=FFTW.MEASURE, num_threads=Threads.nthreads())
    nfftplan = PlanNUFFT(T, img_shape_os)
    set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(trj[:, sample_mask]))

    verbose && println("calculating decomposed non-Cartesian kernel (real)...")
    t = @elapsed for ic2 ∈ 1:Ncoeff, ic1 ∈ 1:Ncoeff
        if ic2 >= ic1
            @simd for it ∈ axes(U, 1)
                idx1 = cumsum_nsamp[it]
                idx2 = cumsum_nsamp[it + 1] - 1
                @inbounds S[idx1:idx2] .= U[it, ic1] * U[it, ic2]
            end

            NonuniformFFTs.exec_type1!(λ2, nfftplan, vec(S))
            λ2 .= conj.(λ2)
            mul!(λ, brfftplan, λ2)

            ind_packed = ic1 + ic2 * (ic2 - 1) ÷ 2
            Threads.@threads for ios ∈ eachindex(kmask_indcs_os)
                @inbounds Λ_decomp[ind_packed, map_1x[ios], map_shift[ios]] = λ[kmask_indcs_os[ios]]
            end
        end
    end
    verbose && println("time to compute kernel: t = $t s")
    return Λ_decomp, kmask_indcs_1x
end

#############################################################################
# Linear phase computation
#############################################################################

"""
    _compute_linphases(img_shape, ::Type{T})

Precompute linear phase ramps of size `(img_shape..., 2^D)`.

The 2×-oversampled FFT of a zero-padded signal satisfies:
    FFT_2N[2k + b] = FFT_N( x[n] · exp(-iπ·b·n/N) )[k]

The factor `1/√(2^D)` per phase application (forward and adjoint) gives
the `1/2^D` normalization from summing over all 2^D shifts.
"""
function _compute_linphases(img_shape, ::Type{T}) where {T}
    D = length(img_shape)
    Nshift = 2^D
    Tr = real(T)
    phases = Array{Complex{Tr}}(undef, img_shape..., Nshift)
    scale = one(Tr) / sqrt(Tr(Nshift))

    for s in 1:Nshift
        bits = s - 1
        for I in CartesianIndices(img_shape)
            θ = zero(Tr)
            for d in 1:D
                if (bits >> (d - 1)) & 1 == 1
                    θ += Tr(I[d] - 1) / Tr(img_shape[d])
                end
            end
            phases[I, s] = scale * cispi(Tr(-1) * θ)
        end
    end
    return phases
end

#############################################################################
# mul! implementation (CPU)
#############################################################################

function LinearAlgebra.mul!(x::AbstractVector{T}, S::_NFFTNormalOpLowmem, b, α, β) where {T}
    idx = CartesianIndices(S.shape)
    D = length(S.shape)
    Nshift = 2^D

    b = reshape(b, S.shape..., S.Ncoeff)
    if β == 0
        fill!(x, zero(T))
    else
        x .*= β
    end
    xr = reshape(x, S.shape..., S.Ncoeff)

    bthreads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        for cmap in S.cmaps
            for s in 1:Nshift
                phase_s = @view S.phases[idx, s]

                # 1) Multiply by phase and coil map
                Threads.@threads for i in 1:S.Ncoeff
                    @views S.kL1[idx, i] .= phase_s .* cmap .* b[idx, i]
                end

                # 2) FFT on non-oversampled grid
                Threads.@threads for i in 1:S.Ncoeff
                    @views S.fftplan * S.kL1[idx, i]
                end

                # 3) Apply packed kernel at masked positions
                kL1_rs = reshape(S.kL1, :, S.Ncoeff)
                kL2_rs = reshape(S.kL2, :, S.Ncoeff)
                Threads.@threads for i in eachindex(kL2_rs)
                    kL2_rs[i] = 0
                end
                @tasks for ik in axes(S.Λ_decomp, 2)
                    kind = S.kmask_indcs[ik]
                    @inbounds for ic1 in 1:S.Ncoeff
                        acc = zero(Complex{real(T)})
                        for ic2 in 1:S.Ncoeff
                            ip = S.ind_lookup[ic1, ic2]
                            if ic1 <= ic2
                                acc += S.Λ_decomp[ip, ik, s] * kL1_rs[kind, ic2]
                            else
                                acc += conj(S.Λ_decomp[ip, ik, s]) * kL1_rs[kind, ic2]
                            end
                        end
                        kL2_rs[kind, ic1] = acc
                    end
                end

                # 4) IFFT on non-oversampled grid
                Threads.@threads for i in 1:S.Ncoeff
                    @views S.ifftplan * S.kL2[idx, i]
                end

                # 5) Accumulate with conjugate phase and coil map
                Threads.@threads for i in 1:S.Ncoeff
                    @views xr[idx, i] .+= α .* conj.(cmap) .* conj.(phase_s) .* S.kL2[idx, i]
                end
            end
        end
    finally
        BLAS.set_num_threads(bthreads)
    end
    return x
end