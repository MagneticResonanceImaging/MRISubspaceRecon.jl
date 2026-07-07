## ##########################################################################
# NFFTNormalOpLowMem - Decomposed Toeplitz trick for reduced memory usage
#############################################################################

"""
    NFFTNormalOpLowmem(img_shape, Λ_decomp, kmask_indcs; cmaps, num_fft_threads)

Low-memory Toeplitz normal operator. Buffers `kL1`, `kL2` are allocated on the
non-oversampled grid `img_shape` (2^D times smaller than the standard operator).
The decomposed kernel `Λ_decomp` has shape `(Ncoeff, Ncoeff, n_kmask_1x, 2^D)`.

Instead of performing a single FFT on a 2×-oversampled grid, this operator
performs 2^D FFTs on the original grid with different linear phase shifts.
The result is mathematically equivalent but uses 2^D less buffer memory.

# References
- Uecker M, et al. "BART toolbox" — lowmem/decomp mode in src/noncart/nufft.c
"""
function NFFTNormalOpLowmem(
    img_shape,
    Λ_decomp::AbstractArray{Tc,4},
    kmask_indcs::Vector{<:Integer};
    cmaps=(1,),
    num_fft_threads=round(Int, Threads.nthreads()/size(Λ_decomp, 1))
    ) where {T, Tc <: Union{T, Complex{T}}}

    Ncoeff = size(Λ_decomp, 1)
    D = length(img_shape)
    Nshift = 2^D
    @assert size(Λ_decomp, 4) == Nshift
    @assert length(kmask_indcs) == size(Λ_decomp, 3)
    @assert all(kmask_indcs .> 0)
    @assert all(kmask_indcs .<= prod(img_shape))

    kL1 = Array{Complex{T}}(undef, img_shape..., Ncoeff)
    kL2 = similar(kL1)

    ktmp = @view kL1[CartesianIndices(img_shape), 1]
    fftplan  = plan_fft!(ktmp; flags=FFTW.MEASURE, num_threads=num_fft_threads)
    ifftplan = plan_ifft!(ktmp; flags=FFTW.MEASURE, num_threads=num_fft_threads)

    phases = _compute_linphases(img_shape, T)

    A = _NFFTNormalOpLowmem(img_shape, Ncoeff, fftplan, ifftplan, Λ_decomp, kmask_indcs, kL1, kL2, cmaps, phases)

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
struct _NFFTNormalOpLowmem{S,E,F,G,H,I,J,K,P}
    shape::S
    Ncoeff::Int
    fftplan::E
    ifftplan::F
    Λ_decomp::G    # (Ncoeff, Ncoeff, n_kmask_1x, 2^D)
    kmask_indcs::H  # indices into the 1x grid
    kL1::I
    kL2::J
    cmaps::K
    phases::P       # (img_shape..., 2^D)
end

#############################################################################
# Kernel decomposition
#############################################################################

"""
    decompose_kernel(img_shape, Λ_os, kmask_indcs_os) -> (Λ_decomp, kmask_indcs_1x)

Decompose a Toeplitz kernel from the 2×-oversampled grid into 2^D sub-kernels
on the non-oversampled grid.

The 2× grid is split by interleaving even/odd indices in each dimension:
each position `p` on the 2× grid maps to `(p ÷ 2, p % 2)`, giving a 1× grid
position and a shift bit.  The resulting `2^D` sub-kernels are applied
independently, each paired with an FFT on the 1× grid and a linear phase ramp
corresponding to the shift.
"""
function decompose_kernel(img_shape, Λ_os::AbstractArray{Tc,3}, kmask_indcs_os::Vector{<:Integer}) where {Tc}
    img_shape_os = 2 .* img_shape
    D = length(img_shape)
    Nshift = 2^D
    Ncoeff = size(Λ_os, 1)

    # The existing kernel Λ_os lives on the 2× oversampled grid in the
    # Fourier domain.  BART's `compute_psf2` computes the PSF on the 2× grid,
    # applies `fftuc` (centered FFT), then uses `md_decompose` to split
    # even/odd indices: psf_decomp[k, s] = psf_2x[2k + bit(s)].
    #
    # Our Λ_os is already in the Fourier domain (it's the kernel that gets
    # multiplied pointwise after FFT).
    
    # 1) Find union of 1x grid positions touched by any OS index
    kmask_1x_set = Set{Int}()
    ci_os_all = CartesianIndices(img_shape_os)
    li_1x_all = LinearIndices(img_shape)
    for ki in kmask_indcs_os
        ci = ci_os_all[ki]
        ci_1x = CartesianIndex(ntuple(d -> (ci[d] - 1) >> 1 + 1, D))
        push!(kmask_1x_set, li_1x_all[ci_1x])
    end
    kmask_indcs_1x = sort!(collect(kmask_1x_set))

    # Reverse map: 1x linear index -> position in kmask_indcs_1x
    pos_lookup = Dict{Int,Int}()
    for (i, ki) in enumerate(kmask_indcs_1x)
        pos_lookup[ki] = i
    end

    # 2) Scatter OS entries into sub-grid bins (pure interleave, no transform)
    n1x = length(kmask_indcs_1x)
    Λ_decomp = zeros(Tc, Ncoeff, Ncoeff, n1x, Nshift)

    for (ios, ki_os) in enumerate(kmask_indcs_os)
        ci = ci_os_all[ki_os]
        shift_bits = 0
        for d in 1:D
            shift_bits |= ((ci[d] - 1) & 1) << (d - 1)
        end
        ci_1x = CartesianIndex(ntuple(d -> (ci[d] - 1) >> 1 + 1, D))
        i_1x = pos_lookup[li_1x_all[ci_1x]]
        s = shift_bits + 1
        @views Λ_decomp[:, :, i_1x, s] .= Λ_os[:, :, ios]
    end

    return Λ_decomp, kmask_indcs_1x
end

#############################################################################
# Linear phase computation
#############################################################################

"""
    _compute_linphases(img_shape, ::Type{T})

Precompute linear phase ramps of size `(img_shape..., 2^D)`.

The relationship between the 2×-oversampled FFT and the 1× FFT is:
    FFT_2N[2k + b] = FFT_N( x[n] · exp(-iπ·b·n/N) )[k]

where `b ∈ {0,1}` for each dimension selects even/odd frequencies.
For multi-dimensional shift index `s` (1-based), bit `d` of `(s-1)` selects
whether dimension `d` uses the odd-frequency phase.

The inverse relationship (crop of IFFT on 2× grid) gives a factor of `1/2^D`
when summing over all shifts. We split this as `1/√2^D` per phase application
(forward and adjoint).

Phase at position `I` for shift `s`:
    (1/√2^D) · exp(-iπ Σ_d b_d · (I[d]-1) / N[d])
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
            # exp(-iπ * θ) = cispi(-θ)
            phases[I, s] = scale * cispi(Tr(-1) * θ)
        end
    end
    return phases
end

#############################################################################
# mul! implementation
#############################################################################

"""
Low-memory `mul!` using the decomposed Toeplitz trick.
For each shift `s ∈ 1:2^D`:
  1. Multiply input by `phase_s` and coil map
  2. FFT on the non-oversampled grid
  3. Apply `Λ_decomp[:,:,:,s]` at masked k-space positions
  4. IFFT on the non-oversampled grid
  5. Multiply by `conj(phase_s)` and `conj(cmap)`, accumulate into output
"""
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

                # 3) Apply decomposed kernel at masked positions
                kL1_rs = reshape(S.kL1, :, S.Ncoeff)
                kL2_rs = reshape(S.kL2, :, S.Ncoeff)
                Threads.@threads for i in eachindex(kL2_rs)
                    kL2_rs[i] = 0
                end
                @tasks for i in axes(S.Λ_decomp, 3)
                    @views @inbounds mul!(kL2_rs[S.kmask_indcs[i], :], S.Λ_decomp[:, :, i, s], kL1_rs[S.kmask_indcs[i], :])
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