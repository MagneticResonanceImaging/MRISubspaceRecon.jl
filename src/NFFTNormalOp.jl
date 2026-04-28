## ##########################################################################
# NFFTNormalOp
#############################################################################

"""
    NFFTNormalOp(img_shape, trj, U; cmaps, sample_mask, verbose, num_fft_threads)
    NFFTNormalOp(img_shape, Λ, kmask_indcs; cmaps)

Create normal operator of NFFT operator.
Differentiate between functions exploiting a pre-calculated Toeplitz kernel basis `Λ` and the function which calculates Λ based on a passed trajectory `trj`.
When the basis functions `U` are real-valued, a real-only NUFFT is used to compute `Λ`, reducing the data volume and computation time for the spreading and interpolation steps by half.

# Arguments
- `img_shape::Tuple{Int}`: Image dimensions
- `trj::AbstractArray`: Trajectory, use `CuArray` as input type to use CUDA code.
- `U::AbstractMatrix`: Basis coefficients of subspace
- `cmaps::AbstractVector{Matrix}=(1,)`: Coil sensitivities, use `AbstractVector{CuArray}` as type for use with CUDA code.
- `sample_mask::AbstractArray{Bool} = trues(size(trj)[2:end])`: Mask indicating which acquired k-space samples are included in the reconstruction
- `verbose::Boolean`=`false`: Verbose level
- `num_fft_threads::Int`=`round(Int, Threads.nthreads()/size(U, 2))` or `round(Int, Threads.nthreads()/size(Λ, 1))`: Number of threads for FFT
- `Λ`: Toeplitz kernel basis resulting from internal `calculate_kernel_noncartesian` method. Using a real basis `U` reduces its size by a factor of 2.

# References
1. Wajer FTAW, and Pruessmann, KP. “Major Speedup of Reconstruction for Sensitivity Encoding with Arbitrary Trajectories”. In: Proc. Intl. Soc. Mag. Reson. Med 9 (2001).
2. Fessler JA, et al. "Toeplitz-Based Iterative Image Reconstruction for MRI With Correction for Magnetic Field Inhomogeneity". IEEE Trans. Signal Process., 53.9 (2006).
3. Mani M, et al. “Fast iterative algorithm for the reconstruction of multishot non-cartesian diffusion data”. Magn Reson Med. 74.4 (2015), pp. 1086–1094. https://doi.org/10.1002/mrm.25486
4. Uecker M, Zhang S, and Frahm J. “Nonlinear inverse reconstruction for real-time MRI of the human heart using undersampled radial FLASH". Magn Res Med. 63 (2010), pp. 1456–1462. https://doi.org/10.1002/mrm.22453
"""
function NFFTNormalOp(
    img_shape,
    trj::AbstractArray{T,3},
    U::AbstractArray{Tc};
    cmaps=(1,),
    sample_mask=trues(size(trj)[2:end]),
    verbose=false,
    num_fft_threads=round(Int, Threads.nthreads()/size(U, 2)),
    ) where {T <: Real, Tc <: Union{T, Complex{T}}}

    Λ, kmask_indcs = calculate_kernel_noncartesian(2 .* img_shape, trj, U; sample_mask, verbose)

    return NFFTNormalOp(img_shape, Λ, kmask_indcs; cmaps=cmaps, num_fft_threads=num_fft_threads)
end

# Wrapper for 4D data arrays
function NFFTNormalOp(img_shape, trj::AbstractArray{T,4}, U::AbstractArray{Tc,2}; sample_mask=trues(size(trj)[2:end]), kwargs...) where {T, Tc <: Union{T, Complex{T}}}
    trj = reshape(trj, size(trj, 1), :, size(trj,4))
    sample_mask = reshape(sample_mask, :, size(sample_mask,3))
    return NFFTNormalOp(img_shape, trj, U; sample_mask, kwargs...)
end

function NFFTNormalOp(
    img_shape,
    Λ::AbstractArray{Tc,3},
    kmask_indcs::Vector{<:Integer};
    cmaps=(1,),
    num_fft_threads=round(Int, Threads.nthreads()/size(Λ, 1))
    ) where {T, Tc <:Union{T, Complex{T}}}

    @assert length(kmask_indcs) == size(Λ,3) # ensure that kmask is not out of bound as we use `@inbounds` in `mul!`
    @assert all(kmask_indcs .> 0)
    @assert all(kmask_indcs .<= prod(2 .* img_shape))
    eltype(Λ) <: Real && @warn "The Toeplitz kernel is real-valued; using a complex kernel greatly improves performance when applying the normal operator on CPUs." maxlog=1

    Ncoeff = size(Λ, 1)
    img_shape_os = 2 .* img_shape
    kL1 = Array{Complex{T}}(undef, img_shape_os..., Ncoeff)
    kL2 = similar(kL1)

    ktmp = @view kL1[CartesianIndices(img_shape_os),1]

    fftplan  = plan_fft!(ktmp; flags = FFTW.MEASURE, num_threads=num_fft_threads)
    ifftplan = plan_ifft!(ktmp; flags = FFTW.MEASURE, num_threads=num_fft_threads)

    A = _NFFTNormalOp(img_shape, Ncoeff, fftplan, ifftplan, Λ, kmask_indcs, kL1, kL2, cmaps, nothing, nothing, nothing)

	return LinearOperator(
        Complex{T},
        prod(A.shape) * A.Ncoeff,
        prod(A.shape) * A.Ncoeff,
        true,
        true,
        (res, x, α, β) -> mul!(res, A, x, α, β),
        nothing,
        (res, x, α, β) -> mul!(res, A, x, α, β),
    )
end

#############################################################################
# Internal use
#############################################################################
struct _NFFTNormalOp{S,E,F,G,H,I,J,K,L,M,N}
    shape::S
    Ncoeff::Int
    fftplan::E
    ifftplan::F
    Λ::G
    kmask_indcs::H
    kL1::I
    kL2::J
    cmaps::K
    ind_lookup::L
    threads::M
    blocks::N
end

function calculate_kmask_indcs(img_shape_os, trj; sample_mask=trues(size(trj)[2:end]))
    @assert all([i .== nextprod((2, 3, 5), i) for i ∈ img_shape_os]) "img_shape_os has to be composed of the prime factors 2, 3, and 5 (cf. NonuniformFFTs.jl documentation)."

    T = eltype(trj)
    backend = CPU()
    p = PlanNUFFT(Complex{T}, img_shape_os; σ=1, kernel=GaussianKernel(), backend=backend) # default is without fftshift
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj[:, sample_mask]))

    S = ones(Complex{T}, size(p.points[1]))
    NonuniformFFTs.spread_from_points!(p.backend, NUFFTCallbacks().nonuniform, p.point_transform_fold, p.blocks, p.kernels, p.kernel_evalmode, p.data.us, p.points, (S,))
    kmask_indcs = findall(vec(p.data.us[1] .!= 0))
    return kmask_indcs
end

# Calculation for complex-valued basis U
function calculate_kernel_noncartesian(img_shape_os, trj::AbstractArray{T,3}, U::AbstractArray{Tc}; sample_mask=trues(size(trj)[2:end]), verbose=false) where {T, Tc <: Complex{T}}
    kmask_indcs = calculate_kmask_indcs(img_shape_os, trj; sample_mask)
    @assert all(kmask_indcs .> 0) # ensure that kmask is not out of bound
    @assert all(kmask_indcs .<= prod(img_shape_os))

    # count the number of samples per time frame using the mask
    nsamp_t = vec(sum(sample_mask; dims=1))
    @assert sum(nsamp_t) > 0 "Mask removes all samples, cannot compute kernel."

    cumsum_nsamp = cumsum(nsamp_t) .+ 1
    prepend!(cumsum_nsamp, 1)

    λ  = Array{Complex{T}}(undef, img_shape_os)
    λ2 = similar(λ)

    Ncoeff = size(U, 2)
    Λ = Array{Complex{T}}(undef, Ncoeff, Ncoeff, length(kmask_indcs))
    S = Vector{Complex{T}}(undef, sum(nsamp_t))

    # Prep FFT and NUFFT plans
    fftplan  = plan_fft(λ; flags=FFTW.MEASURE, num_threads=Threads.nthreads())
    nfftplan = PlanNUFFT(Complex{T}, img_shape_os) # default is without fftshift
    set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(trj[:, sample_mask])) # transform matrix to tuples, change sign of FT exponent, change range to (0,2π)

    # Evaluating only the upper triangular matrix assumes that the PSF from the rightmost voxel to the leftmost voxel is the adjoint of the PSF in the opposite direction.
    # For the outmost voxel, this is not correct, but the resulting images are virtually identical in our test cases.
    # To avoid this error, remove the `if ic2 >= ic1` and the `Λ[ic2,ic1,it] = conj.(λ[kmask_indcs[it]])` statements at the cost of computation time.
    verbose && println("calculating non-Cartesian kernel...")
    t = @elapsed for ic2 ∈ axes(Λ, 2), ic1 ∈ axes(Λ, 1)
        if ic2 >= ic1 # eval. only upper triangular matrix
            @simd for it ∈ axes(U,1)
                idx1 = cumsum_nsamp[it]
                idx2 = cumsum_nsamp[it + 1] - 1
                @inbounds S[idx1:idx2] .= conj(U[it,ic1]) * U[it,ic2]
            end

            NonuniformFFTs.exec_type1!(λ2, nfftplan, vec(S)) # type 1: non-uniform points to uniform grid
            mul!(λ, fftplan, λ2)

            Threads.@threads for it ∈ eachindex(kmask_indcs)
                @inbounds Λ[ic2,ic1,it] = conj.(λ[kmask_indcs[it]])
                @inbounds Λ[ic1,ic2,it] =       λ[kmask_indcs[it]]
            end
        end
    end
    verbose && println("time to compute kernel: t = $t s")
    return Λ, kmask_indcs
end

# Kernel is assumed to be real-valued for faster computation, highly accurate if U is a set of real basis functions
function calculate_kernel_noncartesian(img_shape_os, trj::AbstractArray, U::AbstractArray{T}; sample_mask=trues(size(trj)[2:end]), verbose=false) where {T <: Real}
    kmask_indcs = calculate_kmask_indcs(img_shape_os, trj; sample_mask)
    @assert all(kmask_indcs .> 0) # ensure that kmask is not out of bound
    @assert all(kmask_indcs .<= prod(img_shape_os))

    # count the number of samples per time frame using the mask
    nsamp_t = vec(sum(sample_mask; dims=1))
    @assert sum(nsamp_t) > 0 "Mask removes all samples, cannot compute kernel."

    cumsum_nsamp = cumsum(nsamp_t) .+ 1
    prepend!(cumsum_nsamp, 1)

    λ  = Array{T}(undef, img_shape_os)
    λ2 = Array{Complex{T}}(undef, img_shape_os[1] ÷ 2 + 1, Base.tail(img_shape_os)...)

    Ncoeff = size(U, 2)

    # Complex kernel because mul! is 2.5 faster as complex * complex than real * complex
    # Set Λ to type Array{T} for equivalent results with conserved memory, but at the cost of computation time
    Λ = Array{Complex{T}}(undef, Ncoeff, Ncoeff, length(kmask_indcs)) 
    S = Array{T}(undef, sum(nsamp_t))

    # Prep FFT and NUFFT plans specific to real non-uniform data
    # Use brfft (and conjugate λ2) because an rfft with a complex input does not exist in FFTW
    # That is, a forward transform with Hermitian input that outputs only real values is not defined
    brfftplan = plan_brfft(λ2, img_shape_os[1]; flags=FFTW.MEASURE, num_threads=Threads.nthreads())
    nfftplan = PlanNUFFT(T, img_shape_os)
    set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(trj[:, sample_mask]))

    # Evaluating only the upper triangular matrix assumes that the PSF from the rightmost voxel to the leftmost voxel is the adjoint of the PSF in the opposite direction.
    # For the outmost voxel, this is not correct, but the resulting images are virtually identical in our test cases.
    verbose && println("calculating non-Cartesian kernel...")
    t = @elapsed for ic2 ∈ axes(Λ, 2), ic1 ∈ axes(Λ, 1)
        if ic2 >= ic1 # eval. only upper triangular matrix
            @simd for it ∈ axes(U,1)
                idx1 = cumsum_nsamp[it]
                idx2 = cumsum_nsamp[it + 1] - 1
                @inbounds S[idx1:idx2] .= U[it,ic1] * U[it,ic2]
            end

            NonuniformFFTs.exec_type1!(λ2, nfftplan, vec(S))
            λ2 .= conj.(λ2) # conjugate input to flip the sign of the exponential in brfft
            mul!(λ, brfftplan, λ2)

            Threads.@threads for it ∈ eachindex(kmask_indcs)
                @inbounds Λ[ic2,ic1,it] = λ[kmask_indcs[it]]
                @inbounds Λ[ic1,ic2,it] = λ[kmask_indcs[it]]
            end
        end
    end
    verbose && println("time to compute kernel: t = $t s")
    return Λ, kmask_indcs
end

function LinearAlgebra.mul!(x::AbstractVector{T}, S::_NFFTNormalOp, b, α, β) where {T}
    idx = CartesianIndices(S.shape)
    idxos = CartesianIndices(2 .* S.shape)

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
            Threads.@threads for i ∈ 1:S.Ncoeff
                S.kL1[idxos, i] .= zero(T)
                @views S.kL1[idx, i] .= cmap .* b[idx, i]
                @views S.fftplan * S.kL1[idxos, i]
            end

            kL1_rs = reshape(S.kL1, :, S.Ncoeff)
            kL2_rs = reshape(S.kL2, :, S.Ncoeff)
            Threads.@threads for i in eachindex(kL2_rs)
                kL2_rs[i] = 0
            end
            @tasks for i ∈ axes(S.Λ, 3) # @tasks is 2x faster, because @threads conflicts with multithreading scheduler in LinearOperatorCollection
                @views @inbounds mul!(kL2_rs[S.kmask_indcs[i], :], S.Λ[:, :, i], kL1_rs[S.kmask_indcs[i], :])
            end

            Threads.@threads for i ∈ 1:S.Ncoeff
                @views S.ifftplan * S.kL2[idxos, i]
                @views xr[idx,i] .+= α .* conj.(cmap) .* S.kL2[idx,i]
            end
        end
    finally
        BLAS.set_num_threads(bthreads)
    end
    return x
end