## ##########################################################################
# Helpers for memory-efficient handling of `sample_mask`
#
# On the GPU, logical indexing `A[mask]` is *not* lazy: CUDA.jl defines
# `Base.to_index(::CuArray, ::AbstractArray{Bool}) = findall(I)`, so every
# `A[mask]` materialises an index array. For a 2-D mask that array has element
# type `CartesianIndex{2}` (16 B/sample), and `findall` additionally allocates a
# `cumsum` temporary of `8 * length(mask)` bytes. Repeating this inside a loop,
# or in several functions that each re-derive the same indices, can add tens of
# GiB of avoidable traffic for large trajectories.
#
# The helpers below therefore
#   * compute the indices **once** and reuse them,
#   * store them as linear `Int` indices (8 B) rather than `CartesianIndex{2}`,
#   * and skip masking entirely (`nothing`) when the mask selects all samples,
#     in which case a plain `reshape` is zero-copy.
#############################################################################

"""
    _sample_indices(sample_mask) -> Union{Nothing, CuVector{Int}}

Linear indices of the selected samples, or `nothing` if *all* samples are
selected. `nothing` acts as a fast path that avoids materialising any index
array; downstream helpers then operate on zero-copy reshapes.
"""
_sample_indices(sample_mask::AbstractArray{Bool}) =
    all(sample_mask) ? nothing : findall(vec(sample_mask))

# Allow callers to pass indices through unchanged (e.g. when they were computed
# once and are reused by several kernels).
_sample_indices(idx::Union{Nothing,AbstractVector{<:Integer}}) = idx

"""
    _flatten_samples(A)

Reshape an array whose leading axes enumerate samples and whose trailing axis
enumerates something else (coils, coefficients) into a matrix
`(sample, trailing)`. Zero-copy for `CuArray`s and for views that remain
strided.
"""
_flatten_samples(A::AbstractArray) = reshape(A, :, size(A)[end])

"""
    _gather_points(trj, idx) -> (Ndim, Nsamples) matrix

Trajectory coordinates of the selected samples. With `idx === nothing` this is a
zero-copy `reshape`; otherwise the selected columns are gathered once.
"""
_gather_points(trj::AbstractArray, ::Nothing) = reshape(trj, size(trj, 1), :)
_gather_points(trj::AbstractArray, idx::AbstractVector{<:Integer}) =
    reshape(trj, size(trj, 1), :)[:, idx]

"""
    _gather_samples!(buf, data_flat, idx, itrailing) -> buf

Copy the selected samples of column `itrailing` of `data_flat` into `buf`.
`data_flat` is expected to come from [`_flatten_samples`](@ref).
"""
function _gather_samples!(buf, data_flat, ::Nothing, itrailing)
    copyto!(buf, @view data_flat[:, itrailing])
    return buf
end

function _gather_samples!(buf, data_flat, idx::AbstractVector{<:Integer}, itrailing)
    @views buf .= data_flat[idx, itrailing]
    return buf
end

"""
    _nsamples_per_frame(sample_mask, idx)

Number of selected samples per time frame, as a `CuVector`. Uses a cheap
closed form when no sample is masked out.
"""
_nsamples_per_frame(sample_mask, ::Nothing) =
    CUDA.fill(size(sample_mask, 1), size(sample_mask)[end])
_nsamples_per_frame(sample_mask, ::AbstractVector{<:Integer}) =
    vec(sum(sample_mask; dims=1))
