function MRISubspaceRecon.calculate_backprojection(data::CuArray{Tc,3}, trj::CuArray{T,3}, img_shape; U=cu(I(size(trj, 3))), sample_mask=CUDA.ones(Bool, size(trj)[2:end]), density_compensation=:none, verbose=false) where {T<:Real,Tc<:Complex{T}}
    nsamp_t = dropdims(sum(sample_mask; dims=1); dims=1) # number of samples per time frame
    Ncoef = size(U, 2)
    Ncoil = size(data, 3)
    Uc = conj(U)

    # Compute the last index per time frame, and insert zero at position 1
    cumsum_nsamp = CUDA.zeros(eltype(nsamp_t), size(nsamp_t))
    cumsum_nsamp[2:end] = cumsum(nsamp_t[1:end-1]) # cumulative sum indicates to which time frame a sample belongs

    p = PlanNUFFT(Complex{T}, img_shape; fftshift=true, backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size=Val(200))
    trj_rs = trj[:, sample_mask] # select subset of trj
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj_rs)) # transform matrix to tuples, change sign of FT exponent, change range to (0,2π)

    img_idx = CartesianIndices(img_shape)
    xbp = CUDA.zeros(Tc, img_shape..., Ncoef, Ncoil)
    data_temp = CuArray{Tc}(undef, sum(nsamp_t))

    # Apply sampling mask to data and perform backprojection
    threads, blocks = default_launch_config(nsamp_t)
    verbose && println("calculating backprojection...")
    flush(stdout)
    t = @elapsed for icoil ∈ axes(data, 3)
        for icoef ∈ axes(U, 2)
            copyto!(data_temp, @view data[sample_mask, icoil])
            @cuda threads=threads blocks=blocks multiply_data_with_basis!(data_temp, Uc, nsamp_t, cumsum_nsamp, icoef)
            MRISubspaceRecon.apply_density_compensation!(data_temp, trj_rs; density_compensation)
            @views NonuniformFFTs.exec_type1!(xbp[img_idx, icoef, icoil], p, data_temp) # type 1: non-uniform points to uniform grid
        end
    end
    verbose && println("time to compute backprojection: t = $t s")
    flush(stdout)
    return xbp
end

function MRISubspaceRecon.calculate_backprojection(data::CuArray{Tc,3}, trj::CuArray{T,3}, cmaps::AbstractVector{<:CuArray{Tc,N}}; U=cu(I(size(trj, 3))), sample_mask=CUDA.ones(Bool, size(trj)[2:end]), density_compensation=:none, verbose=false) where {T<:Real,Tc<:Complex{T},N}
    nsamp_t = dropdims(sum(sample_mask; dims=1); dims=1) # number of samples per time frame
    Ncoef = size(U, 2)
    img_shape = size(cmaps[1])
    Uc = conj(U)

    # Compute the last index per time frame, and insert zero at position 1
    cumsum_nsamp = CUDA.zeros(eltype(nsamp_t), size(nsamp_t))
    cumsum_nsamp[2:end] = cumsum(nsamp_t[1:end-1]) # cumulative sum indicates to which time frame a sample belongs

    p = PlanNUFFT(Complex{T}, img_shape; fftshift=true, backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size=Val(200))
    trj_rs = trj[:, sample_mask]
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj_rs))

    img_idx = CartesianIndices(img_shape)
    xbp = CUDA.zeros(Tc, img_shape..., Ncoef)
    xtmp = CuArray{Tc}(undef, img_shape)
    data_temp = CuArray{Tc}(undef, sum(nsamp_t))

    # Apply sampling sample_mask to data and perform backprojection
    threads, blocks = default_launch_config(nsamp_t)
    verbose && println("calculating backprojection...")
    flush(stdout)
    t = @elapsed for icoil ∈ eachindex(cmaps)
        for icoef ∈ axes(U, 2)
            copyto!(data_temp, @view data[sample_mask, icoil])
            @cuda threads=threads blocks=blocks multiply_data_with_basis!(data_temp, Uc, nsamp_t, cumsum_nsamp, icoef)
            MRISubspaceRecon.apply_density_compensation!(data_temp, trj_rs; density_compensation)
            NonuniformFFTs.exec_type1!(xtmp, p, data_temp) # type 1: non-uniform points to uniform grid
            xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* xtmp
        end
    end
    verbose && println("time to compute backprojection: t = $t s")
    flush(stdout)
    return xbp
end

# Wrapper for use with 4D arrays, where nr of ADC samples per readout is in a separate at 2ⁿᵈ dim
function MRISubspaceRecon.calculate_backprojection(data::CuArray{Tc,4}, trj::CuArray{T,4}, arg3; sample_mask=CUDA.ones(Bool, size(trj)[2:end]), kwargs...) where {T,Tc<:Complex}
    data = reshape(data, :, size(data, 3), size(data, 4))
    trj = reshape(trj, size(trj, 1), :, size(trj, 4))
    sample_mask = reshape(sample_mask, :, size(sample_mask, 3))
    return MRISubspaceRecon.calculate_backprojection(data, trj, arg3; sample_mask, kwargs...)
end


## ##########################################################################
# Internal helper functions
#############################################################################
function multiply_data_with_basis!(data_temp, Uc, nsamp_t, cumsum_nsamp, icoef)
    ik = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    it = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    # Grid-stride loop over time frames to handle cases where the number of
    # time frames exceeds the maximum number of blocks in the y-dimension
    stride_y = gridDim().y * blockDim().y
    while it <= length(nsamp_t)
        if ik <= nsamp_t[it]
            ik_abs = cumsum_nsamp[it] + ik # absolute sample index
            data_temp[ik_abs] *= Uc[it, icoef]
        end
        it += stride_y
    end
    return
end

# Default threading for back projection
function default_launch_config(nsamp_t)
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    max_blocks_y = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)

    threads_x = min(max_threads, maximum(nsamp_t))
    threads_y = min(max_threads ÷ threads_x, length(nsamp_t))
    threads = (threads_x, threads_y)

    blocks_x = ceil(Int, maximum(nsamp_t) / threads_x)
    blocks_y = min(ceil(Int, length(nsamp_t) / threads_y), max_blocks_y)
    blocks = (blocks_x, blocks_y)

    return threads, blocks
end