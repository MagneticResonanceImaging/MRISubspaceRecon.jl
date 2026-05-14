"""
    calculate_coil_maps(data, trj, img_shape; U, density_compensation, kernel_size, calib_size, eigThresh_1, eigThresh_2, nmaps, verbose)

Estimate coil sensitivity maps using ESPIRiT [1].

# Arguments
- `data::AbstractArray`: Complex dataset with axes (samples, time frames, channels). Time frames are reconstructed using the subspace defined in U. Use `CuArray` as input type to use CUDA code.
- `trj::AbstractArray`: Trajectory with sample coordinates corresponding to the dataset Use `CuArray` as input type to use GPU code.
- `img_shape::NTuple{N,Int}`: Shape of output image

# Keyword Arguments
- `U::Matrix`=`ones(size(trj)[end])` : `I(1)`: Basis coefficients of subspace
- `sample_mask::AbstractArray{Bool}`=`trues(size(trj)[2:end])`: Mask indicating which acquired k-space samples are included in the reconstruction
- `density_compensation`=`:radial_3D`: Values of `:radial_3D`, `:radial_2D`, `:none`, or of type  `AbstractArray`
- `kernel_size`=`ntuple(_ -> 6, N)`: Kernel size
- `calib_size`=`ntuple(_ -> 24, N)`: Size of calibration region
- `eigThresh_1`=0.01: Threshold of first eigenvalue
- `eigThresh_2`=0.9: Threshold of second eigenvalue
- `nmaps`=1: Number of estimated maps
- `Niter_cg`: Number of CG iterations used for the reconstruction. The default is 100 and 5 for non-Cartesian and Cartesian trajectories, respectively. Using a number of iterations too large for a given dataset can introduce errors.
- `verbose::Boolean`=`false`: Verbosity level

# Return
- `cmaps::Vector{Array}`: Coil sensitivities as `Vector` of arrays

# References
1. Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, and Lustig M. "ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: Where SENSE meets GRAPPA". Magn. Reson. Med. 71 (2014), pp. 990-1001. https://doi.org/10.1002/mrm.24751
"""
function calculate_coil_maps(
    data::AbstractArray{Complex{T}},
    trj::AbstractArray{T},
    img_shape::NTuple{N,Int};
    U=ones(Complex{T}, size(trj)[end]),
    sample_mask=trues(size(trj)[2:end]),
    kernel_size=ntuple(_ -> 6, N),
    calib_size=ntuple(i -> nextprod((2, 3, 5), img_shape[i] ÷ (maximum(img_shape) ÷ 32)), length(img_shape)),
    eigThresh_1=0.01,
    eigThresh_2=0.9,
    nmaps=1,
    Niter_cg=100,
    verbose=false) where {N,T}
    @assert all([icalib .== nextprod((2, 3, 5), icalib) for icalib ∈ calib_size]) "calib_size has to be composed of the prime factors 2, 3, and 5 (cf. NonuniformFFTs.jl documentation)."

    trj_calib = trj .* T.(img_shape ./ calib_size) # scale trj for correct image dims
    mask_calib = reshape(all(abs.(trj_calib) .< 0.5; dims=1), size(trj_calib, 2), :)
    mask_calib .&= sample_mask # new mask only takes data within calib region

    x = reconstruct_coilwise(data, trj_calib, calib_size; U, sample_mask=mask_calib, Niter_cg)

    imdims = ntuple(i -> i, length(img_shape))
    kbp = fftshift(x, imdims)
    fft!(kbp, imdims)
    kbp = fftshift(kbp, imdims)

    t = @elapsed begin
        cmaps = espirit(kbp, img_shape, kernel_size; eigThresh_1, eigThresh_2, nmaps)
    end
    verbose && println("espirit: $t s")

    cmaps = [cmaps[CartesianIndices(img_shape), ic, in] for ic ∈ axes(cmaps, length(img_shape) + 1), in = (nmaps == 1 ? 1 : 1:nmaps)]
    return cmaps
end

function calculate_coil_maps(
    data::AbstractArray{Complex{T}},
    trj::AbstractArray{<:Integer},
    img_shape::NTuple{N,Int};
    U=ones(Complex{T}, size(data, 2)),
    sample_mask=trues(size(trj)[2:end]),
    kernel_size=ntuple(_ -> 6, N),
    calib_size=img_shape .÷ (img_shape[1] ÷ 32),
    eigThresh_1=0.01,
    eigThresh_2=0.9,
    nmaps=1,
    Niter_cg=5,
    verbose=false) where {N,T}

    lower_bound = @. Int(ceil((img_shape - calib_size) / 2))
    upper_bound = @. lower_bound + calib_size + 1
    mask_calib = dropdims(all(trj .> lower_bound; dims=1) .& all(trj .< upper_bound; dims=1); dims=1)
    mask_calib .&= sample_mask

    # Shift trajectory to 1:calib_size range for reconstruction on the calib grid
    trj_calib = Int16.(trj .- lower_bound)
    x = reconstruct_coilwise(data, trj_calib, calib_size; U, sample_mask=mask_calib, Niter_cg)

    imdims = ntuple(i -> i, length(img_shape))
    kbp = fftshift(x, imdims)
    fft!(kbp, imdims)
    kbp = fftshift(kbp, imdims)

    t = @elapsed begin
        cmaps = espirit(kbp, img_shape, kernel_size; eigThresh_1, eigThresh_2, nmaps)
    end
    verbose && println("espirit: $t s")

    cmaps = [cmaps[CartesianIndices(img_shape), ic, in] for ic ∈ axes(cmaps, length(img_shape) + 1), in = (nmaps == 1 ? 1 : 1:nmaps)]
    return cmaps
end

# wrappers for use with 4D arrays where the nr of ADC samples per readout is within a separate 2ⁿᵈ axis
function calculate_coil_maps(
    data::AbstractArray{Tc,4},
    trj::AbstractArray{T,4},
    img_shape::NTuple{N,Int};
    sample_mask=trues(size(trj)[2:end]),
    kwargs...) where {T, Tc <: Complex, N}

    data = reshape(data, :, size(data,3), size(data,4))
    trj = reshape(trj, size(trj, 1), :, size(trj,4))
    sample_mask = reshape(sample_mask, :, size(sample_mask,3))
    return calculate_coil_maps(data, trj, img_shape; kwargs..., sample_mask)
end


## ##########################################################################
# Internal helper functions
#############################################################################
function reconstruct_coilwise(
    data::AbstractArray{Tc,3},
    trj::AbstractArray{T,3},
    img_shape;
    U=I(size(trj)[end]),
    sample_mask=trues(size(trj)[2:end]),
    Niter_cg=100,
    verbose=false) where {T <: Real, Tc <: Complex{T}}

    AᴴA = NFFTNormalOp(img_shape, trj, U[:, 1]; sample_mask, verbose)
    xbp = calculate_backprojection(data, trj, img_shape; U=U[:, 1], sample_mask, verbose)

    Ncoil = size(data, 3)
    x = zeros(Tc, img_shape..., Ncoil)

    for icoil ∈ axes(xbp, length(img_shape) + 2)
        bi = vec(@view xbp[CartesianIndices(img_shape), 1, icoil])
        xi = vec(@view x[CartesianIndices(img_shape), icoil])
        cg!(xi, AᴴA, bi; maxiter=Niter_cg, verbose, reltol=0)
    end
    return x
end

function reconstruct_coilwise(
    data::AbstractArray{Tc,3},
    trj::AbstractArray{<:Integer,3},
    img_shape;
    U=I(size(trj)[end]),
    sample_mask=trues(size(trj)[2:end]),
    Niter_cg=5,
    verbose=false) where {Tc <: Complex}

    AᴴA = FFTNormalOp(img_shape, trj, U[:, 1]; sample_mask)
    xbp = calculate_backprojection(data, trj, img_shape; U=U[:, 1], sample_mask)

    Ncoil = size(data, 3)
    x = zeros(Tc, img_shape..., Ncoil)

    for icoil ∈ axes(xbp, length(img_shape) + 2)
        bi = vec(@view xbp[CartesianIndices(img_shape), 1, icoil])
        xi = vec(@view x[CartesianIndices(img_shape), icoil])
        cg!(xi, AᴴA, bi; maxiter=Niter_cg, verbose) # maxiter <8 to avoid diverging
    end
    return x
end