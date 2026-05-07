function MRISubspaceRecon.calculate_coil_maps(
    data::CuArray{Complex{T}},
    trj::CuArray{T},
    img_shape::NTuple{N,Int};
    U=CUDA.ones(T, size(trj)[end]),
    sample_mask=CUDA.ones(Bool, size(trj)[2:end]),
    kernel_size=ntuple(_ -> 6, N),
    calib_size=ntuple(i -> nextprod((2, 3, 5), img_shape[i] ÷ (maximum(img_shape) ÷ 32)), length(img_shape)),
    eigThresh_1=0.01,
    eigThresh_2=0.9,
    nmaps=1,
    Niter_cg=100,
    verbose=false) where {N,T}

    @assert all([icalib .== nextprod((2, 3, 5), icalib) for icalib ∈ calib_size]) "calib_size has to be composed of the prime factors 2, 3, and 5 (cf. NonuniformFFTs.jl documentation)."

    trj_calib = trj .* cu(collect(T.(img_shape ./ calib_size))) # scale trj for correct FOV
    mask_calib = reshape(all(abs.(trj_calib) .< 0.5; dims=1), size(trj_calib, 2), :)
    mask_calib .&= sample_mask # update mask to only take calib region of k-space in CoilwiseCG

    x = MRISubspaceRecon.reconstruct_coilwise(data, trj_calib, calib_size; U, sample_mask=mask_calib, Niter_cg)

    imdims = ntuple(i -> i, length(img_shape))
    kbp = fftshift(x, imdims)
    fft!(kbp, imdims)
    kbp = fftshift(kbp, imdims)

    t = @elapsed begin
        cmaps = espirit(Array(kbp), img_shape, kernel_size; eigThresh_1, eigThresh_2, nmaps)
    end
    verbose && println("espirit: $t s")

    cmaps = [cu(cmaps[CartesianIndices(img_shape), ic, in]) for ic ∈ axes(cmaps, length(img_shape) + 1), in = (nmaps == 1 ? 1 : 1:nmaps)]
    return cmaps
end

# wrapper for 4D data arrays with readout lines in separate axis
function MRISubspaceRecon.calculate_coil_maps(
    data::CuArray{Tc,4},
    trj::CuArray{T,4},
    img_shape::NTuple{N,Int};
    sample_mask=CUDA.ones(Bool, size(trj)[2:end]),
    kwargs...) where {N,T,Tc<:Complex}

    data = reshape(data, :, size(data,3), size(data,4))
    trj = reshape(trj, size(trj,1), :, size(trj,4))
    sample_mask = reshape(sample_mask, :, size(sample_mask,3))
    return calculate_coil_maps(data, trj, img_shape; sample_mask, kwargs...)
end

## ##########################################################################
# Cartesian coil maps (Integer trajectory) on GPU
#############################################################################
function MRISubspaceRecon.calculate_coil_maps(
    data::CuArray{Complex{T}},
    trj::CuArray{<:Integer},
    img_shape::NTuple{N,Int};
    U=CUDA.ones(Complex{T}, size(data, 2)),
    sample_mask=CUDA.ones(Bool, size(trj)[2:end]),
    kernel_size=ntuple(_ -> 6, N),
    calib_size=img_shape .÷ (img_shape[1] ÷ 32),
    eigThresh_1=0.01,
    eigThresh_2=0.9,
    nmaps=1,
    Niter_cg=5,
    verbose=false) where {N,T}

    lower_bound = @. Int(ceil((img_shape - calib_size) / 2))
    upper_bound = @. lower_bound + calib_size + 1
    mask_calib = dropdims(all(Array(trj) .> lower_bound; dims=1) .& all(Array(trj) .< upper_bound; dims=1); dims=1)
    mask_calib .&= Array(sample_mask)
    mask_calib = cu(mask_calib)

    x = MRISubspaceRecon.reconstruct_coilwise(data, trj, calib_size; U, sample_mask=mask_calib, Niter_cg)

    imdims = ntuple(i -> i, length(img_shape))
    kbp = fftshift(x, imdims)
    fft!(kbp, imdims)
    kbp = fftshift(kbp, imdims)

    t = @elapsed begin
        cmaps = espirit(Array(kbp), img_shape, kernel_size; eigThresh_1, eigThresh_2, nmaps)
    end
    verbose && println("espirit: $t s")

    cmaps = [cu(cmaps[CartesianIndices(img_shape), ic, in]) for ic in axes(cmaps, length(img_shape) + 1), in = (nmaps == 1 ? 1 : 1:nmaps)]
    return cmaps
end


## ##########################################################################
# Internal helper functions
#############################################################################
function MRISubspaceRecon.reconstruct_coilwise(
    data::CuArray{Tc},
    trj::CuArray{<:Integer,3},
    img_shape;
    U=CUDA.ones(Tc, size(data, 2)),
    sample_mask=CUDA.ones(Bool, size(trj)[2:end]),
    Niter_cg=5,
    verbose=false) where {Tc <: Complex}

    AᴴA = MRISubspaceRecon.FFTNormalOp(img_shape, trj, U[:, 1]; sample_mask)
    xbp = MRISubspaceRecon.calculate_backprojection(data, trj, img_shape; U=U[:, 1], sample_mask)

    Ncoil = size(data, 3)
    x = CUDA.zeros(Tc, img_shape..., Ncoil)

    for icoil in axes(xbp, length(img_shape) + 2)
        bi = vec(@view xbp[CartesianIndices(img_shape), 1, icoil])
        xi = vec(@view x[CartesianIndices(img_shape), icoil])
        cg!(xi, AᴴA, bi; maxiter=Niter_cg, verbose)
    end
    return x
end

function MRISubspaceRecon.reconstruct_coilwise(
    data::CuArray{Tc,3},
    trj::CuArray{T,3},
    img_shape;
    U=CUDA.ones(T, size(trj)[end]),
    sample_mask=CUDA.ones(Bool, size(trj)[2:end]),
    Niter_cg=100,
    verbose=false) where {T <: Real,Tc <: Complex{T}}

    AᴴA = MRISubspaceRecon.NFFTNormalOp(img_shape, trj, U[:, 1]; sample_mask=sample_mask, verbose)
    xbp = MRISubspaceRecon.calculate_backprojection(data, trj, img_shape; U=U[:, 1], sample_mask=sample_mask, verbose)

    Ncoil = size(data, 3)
    x = CUDA.zeros(Tc, img_shape..., Ncoil)

    for icoil = axes(xbp, length(img_shape) + 2)
        bi = vec(@view xbp[CartesianIndices(img_shape), 1, icoil])
        xi = vec(@view x[CartesianIndices(img_shape), icoil])
        cg!(xi, AᴴA, bi; maxiter=Niter_cg, verbose, reltol=0)
    end
    return x
end