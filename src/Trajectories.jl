"""
    traj_cartesian(T, Nx, Ny, Nz, Nt)

Generate a 3D Cartesian trajectory.

# Arguments
- `Nx::Int`: Number of readout samples
- `Ny::Int`: Number of phase encoding lines
- `Nz::Int`: Number of phase encoding lines (third dimension)
- `Nt::Int`: Number of time steps in the trajectory

# Optional Keyword Argument
- `T::Type=Int`: Type of output trajectory. If `T <: Float`, trajectory is defined ∈ (-0.5, 0.5). If `T <: Int`, trajectory consists of values ∈ (1, N) instead.
"""
function traj_cartesian(Nx, Ny, Nz, Nt; T=Int)
    kx = T <: Integer ? collect(1:Nx) : collect(-Nx/2:Nx/2-1) / Nx
    ky = T <: Integer ? collect(1:Ny) : collect(-Ny/2:Ny/2-1) / Ny
    kz = T <: Integer ? collect(1:Nz) : collect(-Nz/2:Nz/2-1) / Nz

    k = Array{T}(undef, 3, Nx * Ny * Nz, Nt)
    for it ∈ axes(k, 3)
        ki = Array{T,4}(undef, 3, Nx, Ny, Nz)
        Threads.@threads for z ∈ 1:Nz
            for y ∈ 1:Ny, x ∈ 1:Nx
                ki[1, x, y, z] = kx[x]
                ki[2, x, y, z] = ky[y]
                ki[3, x, y, z] = kz[z]
            end
        end
        k[:, :, it] = reshape(ki, 3, :)
    end
    return k
end

"""
    traj_kooshball_goldenratio(Nr, Ncyc, Nt; theta_rot, phi_rot, delay, adc_dim)

Function to calculate a 3D radial kooshball trajectory with a golden-means angular spacing of k-space readouts [1].

# Arguments
- `Nr::Int`: Number of read out samples
- `Ncyc::Int`: Number of cycles
- `Nt::Int`: Number of time steps in the trajectory
- `theta_rot::Float` = 0: Fixed rotation angle along theta
- `phi_rot::Float` = 0: Fixed rotation angle along phi
- `delay::Tuple{Float, Float, Float}`= `(0, 0, 0)`: Gradient delays in (HF, AP, LR)
- `adc_dim::Bool` = `true`: Place ADC samples in a separate axis from each time frame for compatibility with wrapper functions.

# References
1. Chan RW, Ramsay EA, Cunningham CH, and Plewes DB. "Temporal stability of adaptive 3D radial MRI using multidimensional golden means". Magn. Reson. Med. 61 (2009) pp. 354-363. https://doi.org/10.1002/mrm.21837
"""
function traj_kooshball_goldenratio(Nr, Ncyc, Nt; theta_rot=0, phi_rot=0, delay=(0, 0, 0), adc_dim=true)
    gm1, gm2 = calculate_golden_means()
    theta = acos.(mod.((0:(Ncyc*Nt-1)) * gm1, 1))
    phi = (0:(Ncyc*Nt-1)) * 2π * gm2

    theta = reshape(theta, Nt, Ncyc)
    phi = reshape(phi, Nt, Ncyc)

    return traj_kooshball(Nr, theta', phi'; theta_rot=theta_rot, phi_rot=phi_rot, delay=delay, adc_dim=adc_dim)
end

"""
    traj_2d_radial_goldenratio(Nr, Ncyc, Nt; theta_rot, phi_rot, delay, N, adc_dim)

Function to calculate a 2D radial trajectory with golden-angle spacing between subsequent readouts [1].
The use of tiny golden angles [2] is supported by modifying `N`.

# Arguments
- `Nr::Int`: Number of read out samples
- `Ncyc::Int`: Number of cycles
- `Nt::Int`: Number of time steps in the trajectory
- `theta_rot::Float` = 0: Fixed rotation angle along theta
- `phi_rot::Float` = 0: Fixed rotation angle along phi
- `delay::Tuple{Float, Float, Float}` = `(0, 0, 0)`: Gradient delays in (HF, AP, LR)
- `N::Int` = 1: Number of tiny golden angle
- `adc_dim::Bool` = `true`: Place ADC samples in a separate axis from each time frame for compatibility with wrapper functions.

# References
1. Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O. "An optimal radial profile order based on the Golden Ratio for time-resolved MRI". IEEE TMI 26:68-76 (2007)
2. Wundrak S, Paul J, Ulrici J, Hell E, Geibel MA, Bernhardt P, Rottbauer W, Rasche V. "Golden ratio sparse MRI using tiny golden angles". Magn. Reson. Med. 75:2372-2378 (2016)
"""
function traj_2d_radial_goldenratio(Nr, Ncyc, Nt; theta_rot=0, phi_rot=0, delay=(0, 0, 0), N=1, T=Float32, adc_dim=true)
    τ = (sqrt(5) + 1) / 2
    angle_GR = T.(π / (τ + N - 1))
    phi = (0:(Ncyc*Nt-1)) .* angle_GR
    phi = reshape(phi, Nt, Ncyc)

    theta = similar(phi)
    theta .= π / 2 # 2D

    k = traj_kooshball(Nr, theta', phi'; theta_rot=theta_rot, phi_rot=phi_rot, delay=delay, adc_dim=adc_dim)
    k = k[1:2, :, :] # remove 3rd dimension
    return k
end

"""
    traj_kooshball(Nr, theta, phi; theta_rot, phi_rot, delay, adc_dim)

Function to calculate a 3D radial kooshball trajectory with custom sets of projection angles.

# Arguments
- `Nr::Int`: Number of read out samples
- `theta::Array{Float,2}`: Array with dimensions: `Ncyc, Nt` defining the angles `theta` for each cycle and time step.
- `phi::Array{Float,2}`: Array with dimensions: `Ncyc, Nt` defining the angles `phi` for each cycle and time step.
- `theta_rot::Float` = 0: Fixed rotation angle along theta
- `phi_rot::Float` = 0: Fixed rotation angle along phi
- `delay::Tuple{Float, Float, Float}` = `(0, 0, 0)`: Gradient delays in (HF, AP, LR)
- `adc_dim::Bool` = `true`: Place ADC samples in a separate axis from each time frame for compatibility with wrapper functions.
"""
function traj_kooshball(Nr, theta, phi; theta_rot=0, phi_rot=0, delay=(0, 0, 0), adc_dim=true)
    @assert (eltype(theta) == eltype(phi)) "Mismatch between input types of `theta` and `phi`"

    Ncyc, Nt = size(theta)

    kr = collect(((-Nr+1)/2:(Nr-1)/2) / Nr)
    stheta = sin.(theta)
    ctheta = cos.(theta)
    sphi = sin.(phi)
    cphi = cos.(phi)

    k = Array{eltype(theta),3}(undef, 3, Nr * Ncyc, Nt)
    if theta_rot == 0 && phi_rot == 0
        for it ∈ axes(k, 3)
            ki = Array{eltype(theta),3}(undef, 3, Nr, Ncyc)
            Threads.@threads for ic ∈ 1:Ncyc
                for ir ∈ 1:Nr
                    ki[1, ir, ic] = -stheta[ic, it] * cphi[ic, it] * (kr[ir] + delay[1])
                    ki[2, ir, ic] =  stheta[ic, it] * sphi[ic, it] * (kr[ir] + delay[2])
                    ki[3, ir, ic] =  ctheta[ic, it]                * (kr[ir] + delay[3])
                end
            end
            k[:, :, it] = reshape(ki, 3, :)
            @. k[:, :, it] = max(min(k[:, :, it], 0.5), -0.5) # avoid NFFT.jl to throw errors. This should alter only very few points
        end
    else
        stheta_rot = sin(theta_rot)
        ctheta_rot = cos(theta_rot)
        sphi_rot   = sin(phi_rot)
        cphi_rot   = cos(phi_rot)

        k = Array{eltype(theta),3}(undef, 3, Nr * Ncyc, Nt)
        for it ∈ axes(k, 3)
            ki = Array{eltype(theta),3}(undef, 3, Nr, Ncyc)
            Threads.@threads for ic ∈ 1:Ncyc
                for ir ∈ 1:Nr
                    ki[1, ir, ic] = -(cphi_rot * cphi[ic, it] * ctheta_rot * stheta[ic, it] - sphi_rot *  sphi[ic, it] * stheta[ic, it] + cphi_rot * ctheta[ic, it] * stheta_rot)    * (kr[ir] + delay[1])
                    ki[2, ir, ic] =  (cphi_rot * sphi[ic, it]             * stheta[ic, it] + sphi_rot * (cphi[ic, it] * ctheta_rot * stheta[ic, it] + ctheta[ic, it] * stheta_rot)) * (kr[ir] + delay[2])
                    ki[3, ir, ic] =  (ctheta_rot * ctheta[ic, it] - stheta_rot * cphi[ic, it] * stheta[ic, it])                                                                   * (kr[ir] + delay[3])
                end
            end
            k[:, :, it] = reshape(ki, 3, :)
            @. k[:, :, it] = max(min(k[:, :, it], 0.5), -0.5) # avoid NFFT.jl to throw errors. This should alter only very few points
        end
    end
    k = adc_dim ? reshape(k, 3, Nr, Ncyc, Nt) : k
    return k
end

## ############################################
# Helper Functions
###############################################

"""
    calculate_golden_means()

Function to calculate the 3D golden means [1].

# References
1. Chan RW, Ramsay EA, Cunningham CH, and Plewes DB. "Temporal stability of adaptive 3D radial MRI using multidimensional golden means". Magn. Reson. Med. (2009), 61: 354-363. https://doi.org/10.1002/mrm.21837
"""
function calculate_golden_means()
    M = [0 1 0; 0 0 1; 1 0 1]
    v = eigvecs(M)
    gm1 = real(v[1, 3] / v[3, 3])
    gm2 = real(v[2, 3] / v[3, 3])
    return gm1, gm2
end