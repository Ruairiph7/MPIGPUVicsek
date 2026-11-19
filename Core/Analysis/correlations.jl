using FFTW

include("./load_coords.jl")
include("./coarse_grain.jl")

function get_pert_fft(f,box_width)
    df = f .- mean(f)
    df_hat = fftshift(fft(df))
    kxs = fftshift(2π .* fftfreq(size(f,1),1/box_width))
    kys = fftshift(2π .* fftfreq(size(f,2),1/box_width))
    fft_norm = size(f,1) * size(f,2)
    return df_hat, kxs, kys, fft_norm
end #function

#Get (density-density structure factor) in the (k,θ) plane
function get_Sc(xs,ys,θs,Lx,Ly;R=2/sqrt(π),box_width=R,apply_filter=false,filter_strength=1)

    cs,ps,argps = coarse_grain(xs,ys,θs,Lx,Ly,box_width,apply_filter=apply_filter,filter_strength=filter_strength)
    argps[isnan.(argps)] .= 0.0

    dc_hats, kxs, kys, fft_norm = get_pert_fft(cs,box_width)
    S = abs2.(dc_hats) ./ fft_norm

    return S, kxs, kys
end #function

#Get (magnetisation-magnetisation structure factor) in the (k,θ) plane
function get_Sm(xs,ys,θs,Lx,Ly;R=2/sqrt(π),box_width=R,apply_filter=false,filter_strength=1)

    cs,ps,argps = coarse_grain(xs,ys,θs,Lx,Ly,box_width,apply_filter=apply_filter,filter_strength=filter_strength)
    argps[isnan.(argps)] .= 0.0

    mxs = cs .* ps .* cos.(argps)
    mys = cs .* ps .* sin.(argps)

    dmx_hats, kxs, kys, fft_norm = get_pert_fft(mxs,box_width)
    dmy_hats, _, _, _ = get_pert_fft(mys,box_width)

    S_tot = (abs2.(dmx_hats) .+ abs2.(dmy_hats)) ./ fft_norm
    S_L = zeros(size(S_tot))
    S_T = zeros(size(S_tot))
    for j in eachindex(kys), i in eachindex(kxs)
        k = hypot(kxs[i],kys[j])
        k == 0 && continue
        S_L[i,j] = abs2((kxs[i] * dmx_hats[i,j] + kys[j] * dmy_hats[i,j])/k) / fft_norm
        S_T[i,j] = abs2((kxs[i] * dmy_hats[i,j] - kys[j] * dmx_hats[i,j])/k) / fft_norm
    end #for i,j

    S = (tot=S_tot,L=S_L,T=S_T)

    return S, kxs, kys
end #function


#Get (polarisation-polarisation structure factor) in the (k,θ) plane
function get_Sp(xs,ys,θs,Lx,Ly;R=2/sqrt(π),box_width=R,apply_filter=false,filter_strength=1)

    cs,ps,argps = coarse_grain(xs,ys,θs,Lx,Ly,box_width,apply_filter=apply_filter,filter_strength=filter_strength)
    argps[isnan.(argps)] .= 0.0

    pxs = ps .* cos.(argps)
    pys = ps .* sin.(argps)

    dpx_hats, kxs, kys, fft_norm = get_pert_fft(pxs,box_width)
    dpy_hats, _, _, _ = get_pert_fft(pys,box_width)

    S_tot = (abs2.(dpx_hats) .+ abs2.(dpy_hats)) ./ fft_norm
    S_L = zeros(size(S_tot))
    S_T = zeros(size(S_tot))
    for j in eachindex(kys), i in eachindex(kxs)
        k = hypot(kxs[i],kys[j])
        k == 0 && continue
        S_L[i,j] = abs2((kxs[i] * dpx_hats[i,j] + kys[j] * dpy_hats[i,j])/k) / fft_norm
        S_T[i,j] = abs2((kxs[i] * dpy_hats[i,j] - kys[j] * dpx_hats[i,j])/k) / fft_norm
    end #for i,j

    S = (tot=S_tot,L=S_L,T=S_T)

    return S, kxs, kys
end #function


function save_structure_factors(dir,Lx,Ly,name;R=2/sqrt(π))
    mkpath("correlations/")

    t,xs,ys,θs = load_coords_from_timestep(dir)
    Sc, kxs, kys = get_Sc(xs,ys,θs,Lx,Ly,R=R)
    Sm_full, _, _ = get_Sm(xs,ys,θs,Lx,Ly,R=R)
    Sp_full, _, _ = get_Sp(xs,ys,θs,Lx,Ly,R=R)
    Sm = Sm_full.tot
    Sp = Sp_full.tot

    Sc_radial, ks_radial = get_S_radial(Sc,kxs,kys,R)
    Sm_radial, ks_radial = get_S_radial(Sm,kxs,kys,R)
    Sp_radial, ks_radial = get_S_radial(Sp,kxs,kys,R)

    @show ks_radial[1], Sc_radial[1], Sm_radial[1], Sp_radial[1]
    Sc_radial = Sc_radial[2:end]
    Sm_radial = Sm_radial[2:end]
    Sp_radial = Sp_radial[2:end]
    ks_radial = ks_radial[2:end]

    @save "correlations/$(name).jld2" Sc Sc_radial Sm Sm_radial Sp Sp_radial kxs kys ks_radial
end #function


function get_S_radial(S,kxs,kys,box_width)
    Nx,Ny=size(S)

    ks = [hypot(kxs[i],kys[j]) for i in 1:Nx, j in 1:Ny]
    dk = 2π/(Nx*box_width)
    k_max = maximum(ks)

    get_bin_idx(k_ij) = floor(Int, k_ij/dk + 0.5) + 1
    n_bins = get_bin_idx(k_max)

    sums = zeros(n_bins)
    counts = zeros(Int,n_bins)
    for j in 1:Ny, i in 1:Nx
        bidx = get_bin_idx(ks[i,j])
        sums[bidx] += S[i,j]
        counts[bidx] += 1
    end

    S_radial = [c > 0 ? s/c : NaN for (s,c) in zip(sums,counts)]
    ks_radial = collect(0:n_bins-1) .* dk

    return S_radial, ks_radial
end #function


function get_correlation_length_hwhm(ks,Sks)
    valid_idxs = isfinite.(Sks)
    ks_valid = ks[valid_idxs]
    Sks_valid = Sks[valid_idxs]

    isempty(Sks_valid) && error("No valid bins")
    peak_Sk, peak_idx = findmax(Sks_valid)
    halfmax = peak_Sk/2

    for i in (peak_idx + 1):length(Sks_valid)
        if Sks_valid[i] <= peak_Sk/2
            #Linearly Interpolate
            k_1,k_2 = ks_valid[i-1], ks_valid[i]
            Sk_1,Sk_2 = Sks_valid[i-1], Sks_valid[i]
            k_half = k_1 + (peak_Sk/2-Sk_1)*(k_2-k_1)/(Sk_2-Sk_1)
            return 2π/k_half
        end #if
    end #for i
    return NaN
end #function
