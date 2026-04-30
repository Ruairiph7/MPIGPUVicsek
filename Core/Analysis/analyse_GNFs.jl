using DelimitedFiles, Statistics
using CairoMakie
using UnicodeFun
using LsqFit

function do_analysis(xs, ys, ρ₀, Lx, Ly=Lx; min_box_size=10, max_box_size=Lx / 2, num_box_sizes=30, num_samples=100, scaled_num_samples=false)
    log_box_sizes = collect(range(log(min_box_size), log(max_box_size), num_box_sizes))
    box_sizes = exp.(log_box_sizes)
    maximum(box_sizes) > minimum((Lx, Ly)) && error("Maximum box size doesn't fit")

    n_samples = [zeros(num_samples) for size in box_sizes]
    Δn²_samples = [zeros(num_samples) for size in box_sizes]
    if !scaled_num_samples
        Threads.@threads for (size_idx, size) in enumerate(box_sizes)
            n₀ = ρ₀ * size^2
            for sample_idx = 1:num_samples
                lower_x = (Lx - size) * rand()
                lower_y = (Ly - size) * rand()
                upper_x = lower_x + size
                upper_y = lower_y + size
                this_n = length(findall(x -> lower_x < x < upper_x, xs) ∩ findall(y -> lower_y < y < upper_y, ys))
                this_Δn² = (this_n - n₀)^2

                n_samples[size_idx][sample_idx] = this_n
                Δn²_samples[size_idx][sample_idx] = this_Δn²
            end #for sample_idx
        end #for (size_idx,size)
    elseif scaled_num_samples
        num_samples = round.(Int, 2 .* Lx ./ box_sizes)
        n_samples = [zeros(num_samples[idx]) for idx in eachindex(box_sizes)]
        Δn²_samples = [zeros(num_samples[idx]) for idx in eachindex(box_sizes)]
        Threads.@threads for (size_idx, size) in enumerate(box_sizes)
            n₀ = ρ₀ * size^2
            for sample_idx = 1:num_samples[size_idx]
                lower_x = (Lx - size) * rand()
                lower_y = (Ly - size) * rand()
                upper_x = lower_x + size
                upper_y = lower_y + size
                this_n = length(findall(x -> lower_x < x < upper_x, xs) ∩ findall(y -> lower_y < y < upper_y, ys))
                this_Δn² = (this_n - n₀)^2

                n_samples[size_idx][sample_idx] = this_n
                Δn²_samples[size_idx][sample_idx] = this_Δn²
            end #for sample_idx
        end #for (size_idx,size)
    end

    ns = mean.(n_samples)
    Δn²s = mean.(Δn²_samples)

    return ns, Δn²s
end #function

function save_analysis(file_name_addon, dir, xs::Vector{<:Real}, ys::Vector{<:Real}, ρ₀, Lx, Ly=Lx; min_box_size=10, max_box_size=Lx / 2, num_box_sizes=30, num_samples=100, scaled_num_samples=false)
    ns, Δn²s = do_analysis(xs, ys, ρ₀, Lx, Ly, min_box_size=min_box_size, max_box_size=max_box_size, num_box_sizes=num_box_sizes, num_samples=num_samples, scaled_num_samples=scaled_num_samples)
    writedlm(dir * "/ns_" * file_name_addon * ".txt", ns)
    writedlm(dir * "/vars_" * file_name_addon * ".txt", Δn²s)
    return nothing
end #function


function plot_GNFs(nss, Δn²ss, labels)
    # log_nss = [log10.(ns) for ns in nss]
    # log_Δn²_over_nss = [log10.(Δn²ss[i] ./ nss[i]) for i in eachindex(Δn²ss)]

    Δn²_over_nss = [(Δn²ss[i] ./ nss[i]) for i in eachindex(Δn²ss)]

    fig = Figure()
    ax = Axis(fig[1, 1])
    [scatter!(ax, nss[i], Δn²_over_nss[i], label=labels[i], color=i, colormap=:solar, colorrange=(1, length(nss))) for i in eachindex(nss)]

    TT_nss = collect(range(minimum(nss[1]), maximum(nss[1]), 100))
    lines!(ax, TT_nss, 100 * TT_nss .^ 0.6, linestyle=:dash, color=:black, label=L"\Delta n^2 \sim n^{1.6}")

    ax.xlabel = L"n"
    ax.ylabel = L"\Delta n^2 / n"
    lgd = Legend(fig[1, 2], ax)

    min_n = minimum([minimum(ns) for ns in nss])
    min_Δn²_over_n = minimum([minimum(Δn²ss[i] ./ nss[i]) for i in eachindex(nss)])
    max_n = maximum([maximum(ns) for ns in nss])
    max_Δn²_over_n = maximum([maximum(Δn²ss[i] ./ nss[i]) for i in eachindex(nss)])

    x_min_power_of_10 = floor(Int, log10(min_n))
    x_max_power_of_10 = ceil(Int, log10(max_n))

    y_min_power_of_10 = floor(Int, log10(min_Δn²_over_n))
    y_max_power_of_10 = ceil(Int, log10(max_Δn²_over_n))

    ax.xticks = [10.0^i for i = x_min_power_of_10:x_max_power_of_10]
    ax.xminorticksvisible = true
    ax.xminorticks = IntervalsBetween(10)
    ax.xscale = log10

    ax.yticks = [10.0^i for i = y_min_power_of_10:y_max_power_of_10]
    ax.yminorticksvisible = true
    ax.yminorticks = IntervalsBetween(10)
    ax.yscale = log10

    ax.xtickformat = vals -> to_latex.(raw"10^{" .* string.(Int.(log10.(vals))) .* "}")
    ax.ytickformat = vals -> to_latex.(raw"10^{" .* string.(Int.(log10.(vals))) .* "}")

    return fig
end

function do_GNFs_fits(nss, Δn²ss)
    log_nss = [log10.(ns) for ns in nss]
    log_Δn²_over_nss = [log10.(Δn²ss[i] ./ nss[i]) for i in eachindex(Δn²ss)]

    model(x, p) = p[1] + p[2] * x
    p1s = zeros(length(nss))
    p2s = zeros(length(nss))

    for idx = eachindex(nss)
        xdata = log_nss[idx]
        ydata = log_Δn²_over_nss[idx]
        # min_n_idx = findfirst(>(2), xdata)
        # max_n_idx = findlast(<(4), xdata)
        reduced_xdata = xdata[min_n_idx:max_n_idx]
        reduced_ydata = ydata[min_n_idx:max_n_idx]
        fit = curve_fit((x, p) -> model.(x, Ref(p)), reduced_xdata, reduced_ydata, [1.0, 1.0])
        p1s[idx] = fit.param[1]
        p2s[idx] = fit.param[2]
    end #for idx

    return p1s, p2s
end

function plot_GNFs_with_fits(nss, Δn²ss, labels)

    Δn²_over_nss = [(Δn²ss[i] ./ nss[i]) for i in eachindex(Δn²ss)]

    fig = Figure()
    ax = Axis(fig[1, 1])
    if length(nss) == 1
        [scatter!(ax, nss[i], Δn²_over_nss[i], label=labels[i], alpha=0.5) for i in eachindex(nss)]
    else
        [scatter!(ax, nss[i], Δn²_over_nss[i], label=labels[i], color=i, colormap=:solar, colorrange=(1, length(nss)), alpha=0.5) for i in eachindex(nss)]
    end

    TT_nss = collect(range(minimum(nss[1]), maximum(nss[1]), 100))
    lines!(ax, TT_nss, 100 * TT_nss .^ 0.6, linestyle=:dash, color=:black, label=L"\Delta n^2 \sim n^{1.6}")

    ax.xlabel = L"n"
    ax.ylabel = L"\Delta n^2 / n"
    lgd = Legend(fig[1, 2], ax)

    min_n = minimum([minimum(ns) for ns in nss])
    min_Δn²_over_n = minimum([minimum(Δn²ss[i] ./ nss[i]) for i in eachindex(nss)])
    max_n = maximum([maximum(ns) for ns in nss])
    max_Δn²_over_n = maximum([maximum(Δn²ss[i] ./ nss[i]) for i in eachindex(nss)])

    x_min_power_of_10 = floor(Int, log10(min_n))
    x_max_power_of_10 = ceil(Int, log10(max_n))

    y_min_power_of_10 = floor(Int, log10(min_Δn²_over_n))
    y_max_power_of_10 = ceil(Int, log10(max_Δn²_over_n))

    ax.xticks = [10.0^i for i = x_min_power_of_10:x_max_power_of_10]
    ax.xminorticksvisible = true
    ax.xminorticks = IntervalsBetween(10)
    ax.xscale = log10

    ax.yticks = [10.0^i for i = y_min_power_of_10:y_max_power_of_10]
    ax.yminorticksvisible = true
    ax.yminorticks = IntervalsBetween(10)
    ax.yscale = log10

    ax.xtickformat = vals -> to_latex.(raw"10^{" .* string.(Int.(log10.(vals))) .* "}")
    ax.ytickformat = vals -> to_latex.(raw"10^{" .* string.(Int.(log10.(vals))) .* "}")


    p1s, p2s = do_GNFs_fits(nss, Δn²ss)

    fit_nss = collect(range(minimum(nss[1]), maximum(nss[1]), 100))
    if length(nss) == 1
        [lines!(ax, fit_nss, (10^p1s[i]) * fit_nss .^ (p2s[i]), linestyle=:dash) for i = eachindex(nss)]
    else
        [lines!(ax, fit_nss, (10^p1s[i]) * fit_nss .^ (p2s[i]), linestyle=:dash, color=i, colormap=:solar, colorrange=(1, length(nss))) for i = eachindex(nss)]
    end

    return fig
end

function plot_fit_params(nss, Δn²ss, γ_ns)
    p1s, p2s = do_GNFs_fits(nss, Δn²ss)

    fig = Figure(size=(800, 400))
    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[1, 2])

    scatterlines!(ax1, γ_ns, p1s, color=1:length(nss), colormap=:solar, colorrange=(1, length(nss)))
    scatterlines!(ax2, γ_ns, p2s, color=1:length(nss), colormap=:solar, colorrange=(1, length(nss)))

    ax1.xlabel = L"\gamma_n"
    ax1.ylabel = "loglog fit y-intercept"
    ax2.xlabel = L"\gamma_n"
    ax2.ylabel = "loglog fit gradient"

    return fig
end
