using CairoMakie
include("./coarse_grain.jl")

###########################################################################
# Plotting
###########################################################################

function plot_fields(c_array::Matrix{<:Real}, m_array::Matrix{<:Real}, d_array::Matrix{<:Real})

    # WARN: Maybe transpose the arrays to get the correct orientations
    fig = Figure()
    c_subgrid = GridLayout()
    m_subgrid = GridLayout()
    d_subgrid = GridLayout()
    fig.layout[1, 1] = c_subgrid
    fig.layout[1, 2] = m_subgrid
    fig.layout[2, 1] = d_subgrid
    rowsize!(fig.layout, 1, Relative(4 / 7))

    #Colorwheel
    w_ax = PolarAxis(fig[2, 2])
    hidedecorations!(w_ax)
    rs = range(0, 1, 300)
    ϕs = range(0, 2π, 600)
    ϕmat = [ϕ for r in rs, ϕ in ϕs]
    rmat = [r for r in rs, ϕ in ϕs]
    w_ax.rlimits = (0, 1)
    surface!(w_ax, ϕmat, rmat, zeros(300, 600), color=ϕmat, colormap=:hsv, shading=false)

    #Concentration
    c_ax = Axis(c_subgrid[1, 1], aspect=DataAspect(), title="c")
    hidedecorations!(c_ax, label=false)
    c_hm = heatmap!(c_ax, c_array, colormap=:balance)
    c_cb = Colorbar(c_subgrid[2, 1], c_hm, vertical=false)

    #|polarisation|
    m_ax = Axis(m_subgrid[1, 1], aspect=DataAspect(), title="|m|")
    hidedecorations!(m_ax, label=false)
    m_hm = heatmap!(m_ax, m_array, colormap=:plasma, nan_color=:black)
    m_cb = Colorbar(m_subgrid[2, 1], m_hm, vertical=false)

    #Direction [arg(polarisation)]
    d_ax = Axis(d_subgrid[1, 1], aspect=DataAspect(), title="arg(m)")
    hidedecorations!(d_ax, label=false)
    d_hm = heatmap!(d_ax, mod.(d_array, 2π), colormap=:hsv, colorrange=(0, 2π), nan_color=:black)

    return fig
end #function

function plot_fields(xs::Vector{<:Real}, ys::Vector{<:Real}, θs::Vector{<:Real}, Lx::Int, Ly::Int, cell_width::Real; apply_filter=false, filter_strength=2)
    c_array, m_array, d_array = coarse_grain(xs, ys, θs, Lx, Ly, cell_width, apply_filter=apply_filter, filter_strength=filter_strength)
    fig = plot_fields(c_array, m_array, d_array)
    return fig
end #function

function plot_fields_nematic_too(c_array::Matrix{<:Real}, m_array::Matrix{<:Real}, d_array::Matrix{<:Real}, S_array::Matrix{<:Real})

    # WARN: Maybe transpose the arrays to get the correct orientations
    fig = Figure()
    c_subgrid = GridLayout()
    m_subgrid = GridLayout()
    d_subgrid = GridLayout()
    S_subgrid = GridLayout()
    fig.layout[1, 1] = c_subgrid
    fig.layout[1, 2] = m_subgrid
    fig.layout[2, 1] = d_subgrid
    fig.layout[2, 2] = S_subgrid
    rowsize!(fig.layout, 1, Relative(4 / 7))

    #Concentration
    c_ax = Axis(c_subgrid[1, 1], aspect=DataAspect(), title="c")
    hidedecorations!(c_ax, label=false)
    c_hm = heatmap!(c_ax, c_array, colormap=:balance)
    c_cb = Colorbar(c_subgrid[2, 1], c_hm, vertical=false)

    #|polarisation|
    m_ax = Axis(m_subgrid[1, 1], aspect=DataAspect(), title="|m|")
    hidedecorations!(m_ax, label=false)
    m_hm = heatmap!(m_ax, m_array, colormap=:plasma, nan_color=:black)
    m_cb = Colorbar(m_subgrid[2, 1], m_hm, vertical=false)

    #Direction [arg(polarisation)]
    d_ax = Axis(d_subgrid[1, 1], aspect=DataAspect(), title="arg(m)")
    hidedecorations!(d_ax, label=false)
    d_hm = heatmap!(d_ax, mod.(d_array, 2π), colormap=:hsv, colorrange=(0, 2π), nan_color=:black)

    #S
    S_ax = Axis(S_subgrid[1, 1], aspect=DataAspect(), title="S")
    hidedecorations!(S_ax, label=false)
    S_hm = heatmap!(S_ax, S_array, colormap=:plasma, nan_color=:black)
    S_cb = Colorbar(S_subgrid[2, 1], S_hm, vertical=false)

    return fig
end #function

function plot_fields_nematic_too(xs::Vector{<:Real}, ys::Vector{<:Real}, θs::Vector{<:Real}, Lx::Int, Ly::Int, cell_width::Real; apply_filter=false, filter_strength=2)
    c_array, m_array, d_array, S_array = coarse_grain_nematic_too(xs, ys, θs, Lx, Ly, cell_width, apply_filter=apply_filter, filter_strength=filter_strength)
    fig = plot_fields_nematic_too(c_array, m_array, d_array, S_array)
    return fig
end #function


