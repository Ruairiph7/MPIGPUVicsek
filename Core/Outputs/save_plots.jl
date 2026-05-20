function _save_plots(time_step, particles,
    output_params, num_params, mpi_params)

    local_xs, local_ys, local_θs, local_uids = unpack_coords(Array(particles))

    global_xs = MPI.gather(local_xs, mpi_params.comm)
    global_ys = MPI.gather(local_ys, mpi_params.comm)
    global_θs = MPI.gather(local_θs, mpi_params.comm)

    if mpi_params.rank == 0
        output_dir = "plots/"
        xs = vcat(global_xs...)
        ys = vcat(global_ys...)
        θs = vcat(global_θs...)

        fig = Figure()
        ax = Axis(fig[1, 1], backgroundcolor=:black, aspect=DataAspect())
        xlims!(ax, (0, num_params.Lx))
        ylims!(ax, (0, num_params.Ly))
        colors = mod.(θs, 2π)
        colormap = :hsv
        colorrange = (0, 2π)
        scatter!(ax, xs, ys, markersize=output_params.markersize,
            color=colors, colormap=colormap, colorrange=colorrange)
        file_name = "snapshot_$(output_params.file_name_addon)_$(lpad(time_step, 10, "0")).png"
        CairoMakie.save(output_dir * file_name, fig)
    end #if (rank == 0)

    return nothing
end #function

