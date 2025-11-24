function save_plots_and_OPs(time_step, steps_to_save, particles, save_snapshots, save_OPs, file_name_addon, markersize, OP_m_file, OP_S_file, rank, comm)
    if time_step % steps_to_save == 0

        local_rs, local_θs = unpack_coords(Array(particles))

        if save_snapshots
            global_rs = MPI.gather(local_rs, 0, comm)
            global_θs = MPI.gather(local_θs, 0, comm)

            if rank == 0
                output_dir = "plots/"
                rs = vcat(global_rs...)
                θs = vcat(global_θs...)

                fig = Figure()
                ax = Axis(fig[1, 1], backgroundcolor=:black, aspect=DataAspect())
                xlims!(ax, (0, Lx))
                ylims!(ax, (0, Ly))
                colors = mod.(θs, 2π)
                colormap = :hsv
                colorrange = (0, 2π)
                scatter!(ax, rs, markersize=markersize, color=colors, colormap=colormap, colorrange=colorrange)
                file_name = "snapshot_" * file_name_addon * "_" * lpad(time_step, 8, "0") * ".png"
                CairoMakie.save(output_dir * file_name, fig)
            end #if (rank == 0)

        end #if save_snapshots
        if save_OPs
            local_cos = sum(cos.(local_θs))
            local_sin = sum(sin.(local_θs))
            local_cos2 = sum(cos.(2 .* local_θs))
            local_sin2 = sum(sin.(2 .* local_θs))
            local_count = length(local_θs)

            global_cos = MPI.Allreduce(local_cos, +, comm)
            global_sin = MPI.Allreduce(local_sin, +, comm)
            global_cos2 = MPI.Allreduce(local_cos2, +, comm)
            global_sin2 = MPI.Allreduce(local_sin2, +, comm)
            global_count = MPI.Allreduce(local_count, +, comm)

            if rank == 0
                magnetisation = sqrt(global_cos^2 + global_sin^2) / global_count
                S = sqrt(global_cos2^2 + global_sin2^2) / global_count
                writedlm(OP_m_file, [magnetisation, time_step * dt])
                writedlm(OP_S_file, [S, time_step * dt])
            end #if (rank == 0)
        end #if save_OPs

    end #if time_step
    return nothing
end #function
