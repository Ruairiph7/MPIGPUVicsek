function save_plots_and_OPs(time_step, particles, output_params, num_params, OP_m_file, OP_S_file, rank, comm)
    if (time_step % output_params.steps_to_save_plots == 0) || (time_step % output_params.steps_to_save_OPs == 0)

        local_rs, local_θs, local_uids = unpack_coords(Array(particles))

        if output_params.save_plots && (time_step % output_params.steps_to_save_plots == 0)
            global_rs = MPI.gather(local_rs, comm)
            global_θs = MPI.gather(local_θs, comm)

            if rank == 0
                output_dir = "plots/"
                rs = vcat(global_rs...)
                θs = vcat(global_θs...)

                fig = Figure()
                ax = Axis(fig[1, 1], backgroundcolor=:black, aspect=DataAspect())
                xlims!(ax, (0, num_params.Lx))
                ylims!(ax, (0, num_params.Ly))
                colors = mod.(θs, 2π)
                colormap = :hsv
                colorrange = (0, 2π)
                scatter!(ax, rs, markersize=output_params.markersize, color=colors, colormap=colormap, colorrange=colorrange)
                file_name = "snapshot_" * output_params.file_name_addon * "_" * lpad(time_step, 8, "0") * ".png"
                CairoMakie.save(output_dir * file_name, fig)
            end #if (rank == 0)

        end #if save_plots

        if output_params.save_OPs && (time_step % output_params.steps_to_save_OPs == 0)
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
                writedlm(OP_m_file, [magnetisation, time_step * num_params.dt])
                writedlm(OP_S_file, [S, time_step * num_params.dt])
            end #if (rank == 0)
        end #if save_OPs

    end #if time_step
    return nothing
end #function
