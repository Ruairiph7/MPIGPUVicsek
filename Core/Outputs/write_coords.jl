function write_coords(time_step, particles, output_params, mpi_params)
    if time_step % output_params.steps_to_save_coords == 0

        output_dir = "outputs/timestep_" * lpad(time_step, 10, "0")
        if mpi_params.rank == 0
            ispath(output_dir) || mkpath(output_dir)
        end #if rank
        MPI.Barrier(mpi_params.comm)

        file_name = output_dir * "/" * output_params.file_name_addon *
                    "_rank_" * lpad(mpi_params.rank, 4, "0") * ".jld2"

        particles_cpu = Array(particles)
        @save file_name particles = particles_cpu
    end #if time_step
end #function
