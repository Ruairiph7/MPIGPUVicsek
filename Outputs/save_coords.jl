function save_coords(time_step, steps_to_save_on_the_go, file_name_addon, particles, rank, comm)
    if time_step % steps_to_save_on_the_go == 0

        output_dir = "outputs/timestep_" * lpad(time_step, 10, "0")
        if rank == 0
            ispath(output_dir) || mkpath(output_dir)
        end #if rank
        MPI.Barrier(comm)

        file_name = output_dir * "/" * file_name_addon * "_rank_" * lpad(rank, 4, "0") * ".jld2"

        particles_cpu = Array(particles)
        @save file_name particles = particles_cpu
    end #if time_step
end #function
