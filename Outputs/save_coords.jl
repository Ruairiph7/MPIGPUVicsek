function save_coords(time_step, steps_to_save_on_the_go, file_name_addon, particles, rank)
    if time_step % steps_to_save_on_the_go == 0

        output_dir = "outputs/timestep_" * lpad(time_step, 8, "0")
        ispath(output_dir) || mkpath(output_dir)

        file_name = output_dir * "/" * file_name_addon * "_rank_" * lpad(rank, 4, "0") * ".jld2"

        particles_cpu = Array(particles)
        @save file_name timestep = particles_cpu
    end #if time_step
end #function
