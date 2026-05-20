function save_coords(time_step, particles, num_particles, save_bufs, output_params, mpi_params)
    time_step % output_params.steps_to_save_coords != 0 && return

    output_dir = "outputs/timestep_$(lpad(time_step, 10, "0"))"
    mpi_params.rank == 0 && mkpath(output_dir)
    MPI.Barrier(mpi_params.comm)

    file_name = "$output_dir/$(output_params.file_name_addon)" *
                "_rank_$(lpad(mpi_params.rank, 4, "0")).jld2"

    if save_bufs.ASYNC_SAVES
        if !isnothing(save_bufs.save_task)
            println("Rank $(mpi_params.rank): Waiting before next save.")
            wait(save_bufs.save_task)
        end #if
        t_transfer = @elapsed copyto!(save_bufs.pinned_buf, 1, particles, 1, num_particles)
        t_transfer_old = @elapsed tmp_array = Array(particles)

        main_thread_id = Threads.threadid()
        save_bufs.save_task = Threads.@spawn begin
            write_thread_id = Threads.threadid()
            if write_thread_id == main_thread_id
                @warn "Rank $(mpi_params.rank): write running on main thread - async not working."
            end #if
            t_write = @elapsed @save file_name particles = save_bufs.pinned_buf[1:num_particles]
            println("Rank $(mpi_params.rank): backround write took $(round(t_write, digits=3))s")
        end #begin
        println("Rank $(mpi_params.rank): transfer took $(round(t_transfer, digits=3))s")
        println("Rank $(mpi_params.rank): old transfer took $(round(t_transfer_old, digits=3))s")
    else
        copyto!(save_bufs.pinned_buf, 1, particles, 1, num_particles)
        @save file_name particles = save_bufs.pinned_buf[1:num_particles]
    end #if

    return nothing
end #function
