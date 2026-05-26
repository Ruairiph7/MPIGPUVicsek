function _save_coords(time_step, particles, num_particles,
    save_bufs, output_params, mpi_params)

    output_dir = "outputs/timestep_$(lpad(time_step, 10, "0"))"
    mpi_params.rank == 0 && mkpath(output_dir)
    MPI.Barrier(mpi_params.comm)

    file_name = "$output_dir/$(output_params.file_name_addon)" *
                "_rank_$(lpad(mpi_params.rank, 10, "0")).jld2"

    if save_bufs.ASYNC_SAVES
        #Wait for previous save to finish if it is still running
        if !isnothing(save_bufs.save_task)
            t_wait = @elapsed wait(save_bufs.save_task)
            if output_params.LOG_WRITE_TIMES && t_wait > 0.01
                println("Rank $(mpi_params.rank): waited $(round(t_wait, digits=3))s for previous save")
            end #if
        end #if
        copyto!(save_bufs.pinned_buf, 1, particles, 1, num_particles)

        #Launch saving from the buffer to file asynchronousy on a background thread.
        main_thread_id = Threads.threadid()
        save_bufs.save_task = Threads.@spawn begin
            write_thread_id = Threads.threadid()
            if write_thread_id == main_thread_id
                @warn "Rank $(mpi_params.rank): write running on main thread - async not working."
            end #if
            t_write = @elapsed @save file_name particles = save_bufs.pinned_buf[1:num_particles]
            output_params.LOG_WRITE_TIMES && println("Rank $(mpi_params.rank): backround write took $(round(t_write, digits=3))s")
        end #begin

    else
        copyto!(save_bufs.pinned_buf, 1, particles, 1, num_particles)
        t_write = @elapsed @save file_name particles = save_bufs.pinned_buf[1:num_particles]
        output_params.LOG_WRITE_TIMES && println("Rank $(mpi_params.rank): write took $(round(t_write, digits=3))s")
    end #if

    return nothing
end #function
