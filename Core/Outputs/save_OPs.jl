function _save_OPs(time_step, particles, OP_m_file,
    num_params, mpi_params)

    local_xs, local_ys, local_θs, local_uids = unpack_coords(Array(particles))

    local_cos = sum(cos.(local_θs))
    local_sin = sum(sin.(local_θs))
    local_count = length(local_θs)

    global_cos = MPI.Allreduce(local_cos, +, mpi_params.comm)
    global_sin = MPI.Allreduce(local_sin, +, mpi_params.comm)
    global_count = MPI.Allreduce(local_count, +, mpi_params.comm)

    if mpi_params.rank == 0
        magnetisation = sqrt(global_cos^2 + global_sin^2) / global_count
        writedlm(OP_m_file, [magnetisation, time_step * num_params.dt])
    end #if (rank == 0)

    return nothing
end #function

