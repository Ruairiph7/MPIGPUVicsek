function _save_OPs(time_step, particles, OP_m_file,
    num_params, mpi_params)

    local_cos, local_sin, local_count = compute_local_OP_sums(particles)

    global_cos = MPI.Allreduce(local_cos, +, mpi_params.comm)
    global_sin = MPI.Allreduce(local_sin, +, mpi_params.comm)
    global_count = MPI.Allreduce(local_count, +, mpi_params.comm)

    if mpi_params.rank == 0
        magnetisation = sqrt(global_cos^2 + global_sin^2) / global_count
        writedlm(OP_m_file, [magnetisation, time_step * num_params.dt])
    end #if (rank == 0)

    return nothing
end #function


# --------- Compute sums of sin() and cos() local particle θs on GPU --------- #

function compute_local_OP_sums(particles)
    num_particles = length(particles)
    cos_sum = CUDA.mapreduce(p -> cos(p.θ), +, particles)
    sin_sum = CUDA.mapreduce(p -> sin(p.θ), +, particles)
    return cos_sum, sin_sum, num_particles
end #function
