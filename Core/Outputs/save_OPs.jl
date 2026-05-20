function _save_OPs(time_step, particles, OP_file,
    num_params, mpi_params)

    local_cos, local_sin, local_count = compute_local_OP_sums(particles)

    #Combine into one array for reduced MPI latency
    #Store as Float64 to reduce errors when summing many values
    local_sums = Float64[local_cos, local_sin, local_count]
    global_sums = MPI.Allreduce(local_sums, +, mpi_params.comm)
    global_cos, global_sin, global_count = global_sums

    if mpi_params.rank == 0
        magnetisation = sqrt(global_cos^2 + global_sin^2) / global_count
        writedlm(OP_file, [magnetisation, time_step * num_params.dt])
    end #if (rank == 0)

    return nothing
end #function


# --------- Compute sums of sin() and cos() of local particle θs on GPU --------- #

function compute_local_OP_sums(particles)
    num_particles = length(particles)
    cos_sum = CUDA.mapreduce(p -> cos(p.θ), +, particles)
    sin_sum = CUDA.mapreduce(p -> sin(p.θ), +, particles)
    return cos_sum, sin_sum, num_particles
end #function
