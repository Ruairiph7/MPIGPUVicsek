# --------- Sort particles into sorted_particles --------- #

function sort_particles!(cells_data, particles, num_particles)
    # Reload cell_starts_scratch with cell starts; will increment elements atomically
    # each time a particle in the corresponding cell is inserted into sorted_particles.
    copyto!(cells_data.cell_starts_scratch, cells_data.cell_starts)

    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups

    kernel! = sort_particles_kernel!(CUDABackend(), workgroup_size)
    kernel!(
        cells_data.sorted_particles,
        cells_data.perm,
        cells_data.cell_starts_scratch,
        cells_data.cell_indices,
        particles,
        num_particles;
        ndrange=total_num_threads)
    # KernelAbstractions.synchronize(CUDABackend())
end #function

@kernel function sort_particles_kernel!(
    sorted_particles,
    perm,
    cell_starts_scratch,
    @Const(cell_indices),
    @Const(particles),
    num_particles)

    I = Int32(@index(Global, Linear))
    stride = Int32(@ndrange()[1])

    for i = I:stride:num_particles
        c = cell_indices[i]
        pos = CUDA.atomic_add!(pointer(cell_starts_scratch, c), Int32(1))
        sorted_particles[pos] = particles[i]
        perm[pos] = i
    end #for i
end #function
