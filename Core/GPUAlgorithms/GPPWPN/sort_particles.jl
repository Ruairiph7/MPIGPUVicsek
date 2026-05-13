# --------- Sort particles into sorted_particles based on perm ---------

function sort_particles!(cells_data, particles, num_particles)
    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups

    kernel! = sort_particles_kernel!(CUDABackend(), workgroup_size)
    kernel!(cells_data.sorted_particles cells_data.sorted_cells, particles, cells_data.perm, cells_data.cell_indices, num_particles; ndrange=total_num_threads)
    KernelAbstractions.synchronize(CUDABackend())
end #function

@kernel function sort_particles_kernel!(sorted_particles, sorted_cells, @Const(particles), @Const(perm), @Const(cell_indices), num_particles)
    I = Int32(@index(Global, Linear))
    stride = Int32(@ndrange()[1])
    for i = I:stride:num_particles
        sorted_particles[i] = particles[perm[i]]
        sorted_cells[i] = cell_indices[perm[i]]
    end #for i
end #function

