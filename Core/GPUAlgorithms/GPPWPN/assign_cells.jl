# --------- Assign particles to cells ---------
function assign_cells_gppwpn!(cells_data, particles, cell_list_params, num_particles)
    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups

    kernel! = assign_cells_gppwpn_kernel!(CUDABackend(), workgroup_size)
    kernel!(cells_data.cell_indices, particles, cell_list_params, num_particles; ndrange=total_num_threads)
    KernelAbstractions.synchronize(CUDABackend())
end #function

@kernel function assign_cells_gppwpn_kernel!(cell_indices, @Const(particles), cell_list_params, num_particles)
    I = Int32(@index(Global, Linear))
    stride = Int32(@ndrange()[1])
    for i = I:stride:num_particles
        cell_indices[i] = get_cell_ID(particles[i].r, cell_list_params) #One-based indexing
    end #for i
end #function

