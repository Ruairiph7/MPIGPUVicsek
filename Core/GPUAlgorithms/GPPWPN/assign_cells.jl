# --------- Assign particles to cells ---------
function assign_cells_gppwpn!(cells_data, cell_list_params, particles, num_particles)
    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups

    kernel! = assign_cells_gppwpn_kernel!(CUDABackend(), workgroup_size)
    kernel!(cells_data.cell_indices,
        cell_list_params.num_cells_x,
        cell_list_params.num_cells_y,
        cell_list_params.cell_size_x,
        cell_list_params.cell_size_y,
        particles,
        num_particles;
        ndrange=total_num_threads)
    KernelAbstractions.synchronize(CUDABackend())
end #function

@kernel function assign_cells_gppwpn_kernel!(
    cell_indices,
    @Const(particles),
    num_cells_x,
    num_cells_y,
    cell_size_x,
    cell_size_y,
    num_particles)
    I = Int32(@index(Global, Linear))
    stride = Int32(@ndrange()[1])
    for i = I:stride:num_particles
        cell_indices[i] = get_cell_ID(particles[i].r, num_cells_x, num_cells_y, cell_size_x, cell_size_y) #One-based indexing
    end #for i
end #function

