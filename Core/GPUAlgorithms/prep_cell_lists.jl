# --------- Calculate num occupied cells, and assign addresses (Algorithm 1) ---------

function prep_cell_lists!(cells_data, num_occupied_cells, particles, cell_list_params, num_particles)
    cell_num_particles_list .= Int32(0)
    num_occupied_cells .= Int32(0)

    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups

    kernel! = prep_cell_lists_kernel!(CUDABackend())
    kernel!(cells_data.addresses, cells_data.num_particles, cells_data.occupied_IDs, num_occupied_cells, particles, cell_list_params, num_particles; ndrange=total_num_threads)
    KernelAbstractions.synchronize(CUDABackend())
end #function


@kernel function prep_cell_lists_kernel!(cell_address_list, cell_num_particles_list, occupied_cells_ID_list, num_occupied_cells, @Const(particles), cell_list_params, num_particles)
    I = @index(Global, Linear)
    # stride = @ndrange()
    stride = 256 * 256

    for i = I:stride:num_particles
        r_i = particles[i].r
        cell_ID = get_cell_ID(r_i, cell_list_params)
        old_num_particles = CUDA.atomic_add!(pointer(cell_num_particles_list, cell_ID), Int32(1))
        #^^Add one to num_particles in this cell, and return the old value before you added one
        if old_num_particles == 0 #If this is the first particle in this cell, assign an address for the cell
            #Perform atomically so threads don't interfere and we can assign IDs in sequence
            old_num_occupied_cells = CUDA.atomic_add!(pointer(num_occupied_cells, 1), Int32(1))
            cell_address_list[cell_ID] = old_num_occupied_cells + 1
            occupied_cells_ID_list[old_num_occupied_cells+1] = cell_ID
        end #if
    end #for i
end #function


