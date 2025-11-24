# --------- Calculate num occupied cells, and assign addresses (Algorithm 1) ---------

#NOTE:
#TODO:
#WARN: IVE changed the code from passing each element of cell_list_params individually to just passing the whole struct as an argument --> check this still works
function prep_cell_lists!(cell_address_list, cell_num_particles_list, occupied_cells_ID_list, num_occupied_cells, particles, cell_list_params, backend)
    cell_num_particles_list .= Int32(0)
    num_occupied_cells .= Int32(0)
    KernelAbstractions.synchronize(backend)
    #----------------------------
    kernel! = prep_cell_lists_kernel!(backend)
    kernel!(cell_address_list, cell_num_particles_list, occupied_cells_ID_list, num_occupied_cells, particles, cell_list_params; ndrange=length(particles))
    KernelAbstractions.synchronize(backend)
end #function

@kernel function prep_cell_lists_kernel!(cell_address_list, cell_num_particles_list, occupied_cells_ID_list, num_occupied_cells, @Const(particles), cell_list_params)
    i = @index(Global, Linear)
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
end #function


