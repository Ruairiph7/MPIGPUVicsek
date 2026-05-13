# Carry out cell lists algorithms
#---------------------------------------------#
function get_updates_gppwpn!(θ_updates, particles, cells_data, cell_list_params, num_particles, numerical_params, min_cell_width)

    # --------- Create cell list data structures ---------    

    #1) Assign particles to cells, store in cell_indices
    assign_cells_gppwpn!(cells_data, particles, cell_list_params, num_particles)

    #2) Reset perm buffer to the identity
    reset_perm!(cells_data, num_particles)

    #3) Store permutation to order cell_indices (cell indices is not mutated)
    CUDA.sortperm!(cells_data.perm, cells_data.cell_indices)

    #4) Store sorted particles in sorted_particles (each cell in contiguous memory, tracked with sorted_cells)
    sort_particles!(cells_data, particles, num_particles)

    #6) Find first and last index for each cell in sorted_particles, using sorted_cells
    find_cell_bounds!(cells_data, num_particles)

    # --------- Calculate θ_updates ---------    

    calculate_θ_updates_gppwpn!(θ_updates, cells_data, cell_list_params, num_particles, numerical_params, min_cell_width)

end #function
#---------------------------------------------#

