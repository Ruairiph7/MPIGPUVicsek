# Carry out cell lists algorithms
#---------------------------------------------#
function get_updates!(θ_updates, particles, cells_data, cell_list_params, num_particles, numerical_params, min_cell_width)
    #Algorithm 1
    prep_cell_lists!(cells_data, cells_data.num_occupied, particles, cell_list_params, num_particles)

    #Check if we can lower max_num_occupied_cells at regular intervals
    if time_step % steps_to_shrink_buffers == 0
        lower_max_num_occupied_cells = lower_max_num_occupied_cells_check(cells_data.num_occupied, cells_data.occupied_IDs, cell_list_params)
        if lower_max_num_occupied_cells != false
            new_max, max_particles_in_cell = lower_max_num_occupied_cells
            println("Rank " * string(rank) * ": Lowering max_num_occupied_cells to " * string(new_max))
            reallocate_occupied_cells_lists!(cells_data, new_max, max_particles_in_cell, ArrayType)
            KernelAbstractions.synchronize(backend)
        end #if lower_max_num_occupied_cells
    end #if time_step

    #Check if we need to update max_num_occupied_cells
    update_max_num_occupied_cells = check_num_occupied_cells(cells_data.num_occupied,
        cells_data.occupied_IDs, cell_list_params)
    if update_max_num_occupied_cells != false
        new_max, max_particles_in_cell = update_max_num_occupied_cells
        println("Rank " * string(rank) * ": Raising max_num_occupied_cells to " * string(new_max))
        reallocate_occupied_cells_lists!(cells_data, new_max, max_particles_in_cell, ArrayType)
        KernelAbstractions.synchronize(backend)
    end #if

    #Algorithm 2
    assign_particles!(cells_data, particles, cell_list_params, num_particles)

    #Algorithm 3
    calculate_θ_updates!(θ_updates, cells_data, numerical_params, min_cell_width, cells_data.num_occupied, cell_list_params.num_boxes)

end #function
#---------------------------------------------#
