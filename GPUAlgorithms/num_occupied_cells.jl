# --------- Check num_occupied_cells < max_num_occupied_cells, and reallocate if not ---------

function check_num_occupied_cells(num_occupied_cells, occupied_cells_particle_IDs, cell_list_params)
    CUDA.@allowscalar num_occupied = num_occupied_cells[1]
    old_max, max_particles_in_cell = size(occupied_cells_particle_IDs)
    if num_occupied > old_max
        new_max = minimum([ceil(Int32, 1.1 * num_occupied), cell_list_params.num_boxes])
        return new_max, Int32(max_particles_in_cell)
    else
        return false
    end #if
end #function

# New function used to lower max_num_occupied_cells after many steps
function lower_max_num_occupied_cells_check(num_occupied_cells, occupied_cells_particle_IDs, cell_list_params)
    CUDA.@allowscalar num_occupied = num_occupied_cells[1]
    old_max, max_particles_in_cell = size(occupied_cells_particle_IDs)
    new_max = minimum([ceil(Int32, 1.7 * num_occupied), cell_list_params.num_boxes])
    if new_max < old_max
        return new_max, Int32(max_particles_in_cell)
    else
        return false
    end #if
end #function

function reallocate_occupied_cells_lists(new_max, max_particles_in_cell, ArrayType)
    IDs = build_occupied_cells_particle_IDs(new_max, max_particles_in_cell, ArrayType)
    rs = build_occupied_cells_particle_rs(new_max, max_particles_in_cell, ArrayType)
    θs = build_occupied_cells_particle_θs(new_max, max_particles_in_cell, ArrayType)
    return IDs, rs, θs
end #function

