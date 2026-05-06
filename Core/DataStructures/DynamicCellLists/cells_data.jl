# --------- Initialise data structures ---------

function initialise_θ_updates(N; ArrayType=CuArray)
    return ArrayType(zeros(Float32, N))
end #function

function initialise_data_structures(params::CellListParams, max_num_occupied_cells, max_particles_in_cell, num_occupied_cells, ArrayType)
    neighbours = build_cell_neighbours_list(params, ArrayType)
    addresses = build_cell_address_list(params, ArrayType)
    num_particles = build_cell_num_particles_list(params, ArrayType)
    occupied_IDs = build_occupied_cells_particle_IDs(max_num_occupied_cells, max_particles_in_cell, ArrayType)
    occupied_rs = build_occupied_cells_particle_rs(max_num_occupied_cells, max_particles_in_cell, ArrayType)
    occupied_θs = build_occupied_cells_particle_θs(max_num_occupied_cells, max_particles_in_cell, ArrayType)
    occupied_ID_list = build_occupied_cells_ID_list(params::CellListParams, ArrayType)
    max_num_occupied = max_num_occupied_cells
    num_occupied = num_occupied_cells
    return (; neighbours, addresses, num_particles, occupied_IDs, occupied_rs, occupied_θs, occupied_ID_list, max_num_occupied, num_occupied)
end #function

function build_cell_neighbours_list(params::CellListParams, ArrayType)
    neighbours_list = build_cell_neighbours(params::CellListParams)
    return ArrayType([SVector{8,Int32}(neighbours_list[i]) for i = 1:params.num_boxes])
end #function

function build_cell_address_list(params::CellListParams, ArrayType)
    return ArrayType(zeros(Int32, params.num_boxes))
end #function

function build_cell_num_particles_list(params::CellListParams, ArrayType)
    return ArrayType(zeros(Int32, params.num_boxes))
end #function

function build_occupied_cells_particle_IDs(max_num_occupied_cells, max_particles_in_cell, ArrayType)
    return ArrayType([Int32(0) for i = 1:max_num_occupied_cells, j = 1:max_particles_in_cell])
end #function

function build_occupied_cells_particle_rs(max_num_occupied_cells, max_particles_in_cell, ArrayType)
    return ArrayType([zeros(SVector{2,Float32}) for i = 1:max_num_occupied_cells, j = 1:max_particles_in_cell])
end #function

function build_occupied_cells_particle_θs(max_num_occupied_cells, max_particles_in_cell, ArrayType)
    return ArrayType([Float32(0) for i = 1:max_num_occupied_cells, j = 1:max_particles_in_cell])
end #function

function build_occupied_cells_ID_list(params::CellListParams, ArrayType)
    return ArrayType(zeros(Int32, params.num_boxes))
end #function

