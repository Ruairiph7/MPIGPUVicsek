function get_updates!(θ_updates, particles, cells_data, cell_list_params, num_particles, numerical_params)

    # --------- Create cell list data structures --------- #

    #1) Build histogram - count particles per cell; 
    #   Also record each particle's cell index and track occupied cells
    build_histogram!(cells_data, cell_list_params, particles, num_particles)

    # Sort occupied_cells by hilbert idx so spatially adjacent cell
    # are processed by temporally adjacent workgroups - increase cache resuse
    n_occ = Array(cells_data.num_occupied)[1]
    occ_view = @view cells_data.occupied_cells[1:n_occ]
    CUDA.sort!(occ_view)

    #2) Assign cell_starts
    assign_cell_starts!(cells_data, cell_list_params)

    #3) Sort particles
    sort_particles!(cells_data, particles, num_particles)


    # --------- Store interactions in θ_updates --------- #

    calculate_interactions!(θ_updates, cells_data, cell_list_params, numerical_params)

end #function
