# --------- Carry out cell lists algorithms ---------
function get_updates_bpc!(θ_updates, particles, cells_data, cell_list_params, num_particles, numerical_params, R_max)

    # --------- Create cell list data structures ---------    

    #1) Build histogram - count particles per cell; 
    #   Also record each particle's cell index and track occupied cells
    build_histogram!(cells_data, cell_list_params, particles, num_particles)

    #2) Assign cell_starts
    assign_cell_starts!(cells_data, cell_list_params)

    #3) Sort particles
    sort_particles_bpc!(cells_data, particles, num_particles)

    # --------- Calculate θ_updates ---------    

    calculate_θ_updates_bpc!(θ_updates, cells_data, numerical_params, R_max)

end #function
