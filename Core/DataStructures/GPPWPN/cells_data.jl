# --------- Initialise data structures ---------

function initialise_θ_updates_gppwpn(N; ArrayType=CuArray)
    return ArrayType(zeros(Float32, N))
end #function

function initialise_rand_bufs_gppwpn(N; ArrayType=CuArray)
    rand1 = ArrayType(zeros(Float32, N))
    rand2 = ArrayType(zeros(Float32, N))
    return (; rand1, rand2)
end #function

function initialise_data_structures_gppwpn(cell_list_params::CellListParams, N)
    cell_indices = CuArray(zeros(Int32, N)) #Cell index of each particle
    perm = CuArray(zeros(Int32, N)) #Permutation to sort cell_indices
    sorted_particles = CuArray(Vector{Particle}(undef, N)) #Particles sorted by cell index
    sorted_cells = CuArray(zeros(Int32, N)) #Corresponding cells to sorted_particles
    cell_starts = CuArray(zeros(Int32, cell_list_params.num_boxes)) #First index in sorted_particles for cell c
    cell_ends = CuArray(zeros(Int32, cell_list_params.num_boxes)) #Last index in sorted_particles for cell c
    return (; cell_indices, perm, sorted_particles, sorted_cells, cell_starts, cell_ends)
end #function
error("CHECK cell_ends is correct; i.e. last index or one past last index")


