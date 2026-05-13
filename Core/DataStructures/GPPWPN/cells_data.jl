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
    cell_indices = CuArray(zeros(Int32, N)) #Cell index of each particle; 1-based
    perm = CuArray(zeros(Int32, N)) #Permutation to sort cell_indices
    sorted_particles = CuArray(Vector{Particle}(undef, N)) #Particles sorted by cell index
    sorted_cells = CuArray(zeros(Int32, N)) #Corresponding cells to sorted_particles
    cell_starts = CuArray(zeros(Int32, cell_list_params.num_boxes)) #First index in sorted_particles for cell c
    cell_ends = CuArray(zeros(Int32, cell_list_params.num_boxes)) #Last index in sorted_particles for cell c (inclusive)
    cell_neighbours = initialise_cell_neighbours_gppwpn(cell_list_params)
    return (; cell_indices, perm, sorted_particles, sorted_cells, cell_starts, cell_ends, cell_neighbours)
end #function

# Get array to store neighbours of each cell, including itself
# Take shape (num_neighbours=9, num_cells) as each workgroup in the calculate_θ_updates
#    kernel will belong to a single cell, with the warps corresponding to neighbours.
#    As Julia is column-major, we therefore want to have neighbours vary along columns
#    for warps to read from contiguous memory.
function initialise_cell_neighbours_gppwpn(cell_list_params::CellListParams)
    nx = cell_list_params.num_boxes_x
    ny = cell_list_params.num_boxes_y
    n = cell_list_params.num_boxes

    cell_neighbours = zeros(Int32, 9, n)

    for yidx in Int32(1):ny
        for xidx in Int32(1):nx
            cell = xidx + nx * (yidx - Int32(1))
            nghbr = Int32(1)
            for Δy in -Int32(1):Int32(1)
                for Δx in -Int32(1):Int32(1)
                    nghbr_xidx = mod1(xidx + Δx, nx)
                    nghbr_yidx = mod1(yidx + Δy, ny)
                    cell_neighbours[nghbr, cell] = nghbr_xidx + nx * (nghbr_yidx - Int32(1))
                    nghbr += 1
                end #for Δx
            end #for Δy
        end #for xidx
    end #for yidx

    return CuArray(cell_neighbours)
end #function
