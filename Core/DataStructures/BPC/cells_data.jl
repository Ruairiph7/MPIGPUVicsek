# --------- Initialise cell list data structures ---------

struct CellList
    # Per-particle (length N)
    cell_indices::CuVector{Int32} #Cell index of each particle; 1-based
    sorted_particles::CuVector{Particle} #Particles sorted by cell index
    perm::CuVector{Int32} #Permutation from sorted particle indices to absolute ones

    # Per-cell (length num_cells)
    cell_counts::CuVector{Int32} #Number of particles in each cell
    cell_starts::CuVector{Int32} #First index in sorted particles for cell c
    cell_starts_scratch::CuVector{Int32} #Copy of cell_starts, used during sorting

    # Occupied cell tracking
    occupied_cells::CuVector{Int32} #Indices of non-empty cells; length == num_cells
    num_occupied::CuVector{Int32} #Scalar counter; length == 1

    # Initialised once at startup, never changes
    cell_neighbours::CuArray{Int32,2}   # size == (9, ncells)
end #struct
CellList(cell_list_params::CellListParams, N) = CellList(
    CUDA.zeros(Int32, N),
    CuArray(Vector{Particle}(undef, N)),
    CUDA.zeros(Int32, N),
    CUDA.zeros(Int32, cell_list_params.num_cells),
    CUDA.zeros(Int32, cell_list_params.num_cells),
    CUDA.zeros(Int32, cell_list_params.num_cells),
    CUDA.zeros(Int32, cell_list_params.num_cells),
    CUDA.zeros(Int32, 1),
    initialise_cell_neighbours_bpc(cell_list_params)
)

# Get array to store neighbours of each cell, including itself
# Take shape (num_neighbours=9, num_cells) as each workgroup in the calculate_θ_updates
#    kernel will belong to a single cell, with the warps corresponding to neighbours.
#    As Julia is column-major, we therefore want to have neighbours vary along columns
#    for warps to read from contiguous memory.
function initialise_cell_neighbours_bpc(cell_list_params::CellListParams)
    nx = cell_list_params.num_cells_x
    ny = cell_list_params.num_cells_y
    n = cell_list_params.num_cells

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
