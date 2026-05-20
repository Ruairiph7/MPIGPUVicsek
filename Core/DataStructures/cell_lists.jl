# --------- Cell list parameters --------- #

struct CellListParams
    x_min_cells::Float32 #Start of cell lists in x
    x_max_cells::Float32 #End of cell lists in x
    x_min_local::Float32 #Start of local domain in x
    x_max_local::Float32 #End of local domain in x
    inv_cell_size_x::Float32 #1/cell_size_x - used for get_cell_ID
    inv_cell_size_y::Float32 #1/cell_size_y - used for get_cell_ID
    num_cells_x::Int32
    num_cells_y::Int32
    num_cells::Int32
end #struct

function CellListParams(numerical_params; SINGLE_RANK::Bool=false)
    x_min_local = numerical_params.x_min_local
    Lx_local = numerical_params.Lx_local
    Ly = numerical_params.Ly
    R = numerical_params.R
    extended_Lx_local = SINGLE_RANK ? Lx_local : Lx_local + 2 * R

    num_cells_x = floor(Int32, max(1, extended_Lx_local / R))
    num_cells_y = floor(Int32, max(1, Ly / R))

    cell_size_x = Float32(extended_Lx_local / num_cells_x)
    cell_size_y = Float32(Ly / num_cells_y)

    inv_cell_size_x = 1.0f0 / cell_size_x
    inv_cell_size_y = 1.0f0 / cell_size_y

    num_cells = num_cells_x * num_cells_y
    num_cells >= typemax(Int32) && error("Too many cells to resolve with Int32")

    x_max_local = x_min_local + Lx_local
    x_min_cells = SINGLE_RANK ? x_min_local : x_min_local - R
    x_max_cells = SINGLE_RANK ? x_max_local : x_max_local + R

    return CellListParams(
        Float32(x_min_cells), Float32(x_max_cells),
        Float32(x_min_local), Float32(x_max_local),
        inv_cell_size_x, inv_cell_size_y,
        num_cells_x, num_cells_y, num_cells)
end #function


# --------- Cell list data structures --------- #

struct CellList
    # Per-particle (length N)
    cell_indices::CuVector{Int32} #Cell index of each particle; 1-based
    sorted_particles::CuVector{Particle} #Particles sorted by cell index
    perm::CuVector{Int32} #Permutation from sorted particle indices to absolute ones

    # Per-cell (length num_cells)
    cell_counts::CuVector{Int32} #Number of particles in each cell
    cell_starts::CuVector{Int32} #First index in sorted particles for cell c
    cell_starts_scratch::CuVector{Int32} #Copy of cell_starts, used during sorting

    row_major_to_hilbert::CuVector{Int32} #Look-up table to map from row-major to hilbert indices

    occupied_cells::CuVector{Int32} #Indices of non-empty cells; length == num_cells
    num_occupied::CuVector{Int32} #Scalar counter; length == 1

    cell_neighbours::CuArray{Int32,2}   # size == (9, ncells)
end #struct

CellList(cell_list_params::CellListParams, rm2h, h2rm, N) = CellList(
    CUDA.zeros(Int32, N),
    CuVector{Particle}(undef, N),
    CUDA.zeros(Int32, N),
    CUDA.zeros(Int32, cell_list_params.num_cells),
    CUDA.zeros(Int32, cell_list_params.num_cells),
    CUDA.zeros(Int32, cell_list_params.num_cells),
    CuArray(rm2h),
    CUDA.zeros(Int32, cell_list_params.num_cells),
    CUDA.zeros(Int32, 1),
    initialise_cell_neighbours(cell_list_params, rm2h, h2rm)
)

# --------- Cell List Functions --------- #

function initialise_cell_neighbours(cell_list_params::CellListParams, rm2h, h2rm)
    nx = cell_list_params.num_cells_x
    ny = cell_list_params.num_cells_y
    n = cell_list_params.num_cells
    cell_neighbours = zeros(Int32, 9, n)

    for h_idx in 1:n
        rm_idx = h2rm[h_idx] - 1
        cx = Int32(rm_idx % nx)
        cy = Int32(rm_idx ÷ nx)

        nghbr = Int32(1)
        for Δy in -Int32(1):Int32(1)
            for Δx in -Int32(1):Int32(1)
                nghbr_xidx = mod(cx + Δx, nx)
                nghbr_yidx = mod(cy + Δy, ny)
                nghbr_rm = 1 + nghbr_xidx + nghbr_yidx * nx
                cell_neighbours[nghbr, h_idx] = rm2h[nghbr_rm]
                nghbr += 1
            end #for Δx
        end #for Δy
    end #for h_idx

    return CuArray(cell_neighbours)
end #function

@inline function get_cell_ID(x, y, x_min_cells, num_cells_x, num_cells_y, inv_cell_size_x, inv_cell_size_y, rm2h)
    #Assume particle is inside cell list domain (so ghosts must be correctly wrapped already)
    x_idx = clamp(Int32(floor((x - x_min_cells) * inv_cell_size_x)), Int32(0), num_cells_x - Int32(1))
    y_idx = clamp(Int32(floor(y * inv_cell_size_y)), Int32(0), num_cells_y - Int32(1))
    rm_idx = Int32(1) + x_idx + num_cells_x * y_idx #1-based

    return rm2h[rm_idx]
end #function
