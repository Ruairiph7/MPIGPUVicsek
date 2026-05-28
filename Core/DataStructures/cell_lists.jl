# --------- Cell list parameters --------- #

struct CellListParams
    x_min_cells::Float64 #Start of cell lists in x
    x_max_cells::Float64 #End of cell lists in x
    x_min_local::Float64 #Start of local domain in x
    x_max_local::Float64 #End of local domain in x
    inv_cell_size_x::Float64 #1/cell_size_x - used for get_cell_ID
    inv_cell_size_y::Float64 #1/cell_size_y - used for get_cell_ID
    num_cells_x::Int64
    num_cells_y::Int64
    num_cells::Int64
end #struct

function CellListParams(numerical_params; SINGLE_RANK::Bool=false)
    x_min_local = numerical_params.x_min_local
    Lx_local = numerical_params.Lx_local
    Ly = numerical_params.Ly
    R = numerical_params.R
    extended_Lx_local = SINGLE_RANK ? Lx_local : Lx_local + 2 * R

    num_cells_x = floor(Int64, max(1, extended_Lx_local / R))
    num_cells_y = floor(Int64, max(1, Ly / R))

    cell_size_x = Float64(extended_Lx_local / num_cells_x)
    cell_size_y = Float64(Ly / num_cells_y)

    inv_cell_size_x = 1.0 / cell_size_x
    inv_cell_size_y = 1.0 / cell_size_y

    num_cells = num_cells_x * num_cells_y
    num_cells >= typemax(Int64) && error("Too many cells to resolve with Int64")

    x_max_local = x_min_local + Lx_local
    x_min_cells = SINGLE_RANK ? x_min_local : x_min_local - R
    x_max_cells = SINGLE_RANK ? x_max_local : x_max_local + R

    return CellListParams(
        Float64(x_min_cells), Float64(x_max_cells),
        Float64(x_min_local), Float64(x_max_local),
        inv_cell_size_x, inv_cell_size_y,
        num_cells_x, num_cells_y, num_cells)
end #function


# --------- Cell list data structures --------- #

struct CellList
    # Per-particle (length N)
    cell_indices::CuVector{Int64} #Cell index of each particle; 1-based
    sorted_particles::CuVector{Particle} #Particles sorted by cell index
    perm::CuVector{Int64} #Permutation from sorted particle indices to absolute ones

    # Per-cell (length num_cells)
    cell_counts::CuVector{Int64} #Number of particles in each cell
    cell_starts::CuVector{Int64} #First index in sorted particles for cell c
    cell_starts_scratch::CuVector{Int64} #Copy of cell_starts, used during sorting

    occupied_cells::CuVector{Int64} #Indices of non-empty cells; length == num_cells
    num_occupied::CuVector{Int64} #Scalar counter; length == 1

    cell_neighbours::CuArray{Int64,2}   # size == (9, ncells)
end #struct

CellList(cell_list_params::CellListParams, N) = CellList(
    CUDA.zeros(Int64, N),
    CuVector{Particle}(undef, N),
    CUDA.zeros(Int64, N),
    CUDA.zeros(Int64, cell_list_params.num_cells),
    CUDA.zeros(Int64, cell_list_params.num_cells),
    CUDA.zeros(Int64, cell_list_params.num_cells),
    CUDA.zeros(Int64, cell_list_params.num_cells),
    CUDA.zeros(Int64, 1),
    initialise_cell_neighbours(cell_list_params)
)

# --------- Cell List Functions --------- #

function initialise_cell_neighbours(cell_list_params::CellListParams)
    nx = cell_list_params.num_cells_x
    ny = cell_list_params.num_cells_y
    n = cell_list_params.num_cells
    cell_neighbours = zeros(Int64, 9, n)
    for yidx in Int64(1):ny
        for xidx in Int64(1):nx
            cell = xidx + nx * (yidx - Int64(1))
            nghbr = Int64(1)
            for Δy in -Int64(1):Int64(1)
                for Δx in -Int64(1):Int64(1)
                    nghbr_xidx = mod1(xidx + Δx, nx)
                    nghbr_yidx = mod1(yidx + Δy, ny)
                    cell_neighbours[nghbr, cell] = nghbr_xidx + nx * (nghbr_yidx - Int64(1))
                    nghbr += 1
                end #for Δx
            end #for Δy
        end #for xidx
    end #for yidx
    return CuArray(cell_neighbours)
end #function

@inline function get_cell_ID(x, y, x_min_cells, num_cells_x, num_cells_y, inv_cell_size_x, inv_cell_size_y)
    #Assume particle is inside cell list domain (so ghosts must be correctly wrapped already)
    x_idx = clamp(Int64(floor((x - x_min_cells) * inv_cell_size_x)), Int64(0), num_cells_x - Int64(1))
    y_idx = clamp(Int64(floor(y * inv_cell_size_y)), Int64(0), num_cells_y - Int64(1))
    return Int64(1) + x_idx + num_cells_x * y_idx #1-based
end #function
