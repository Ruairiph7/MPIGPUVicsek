# --------- Data structures ---------

struct CellListParams
    x_min_cells::Float32 #Start of cell lists in x
    x_max_cells::Float32 #End of cell lists in x
    x_min_local::Float32 #Start of local domain in x
    x_max_local::Float32 #End of local domain in x
    cell_size_x::Float32
    cell_size_y::Float32
    num_cells_x::Int32
    num_cells_y::Int32
    num_cells::Int32
end #struct

function CellListParams(x_min_local::Real, Lx_local::Real, Ly::Real, R_max::Real; SINGLE_RANK::Bool=false)
    extended_Lx_local = SINGLE_RANK ? Lx_local : Lx_local + 2 * R_max
    num_cells_x = floor(Int32, max(1, extended_Lx_local / R_max))
    num_cells_y = floor(Int32, max(1, Ly / R_max))
    cell_size_x = Float32(extended_Lx_local / num_cells_x)
    cell_size_y = Float32(Ly / num_cells_y)
    num_cells = num_cells_x * num_cells_y
    num_cells >= typemax(Int32) && error("Too many cells to resolve with Int32")
    x_max_local = x_min_local + Lx_local
    x_min_cells = SINGLE_RANK ? x_min_local : x_min_local - R_max
    x_max_cells = SINGLE_RANK ? x_max_local : x_max_local + R_max
    return CellListParams(Float32(x_min_cells), Float32(x_max_cells), Float32(x_min_local), Float32(x_max_local), cell_size_x, cell_size_y, num_cells_x, num_cells_y, num_cells)
end #function


# --------- Cell List Functions ---------

#Assume particle is inside cell list domain (so ghosts must be correctly wrapped already)
@inline function get_cell_ID(r, num_cells_x, num_cells_y, cell_size_x, cell_size_y)
    x_idx = min(floor(Int32, r[1] / cell_size_x), num_cells_x - Int32(1))
    y_idx = min(floor(Int32, r[2] / cell_size_y), num_cells_y - Int32(1))
    return Int32(1) + x_idx + num_cells_x * y_idx
end
