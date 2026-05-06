using CUDA

# --------- Data structures ---------

struct CellListParams
    x_start::Float32
    x_end::Float32
    box_size_x::Float32
    box_size_y::Float32
    num_boxes_x::Int32
    num_boxes_y::Int32
    num_boxes::Int32
end

function CellListParams(x_start::Real, Lx_local::Real, Ly::Real, R::Real)
    num_boxes_x = floor(Int32, max(1, Lx_local / R))
    num_boxes_y = floor(Int32, max(1, Ly / R))
    box_size_x = Float32(Lx_local / num_boxes_x)
    box_size_y = Float32(Ly / num_boxes_y)
    num_boxes = num_boxes_x * num_boxes_y
    x_end = x_start + Lx_local
    return CellListParams(Float32(x_start), Float32(x_end), box_size_x, box_size_y, num_boxes_x, num_boxes_y, num_boxes)
end

# --------- Cell List Functions ---------

function get_cell_ID(r, params::CellListParams)
    if r[1] > params.x_end #We're on rank 0 and r has wrapped round to the end of the domain
        x_idx = Int32(1)
    elseif r[1] < params.x_start #We're on rank nprocs-1 and r has wrapped round to the start of the domain
        x_idx = params.num_boxes_x
    else
        x_idx = ceil(Int32, (r[1] - params.x_start) / params.box_size_x)
    end #if r[1]
    y_idx = ceil(Int32, r[2] / params.box_size_y)
    x_idx = clamp(x_idx, 1, params.num_boxes_x)
    y_idx = clamp(y_idx, 1, params.num_boxes_y)
    return Int32(x_idx + params.num_boxes_x * (y_idx - 1))
end

function scalar_to_vector_cell_ID(sID::Int32, num_boxes_x::Int32, num_boxes_y::Int32)
    vID = Vector{Int32}(undef, 2)
    #x component:
    if sID % num_boxes_x != 0
        vID[1] = sID % num_boxes_x
    else
        vID[1] = num_boxes_x
    end #if
    #y component:
    for j = 1:num_boxes_y
        if (j - 1) * num_boxes_x < sID <= j * num_boxes_x
            vID[2] = j
            break
        end #if
    end #for j
    return Int32.(vID)
end #function

function vector_to_scalar_cell_ID(vID::Vector{Int32}, num_boxes_x::Int32, num_boxes_y::Int32)
    #Apply PBCs (for neighbours outside the grid)
    if vID[2] == num_boxes_y + 1 #If above box, correct down
        true_vID_y = 1
    elseif vID[2] == 0 #If below box, correct up
        true_vID_y = num_boxes_y
    else
        true_vID_y = vID[2]
    end #if
    if vID[1] == num_boxes_x + 1 #If to right, correct left
        true_vID_x = 1
    elseif vID[1] == 0 #If to left, correct right
        true_vID_x = num_boxes_x
    else
        true_vID_x = vID[1]
    end #if
    return Int32(true_vID_x + num_boxes_x * (true_vID_y - 1))
end #function

function build_cell_neighbours(params::CellListParams)
    #Construct list of neighbours for each box
    box_neighbours = Vector{Int32}[Vector{Int32}(undef, 8) for i = 1:params.num_boxes_x*params.num_boxes_y]
    for sID = 1:params.num_boxes_x*params.num_boxes_y
        vID = scalar_to_vector_cell_ID(Int32(sID), params.num_boxes_x, params.num_boxes_y)
        #Box above
        box_neighbours[sID][1] = vector_to_scalar_cell_ID(vID + Int32.([0, -1]), params.num_boxes_x, params.num_boxes_y)
        #Box top right
        box_neighbours[sID][2] = vector_to_scalar_cell_ID(vID + Int32.([1, -1]), params.num_boxes_x, params.num_boxes_y)
        #Box right
        box_neighbours[sID][3] = vector_to_scalar_cell_ID(vID + Int32.([1, 0]), params.num_boxes_x, params.num_boxes_y)
        #Box bottom right
        box_neighbours[sID][4] = vector_to_scalar_cell_ID(vID + Int32.([1, 1]), params.num_boxes_x, params.num_boxes_y)
        #Box below
        box_neighbours[sID][5] = vector_to_scalar_cell_ID(vID + Int32.([0, 1]), params.num_boxes_x, params.num_boxes_y)
        #Box bottom left
        box_neighbours[sID][6] = vector_to_scalar_cell_ID(vID + Int32.([-1, 1]), params.num_boxes_x, params.num_boxes_y)
        #Box left
        box_neighbours[sID][7] = vector_to_scalar_cell_ID(vID + Int32.([-1, 0]), params.num_boxes_x, params.num_boxes_y)
        #Box top left
        box_neighbours[sID][8] = vector_to_scalar_cell_ID(vID + Int32.([-1, -1]), params.num_boxes_x, params.num_boxes_y)
    end #for
    return box_neighbours
end #function
