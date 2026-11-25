using ImageFiltering
using LinearAlgebra

###########################################################################
# Linked-Cell Lists
###########################################################################

struct CellListParams{T<:Real}
    boxSizeX::T
    boxSizeY::T
    numBoxesX::Int64
    numBoxesY::Int64
    numBoxes::Int64
end
function CellListParams(Lx::Real, Ly::Real, R::Real)
    numBoxesX = floor(Int64, Lx / R)
    numBoxesY = floor(Int64, Ly / R)
    numBoxes = numBoxesX * numBoxesY
    boxSizeX = Lx / numBoxesX
    boxSizeY = Ly / numBoxesY
    # (boxSize != Ly / numBoxesY) && error("Cell boxes must be square -- Ly/numBoxesY != Lx/numBoxesX")
    return CellListParams{typeof(boxSizeX)}(boxSizeX, boxSizeY, numBoxesX, numBoxesY, numBoxes)
end

mutable struct CellLists
    listHeads::Vector{Int64}
    listTails::Vector{Int64}
    downList::Vector{Int64}
    upList::Vector{Int64}
    function CellLists(params::CellListParams, N::Int64)
        new(zeros(Int64, params.numBoxes), zeros(Int64, params.numBoxes), zeros(Int64, N), zeros(Int64, N))
    end
end

function buildCellLists!(cellLists::CellLists, cellListParams::CellListParams, xs::Vector{<:Real}, ys::Vector{<:Real})
    #Build downList "upwards"
    for i in eachindex(xs)
        b = getScalarBoxID(cellListParams, xs[i], ys[i])
        cellLists.downList[i] = cellLists.listHeads[b]
        cellLists.listHeads[b] = i
    end #for i
    #Build upList "downwards"
    for i = length(xs):-1:1
        b = getScalarBoxID(cellListParams, xs[i], ys[i])
        cellLists.upList[i] = cellLists.listTails[b]
        cellLists.listTails[b] = i
    end #for i
    return nothing
end

function getScalarBoxID(cellListParams::CellListParams, x::Real, y::Real)
    ix = clamp(ceil(Int, x / cellListParams.boxSizeX), 1, cellListParams.numBoxesX)
    iy = clamp(ceil(Int, y / cellListParams.boxSizeY), 1, cellListParams.numBoxesY)
    return ix + cellListParams.numBoxesX * (iy - 1)
end #function

###########################################################################
###########################################################################
###########################################################################

###########################################################################
# Coarse-graining
###########################################################################

"""
Set up a grid of cells with width "cell_width", store particles in cell lists
"""
function _initialise_grid(xs::Vector{<:Real}, ys::Vector{<:Real}, Lx::Int, Ly::Int, cell_width::Real)
    N = length(xs)
    # Set up cell lists to store which particles are in each cell in our grid
    cell_list_params = CellListParams(Lx, Ly, cell_width)
    cell_lists = CellLists(cell_list_params, N)
    buildCellLists!(cell_lists, cell_list_params, xs, ys)
    return cell_list_params, cell_lists
end #function

"""
Coarse grain density and polarisation fields given grid and θs
"""
function _coarse_grain(cell_list_params::CellListParams, cell_lists::CellLists, θs::Vector{<:Real})
    num_boxes_x = cell_list_params.numBoxesX
    num_boxes_y = cell_list_params.numBoxesY
    box_size_x = cell_list_params.boxSizeX
    box_size_y = cell_list_params.boxSizeY
    box_area = box_size_x * box_size_y

    #Arrays to store fields
    density_array = zeros(num_boxes_x, num_boxes_y)
    vec_polarisation_array = [zeros(2) for i = 1:num_boxes_x, j = 1:num_boxes_y]


    # Loop through boxes
    Threads.@threads for i = 1:num_boxes_x
        for j = 1:num_boxes_y
            box_id = i + (j - 1) * num_boxes_x

            this_box_num_particles = 0
            this_box_px = 0.0
            this_box_py = 0.0

            # Loop through this box's cell list
            particle_id = cell_lists.listHeads[box_id]
            while particle_id != 0
                this_box_num_particles += 1
                this_box_px += cos(θs[particle_id])
                this_box_py += sin(θs[particle_id])
                # Move down the list
                particle_id = cell_lists.downList[particle_id]
            end #while
            if this_box_num_particles != 0
                this_box_px = this_box_px ./ this_box_num_particles
                this_box_py = this_box_py ./ this_box_num_particles
            end #if

            this_box_density = this_box_num_particles / box_area
            density_array[i, j] = this_box_density
            vec_polarisation_array[i, j] = [this_box_px, this_box_py]
        end #for j
    end #for i

    return density_array, vec_polarisation_array
end #function


"""
Coarse grain density and polarisation and nematic S fields given grid and θs
"""
function _coarse_grain_nematic_too(cell_list_params::CellListParams, cell_lists::CellLists, θs::Vector{<:Real})
    num_boxes_x = cell_list_params.numBoxesX
    num_boxes_y = cell_list_params.numBoxesY
    box_size_x = cell_list_params.boxSizeX
    box_size_y = cell_list_params.boxSizeY
    box_area = box_size_x * box_size_y

    #Arrays to store fields
    density_array = zeros(num_boxes_x, num_boxes_y)
    vec_polarisation_array = [zeros(2) for i = 1:num_boxes_x, j = 1:num_boxes_y]
    vec_nematic_array = [zeros(2) for i = 1:num_boxes_x, j = 1:num_boxes_y]


    # Loop through boxes
    Threads.@threads for i = 1:num_boxes_x
        for j = 1:num_boxes_y
            box_id = i + (j - 1) * num_boxes_x

            this_box_num_particles = 0
            this_box_px = 0
            this_box_py = 0
            this_box_Sx = 0
            this_box_Sy = 0

            # Loop through this box's cell list
            particle_id = cell_lists.listHeads[box_id]
            while particle_id != 0
                this_box_num_particles += 1
                this_box_px += cos(θs[particle_id])
                this_box_py += sin(θs[particle_id])
                this_box_Sx += cos(2 * θs[particle_id])
                this_box_Sy += sin(2 * θs[particle_id])
                # Move down the list
                particle_id = cell_lists.downList[particle_id]
            end #while
            if this_box_num_particles != 0
                this_box_px = this_box_px ./ this_box_num_particles
                this_box_py = this_box_py ./ this_box_num_particles
                this_box_Sx = this_box_Sx ./ this_box_num_particles
                this_box_Sy = this_box_Sy ./ this_box_num_particles
            end #if

            this_box_density = this_box_num_particles / box_area
            density_array[i, j] = this_box_density
            vec_polarisation_array[i, j] = [this_box_px, this_box_py]
            vec_nematic_array[i, j] = [this_box_Sx, this_box_Sy]
        end #for j
    end #for i

    return density_array, vec_polarisation_array, vec_nematic_array
end #function


"""
Coarse grain the positions into density, polarisation and orientation fields.
"""
function coarse_grain(xs::Vector{<:Real}, ys::Vector{<:Real}, θs::Vector{<:Real}, Lx::Int, Ly::Int, cell_width::Real; apply_filter=false, filter_strength=2, return_box_size=false)
    cell_list_params, cell_lists = _initialise_grid(xs, ys, Lx, Ly, cell_width)
    densities, vec_polarisations = _coarse_grain(cell_list_params, cell_lists, θs)

    if apply_filter
        vec_polarisations_x = [vec_polarisations[i, j][1] for i = 1:size(densities)[1], j = 1:size(densities)[2]]
        vec_polarisations_y = [vec_polarisations[i, j][2] for i = 1:size(densities)[1], j = 1:size(densities)[2]]

        filtered_densities = imfilter(densities, Kernel.gaussian(filter_strength), "circular")
        filtered_vec_polarisations_x = imfilter(vec_polarisations_x, Kernel.gaussian(filter_strength), "circular")
        filtered_vec_polarisations_y = imfilter(vec_polarisations_y, Kernel.gaussian(filter_strength), "circular")

        filtered_polarisations = similar(densities)
        filtered_orientations = similar(densities)
        for i = 1:cell_list_params.numBoxesX
            for j = 1:cell_list_params.numBoxesY
                filtered_polarisations[i, j] = sqrt(filtered_vec_polarisations_x[i, j]^2 + filtered_vec_polarisations_y[i, j]^2)
                filtered_orientations[i, j] = atan(filtered_vec_polarisations_y[i, j], filtered_vec_polarisations_x[i, j])
            end #for j
        end #for i
        if return_box_size
            return filtered_densities, filtered_polarisations, filtered_orientations, cell_list_params.boxSizeX, cell_list_params.boxSizeY
        else
            return filtered_densities, filtered_polarisations, filtered_orientations
        end #if
    else
        polarisations = similar(densities)
        orientations = similar(densities)
        for i = 1:cell_list_params.numBoxesX
            for j = 1:cell_list_params.numBoxesY
                polarisations[i, j] = sqrt(vec_polarisations[i, j][1]^2 + vec_polarisations[i, j][2]^2)
                orientations[i, j] = atan(vec_polarisations[i, j][2], vec_polarisations[i, j][1])
                vec_polarisations[i, j] == [0, 0] && (orientations[i, j] = NaN)
            end #for j
        end #for i
        if return_box_size
            return densities, polarisations, orientations, cell_list_params.boxSizeX, cell_list_params.boxSizeY
        else
            return densities, polarisations, orientations
        end #if
    end #if

end #function

"""
Coarse grain the positions into density, polarisation, orientation and nematic S fields.
"""
function coarse_grain_nematic_too(xs::Vector{<:Real}, ys::Vector{<:Real}, θs::Vector{<:Real}, Lx::Int, Ly::Int, cell_width::Real; apply_filter=false, filter_strength=2, return_box_size=false)
    cell_list_params, cell_lists = _initialise_grid(xs, ys, Lx, Ly, cell_width)
    densities, vec_polarisations, vec_Ss = _coarse_grain_nematic_too(cell_list_params, cell_lists, θs)

    if apply_filter
        vec_polarisations_x = [vec_polarisations[i, j][1] for i = 1:size(densities)[1], j = 1:size(densities)[2]]
        vec_polarisations_y = [vec_polarisations[i, j][2] for i = 1:size(densities)[1], j = 1:size(densities)[2]]
        vec_Ss_x = [vec_Ss[i, j][1] for i = 1:size(densities)[1], j = 1:size(densities)[2]]
        vec_Ss_y = [vec_Ss[i, j][2] for i = 1:size(densities)[1], j = 1:size(densities)[2]]

        filtered_densities = imfilter(densities, Kernel.gaussian(filter_strength), "circular")
        filtered_vec_polarisations_x = imfilter(vec_polarisations_x, Kernel.gaussian(filter_strength), "circular")
        filtered_vec_polarisations_y = imfilter(vec_polarisations_y, Kernel.gaussian(filter_strength), "circular")
        filtered_vec_Ss_x = imfilter(vec_Ss_x, Kernel.gaussian(filter_strength), "circular")
        filtered_vec_Ss_y = imfilter(vec_Ss_y, Kernel.gaussian(filter_strength), "circular")

        filtered_polarisations = similar(densities)
        filtered_orientations = similar(densities)
        filtered_Ss = similar(densities)
        for i = 1:cell_list_params.numBoxesX
            for j = 1:cell_list_params.numBoxesY
                filtered_polarisations[i, j] = sqrt(filtered_vec_polarisations_x[i, j]^2 + filtered_vec_polarisations_y[i, j]^2)
                filtered_orientations[i, j] = atan(filtered_vec_polarisations_y[i, j], filtered_vec_polarisations_x[i, j])
                filtered_Ss[i, j] = sqrt(filtered_vec_Ss_x[i, j]^2 + filtered_vec_Ss_y[i, j]^2)
            end #for j
        end #for i
        if return_box_size
            return filtered_densities, filtered_polarisations, filtered_orientations, filtered_Ss, cell_list_params.boxSizeX, cell_list_params.boxSizeY
        else
            return filtered_densities, filtered_polarisations, filtered_orientations, filtered_Ss
        end #if
    else
        polarisations = similar(densities)
        orientations = similar(densities)
        Ss = similar(densities)
        for i = 1:cell_list_params.numBoxesX
            for j = 1:cell_list_params.numBoxesY
                polarisations[i, j] = sqrt(vec_polarisations[i, j][1]^2 + vec_polarisations[i, j][2]^2)
                orientations[i, j] = atan(vec_polarisations[i, j][2], vec_polarisations[i, j][1])
                vec_polarisations[i, j] == [0, 0] && (orientations[i, j] = NaN)
                Ss[i, j] = sqrt(vec_Ss[i, j][1]^2 + vec_Ss[i, j][2]^2)
            end #for j
        end #for i
        if return_box_size
            return densities, polarisations, orientations, Ss, cell_list_params.boxSizeX, cell_list_params.boxSizeY
        else
            return densities, polarisations, orientations, Ss
        end #if
    end #if

end #function
