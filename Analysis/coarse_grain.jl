using ImageFiltering
using LinearAlgebra
using CairoMakie

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

function buildCellNeighbours(params::CellListParams)
    #Construct list of neighbours for each box
    boxNeighbours = Vector{Int64}[Vector{Int64}(undef, 9) for i = 1:params.numBoxes]
    for sID = 1:params.numBoxes
        vID = scalarToVectorBoxID(sID, params.numBoxesX, params.numBoxesY)
        #Itself
        boxNeighbours[sID][1] = vectorToScalarBoxID(vID, params.numBoxesX, params.numBoxesY)
        #Box above
        boxNeighbours[sID][2] = vectorToScalarBoxID(vID + [0, -1], params.numBoxesX, params.numBoxesY)
        #Box top right
        boxNeighbours[sID][3] = vectorToScalarBoxID(vID + [1, -1], params.numBoxesX, params.numBoxesY)
        #Box right
        boxNeighbours[sID][4] = vectorToScalarBoxID(vID + [1, 0], params.numBoxesX, params.numBoxesY)
        #Box bottom right
        boxNeighbours[sID][5] = vectorToScalarBoxID(vID + [1, 1], params.numBoxesX, params.numBoxesY)
        #Box below
        boxNeighbours[sID][6] = vectorToScalarBoxID(vID + [0, 1], params.numBoxesX, params.numBoxesY)
        #Box bottom left
        boxNeighbours[sID][7] = vectorToScalarBoxID(vID + [-1, 1], params.numBoxesX, params.numBoxesY)
        #Box left
        boxNeighbours[sID][8] = vectorToScalarBoxID(vID + [-1, 0], params.numBoxesX, params.numBoxesY)
        #Box top left
        boxNeighbours[sID][9] = vectorToScalarBoxID(vID + [-1, -1], params.numBoxesX, params.numBoxesY)
    end #for
    return boxNeighbours
end #function

function newCellCheck(params::CellListParams, sID::Int64, x::Real, y::Real, prev_x::Real, prev_y::Real)
    #Flag if the particle has changed cell, and store the old cell 
    oldBox = getScalarBoxID(params,prev_x,prev_y)
    newBox = getScalarBoxID(params,x,y)
    return (oldBox != newBox) ? sID : 0
end

function updateCellLists!(cellLists::CellLists, params::CellListParams, xs::Vector{<:Real}, ys::Vector{<:Real}, changedBoxFlags::Vector{Int64})
    #Update cell lists
    # changedBoxIndices = ThreadsX.findall(!iszero, changedBoxFlags)
    changedBoxIndices = findall(!iszero, changedBoxFlags)
    for particleIdx = changedBoxIndices
        oldBox = changedBoxFlags[particleIdx]
        newBox = getScalarBoxID(params, xs[particleIdx], ys[particleIdx])
        removeParticle!(cellLists, particleIdx, oldBox)
        addParticle!(cellLists, particleIdx, newBox)
    end #for particleIdx
end


function removeParticle!(cellLists::CellLists, i::Int64, b::Int64)
    #Remove particle i from it's box
    ## b = current box ID for particle i
    nodeBelow = cellLists.downList[i]
    nodeAbove = cellLists.upList[i]
    if (nodeBelow != 0) && (nodeAbove != 0) #Node is in the middle
        cellLists.downList[nodeAbove] = nodeBelow
        cellLists.upList[nodeBelow] = nodeAbove
    elseif (nodeBelow != 0) && (nodeAbove == 0) #Node is at the top with more below
        cellLists.listHeads[b] = nodeBelow
        cellLists.upList[nodeBelow] = 0
    elseif (nodeBelow == 0) && (nodeAbove != 0) #Node is at the bottom with more above
        cellLists.listTails[b] = nodeAbove
        cellLists.downList[nodeAbove] = 0
    else                                        #It is the only node
        cellLists.listHeads[b] = 0
        cellLists.listTails[b] = 0
    end #if
    return nothing
end #function

function addParticle!(cellLists::CellLists, i::Int64, b::Int64)
    #Add particle i to the bottom of a new box
    ## b = new box ID for particle i
    oldTail = cellLists.listTails[b]
    if oldTail != 0                  #There was a particle in the box before
        cellLists.downList[oldTail] = i
        cellLists.downList[i] = 0
        cellLists.upList[i] = oldTail
        cellLists.listTails[b] = i
    else
        cellLists.listHeads[b] = i
        cellLists.listTails[b] = i
        cellLists.downList[i] = 0
        cellLists.upList[i] = 0
    end #if
    return nothing
end #function

function getScalarBoxID(cellListParams::CellListParams, x::Real, y::Real)
    ix = clamp(ceil(Int,x/cellListParams.boxSizeX), 1, cellListParams.numBoxesX)
    iy = clamp(ceil(Int,y/cellListParams.boxSizeY), 1, cellListParams.numBoxesY)
    return ix + cellListParams.numBoxesX * (iy-1)
end #function

function scalarToVectorBoxID(sID::Int, numBoxesX::Int, numBoxesY::Int)
    vID = Vector{Int64}(undef, 2)
    #x component:
    if sID % numBoxesX != 0
        vID[1] = sID % numBoxesX
    else
        vID[1] = numBoxesX
    end #if
    #y component:
    for j = 1:numBoxesY
        if (j - 1) * numBoxesX < sID <= j * numBoxesX
            vID[2] = j
            break
        end #if
    end #for j
    return vID
end #function

function vectorToScalarBoxID(vID::Vector{<:Int}, numBoxesX::Int, numBoxesY::Int)
    #Apply PBCs (for neighbours outside the grid)
    if vID[2] == numBoxesY + 1 #If above box, correct down
        true_vID_y = 1
    elseif vID[2] == 0 #If below box, correct up
        true_vID_y = numBoxesY
    else
        true_vID_y = vID[2]
    end #if
    if vID[1] == numBoxesX + 1 #If to right, correct left
        true_vID_x = 1
    elseif vID[1] == 0 #If to left, correct right
        true_vID_x = numBoxesX
    else
        true_vID_x = vID[1]
    end #if
    return true_vID_x + numBoxesX * (true_vID_y - 1)
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
Coarse grain density and magnetisation fields given grid and θs
"""
function _coarse_grain(cell_list_params::CellListParams, cell_lists::CellLists, θs::Vector{<:Real})
    num_boxes_x = cell_list_params.numBoxesX
    num_boxes_y = cell_list_params.numBoxesY
    box_size_x = cell_list_params.boxSizeX
    box_size_y = cell_list_params.boxSizeY
    box_area = box_size_x * box_size_y

    #Arrays to store fields
    density_array = zeros(num_boxes_x, num_boxes_y)
    vec_magnetisation_array = [zeros(2) for i = 1:num_boxes_x, j = 1:num_boxes_y]


    # Loop through boxes
    Threads.@threads for i = 1:num_boxes_x
        for j = 1:num_boxes_y
            box_id = i + (j - 1) * num_boxes_x

            this_box_num_particles = 0
            this_box_mx = 0
            this_box_my = 0

            # Loop through this box's cell list
            particle_id = cell_lists.listHeads[box_id]
            while particle_id != 0
                this_box_num_particles += 1
                this_box_mx += cos(θs[particle_id])
                this_box_my += sin(θs[particle_id])
                # Move down the list
                particle_id = cell_lists.downList[particle_id]
            end #while
            if this_box_num_particles != 0
                this_box_mx = this_box_mx ./ this_box_num_particles
                this_box_my = this_box_my ./ this_box_num_particles
            end #if

            density_array[i, j] = this_box_num_particles / box_area #Calculate density
            vec_magnetisation_array[i, j] = [this_box_mx, this_box_my]
        end #for j
    end #for i

    return density_array, vec_magnetisation_array
end #function


"""
Coarse grain density and magnetisation and nematic S fields given grid and θs
"""
function _coarse_grain_nematic_too(cell_list_params::CellListParams, cell_lists::CellLists, θs::Vector{<:Real})
    num_boxes_x = cell_list_params.numBoxesX
    num_boxes_y = cell_list_params.numBoxesY
    box_size_x = cell_list_params.boxSizeX
    box_size_y = cell_list_params.boxSizeY
    box_area = box_size_x * box_size_y

    #Arrays to store fields
    density_array = zeros(num_boxes_x, num_boxes_y)
    vec_magnetisation_array = [zeros(2) for i = 1:num_boxes_x, j = 1:num_boxes_y]
    vec_nematic_array = [zeros(2) for i = 1:num_boxes_x, j = 1:num_boxes_y]


    # Loop through boxes
    Threads.@threads for i = 1:num_boxes_x
        for j = 1:num_boxes_y
            box_id = i + (j - 1) * num_boxes_x

            this_box_num_particles = 0
            this_box_mx = 0
            this_box_my = 0
            this_box_Sx = 0
            this_box_Sy = 0

            # Loop through this box's cell list
            particle_id = cell_lists.listHeads[box_id]
            while particle_id != 0
                this_box_num_particles += 1
                this_box_mx += cos(θs[particle_id])
                this_box_my += sin(θs[particle_id])
                this_box_Sx += cos(2 * θs[particle_id])
                this_box_Sy += sin(2 * θs[particle_id])
                # Move down the list
                particle_id = cell_lists.downList[particle_id]
            end #while
            if this_box_num_particles != 0
                this_box_mx = this_box_mx ./ this_box_num_particles
                this_box_my = this_box_my ./ this_box_num_particles
                this_box_Sx = this_box_Sx ./ this_box_num_particles
                this_box_Sy = this_box_Sy ./ this_box_num_particles
            end #if

            density_array[i, j] = this_box_num_particles / box_area #Calculate density
            vec_magnetisation_array[i, j] = [this_box_mx, this_box_my]
            vec_nematic_array[i, j] = [this_box_Sx, this_box_Sy]
        end #for j
    end #for i

    return density_array, vec_magnetisation_array, vec_nematic_array
end #function


"""
Coarse grain the positions into density, magnetisation and orientation fields.
"""
function coarse_grain(xs::Vector{<:Real}, ys::Vector{<:Real}, θs::Vector{<:Real}, Lx::Int, Ly::Int, cell_width::Real; apply_filter=false, filter_strength=2, return_box_size=false)
    cell_list_params, cell_lists = _initialise_grid(xs, ys, Lx, Ly, cell_width)
    densities, vec_magnetisations = _coarse_grain(cell_list_params, cell_lists, θs)

    if apply_filter
        vec_magnetisations_x = [vec_magnetisations[i, j][1] for i = 1:size(densities)[1], j = 1:size(densities)[2]]
        vec_magnetisations_y = [vec_magnetisations[i, j][2] for i = 1:size(densities)[1], j = 1:size(densities)[2]]

        filtered_densities = imfilter(densities, Kernel.gaussian(filter_strength), "circular")
        filtered_vec_magnetisations_x = imfilter(vec_magnetisations_x, Kernel.gaussian(filter_strength), "circular")
        filtered_vec_magnetisations_y = imfilter(vec_magnetisations_y, Kernel.gaussian(filter_strength), "circular")

        filtered_magnetisations = similar(densities)
        filtered_orientations = similar(densities)
        for i = 1:cell_list_params.numBoxesX
            for j = 1:cell_list_params.numBoxesY
                filtered_magnetisations[i, j] = sqrt(filtered_vec_magnetisations_x[i, j]^2 + filtered_vec_magnetisations_y[i, j]^2)
                filtered_orientations[i, j] = atan(filtered_vec_magnetisations_y[i, j], filtered_vec_magnetisations_x[i, j])
            end #for j
        end #for i
        if return_box_size
            return filtered_densities, filtered_magnetisations, filtered_orientations, cell_list_params.boxSizeX, cell_list_params.boxSizeY
        else
            return filtered_densities, filtered_magnetisations, filtered_orientations
        end #if
    else
        magnetisations = similar(densities)
        orientations = similar(densities)
        for i = 1:cell_list_params.numBoxesX
            for j = 1:cell_list_params.numBoxesY
                magnetisations[i, j] = sqrt(vec_magnetisations[i, j][1]^2 + vec_magnetisations[i, j][2]^2)
                orientations[i, j] = atan(vec_magnetisations[i, j][2], vec_magnetisations[i, j][1])
                vec_magnetisations[i, j] == [0, 0] && (orientations[i, j] = NaN)
            end #for j
        end #for i
        if return_box_size
            return densities, magnetisations, orientations, cell_list_params.boxSizeX, cell_list_params.boxSizeY
        else
            return densities, magnetisations, orientations
        end #if
    end #if

end #function

"""
Coarse grain the positions into density, magnetisation, orientation and nematic S fields.
"""
function coarse_grain_nematic_too(xs::Vector{<:Real}, ys::Vector{<:Real}, θs::Vector{<:Real}, Lx::Int, Ly::Int, cell_width::Real; apply_filter=false, filter_strength=2, return_box_size=false)
    cell_list_params, cell_lists = _initialise_grid(xs, ys, Lx, Ly, cell_width)
    densities, vec_magnetisations, vec_Ss = _coarse_grain_nematic_too(cell_list_params, cell_lists, θs)

    if apply_filter
        vec_magnetisations_x = [vec_magnetisations[i, j][1] for i = 1:size(densities)[1], j = 1:size(densities)[2]]
        vec_magnetisations_y = [vec_magnetisations[i, j][2] for i = 1:size(densities)[1], j = 1:size(densities)[2]]
        vec_Ss_x = [vec_Ss[i, j][1] for i = 1:size(densities)[1], j = 1:size(densities)[2]]
        vec_Ss_y = [vec_Ss[i, j][2] for i = 1:size(densities)[1], j = 1:size(densities)[2]]

        filtered_densities = imfilter(densities, Kernel.gaussian(filter_strength), "circular")
        filtered_vec_magnetisations_x = imfilter(vec_magnetisations_x, Kernel.gaussian(filter_strength), "circular")
        filtered_vec_magnetisations_y = imfilter(vec_magnetisations_y, Kernel.gaussian(filter_strength), "circular")
        filtered_vec_Ss_x = imfilter(vec_Ss_x, Kernel.gaussian(filter_strength), "circular")
        filtered_vec_Ss_y = imfilter(vec_Ss_y, Kernel.gaussian(filter_strength), "circular")

        filtered_magnetisations = similar(densities)
        filtered_orientations = similar(densities)
        filtered_Ss = similar(densities)
        for i = 1:cell_list_params.numBoxesX
            for j = 1:cell_list_params.numBoxesY
                filtered_magnetisations[i, j] = sqrt(filtered_vec_magnetisations_x[i, j]^2 + filtered_vec_magnetisations_y[i, j]^2)
                filtered_orientations[i, j] = atan(filtered_vec_magnetisations_y[i, j], filtered_vec_magnetisations_x[i, j])
                filtered_Ss[i, j] = sqrt(filtered_vec_Ss_x[i, j]^2 + filtered_vec_Ss_y[i, j]^2)
            end #for j
        end #for i
        if return_box_size
            return filtered_densities, filtered_magnetisations, filtered_orientations, filtered_Ss, cell_list_params.boxSizeX, cell_list_params.boxSizeY
        else
            return filtered_densities, filtered_magnetisations, filtered_orientations, filtered_Ss
        end #if
    else
        magnetisations = similar(densities)
        orientations = similar(densities)
        Ss = similar(densities)
        for i = 1:cell_list_params.numBoxesX
            for j = 1:cell_list_params.numBoxesY
                magnetisations[i, j] = sqrt(vec_magnetisations[i, j][1]^2 + vec_magnetisations[i, j][2]^2)
                orientations[i, j] = atan(vec_magnetisations[i, j][2], vec_magnetisations[i, j][1])
                vec_magnetisations[i, j] == [0, 0] && (orientations[i, j] = NaN)
                Ss[i, j] = sqrt(vec_Ss[i, j][1]^2 + vec_Ss[i, j][2]^2)
            end #for j
        end #for i
        if return_box_size
            return densities, magnetisations, orientations, Ss, cell_list_params.boxSizeX, cell_list_params.boxSizeY
        else
            return densities, magnetisations, orientations, Ss
        end #if
    end #if

end #function


###########################################################################
###########################################################################
###########################################################################

###########################################################################
# Plotting
###########################################################################

function plot_fields(c_array::Matrix{<:Real}, m_array::Matrix{<:Real}, d_array::Matrix{<:Real})

    # WARN: Maybe transpose the arrays to get the correct orientations
    fig = Figure()
    c_subgrid = GridLayout()
    m_subgrid = GridLayout()
    d_subgrid = GridLayout()
    fig.layout[1, 1] = c_subgrid
    fig.layout[1, 2] = m_subgrid
    fig.layout[2, 1] = d_subgrid
    rowsize!(fig.layout, 1, Relative(4 / 7))

    #Colorwheel
    w_ax = PolarAxis(fig[2, 2])
    hidedecorations!(w_ax)
    rs = range(0, 1, 300)
    ϕs = range(0, 2π, 600)
    ϕmat = [ϕ for r in rs, ϕ in ϕs]
    rmat = [r for r in rs, ϕ in ϕs]
    w_ax.rlimits = (0, 1)
    surface!(w_ax, ϕmat, rmat, zeros(300, 600), color=ϕmat, colormap=:hsv, shading=false)

    #Concentration
    c_ax = Axis(c_subgrid[1, 1], aspect=DataAspect(), title="c")
    hidedecorations!(c_ax, label=false)
    c_hm = heatmap!(c_ax, c_array, colormap=:balance)
    c_cb = Colorbar(c_subgrid[2, 1], c_hm, vertical=false)

    #|Magnetisation|
    m_ax = Axis(m_subgrid[1, 1], aspect=DataAspect(), title="|m|")
    hidedecorations!(m_ax, label=false)
    m_hm = heatmap!(m_ax, m_array, colormap=:plasma, nan_color=:black)
    m_cb = Colorbar(m_subgrid[2, 1], m_hm, vertical=false)

    #Direction [arg(Magnetisation)]
    d_ax = Axis(d_subgrid[1, 1], aspect=DataAspect(), title="arg(m)")
    hidedecorations!(d_ax, label=false)
    d_hm = heatmap!(d_ax, mod.(d_array, 2π), colormap=:hsv, colorrange=(0, 2π), nan_color=:black)

    return fig
end #function

function plot_fields(xs::Vector{<:Real}, ys::Vector{<:Real}, θs::Vector{<:Real}, Lx::Int, Ly::Int, cell_width::Real; apply_filter=false, filter_strength=2)
    c_array, m_array, d_array = coarse_grain(xs, ys, θs, Lx, Ly, cell_width, apply_filter=apply_filter, filter_strength=filter_strength)
    fig = plot_fields(c_array, m_array, d_array)
    return fig
end #function

function plot_fields_nematic_too(c_array::Matrix{<:Real}, m_array::Matrix{<:Real}, d_array::Matrix{<:Real}, S_array::Matrix{<:Real})

    # WARN: Maybe transpose the arrays to get the correct orientations
    fig = Figure()
    c_subgrid = GridLayout()
    m_subgrid = GridLayout()
    d_subgrid = GridLayout()
    S_subgrid = GridLayout()
    fig.layout[1, 1] = c_subgrid
    fig.layout[1, 2] = m_subgrid
    fig.layout[2, 1] = d_subgrid
    fig.layout[2, 2] = S_subgrid
    rowsize!(fig.layout, 1, Relative(4 / 7))

    #Concentration
    c_ax = Axis(c_subgrid[1, 1], aspect=DataAspect(), title="c")
    hidedecorations!(c_ax, label=false)
    c_hm = heatmap!(c_ax, c_array, colormap=:balance)
    c_cb = Colorbar(c_subgrid[2, 1], c_hm, vertical=false)

    #|Magnetisation|
    m_ax = Axis(m_subgrid[1, 1], aspect=DataAspect(), title="|m|")
    hidedecorations!(m_ax, label=false)
    m_hm = heatmap!(m_ax, m_array, colormap=:plasma, nan_color=:black)
    m_cb = Colorbar(m_subgrid[2, 1], m_hm, vertical=false)

    #Direction [arg(Magnetisation)]
    d_ax = Axis(d_subgrid[1, 1], aspect=DataAspect(), title="arg(m)")
    hidedecorations!(d_ax, label=false)
    d_hm = heatmap!(d_ax, mod.(d_array, 2π), colormap=:hsv, colorrange=(0, 2π), nan_color=:black)

    #S
    S_ax = Axis(S_subgrid[1, 1], aspect=DataAspect(), title="S")
    hidedecorations!(S_ax, label=false)
    S_hm = heatmap!(S_ax, S_array, colormap=:plasma, nan_color=:black)
    S_cb = Colorbar(S_subgrid[2, 1], S_hm, vertical=false)

    return fig
end #function

function plot_fields_nematic_too(xs::Vector{<:Real}, ys::Vector{<:Real}, θs::Vector{<:Real}, Lx::Int, Ly::Int, cell_width::Real; apply_filter=false, filter_strength=2)
    c_array, m_array, d_array, S_array = coarse_grain_nematic_too(xs, ys, θs, Lx, Ly, cell_width, apply_filter=apply_filter, filter_strength=filter_strength)
    fig = plot_fields_nematic_too(c_array, m_array, d_array, S_array)
    return fig
end #function

