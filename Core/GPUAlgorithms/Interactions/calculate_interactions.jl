# --------- Store interactions in θ_updates --------- #
# NOTE: Tumbling comes later in the kernel for updating particles.

# Assign a workgroup to each occupied cell, iterate in batches to assign
# one thread per particle even when cell_count > workgroup_size.
#   - Also iterate through neighbour cells in tiles, again handles high cell_counts
#   - In each batch all neighbour tiles are traversed (so tiles are reloaded per batch)
#   - workgroup_size = tile_size, so all threads always participate in tile loading

@inline function F(θ::Float32, inv_πR²::Float32)
    return sin(θ) * inv_πR²
end #function

function calculate_interactions!(θ_updates, cells_data, cell_list_params, numerical_params)

    max_cell_count = maximum(cells_data.cell_counts)

    if max_cell_count < 256
        tile_size = 32
        workgroup_size = tile_size
        num_workgroups = cell_list_params.num_cells
        total_num_threads = workgroup_size * num_workgroups
        kernel! = calculate_interactions_kernel_32!(CUDABackend(), workgroup_size)
    elseif max_cell_count < 512
        tile_size = 64
        workgroup_size = tile_size
        num_workgroups = cell_list_params.num_cells
        total_num_threads = workgroup_size * num_workgroups
        kernel! = calculate_interactions_kernel_64!(CUDABackend(), workgroup_size)
    elseif max_cell_count < 1024
        tile_size = 128
        workgroup_size = tile_size
        num_workgroups = cell_list_params.num_cells
        total_num_threads = workgroup_size * num_workgroups
        kernel! = calculate_interactions_kernel_128!(CUDABackend(), workgroup_size)
    else
        tile_size = 256
        workgroup_size = tile_size
        num_workgroups = cell_list_params.num_cells
        total_num_threads = workgroup_size * num_workgroups
        kernel! = calculate_interactions_kernel_256!(CUDABackend(), workgroup_size)
    end #if


    kernel!(
        θ_updates,
        cells_data.sorted_particles,
        cells_data.perm,
        cells_data.cell_starts,
        cells_data.cell_counts,
        cells_data.cell_neighbours,
        cells_data.occupied_cells,
        cells_data.num_occupied,
        numerical_params.Lx,
        numerical_params.Ly,
        numerical_params.R²,
        numerical_params.inv_πR²,
        numerical_params.γ,
        numerical_params.dt;
        ndrange=total_num_threads)
    # KernelAbstractions.synchronize(CUDABackend())
end #function
