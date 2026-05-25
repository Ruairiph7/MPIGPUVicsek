# --------- Build histogram and track each particle's cell index --------- #

function build_histogram!(cells_data, cell_list_params, particles, num_particles)
    fill!(cells_data.cell_counts, Int32(0))
    fill!(cells_data.num_occupied, Int32(0))

    workgroup_size = STD_WORKGROUP_SIZE
    num_workgroups = STD_NUM_WORKGROUPS
    total_num_threads = workgroup_size * num_workgroups

    kernel! = build_histogram_kernel!(CUDABackend(), workgroup_size)
    kernel!(
        cells_data.cell_counts,
        cells_data.cell_indices,
        cells_data.occupied_cells,
        cells_data.num_occupied,
        cell_list_params.x_min_cells,
        cell_list_params.num_cells_x,
        cell_list_params.num_cells_y,
        cell_list_params.inv_cell_size_x,
        cell_list_params.inv_cell_size_y,
        particles,
        num_particles;
        ndrange=total_num_threads)

    # KernelAbstractions.synchronize(CUDABackend())
end #function

@kernel function build_histogram_kernel!(
    cell_counts,
    cell_indices,
    occupied_cells,
    num_occupied,
    x_min_cells,
    num_cells_x,
    num_cells_y,
    inv_cell_size_x,
    inv_cell_size_y,
    @Const(particles),
    num_particles)

    I = Int32(@index(Global, Linear))
    stride = Int32(@ndrange()[1])

    for i = I:stride:num_particles
        c = get_cell_ID(particles[i].x, particles[i].y, x_min_cells, num_cells_x, num_cells_y, inv_cell_size_x, inv_cell_size_y)

        cell_indices[i] = c
        old_cell_count = CUDA.atomic_add!(pointer(cell_counts, c), Int32(1))

        if old_cell_count == Int32(0)
            old_num_occupied = CUDA.atomic_add!(pointer(num_occupied, Int32(1)), Int32(1))
            occupied_cells[old_num_occupied+Int32(1)] = c
        end #if old_cell_count
    end #for i
end #function
