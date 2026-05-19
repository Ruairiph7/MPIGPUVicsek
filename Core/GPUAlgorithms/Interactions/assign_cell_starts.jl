# --------- Compute the cell_start for each cell via exclusive prefix sum --------- #

function assign_cell_starts!(cells_data, cell_list_params)

    # Inclusive prefix sum into scratch
    CUDA.cumsum!(cells_data.cell_starts_scratch, cells_data.cell_counts)

    # Shift to 1-based exclusive prefix sum in cell_starts
    workgroup_size = 256
    num_workgroups = 512
    total_num_threads = workgroup_size * num_workgroups

    kernel! = assign_cell_starts_kernel!(CUDABackend(), workgroup_size)
    kernel!(
        cells_data.cell_starts,
        cells_data.cell_starts_scratch,
        cell_list_params.num_cells;
        ndrange=total_num_threads)
    # KernelAbstractions.synchronize(CUDABackend())
end #function

@kernel function assign_cell_starts_kernel!(
    cell_starts,
    @Const(cell_starts_inclusive),
    num_cells)

    I = Int32(@index(Global, Linear))
    stride = Int32(@ndrange()[1])

    for c = I:stride:num_cells
        cell_starts[c] = c == Int32(1) ? Int32(1) :
                         cell_starts_inclusive[c-Int32(1)] + Int32(1)
    end #for c
end #function
