# --------- Sort particles into sorted_particles based on perm ---------

function find_cell_bounds!(cells_data, num_particles)

    fill!(cells_data.cell_starts, Int32(-1))
    fill!(cells_data.cell_ends, Int32(-1))

    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups

    kernel! = find_cell_bounds_kernel!(CUDABackend(), workgroup_size)
    kernel!(cells_data.cell_starts, cells_data.cell_ends, cells_data.sorted_cells, num_particles; ndrange=total_num_threads)
    KernelAbstractions.synchronize(CUDABackend())
end #function

@kernel function find_cell_bounds_kernel!(cell_starts, cell_ends, @Const(sorted_cells), num_particles)
    I = Int32(@index(Global, Linear))
    stride = Int32(@ndrange()[1])
    for i = I:stride:num_particles
        c = sorted_cells[i]
        if i == Int32(1) || sorted_cells[i-Int32(1)] != c
            cell_starts[c] = i
        end
        if i == N || sorted_cells[i+Int32(1)] != c
            cell_ends[c] = i
        end
    end #for i
end #function


