# --------- Store interactions in θ_updates --------- #
# NOTE: Tumbling comes later in the kernel for updating particles.

function calculate_interactions!(θ_updates, cells_data, cell_list_params, numerical_params, num_particles)
    num_particles == 0 && return nothing

    sparse_workgroup_size = STD_WORKGROUP_SIZE
    sparse_num_workgroups = STD_NUM_WORKGROUPS
    sparse_total_num_threads = sparse_workgroup_size * sparse_num_workgroups
    sparse_kernel! = sparse_interactions_kernel!(CUDABackend(), sparse_workgroup_size)

    dense_workgroup_size = TILE_SIZE
    dense_num_workgroups = min(cell_list_params.num_cells, num_particles ÷ DENSE_THRESHOLD)
    dense_total_num_threads = dense_workgroup_size * dense_num_workgroups
    dense_kernel! = dense_interactions_kernel!(CUDABackend(), dense_workgroup_size)

    coef = Float32(numerical_params.γ * numerical_params.dt * numerical_params.inv_πR²)
    Lx = Float32(numerical_params.Lx)
    Ly = Float32(numerical_params.Ly)

    sparse_kernel!(
        θ_updates,
        cells_data.sorted_particles,
        cells_data.perm,
        cells_data.cell_starts,
        cells_data.cell_counts,
        cells_data.cell_neighbours,
        cell_list_params.x_min_cells,
        cell_list_params.num_cells_x,
        cell_list_params.num_cells_y,
        cell_list_params.inv_cell_size_x,
        cell_list_params.inv_cell_size_y,
        Lx,
        Ly,
        1.0f0 / Lx,
        1.0f0 / Ly,
        numerical_params.R²,
        coef,
        num_particles,
        ndrange=sparse_total_num_threads)

    dense_kernel!(
        θ_updates,
        cells_data.sorted_particles,
        cells_data.perm,
        cells_data.cell_starts,
        cells_data.cell_counts,
        cells_data.cell_neighbours,
        cells_data.dense_cells,
        cells_data.num_dense,
        Lx,
        Ly,
        1.0f0 / Lx,
        1.0f0 / Ly,
        numerical_params.R²,
        coef,
        ndrange=dense_total_num_threads)
    # KernelAbstractions.synchronize(CUDABackend())
end #function




# --------- Sparse Cells --------- #

@kernel function sparse_interactions_kernel!(
    θ_updates,
    @Const(sorted_particles),
    @Const(perm),
    @Const(cell_starts),
    @Const(cell_counts),
    @Const(cell_neighbours),
    x_min_cells,
    num_cells_x,
    num_cells_y,
    inv_cell_size_x,
    inv_cell_size_y,
    Lx, Ly,
    inv_Lx, inv_Ly,
    R², coef,
    num_particles)

    I = Int32(@index(Global, Linear))
    stride = Int32(@ndrange()[1])

    for i = I:stride:num_particles
        p_i = sorted_particles[i]
        c_i = get_cell_ID(p_i.x, p_i.y, x_min_cells, num_cells_x, num_cells_y, inv_cell_size_x, inv_cell_size_y)

        if cell_counts[c_i] < DENSE_THRESHOLD
            cos_sum = 0.0f0
            sin_sum = 0.0f0
            n = 0.0f0

            for nghbr in Int32(1):Int32(9)
                nghbr_idx = cell_neighbours[nghbr, c_i]
                nghbr_start = cell_starts[nghbr_idx]
                nghbr_end = nghbr_start + cell_counts[nghbr_idx] - Int32(1)

                for j in nghbr_start:nghbr_end
                    p_j = sorted_particles[j]
                    Δx = p_i.x - p_j.x
                    Δy = p_i.y - p_j.y
                    Δx -= Lx * round(Δx * inv_Lx)
                    Δy -= Ly * round(Δy * inv_Ly)
                    WITHIN_R = Float32(Δx * Δx + Δy * Δy < R²)

                    cos_sum += WITHIN_R * p_j.cosθ
                    sin_sum += WITHIN_R * p_j.sinθ
                    n += WITHIN_R
                end #for j
            end #for nghbr

            F_sum = p_i.cosθ * sin_sum - p_i.sinθ * cos_sum
            θ_updates[perm[i]] = n > 0.0f0 ? coef * F_sum / n : 0.0f0
        end #if cell_counts

    end #for i
end #function sparse_interactions_kernel!




# --------- Dense Cells --------- #

# Assign a workgroup to each dense cell, iterate in batches to assign
# one thread per particle even when cell_count > workgroup_size.
#   - Also iterate through neighbour cells in tiles, again handles high cell_counts
#   - In each batch all neighbour tiles are traversed (so tiles are reloaded per batch)
#   - workgroup_size = TILE_SIZE, so all threads always participate in tile loading
#   - For TILE_SIZE=128, shared memory usage of tile_size * sizeof(Particle) = 128 * 16 = 2 KB per workgroup

@kernel function dense_interactions_kernel!(
    θ_updates,
    @Const(sorted_particles),
    @Const(perm),
    @Const(cell_starts),
    @Const(cell_counts),
    @Const(cell_neighbours),
    @Const(dense_cells),
    @Const(num_dense),
    Lx, Ly,
    inv_Lx, inv_Ly,
    R², coef)

    group_idx = Int32(@index(Group, Linear))
    local_tidx = Int32(@index(Local, Linear))

    # Exit immediately for workgroups beyond num_dense
    @uniform max_group_idx = num_dense[1]
    if group_idx <= max_group_idx

        shared_tile = @localmem InteractionParticle TILE_SIZE

        # Uniform values - same for all threads in workgrooup
        @uniform cell_idx = dense_cells[group_idx]
        @uniform cell_start = cell_starts[cell_idx]
        @uniform cell_count = cell_counts[cell_idx]


        # --------- Loop over batches --------- #
        batch_offset = Int32(0)

        while batch_offset < cell_count

            # Get each thread's particle in this batch
            p_offset = batch_offset + local_tidx - Int32(1)
            VALID_IDX = p_offset < cell_count
            p_idx = cell_start + p_offset
            p_i = VALID_IDX ? sorted_particles[p_idx] : InteractionParticle(0.0f0, 0.0f0, 0.0f0, 0.0f0)

            # Load this thread's particle position and angle
            # (will only read later if valid so safe to load unconditionally)
            x_i = p_i.x
            y_i = p_i.y
            sin_θ_i = p_i.sinθ
            cos_θ_i = p_i.cosθ

            F_sum_local = 0.0f0
            n_local = 0.0f0


            # --------- Loop over neighbouring cells (including self) --------- #
            for nghbr in Int32(1):Int32(9)

                nghbr_idx = cell_neighbours[nghbr, cell_idx]
                nghbr_start = cell_starts[nghbr_idx]
                nghbr_count = cell_counts[nghbr_idx]

                if nghbr_count > Int32(0)

                    # --------- Loop over tiles --------- #
                    tile_offset = Int32(0)

                    while tile_offset < nghbr_count

                        #Fix tile size to TILE_SIZE, or number of remaining particles if < TILE_SIZE
                        this_tile_size = min(Int32(TILE_SIZE), nghbr_count - tile_offset)

                        if local_tidx <= this_tile_size
                            shared_tile[local_tidx] = sorted_particles[
                                nghbr_start+tile_offset+local_tidx-Int32(1)]
                        end #if local_tidx
                        @synchronize #Ensure tile is fully loaded before any thread reads it

                        #If the thread corresponds to a valid particle, find its interactions with this tile
                        if VALID_IDX
                            for j in Int32(1):this_tile_size
                                p_j = shared_tile[j]
                                Δx = x_i - p_j.x
                                Δy = y_i - p_j.y
                                Δx -= Lx * round(Δx * inv_Lx)
                                Δy -= Ly * round(Δy * inv_Ly)
                                Δr² = Δx * Δx + Δy * Δy
                                sin_θ_ij = p_j.sinθ * cos_θ_i - p_j.cosθ * sin_θ_i

                                WITHIN_R = Float32(Δr² < R²)
                                F_sum_local += WITHIN_R * sin_θ_ij
                                n_local += WITHIN_R
                            end #for j
                        end #if VALID_IDX
                        @synchronize #Ensure all threads are done before the next load

                        tile_offset += Int32(TILE_SIZE)
                    end #while tile_offset
                end #if nghbr_count
            end #for nghbr

            # Each entry written by exactly one thread in exactly one batch — no atomics needed
            if VALID_IDX
                θ_updates[perm[p_idx]] = n_local > 0.0f0 ? coef * F_sum_local / n_local : 0.0f0
            end #if

            batch_offset += Int32(TILE_SIZE)
        end #while batch_offset
    end #if group_idx
end #function dense_interactions_kernel!
