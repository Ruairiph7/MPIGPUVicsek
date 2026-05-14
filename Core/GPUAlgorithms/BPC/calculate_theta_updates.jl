# --------- Calculate θ_updates ---------
# NOTE: Tumbling comes later in the kernel for updating particles.

# Assign a workgroup to each occupied cell, iterate in batches to assign
# one thread per particle even when cell_count > workgroup_size.
#   - Also iterate through neighbour cells in tiles, again handles high cell_counts
#   - In each batch all neighbour tiles are traversed (so tiles are reloaded per batch)
#   - workgroup_size = tile_size = 128, so all threads always participate in tile loading
#   - Shared memory usage of tile_size * sizeof(Particle) = 128 * 16 = 2 KB per workgroup

function calculate_θ_updates_bpc!(θ_updates, cells_data, numerical_params, R_max)

    # Need num_occupied on the CPU to launch the correct number of workgroups
    num_occupied_cpu = Array(cells_data.num_occupied)[1]

    workgroup_size = 128
    num_workgroups = num_occupied_cpu
    total_num_threads = workgroup_size * num_workgroups

    kernel! = calculate_θ_updates_bpc_kernel!(CUDABackend(), workgroup_size)
    kernel!(
        θ_updates,
        cells_data.sorted_particles,
        cells_data.perm,
        cells_data.cell_starts,
        cells_data.cell_counts,
        cells_data.cell_neighbours,
        cells_data.occupied_cells,
        R_max,
        numerical_params.Lx,
        numerical_params.Ly,
        numerical_params.R²,
        numerical_params.Rn²,
        numerical_params.γ,
        numerical_params.γn,
        numerical_params.dt;
        ndrange=total_num_threads)
    KernelAbstractions.synchronize(CUDABackend())
end #function

@kernel function calculate_θ_updates_bpc_kernel!(
    θ_updates,
    @Const(sorted_particles),
    @Const(perm),
    @Const(cell_starts),
    @Const(cell_counts),
    @Const(cell_neighbours),
    @Const(occupied_cells),
    R_max,
    Lx, Ly,
    R², Rn²,
    γ, γn,
    dt)

    group_idx = Int32(@index(Group, Linear))
    local_tidx = Int32(@index(Local, Linear))

    shared_tile = @localmem Particle 128

    # Uniform values - same for all threads in workgrooup
    @uniform cell_idx = occupied_cells[group_idx]
    @uniform cell_start = cell_starts[cell_idx]
    @uniform cell_count = cell_counts[cell_idx]

    # --------- Loop over batches ---------
    batch_offset = Int32(0)

    while batch_offset < cell_count

        # Get each thread's particle in this batch
        p_offset = batch_offset + local_tidx - Int32(1)
        valid = p_offset < cell_count
        p_idx = cell_start + p_offset
        p_i = valid ? sorted_particles[p_idx] : Particle(@SVector([0.0, 0.0]), 0.0, Int32(0))

        # Load this thread's particle position and angle
        # (will only read later if valid so safe to load unconditionally)
        x_i = valid ? p_i.r[1] : 0.0f0
        y_i = valid ? p_i.r[2] : 0.0f0

        F_sum_local = 0.0f0
        Fn_sum_local = 0.0f0
        n_local = Int32(0)

        # --------- Loop over neighbouring cells (including self) ---------
        for nghbr in Int32(1):Int32(9)

            nghbr_idx = cell_neighbours[nghbr, cell_idx]
            nghbr_start = cell_starts[nghbr_idx]
            nghbr_count = cell_counts[nghbr_idx]

            if nghbr_count > Int32(0)

                tile_offset = Int32(0)

                while tile_offset < nghbr_count

                    #Fix tile size to 128, or number of remaining particles if < 128
                    this_tile_size = min(Int32(128), nghbr_count - tile_offset)

                    if local_tidx <= this_tile_size
                        shared_tile[local_tidx] = sorted_particles[
                            nghbr_start+tile_offset+local_tidx-Int32(1)]
                    end #if local_tidx
                    @synchronize #Ensure tile is fully loaded before any thread reads it

                    #If the thread corresponds to a valid particle, find its interactions with this tile
                    if valid
                        for j in Int32(1):this_tile_size
                            p_j = shared_tile[j]
                            Δx = p_i.r[1] - p_j.r[1]
                            Δy = p_i.r[2] - p_j.r[2]
                            Δx > min_cell_width && (Δx -= Lx)
                            Δx < -min_cell_width && (Δx += Lx)
                            Δy > min_cell_width && (Δy -= Ly)
                            Δy < -min_cell_width && (Δy += Ly)
                            if Δx^2 + Δy^2 < R²
                                θ_ij = p_j.θ - p_i.θ
                                F_sum_local += F(θ_ij, R²)
                                n_local += Int32(1)
                            end #if
                            if Δx^2 + Δy^2 < Rn²
                                θ_ij = p_j.θ - p_i.θ
                                Fn_sum_local += Fn(θ_ij, Rn²)
                            end #if
                        end #for j
                    end #if valid
                    @synchronize   #Ensure all threads are done before the next load

                    tile_offset += Int32(128)
                end #while tile_offset
            end #if nghbr_count
        end #for nghbr

        # ── Write result ────────────────────────────────────────────────────
        # Each p_idx is written by exactly one thread in exactly one batch —
        # direct assignment, no atomics needed.
        if valid
            θ_updates[perm[p_idx]] = γ * F_sum_local * dt / n_local + γn * Fn_sum_local * dt
        end #if

        batch_offset += Int32(128)
    end #while batch_offset
end #function

