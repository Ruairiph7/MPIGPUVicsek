@kernel function calculate_interactions_kernel_64!(
    θ_updates,
    @Const(sorted_particles),
    @Const(perm),
    @Const(cell_starts),
    @Const(cell_counts),
    @Const(cell_neighbours),
    @Const(occupied_cells),
    @Const(num_occupied),
    Lx, Ly,
    R², inv_πR², 
    γ, dt)

    group_idx = Int32(@index(Group, Linear))
    local_tidx = Int32(@index(Local, Linear))

    # Exit immediately for workgroups beyond num_occupied
    @uniform max_group_idx = num_occupied[1]
    if group_idx <= max_group_idx

        shared_tile = @localmem Particle 64

        # Uniform values - same for all threads in workgrooup
        @uniform cell_idx = occupied_cells[group_idx]
        @uniform cell_start = cell_starts[cell_idx]
        @uniform cell_count = cell_counts[cell_idx]


        # --------- Loop over batches --------- #
        batch_offset = Int32(0)

        while batch_offset < cell_count

            # Get each thread's particle in this batch
            p_offset = batch_offset + local_tidx - Int32(1)
            VALID_IDX = p_offset < cell_count
            p_idx = cell_start + p_offset
            p_i = VALID_IDX ? sorted_particles[p_idx] : Particle(0.0f0, 0.0f0, 0.0f0, Int32(0))

            # Load this thread's particle position and angle
            # (will only read later if valid so safe to load unconditionally)
            x_i = p_i.x
            y_i = p_i.y
            θ_i = p_i.θ

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

                        #Fix tile size to 64, or number of remaining particles if < 64
                        this_tile_size = min(Int32(64), nghbr_count - tile_offset)

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
                                Δx -= Lx * round(Δx / Lx)
                                Δy -= Ly * round(Δy / Ly)
                                Δr² = Δx * Δx + Δy * Δy
                                θ_ij = p_j.θ - θ_i

                                WITHIN_R = Float32(Δr² < R²)
                                F_sum_local += WITHIN_R * F(θ_ij, inv_πR²)
                                n_local += WITHIN_R
                            end #for j
                        end #if VALID_IDX
                        @synchronize #Ensure all threads are done before the next load

                        tile_offset += Int32(64)
                    end #while tile_offset
                end #if nghbr_count
            end #for nghbr

            # Each entry written by exactly one thread in exactly one batch — no atomics needed
            if VALID_IDX
                θ_updates[perm[p_idx]] = n_local > 0.0f0 ? γ * F_sum_local * dt / n_local : 0.0f0
            end #if

            batch_offset += Int32(64)
        end #while batch_offset
    end #if group_idx
end #function


