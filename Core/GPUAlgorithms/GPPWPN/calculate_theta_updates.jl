# --------- Calculate θ_updates ---------
# NOTE: Tumbling comes later in the kernel for updating particles.

# Assign a workgroup to each particle, composed of 9 warps of 32 threads/lanes
#   - Each warp is assigned to one of the neighbouring cells to the
#   particle (including its own).
#   - All lanes of a warp stride through that cell's particles, summing interactions
#   - The first lane of each warp gathers its sums
#   - All warps are synchronized, then the first 9 lanes of the first warp collect
#   the sums, before the first lane writes to θ_updates for that particle.
#
function calculate_θ_updates_gppwpn!(θ_updates, cells_data, cell_list_params, num_particles, numerical_params, R_max)
    workgroup_size = 288
    num_workgroups = num_particles
    total_num_threads = workgroup_size * num_workgroups
    kernel! = calculate_θ_updates_gppwpn_kernel!(CUDABackend(), workgroup_size)
    kernel!(θ_updates,
        cells_data.sorted_particles,
        cells_data.cell_starts,
        cells_data.cell_ends,
        cells_data.cell_neighbours,
        cells_data.perm,
        cell_list_params.num_cells_x,
        cell_list_params.num_cells_y,
        cell_list_params.cell_size_x,
        cell_list_params.cell_size_y,
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

@kernel function calculate_θ_updates_gppwpn_kernel!(
    θ_updates,
    @Const(sorted_particles),
    @Const(cell_starts),
    @Const(cell_ends),
    @Const(cell_neighbours),
    @Const(perm),
    num_cells_x,
    num_cells_y,
    cell_size_x,
    cell_size_y,
    R_max,
    Lx, Ly,
    R², Rn²,
    γ, γn,
    dt)

    # --------- Set indices ---------
    global_tidx = Int32(@index(Global, Linear)) #Global linear thread index; 1-based
    local_tidx = Int32(@index(Local, Linear)) - Int32(1) #Local index within workgroup; 0-based, [0,287]

    warp_idx = local_tidx ÷ Int32(32) #Warp index, i.e. which neighbour cell; 0-based, [0,8]
    lane = local_tidx % Int32(32) #Lane within warp; 0-based, [0,31]

    i = Int32(@index(Group, Linear)) # i = particle index = workgroup index; 1-based

    # --------- Allocate local memory ---------
    F_sum_warps = @localmem Float32 9
    Fn_sum_warps = @localmem Float32 9
    n_warps = @localmem Int32 9

    # --------- Prepare variables ---------
    p_i = sorted_particles[i] # = particles[perm[i]]
    this_cell_idx = get_cell_ID(p_i.r, num_cells_x, num_cells_y, cell_size_x, cell_size_y)
    nghbr_cell_idx = cell_neighbours[warp_idx+Int32(1), this_cell_idx]

    nghbr_cell_start = cell_starts[nghbr_cell_idx]
    nghbr_cell_end = cell_ends[nghbr_cell_idx]

    F_sum_lane = 0.0f0
    Fn_sum_lane = 0.0f0
    n_lane = Int32(0)

    # --------- Calculate interactions with particles inside this neighbour ---------

    if nghbr_cell_start != Int32(-1) #Check we have at least one particle in this nghbr
        j = nghbr_cell_start + lane
        num_strides = (nghbr_cell_end - nghbr_cell_start + Int32(32)) ÷ Int32(32)
        for stride in Int32(1):num_strides
            if j <= nghbr_cell_end
                p_j = sorted_particles[j]
                Δx = p_i.r[1] - p_j.r[1]
                Δy = p_i.r[2] - p_j.r[2]
                Δx > min_cell_width && (Δx -= Lx)
                Δx < -min_cell_width && (Δx += Lx)
                Δy > min_cell_width && (Δy -= Ly)
                Δy < -min_cell_width && (Δy += Ly)
                if Δx^2 + Δy^2 < R²
                    θ_ij = p_j.θ - p_i.θ
                    F_sum_lane += F(θ_ij, R²)
                    n_lane += Int32(1)
                end #if
                if Δx^2 + Δy^2 < Rn²
                    θ_ij = p_j.θ - p_i.θ
                    Fn_sum_lane += Fn(θ_ij, Rn²)
                end #if
            end #if j
            j += Int32(32)
        end #for stride
    end #if nghbr_cell_start

    # --------- Collect onto lane 0 ---------
    F_sum_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), F_sum_lane, Int32(16))
    F_sum_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), F_sum_lane, Int32(8))
    F_sum_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), F_sum_lane, Int32(4))
    F_sum_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), F_sum_lane, Int32(2))
    F_sum_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), F_sum_lane, Int32(1))

    Fn_sum_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), Fn_sum_lane, Int32(16))
    Fn_sum_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), Fn_sum_lane, Int32(8))
    Fn_sum_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), Fn_sum_lane, Int32(4))
    Fn_sum_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), Fn_sum_lane, Int32(2))
    Fn_sum_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), Fn_sum_lane, Int32(1))

    n_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), n_lane, Int32(16))
    n_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), n_lane, Int32(8))
    n_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), n_lane, Int32(4))
    n_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), n_lane, Int32(2))
    n_lane += CUDA.shfl_down_sync(UInt32(0xffffffff), n_lane, Int32(1))

    if lane == Int32(0)
        F_sum_warps[warp_idx+Int32(1)] = F_sum_lane
        Fn_sum_warps[warp_idx+Int32(1)] = Fn_sum_lane
        n_warps[warp_idx+Int32(1)] = n_lane
    end #if lane

    @synchronize()

    # --------- Collect onto warp 0, lane 0 ---------
    if warp_idx == Int32(0) && lane == Int32(0)
        F_sum = 0.0f0
        Fn_sum = 0.0f0
        n = Int32(0)

        for lidx in Int32(1):Int32(9)
            F_sum += F_sum_warps[lidx]
            Fn_sum += Fn_sum_warps[lidx]
            n += n_warps[lidx]
        end

        θ_updates[perm[i]] = γ * F_sum * dt / n + γn * Fn_sum * dt
    end

end #function
