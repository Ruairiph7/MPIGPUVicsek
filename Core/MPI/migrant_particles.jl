# ------------------------------------------------------------
# GPU Migrant detection
# ------------------------------------------------------------

@kernel function sort_migrants_kernel!(
    stayers, lefts, rights, counters,
    overflow_flag, max_num,
    @Const(particles), n,
    x_min_local, x_max_local,
    R_max)

    I = Int32(@index(Global, Linear))
    stride = Int32(@ndrange()[1])

    for i = I:stride:n
        p = particles[i]
        x = p.x
        if x_min_local - R_max <= x < x_min_local #Particle is in the cell immediately to the left; moved to left domain
            idx = CUDA.atomic_add!(pointer(counters, 2), Int32(1))
            if idx <= max_num
                lefts[idx+1] = p
            else #No remaining space in buffers - raise overflow flag
                CUDA.atomic_max!(pointer(overflow_flag, 1), Int32(1))
            end #if idx
        elseif x < x_min_local - R_max #Particle has been wrapped round to the left; moved to "right" domain (PBCs)
            idx = CUDA.atomic_add!(pointer(counters, 3), Int32(1))
            if idx <= max_num
                rights[idx+1] = p
            else #No remaining space in buffers - raise overflow flag
                CUDA.atomic_max!(pointer(overflow_flag, 1), Int32(1))
            end #if idx
        elseif x_max_local + R_max >= x > x_max_local #Particle is in the cell immediately to the right; moved to right domain
            idx = CUDA.atomic_add!(pointer(counters, 3), Int32(1))
            if idx <= max_num
                rights[idx+1] = p
            else #No remaining space in buffers - raise overflow flag
                CUDA.atomic_max!(pointer(overflow_flag, 1), Int32(1))
            end #if idx
        elseif x > x_max_local + R_max #Particle has been wrapped round to the right; moved to "left" domain (PBCs)
            idx = CUDA.atomic_add!(pointer(counters, 2), Int32(1))
            if idx <= max_num
                lefts[idx+1] = p
            else #No remaining space in buffers - raise overflow flag
                CUDA.atomic_max!(pointer(overflow_flag, 1), Int32(1))
            end #if idx
        else #Particle has remained in this domain
            idx = CUDA.atomic_add!(pointer(counters, 1), Int32(1))
            stayers[idx+1] = p
        end
    end #for i
end

function sort_migrants!(bufs, particles, x_min_local, x_max_local, R_max, rank)
    n = length(particles)
    n == 0 && return view(bufs.stayers, 1:0), view(bufs.lefts, 1:0), view(bufs.rights, 1:0)

    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups
    kernel! = sort_migrants_kernel!(CUDABackend())

    for attempt in 1:2
        fill!(bufs.counters, Int32(0))
        fill!(bufs.overflow_flag, Int32(0))

        kernel!(
            bufs.stayers, bufs.lefts, bufs.rights, bufs.counters,
            bufs.overflow_flag, bufs.max_num,
            particles, n,
            x_min_local, x_max_local,
            R_max;
            ndrange=total_num_threads)
        KernelAbstractions.synchronize(CUDABackend())

        counters_cpu = Array(bufs.counters)
        overflowed = Array(bufs.overflow_flag)[1] != Int32(0)

        if !overflowed
            return (
                view(bufs.stayers, 1:counters_cpu[1]),
                view(bufs.lefts, 1:counters_cpu[2]),
                view(bufs.rights, 1:counters_cpu[3])
            )
        end #if
        attempt == 2 && error("sort_migrants!: Still overflows on second attempt.")

        # Resize buffers
        max_count = maximum(counters_cpu[2], counters_cpu[3])
        new_buf_size = ceil(Int32, max_count * 1.5f0)
        bufs.lefts = CuVector{Particle}(undef, new_buf_size)
        bufs.rights = CuVector{Particle}(undef, new_buf_size)
        bufs.max_num = new_buf_size
        println("Rank $rank raising migrant buffer size to $new_buf_size")

    end #for attempt
end #function

# ------------------------------------------------------------
# Migration exchange between ranks
# ------------------------------------------------------------
function exchange_migrants!(mpi_bufs, local_particles, migrant_bufs, numerical_params, mpi_params; SINGLE_RANK=false)

    # --- If only a single GPU, all particles are stayers ---
    if SINGLE_RANK
        return local_particles, CuVector{Particle}(undef, 0), CuVector{Particle}(undef, 0)
    end #if SINGLE_RANK

    # --- Otherwise: ---

    left_rank = (mpi_params.rank == 0) ? mpi_params.nprocs - 1 : mpi_params.rank - 1
    right_rank = (mpi_params.rank == mpi_params.nprocs - 1) ? 0 : mpi_params.rank + 1

    # 1. GPU-only detection + compaction of migrants
    stayers_view, migrants_left_view, migrants_right_view = sort_migrants!(
        migrant_bufs,
        local_particles,
        numerical_params.x_min_local,
        numerical_params.x_max_local,
        numerical_params.R_max,
        rank)

    # 2. Serialize migrants
    ensure_send_capacity!(mpi_bufs, ghost_bufs.max_num, mpi_params.rank)
    send_left_count, send_right_count = pack_particles!(
        mpi_bufs,
        migrants_left_view,
        migrants_right_view)

    send_left_buf = view(mpi_bufs.send_left_buf, 1:getindex(send_left_count))
    send_right_buf = view(mpi_bufs.send_right_buf, 1:getindex(send_right_count))

    # 3. Count exchange
    recv_left_count = Ref{Int32}(0)
    recv_right_count = Ref{Int32}(0)

    migrant_count_left_tag = 301
    migrant_count_right_tag = 302

    MPI.Sendrecv!(
        send_left_count,
        recv_right_count,
        mpi_params.comm,
        dest=left_rank,
        source=right_rank,
        sendtag=migrant_count_left_tag,
        recvtag=migrant_count_left_tag)
    MPI.Sendrecv!(
        send_right_count,
        recv_left_count,
        mpi_params.comm,
        dest=right_rank,
        source=left_rank,
        sendtag=migrant_count_right_tag,
        recvtag=migrant_count_right_tag)

    # 4. Allocate recv buffers on GPU
    ensure_recv_capacity!(
        mpi_bufs,
        getindex(recv_left_count),
        getindex(recv_right_count),
        mpi_params.rank)
    recv_left_buf = view(mpi_bufs.recv_left_buf, 1:getindex(recv_left_count))
    recv_right_buf = view(mpi_bufs.recv_right_buf, 1:getindex(recv_right_count))

    migrant_left_tag = 401
    migrant_right_tag = 402

    # 5. Exchange actual data
    MPI.Sendrecv!(
        send_left_buf,
        recv_right_buf,
        mpi_params.comm,
        dest=left_rank,
        source=right_rank,
        sendtag=migrant_left_tag,
        recvtag=migrant_left_tag)

    MPI.Sendrecv!(
        send_right_buf,
        recv_left_buf,
        mpi_params.comm,
        dest=right_rank,
        source=left_rank,
        sendtag=migrant_right_tag,
        recvtag=migrant_right_tag)

    return stayers_view, recv_left_buf, recv_right_buf
end
