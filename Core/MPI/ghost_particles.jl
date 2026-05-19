# ------------------------------------------------------------
# Ghost particle extraction on GPU
# ------------------------------------------------------------

@kernel function extract_ghosts_kernel!(lefts, rights, counters, @Const(particles), n, x_min_local, x_max_local, R_max)
    I = Int32(@index(Global, Linear))
    stride = Int32(@ndrange()[1])

    for i = I:stride:n
        p = particles[i]
        x = p.x
        if x < x_min_local + R_max #Ghost to be sent left
            idx = CUDA.atomic_add!(pointer(counters, 1), Int32(1))
            lefts[idx+1] = p
        elseif x > x_max_local - R_max #Ghost to be sent right
            idx = CUDA.atomic_add!(pointer(counters, 2), Int32(1))
            rights[idx+1] = p
        end #if
    end #for j
end #function

function extract_ghosts!(bufs, particles, x_min_local, x_max_local, R_max)
    n = length(particles)
    if n == 0
        return (CuVector{Particle}(undef, 0), CuVector{Particle}(undef, 0))
    end

    bufs.counters .= Int32(0)

    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups

    kernel! = extract_ghosts_kernel!(CUDABackend())
    kernel!(bufs.lefts, bufs.rights, bufs.counters, particles, n, x_min_local, x_max_local, R_max; ndrange=total_num_threads)
    KernelAbstractions.synchronize(CUDABackend())

    counters_cpu = Array(bufs.counters)
    n_left, n_right = counters_cpu

    return @view(bufs.lefts[1:n_left]), @view(bufs.rights[1:n_right])
end #function


# ------------------------------------------------------------
# Ghost particle exchange (GPU-only serialisation)
# ------------------------------------------------------------

function exchange_ghosts!(mpi_bufs, local_particles, ghost_bufs, numerical_params, mpi_params; SINGLE_RANK=false)

    # --- If only a single GPU, no ghost exchange needed ---
    if SINGLE_RANK
        return CuVector{Particle}(undef, 0), CuVector{Particle}(undef, 0)
    end #if SINGLE_RANK

    # --- Otherwise: ---

    left_rank = (mpi_params.rank == 0) ? mpi_params.nprocs - 1 : mpi_params.rank - 1
    right_rank = (mpi_params.rank == mpi_params.nprocs - 1) ? 0 : mpi_params.rank + 1

    # 1. Identify ghosts
    ghosts_left_view, ghosts_right_view = extract_ghosts!(
        ghost_bufs,
        local_particles,
        numerical_params.x_min_local,
        numerical_params.x_max_local,
        numerical_params.R_max)

    # 2. Serialize on device
    #   - Apply PBCs if needed to ensure coordinates are in the domain covered by the local cell list
    send_left_count, send_right_count = pack_particles!(
        mpi_bufs,
        ghosts_left_view,
        ghosts_right_view,
        GHOST_FLAG=true,
        Lx=mpi_params.Lx,
        rank=mpi_params.rank,
        nprocs=mpi_params.nprocs)

    @assert send_left_count < mpi_bufs.buf_lengths "Too many ghosts"
    @assert send_right_count < mpi_bufs.buf_lengths "Too many ghosts"

    send_left_buf = view(mpi_bufs.send_left_buf, 1:getindex(send_left_count))
    send_right_buf = view(mpi_bufs.send_right_buf, 1:getindex(send_right_count))

    # 3. Exchange counts with neighbors (host-level small messages)
    recv_left_count = Ref{Int32}(0)
    recv_right_count = Ref{Int32}(0)

    ghost_count_left_tag = 101
    ghost_count_right_tag = 102

    MPI.Sendrecv!(
        send_left_count,
        recv_right_count,
        mpi_params.comm,
        dest=left_rank,
        source=right_rank,
        sendtag=ghost_count_left_tag,
        recvtag=ghost_count_left_tag)
    MPI.Sendrecv!(
        send_right_count,
        recv_left_count,
        mpi_params.comm,
        dest=right_rank,
        source=left_rank,
        sendtag=ghost_count_right_tag,
        recvtag=ghost_count_right_tag)

    # allocate receive buffers on GPU
    recv_left_buf = view(mpi_bufs.recv_left_buf, 1:getindex(recv_left_count))
    recv_right_buf = view(mpi_bufs.recv_right_buf, 1:getindex(recv_right_count))

    ghost_left_tag = 201
    ghost_right_tag = 202

    # 4. Actual device-aware data exchange
    MPI.Sendrecv!(
        send_left_buf,
        recv_right_buf,
        mpi_params.comm,
        dest=left_rank,
        source=right_rank,
        sendtag=ghost_left_tag,
        recvtag=ghost_left_tag)
    MPI.Sendrecv!(
        send_right_buf,
        recv_left_buf,
        mpi_params.comm,
        dest=right_rank,
        source=left_rank,
        sendtag=ghost_right_tag,
        recvtag=ghost_right_tag)

    return recv_left_buf, recv_right_buf
end


