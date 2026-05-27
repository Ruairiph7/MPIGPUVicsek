function exchange_ghosts!(
    mpi_bufs,
    local_particles,
    ghost_bufs,
    numerical_params, mpi_params;
    SINGLE_RANK=false)

    #If only a single GPU, no ghost exchange needed
    if SINGLE_RANK
        return CuVector{Particle}(undef, 0), CuVector{Particle}(undef, 0)
    end #if

    left_rank = (mpi_params.rank == 0) ? mpi_params.nprocs - 1 : mpi_params.rank - 1
    right_rank = (mpi_params.rank == mpi_params.nprocs - 1) ? 0 : mpi_params.rank + 1

    #1) Identify ghosts
    ghosts_left_view, ghosts_right_view = extract_ghosts!(
        ghost_bufs,
        local_particles,
        numerical_params.x_min_local,
        numerical_params.x_max_local,
        numerical_params.R,
        mpi_params.rank)

    #2) Serialise ghosts to send over MPI, allocating extra space if required
    #   - Apply PBCs if needed to ensure coordinates are in the domain covered by the local cell list
    ensure_send_capacity!(mpi_bufs, ghost_bufs.buf_lengths, mpi_params.rank)
    send_left_count, send_right_count = pack_particles!(
        mpi_bufs,
        ghosts_left_view,
        ghosts_right_view,
        GHOST_FLAG=true,
        Lx=numerical_params.Lx,
        rank=mpi_params.rank,
        nprocs=mpi_params.nprocs)

    send_left = view(mpi_bufs.send_left, 1:getindex(send_left_count))
    send_right = view(mpi_bufs.send_right, 1:getindex(send_right_count))

    #3) Exchange counts with neighbours and prepare space to receive
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

    ensure_recv_capacity!(
        mpi_bufs,
        getindex(recv_left_count),
        getindex(recv_right_count),
        mpi_params.rank)
    recv_left = view(mpi_bufs.recv_left, 1:getindex(recv_left_count))
    recv_right = view(mpi_bufs.recv_right, 1:getindex(recv_right_count))

    #4) Exchange ghost particles
    ghost_left_tag = 201
    ghost_right_tag = 202

    MPI.Sendrecv!(
        send_left,
        recv_right,
        mpi_params.comm,
        dest=left_rank,
        source=right_rank,
        sendtag=ghost_left_tag,
        recvtag=ghost_left_tag)
    MPI.Sendrecv!(
        send_right,
        recv_left,
        mpi_params.comm,
        dest=right_rank,
        source=left_rank,
        sendtag=ghost_right_tag,
        recvtag=ghost_right_tag)

    return recv_left, recv_right
end #function


# --------- Extract particles to be sent as ghosts and store in local buffers --------- #

function extract_ghosts!(bufs, particles, x_min_local, x_max_local, R, rank)
    n = Int32(length(particles))
    n == Int32(0) && return view(bufs.lefts, 1:0), view(bufs.rights, 1:0)

    workgroup_size = STD_WORKGROUP_SIZE
    num_workgroups = STD_NUM_WORKGROUPS
    total_num_threads = workgroup_size * num_workgroups
    kernel! = extract_ghosts_kernel!(CUDABackend(), workgroup_size)

    for attempt in 1:2 #Make two attempts, allocate extra space if the first overflows
        fill!(bufs.counters, Int32(0))
        fill!(bufs.overflow_flag, Int32(0))

        kernel!(
            bufs.lefts, bufs.rights, bufs.counters,
            bufs.overflow_flag, bufs.buf_lengths,
            particles, n,
            x_min_local, x_max_local,
            R;
            ndrange=total_num_threads)
        KernelAbstractions.synchronize(CUDABackend())

        counters_cpu = Array(bufs.counters)
        overflowed = Array(bufs.overflow_flag)[1] != Int32(0)

        if !overflowed
            return (
                view(bufs.lefts, 1:counters_cpu[1]),
                view(bufs.rights, 1:counters_cpu[2])
            )
        end #if
        attempt == 2 && error("extract_ghosts!: Still overflows on second attempt.")

        # Resize buffers
        max_count = maximum(counters_cpu)
        new_buf_size = maximum((bufs.buf_lengths * 2, ceil(Int32, max_count * 1.5f0)))
        bufs.lefts = CuVector{Particle}(undef, new_buf_size)
        bufs.rights = CuVector{Particle}(undef, new_buf_size)
        bufs.buf_lengths = new_buf_size
        println("Rank $rank raising ghost buffer size to $new_buf_size")

    end #for attempt
end #function

@kernel function extract_ghosts_kernel!(
    lefts, rights, counters,
    overflow_flag, buf_lengths,
    @Const(particles), n,
    x_min_local, x_max_local,
    R)

    I = Int32(@index(Global, Linear))
    stride = Int32(@ndrange()[1])

    for i = I:stride:n
        p = particles[i]
        x = p.x
        if x < x_min_local + R #Ghost to be sent left
            idx = CUDA.atomic_add!(pointer(counters, 1), Int32(1))
            if idx < buf_lengths
                lefts[idx+1] = p
            else #No remaining space in buffers - raise overflow flag
                CUDA.atomic_max!(pointer(overflow_flag, 1), Int32(1))
            end #if idx
        elseif x > x_max_local - R #Ghost to be sent right
            idx = CUDA.atomic_add!(pointer(counters, 2), Int32(1))
            if idx < buf_lengths
                rights[idx+1] = p
            else #No remaining space in buffers - raise overflow flag
                CUDA.atomic_max!(pointer(overflow_flag, 1), Int32(1))
            end #if idx
        end #if x
    end #for j
end #function
