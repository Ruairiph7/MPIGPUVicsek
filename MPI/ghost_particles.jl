# ------------------------------------------------------------
# Ghost particle extraction on GPU
# ------------------------------------------------------------

@kernel function extract_ghosts_kernel!(lefts, rights, counters, @Const(particles), n, x_min, x_max, R)
    I = @index(Global)
    # stride = @ndrange()
    stride = 256*256

    for i = I:stride:n
        p = particles[i]
        x = p.r[1]
        if x < x_min + R #Ghost to be sent left
            idx = CUDA.atomic_add!(pointer(counters, 1), Int32(1))
            lefts[idx+1] = p
        elseif x > x_max - R #Ghost to be sent right
            idx = CUDA.atomic_add!(pointer(counters, 2), Int32(1))
            rights[idx+1] = p
        end #if
    end #for j
end #function

function extract_ghosts!(bufs, particles, x_min, x_max, R)
    n = length(particles)
    if n == 0
        return (CuArray{Particle}(undef, 0), CuArray{Particle}(undef, 0))
    end

    bufs.counters .= Int32(0)

    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups

    kernel! = extract_ghosts_kernel!(CUDABackend())
    kernel!(bufs.lefts, bufs.rights, bufs.counters, particles, n, x_min, x_max, R; ndrange=total_num_threads)
    KernelAbstractions.synchronize(CUDABackend())

    counters_cpu = Array(bufs.counters)
    n_left, n_right = counters_cpu

    return @view(bufs.lefts[1:n_left]), @view(bufs.rights[1:n_right])
end #function


# ------------------------------------------------------------
# Ghost particle exchange (GPU-only serialisation)
# ------------------------------------------------------------

function exchange_ghosts!(particles, local_particles, comm, rank, nprocs, x_min, x_max, R, ghost_bufs, mpi_bufs)
    left_rank = (rank == 0) ? nprocs - 1 : rank - 1
    right_rank = (rank == nprocs - 1) ? 0 : rank + 1
    num_local_particles = length(local_particles)

    # 1. Identify ghosts
    ghosts_left_view, ghosts_right_view = extract_ghosts!(ghost_bufs, local_particles, x_min, x_max, R)

    # 2. Serialize on device
    send_left_count, send_right_count = pack_particles_to_f32!(mpi_bufs, ghosts_left_view, ghosts_right_view)
    send_left_buf = view(mpi_bufs.send_left_buf, 1:getindex(send_left_count))
    send_right_buf = view(mpi_bufs.send_right_buf, 1:getindex(send_right_count))

    # 3. Exchange counts with neighbors (host-level small messages)
    recv_left_count = Ref{Int32}(0)
    recv_right_count = Ref{Int32}(0)

    ghost_count_left_tag = 101
    ghost_count_right_tag = 102

    MPI.Sendrecv!(send_left_count, recv_right_count, comm,
        dest=left_rank,
        source=right_rank,
        sendtag=ghost_count_left_tag,
        recvtag=ghost_count_left_tag)
    MPI.Sendrecv!(send_right_count, recv_left_count, comm,
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
    MPI.Sendrecv!(send_left_buf, recv_right_buf, comm,
        dest=left_rank,
        source=right_rank,
        sendtag=ghost_left_tag,
        recvtag=ghost_left_tag)
    MPI.Sendrecv!(send_right_buf, recv_left_buf, comm,
        dest=right_rank,
        source=left_rank,
        sendtag=ghost_right_tag,
        recvtag=ghost_right_tag)

    # 5. Deserialize and add into particles after local_particles, return new total
    num_particles = unpack_f32_to_particles!(particles, num_local_particles, recv_left_buf, recv_right_buf)

    return num_particles
end


