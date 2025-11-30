@warn "extract_ghosts workgroup_size, num_workgroups hard-coded at 256"

# ------------------------------------------------------------
# Ghost particle extraction on GPU
# ------------------------------------------------------------

@kernel function extract_ghosts_kernel!(lefts, rights, counters, @Const(particles), n, x_min, x_max, R)
    I = @index(Global)
    stride = @ndrange()

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

function extract_ghosts(particles, x_min, x_max, R, bufs)
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

function exchange_ghosts(local_particles_gpu, comm, rank, nprocs, x_min, x_max, R, bufs)
    left_rank = (rank == 0) ? nprocs - 1 : rank - 1
    right_rank = (rank == nprocs - 1) ? 0 : rank + 1

    # 1. Identify ghosts
    ghosts_left, ghosts_right = extract_ghosts(local_particles_gpu, x_min, x_max, R, bufs)

    # 2. Serialize on device
    send_left_buf = pack_particles_to_f32(ghosts_left)
    send_left_count = Ref{Int32}(length(send_left_buf))
    send_right_buf = pack_particles_to_f32(ghosts_right)
    send_right_count = Ref{Int32}(length(send_right_buf))

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
    recv_left_buf = CuArray{Float32}(undef, recv_left_count[])  # size 0 allowed
    recv_right_buf = CuArray{Float32}(undef, recv_right_count[])

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

    # 5. Deserialize
    ghosts_left = unpack_f32_to_particles(recv_left_buf)
    ghosts_right = unpack_f32_to_particles(recv_right_buf)

    return vcat(ghosts_left, ghosts_right)
end


