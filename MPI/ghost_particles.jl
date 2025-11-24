# ------------------------------------------------------------
# Ghost particle extraction on GPU
# ------------------------------------------------------------

# Store whether particles are ghosts left (1), right (2) or are not ghosts (0)
@kernel function mark_ghosts_kernel!(flags, @Const(particles), n, x_min, x_max, R)
    i = @index(Global)
    if i > n
        return nothing
    end
    x = particles[i].r[1]
    if x < x_min + R
        flags[i] = 1
    elseif x > x_max - R
        flags[i] = 2
    else
        flags[i] = 0
    end #if
    return nothing
end #function

@kernel function pack_ghosts_kernel!(left_particles, right_particles, left_counter, right_counter, @Const(particles), @Const(flags), n)
    i = @index(Global)
    if i > n
        return nothing
    end
    if flags[i] == 1
        #NOTE:
        #TODO:
        #WARN: CHECK THIS SYNTAX is correct; chatgpt wanted just "idx = CUDA.atomic_add!(left_counter,1,1)"; but I took this from my old code
        idx = CUDA.atomic_add!(pointer(left_counter, 1), Int32(1))
        left_particles[idx+1] = particles[i]
    elseif flags[i] == 2
        idx = CUDA.atomic_add!(pointer(right_counter, 1), Int32(1))
        right_particles[idx+1] = particles[i]
    end #if
    return nothing
end #function


function pack_ghosts(particles, flags)
    n = length(particles) #WARN: CHANGING n every time could slow down things like calling kernels due to recompilation
    if n == 0
        return CuArray{Particle}(undef, 0)
    end
    #WARN: COULD KEEP these as buffers rather than reallocating every time (if n was constant)
    left_particles = CuArray{Particle}(undef, n)
    right_particles = CuArray{Particle}(undef, n)
    left_counter = CuArray([Int32(0)])
    right_counter = CuArray([Int32(0)])

    kernel! = pack_ghosts_kernel!(CUDABackend())
    kernel!(left_particles, right_particles, left_counter, right_counter, particles, flags, n; ndrange=n)
    KernelAbstractions.synchronize(CUDABackend())

    left_count = Array(left_counter)[]
    right_count = Array(right_counter)[]

    return @view left_particles[1:left_count], @view right_particles[1:right_count]
end #function

function extract_ghosts(particles, x_min, x_max, R)
    n = length(particles)
    if n == 0
        return CuArray{Particle}(undef, 0)
    end
    flags = CuArray{Int32}(undef, n)
    kernel! = mark_ghosts_kernel!(CUDABackend())
    kernel!(flags, particles, n, x_min, x_max, R; ndrange=n)
    KernelAbstractions.synchronize(CUDABackend())
    return pack_ghosts(particles, flags)
end #function


# ------------------------------------------------------------
# Ghost particle exchange (GPU-only serialisation)
# ------------------------------------------------------------

function exchange_ghosts(local_particles_gpu, comm, rank, nprocs, x_min, x_max, R)
    left_rank = (rank == 0) ? nprocs - 1 : rank - 1
    right_rank = (rank == nprocs - 1) ? 0 : rank + 1

    # 1. Identify ghosts
    ghosts_left, ghosts_right = extract_ghosts(local_particles_gpu, x_min, x_max, R)

    # 2. Serialize on device
    send_left_buf = pack_particles_to_f32(ghosts_left)
    send_left_count = Int32(length(send_left_buf))
    send_right_buf = pack_particles_to_f32(ghosts_right)
    send_right_count = Int32(length(send_right_buf))

    # 3. Exchange counts with neighbors (host-level small messages)
    recv_left_count = Ref{Int32}(0)
    recv_right_count = Ref{Int32}(0)

    ghost_count_left_tag = 101
    ghost_count_right_tag = 102

    MPI.Sendrecv!(send_left_count, recv_right_count, comm,
        dest=left_rank,
        source=right_rank,
        sendtag=ghost_count_left_tag,
        recvtag=ghost_count_right_tag)
    MPI.Sendrecv!(send_right_count, recv_left_count, comm,
        dest=right_rank,
        source=left_rank,
        sendtag=ghost_count_right_tag,
        recvtag=ghost_count_left_tag)


    # allocate receive buffers on GPU
    recv_left_buf = CuArray{Float32}(undef, recv_left_count[])  # size 0 allowed
    recv_right_buf = CuArray{Float32}(undef, recv_right_count[])

    ghost_left_tag = 201
    ghost_right_tag = 202

    # 4. Actual device-aware data exchange
    reqs = MPI.Request[]
    push!(reqs, try_Irecv!(recv_left_buf, comm, left_rank, ghost_left_tag))
    push!(reqs, try_Irecv!(recv_right_buf, comm, right_rank, ghost_right_tag))
    push!(reqs, try_isend!(send_left_buf, comm, left_rank, ghost_left_tag))
    push!(reqs, try_isend!(send_right_buf, comm, right_rank, ghost_right_tag))
    MPI.Waitall!(reqs)

    # 5. Deserialize
    ghosts_left = unpack_f32_to_particles(recv_left_buf)
    ghosts_right = unpack_f32_to_particles(recv_right_buf)

    return vcat(ghosts_left, ghosts_right)
end


