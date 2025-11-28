# ------------------------------------------------------------
# GPU Migrant detection
# ------------------------------------------------------------

@kernel function sort_migrants_kernel!(
    stayers, lefts, rights,
    counters, @Const(particles), x_min, x_max, cell_width, n)

    i = @index(Global)
    if i <= n

    p = particles[i]
    x = p.r[1]
    #NOTE:
    #TODO:
    #WARN: CHECK THIS LOGIC is correct for applying PBCs here
    if x_min - cell_width <= x < x_min #Particle is in the cell immediately to the left; moved to left domain
        #NOTE:
        #TODO:
        #WARN: CHECK THIS SYNTAX is correct; chatgpt wanted just "idx = CUDA.atomic_add!(counters,2,1)"; but I took this from my old code
        idx = CUDA.atomic_add!(pointer(counters, 2), Int32(1))
        lefts[idx+1] = p
    elseif x < x_min - cell_width #Particle has been wrapped round to the left; moved to "right" domain (PBCs)
        idx = CUDA.atomic_add!(pointer(counters, 3), Int32(1))
        rights[idx+1] = p
    elseif x_max + cell_width >= x >= x_max #Particle is in the cell immediately to the right; moved to right domain
        idx = CUDA.atomic_add!(pointer(counters, 3), Int32(1))
        rights[idx+1] = p
    elseif x > x_max + cell_width #Particle has been wrapped round to the right; moved to "left" domain (PBCs)
        idx = CUDA.atomic_add!(pointer(counters, 2), Int32(1))
        lefts[idx+1] = p
    else #Particle has remained in this domain
        idx = CUDA.atomic_add!(pointer(counters, 1), Int32(1))
        stayers[idx+1] = p
    end

end #if i <= n
end

function sort_migrants(particles, x_min, x_max, cell_width)
    n = length(particles)

    if n == 0
        return (CuArray{Particle}(undef, 0), CuArray{Particle}(undef,0), CuArray{Particle}(undef,0))
    end

    stayers = CuArray{Particle}(undef, n)
    lefts = CuArray{Particle}(undef, n)
    rights = CuArray{Particle}(undef, n)
    counters = CuArray([Int32(0), Int32(0), Int32(0)])

    kernel! = sort_migrants_kernel!(CUDABackend())
    kernel!(stayers, lefts, rights, counters, particles, x_min, x_max, cell_width, n; ndrange=n)
    KernelAbstractions.synchronize(CUDABackend())

    h = Array(counters)
    n_stay, n_left, n_right = h

    return @view(stayers[1:n_stay]), @view(lefts[1:n_left]), @view(rights[1:n_right])
end

# ------------------------------------------------------------
# Migration exchange between ranks
# ------------------------------------------------------------

function exchange_migrants!(local_particles_gpu, comm, rank, nprocs, x_min, x_max, cell_width)
    left_rank = (rank == 0) ? nprocs - 1 : rank - 1
    right_rank = (rank == nprocs - 1) ? 0 : rank + 1

    # 1. GPU-only detection + compaction of migrants
    stayers, migrants_left, migrants_right = sort_migrants(local_particles_gpu, x_min, x_max, cell_width)

    # 2. Serialize migrants
    send_left_buf = pack_particles_to_f32(migrants_left)
    send_left_count = Ref{Int32}(length(send_left_buf))
    send_right_buf = pack_particles_to_f32(migrants_right)
    send_right_count = Ref{Int32}(length(send_right_buf))

    # 3. Count exchange
    recv_left_count = Ref{Int32}(0)
    recv_right_count = Ref{Int32}(0)

    migrant_count_left_tag = 301
    migrant_count_right_tag = 302

    MPI.Sendrecv!(send_left_count, recv_right_count, comm,
        dest=left_rank,
        source=right_rank,
        sendtag=migrant_count_left_tag,
        recvtag=migrant_count_left_tag)
    MPI.Sendrecv!(send_right_count, recv_left_count, comm,
        dest=right_rank,
        source=left_rank,
        sendtag=migrant_count_right_tag,
        recvtag=migrant_count_right_tag)

    # 4. Allocate recv buffers on GPU
    recv_left_buf = CuArray{Float32}(undef, recv_left_count[])
    recv_right_buf = CuArray{Float32}(undef, recv_right_count[])

    migrant_left_tag = 401
    migrant_right_tag = 402

    # 5. Exchange actual data
    # reqs = MPI.Request[]
    # push!(reqs, try_Irecv!(recv_left_buf, comm, left_rank, migrant_left_tag))
    # push!(reqs, try_Irecv!(recv_right_buf, comm, right_rank, migrant_right_tag))
    # push!(reqs, try_isend(send_left_buf, comm, left_rank, migrant_left_tag))
    # push!(reqs, try_isend(send_right_buf, comm, right_rank, migrant_right_tag))
    # MPI.Waitall!(reqs)

    MPI.Sendrecv!(send_left_buf, recv_right_buf, comm,
                  dest=left_rank,
                  source=right_rank,
                  sendtag=migrant_left_tag,
                  recvtag=migrant_left_tag)

    MPI.Sendrecv!(send_right_buf, recv_left_buf, comm,
                  dest=right_rank,
                  source=left_rank,
                  sendtag=migrant_right_tag,
                  recvtag=migrant_right_tag)


    # 6. Deserialize on device
    incoming_left = unpack_f32_to_particles(recv_left_buf)
    incoming_right = unpack_f32_to_particles(recv_right_buf)

    # 7. Rebuild local set
    return vcat(stayers, incoming_left, incoming_right)
end


