# ------------------------------------------------------------
# GPU Migrant detection
# ------------------------------------------------------------

@kernel function sort_migrants_kernel!(stayers, lefts, rights, counters, @Const(particles), x_min, x_max, min_cell_width, n)
    I = @index(Global)
    # stride = @ndrange()
    stride = 256 * 256

    for i = I:stride:n
        p = particles[i]
        x = p.r[1]
        if x_min - min_cell_width <= x < x_min #Particle is in the cell immediately to the left; moved to left domain
            idx = CUDA.atomic_add!(pointer(counters, 2), Int32(1))
            lefts[idx+1] = p
        elseif x < x_min - min_cell_width #Particle has been wrapped round to the left; moved to "right" domain (PBCs)
            idx = CUDA.atomic_add!(pointer(counters, 3), Int32(1))
            rights[idx+1] = p
        elseif x_max + min_cell_width >= x >= x_max #Particle is in the cell immediately to the right; moved to right domain
            idx = CUDA.atomic_add!(pointer(counters, 3), Int32(1))
            rights[idx+1] = p
        elseif x > x_max + min_cell_width #Particle has been wrapped round to the right; moved to "left" domain (PBCs)
            idx = CUDA.atomic_add!(pointer(counters, 2), Int32(1))
            lefts[idx+1] = p
        else #Particle has remained in this domain
            idx = CUDA.atomic_add!(pointer(counters, 1), Int32(1))
            stayers[idx+1] = p
        end
    end #for i
end

function sort_migrants!(bufs, particles, x_min, x_max, min_cell_width)
    n = length(particles)
    if n == 0
        return (CuArray{Particle}(undef, 0), CuArray{Particle}(undef, 0), CuArray{Particle}(undef, 0))
    end

    bufs.counters .= Int32(0)

    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups

    kernel! = sort_migrants_kernel!(CUDABackend())
    kernel!(bufs.stayers, bufs.lefts, bufs.rights, bufs.counters, particles, x_min, x_max, min_cell_width, n; ndrange=total_num_threads)
    KernelAbstractions.synchronize(CUDABackend())

    counters_cpu = Array(bufs.counters)
    n_stay, n_left, n_right = counters_cpu

    return @view(bufs.stayers[1:n_stay]), @view(bufs.lefts[1:n_left]), @view(bufs.rights[1:n_right])
end

# ------------------------------------------------------------
# Migration exchange between ranks
# ------------------------------------------------------------
function exchange_migrants!(mpi_bufs, local_particles, comm, rank, nprocs, x_min, x_max, min_cell_width, migrant_bufs; SINGLE_RANK=false)

    # --- If only a single GPU, all particles are stayers ---
    if SINGLE_RANK
        return local_particles, CuArray{Particle}(undef, 0), CuArray{Particle}(undef, 0)
    end #if SINGLE_RANK

    # --- Otherwise: ---

    left_rank = (rank == 0) ? nprocs - 1 : rank - 1
    right_rank = (rank == nprocs - 1) ? 0 : rank + 1

    # 1. GPU-only detection + compaction of migrants
    stayers_view, migrants_left_view, migrants_right_view = sort_migrants!(migrant_bufs, local_particles, x_min, x_max, min_cell_width)

    # 2. Serialize migrants
    send_left_count, send_right_count = pack_particles_to_f32!(mpi_bufs, migrants_left_view, migrants_right_view)
    @assert send_left_count < mpi_bufs.buf_lengths "Too many migrants"
    @assert send_right_count < mpi_bufs.buf_lengths "Too many migrants"

    send_left_buf = view(mpi_bufs.send_left_buf, 1:getindex(send_left_count))
    send_right_buf = view(mpi_bufs.send_right_buf, 1:getindex(send_right_count))

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
    recv_left_buf = view(mpi_bufs.recv_left_buf, 1:getindex(recv_left_count))
    recv_right_buf = view(mpi_bufs.recv_right_buf, 1:getindex(recv_right_count))

    migrant_left_tag = 401
    migrant_right_tag = 402

    # 5. Exchange actual data
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

    return stayers_view, recv_left_buf, recv_right_buf
end
