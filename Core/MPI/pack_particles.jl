# --------- Pack particles into UInt64 buffers to send over MPI --------- #

function pack_particles!(
    bufs::SendRecvBuffers,
    lefts::CuVector{Particle},
    rights::CuVector{Particle};
    GHOST_FLAG::Bool=false,
    Lx::Int64=Int64(0),
    rank::Int=0,
    nprocs::Int=0)

    n_left = length(lefts)
    n_right = length(rights)
    left_count = Ref{Int64}(4 * n_left)
    right_count = Ref{Int64}(4 * n_right)
    n_left == n_right == 0 && return left_count, right_count

    workgroup_size = STD_WORKGROUP_SIZE
    num_workgroups = STD_NUM_WORKGROUPS
    total_num_threads = workgroup_size * num_workgroups
    kernel! = pack_particles_kernel!(CUDABackend())

    if n_left != 0
        x_offset = 0.0
        GHOST_FLAG && (rank == 0) && (x_offset = Float64(Lx))
        kernel!(view(bufs.send_left, 1:4*n_left), lefts, n_left, x_offset; ndrange=total_num_threads)
        KernelAbstractions.synchronize(CUDABackend())
    end #if 

    if n_right != 0
        x_offset = 0.0
        GHOST_FLAG && (rank == nprocs - 1) && (x_offset = Float64(-Lx))
        kernel!(view(bufs.send_right, 1:4*n_right), rights, n_right, x_offset; ndrange=total_num_threads)
        KernelAbstractions.synchronize(CUDABackend())
    end #if 

    return left_count, right_count
end #function

@kernel function pack_particles_kernel!(out, @Const(particles), size, x_offset)
    I = Int64(@index(Global, Linear))
    stride = Int64(@ndrange()[1])

    for i = I:stride:size
        p = particles[i]
        base = 4 * (i - 1)
        out[base+1] = reinterpret(UInt64, p.x + x_offset)
        out[base+2] = reinterpret(UInt64, p.y)
        out[base+3] = reinterpret(UInt64, p.θ)
        out[base+4] = reinterpret(UInt64, p.uid)
    end #for i
end #function

