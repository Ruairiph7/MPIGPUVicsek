###############################################
# Packing particles → Float32 buffer on GPU
###############################################

@kernel function serialize_kernel!(out, @Const(particles), size, x_offset)
    I = Int32(@index(Global, Linear))
    stride = Int32(@ndrange()[1])

    for i = I:stride:size
        p = particles[i]
        base = 4 * (i - 1)
        out[base+1] = reinterpret(UInt32, p.x + x_offset)
        out[base+2] = reinterpret(UInt32, p.y)
        out[base+3] = reinterpret(UInt32, p.θ)
        out[base+4] = reinterpret(UInt32, p.uid)
    end #for i
end

function pack_particles!(bufs::SendRecvBuffers, lefts::CuVector{Particle}, rights::CuVector{Particle}; GHOST_FLAG::Bool=false, Lx::Int32=Int32(0), rank::Int=0, nprocs::Int=0)
    n_left = length(lefts)
    n_right = length(rights)
    left_count = Ref{Int32}(4 * n_left)
    right_count = Ref{Int32}(4 * n_right)
    n_left == n_right == 0 && return left_count, right_count

    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups
    kernel! = serialize_kernel!(CUDABackend())

    x_offset = 0.0f0
    if n_left != 0
        GHOST_FLAG && (rank == 0) && (x_offset = Float32(Lx))
        kernel!(view(bufs.send_left_buf, 1:4*n_left), lefts, n_left, x_offset; ndrange=total_num_threads)
        KernelAbstractions.synchronize(CUDABackend())
    end #if (n_left != 0)
    if n_right != 0
        GHOST_FLAG && (rank == nprocs - 1) && (x_offset = Float32(-Lx))
        kernel!(view(bufs.send_right_buf, 1:4*n_right), rights, n_right, x_offset; ndrange=total_num_threads)
        KernelAbstractions.synchronize(CUDABackend())
    end #if (n_right != 0)

    return left_count, right_count
end


