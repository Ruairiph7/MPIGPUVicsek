###############################################
# Packing particles → Float32 buffer on GPU
###############################################

@kernel function serialize_kernel!(out, @Const(particles), size)
    I = @index(Global)
    stride = 256*256

    for i = I:stride:size
        p = particles[i]
        base = 4 * (i - 1)
        out[base+1] = p.r[1]
        out[base+2] = p.r[2]
        out[base+3] = p.θ
        out[base+4] = Float32(p.uid)
    end #for i
end

function pack_particles_to_f32!(bufs::SendRecvBuffers, lefts::CuArray{Particle}, rights::CuArray{Particle})
    n_left = length(lefts)
    n_right = length(rights)
    left_count = Ref{Int32}(4 * n_left)
    right_count = Ref{Int32}(4 * n_right)
    n_left == n_right == 0 && return left_count, right_count

    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups
    kernel! = serialize_kernel!(CUDABackend())

    if n_left != 0
        kernel!(view(bufs.send_left_buf, 1:4*n_left), lefts, n_left; ndrange=total_num_threads)
        KernelAbstractions.synchronize(CUDABackend())
    end #if (n_left != 0)
    if n_right != 0
        kernel!(view(bufs.send_right_buf, 1:4*n_right), rights, n_right; ndrange=total_num_threads)
        KernelAbstractions.synchronize(CUDABackend())
    end #if (n_right != 0)

    return left_count, right_count
end

###############################################
# Unpacking Float32 buffer → particles on GPU
###############################################

@kernel function deserialize_kernel!(out, @Const(buf), size)
    I = @index(Global)
    stride = 256*256

    for i = I:stride:size
        base = 4 * (i - 1)
        x = buf[base+1]
        y = buf[base+2]
        θ = buf[base+3]
        uid = Int32(buf[base+4])
        out[i] = Particle(SVector{2,Float32}(x, y), θ, uid)
    end #for i
end

function unpack_f32_to_particles!(particles, base_num_particles, left_buf, right_buf)
    n_left = length(left_buf) ÷ 4
    n_right = length(right_buf) ÷ 4
    n_left == n_right == 0 && return nothing
    num_particles = base_num_particles + n_left + n_right

    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups
    kernel! = deserialize_kernel!(CUDABackend())

    if n_left != 0
        kernel!(view(particles, base_num_particles+1:base_num_particles+n_left), left_buf, n_left; ndrange=total_num_threads)
        KernelAbstractions.synchronize(CUDABackend())
    end # if (n_left != 0)
    if n_right != 0
        kernel!(view(particles, base_num_particles+n_left+1:num_particles), right_buf, n_right; ndrange=total_num_threads)
        KernelAbstractions.synchronize(CUDABackend())
    end # if (n_right != 0)

    return nothing
end
