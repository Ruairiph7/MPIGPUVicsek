function unpack_particles!(particles, base_num_particles, left_buf, right_buf)
    n_left = length(left_buf) ÷ 4
    n_right = length(right_buf) ÷ 4
    n_left == n_right == 0 && return nothing
    num_particles = base_num_particles + n_left + n_right

    workgroup_size = STD_WORKGROUP_SIZE
    num_workgroups = STD_NUM_WORKGROUPS
    total_num_threads = workgroup_size * num_workgroups
    kernel! = unpack_particles_kernel!(CUDABackend())

    if n_left != 0
        kernel!(view(particles, base_num_particles+1:base_num_particles+n_left), left_buf, n_left; ndrange=total_num_threads)
        KernelAbstractions.synchronize(CUDABackend())
    end # if 

    if n_right != 0
        kernel!(view(particles, base_num_particles+n_left+1:num_particles), right_buf, n_right; ndrange=total_num_threads)
        KernelAbstractions.synchronize(CUDABackend())
    end # if 

    return nothing
end #function

@kernel function unpack_particles_kernel!(out, @Const(buf), size)
    I = Int64(@index(Global, Linear))
    stride = Int64(@ndrange()[1])

    for i = I:stride:size
        base = 4 * (i - 1)
        x = reinterpret(Float64, buf[base+1])
        y = reinterpret(Float64, buf[base+2])
        θ = reinterpret(Float64, buf[base+3])
        uid = reinterpret(Int64, buf[base+4])
        out[i] = Particle(x, y, θ, uid)
    end #for i
end #function
