# --------- Reset permutation buffer to the identity ---------

function reset_perm!(cells_data, num_particles)
    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups

    kernel! = reset_perm_kernel!(CUDABackend(), workgroup_size)
    kernel!(cells_data.perm, num_particles; ndrange=total_num_threads)
    KernelAbstractions.synchronize(CUDABackend())
end #function

@kernel function reset_perm_kernel!(perm, num_particles)
    I = Int32(@index(Global, Linear))
    stride = Int32(@ndrange()[1])
    for i = I:stride:num_particles
        perm[i] = Int32(i)
    end #for i
end #function
