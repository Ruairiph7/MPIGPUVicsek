const two_π_f32 = Float32(2*π)

function update_particles!(particles, θ_updates, numerical_params, rand_bufs)
    num_particles = length(particles)

    cuRAND.rand!(rand_bufs.rand1)
    cuRAND.rand!(rand_bufs.rand2)

    workgroup_size = 256
    num_workgroups = 512
    total_num_threads = workgroup_size * num_workgroups
    kernel! = update_particles_kernel!(CUDABackend(), workgroup_size)

    kernel!(
        particles,
        θ_updates,
        numerical_params.λ,
        numerical_params.dt,
        numerical_params.v,
        numerical_params.Lx,
        numerical_params.Ly,
        rand_bufs.rand1,
        rand_bufs.rand2,
        num_particles;
        ndrange=total_num_threads
    )
    KernelAbstractions.synchronize(CUDABackend())
end #function

@kernel function update_particles_kernel!(
    particles,
    @Const(θ_updates),
    λ,
    dt,
    v,
    Lx,
    Ly,
    @Const(rand1),
    @Const(rand2),
    num_particles
)
    I = Int32(@index(Global, Linear))
    stride = Int32(@ndrange()[1])

    for i = I:stride:num_particles
        this_particle = particles[i]
        x_i = this_particle.x
        y_i = this_particle.y
        θ_i = this_particle.θ
        uid_i = this_particle.uid
        this_rand1 = rand1[i]
        if this_rand1 < λ * dt
            this_rand2 = rand2[i]
            this_θ_update = two_π_f32 * this_rand2 - θ_i
        else
            this_θ_update = θ_updates[i]
        end #if

        x_i = mod(x_i + dt * v * cos(θ_i), Lx)
        y_i = mod(y_i + dt * v * sin(θ_i), Ly)
        θ_i = θ_i + this_θ_update
        particles[i] = Particle(x_i, y_i, θ_i, uid_i)
    end #for i
end #function
