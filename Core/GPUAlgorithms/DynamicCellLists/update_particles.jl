# --------- Update particles ---------

function update_particles!(particles, θ_updates, numerical_params)
    num_particles = length(particles)
    rand1 = CUDA.rand(num_particles)
    rand2 = CUDA.rand(num_particles)

    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups

    kernel! = update_particles_kernel!(CUDABackend())
    kernel!(
        particles,
        θ_updates,
        numerical_params.λ,
        numerical_params.dt,
        numerical_params.v,
        numerical_params.Lx,
        numerical_params.Ly,
        rand1,
        rand2,
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
    I = @index(Global, Linear)
    # stride = @ndrange()
    stride = 256 * 256

    for i = I:stride:num_particles
        this_particle = particles[i]
        r_i = this_particle.r
        θ_i = this_particle.θ
        uid_i = this_particle.uid
        this_rand1 = rand1[i]
        if this_rand1 < λ * dt
            this_rand2 = rand2[i]
            this_θ_update = π * this_rand2 * 2 - θ_i
        else
            this_θ_update = θ_updates[i]
        end #if

        r_i = get_new_r(r_i, θ_i, dt, v, Lx, Ly)
        θ_i = θ_i + this_θ_update
        particles[i] = Particle(r_i, θ_i, uid_i)
    end #for i
end #function

@inline function get_new_r(r, θ, dt, v, Lx, Ly)
    return @SVector [
        mod(r[1] + dt * v * cos(θ), Lx),
        mod(r[2] + dt * v * sin(θ), Ly)
    ]
end #function

