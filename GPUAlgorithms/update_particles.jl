# --------- Update particles ---------

function update_particles!(particles, θ_updates, λ, dt, v, Lx, Ly, backend)
    kernel! = update_particles_kernel!(backend)
    if backend == CUDABackend()
        rand1 = CUDA.rand(length(particles))
        rand2 = CUDA.rand(length(particles))
    else
        rand1 = rand(Float32, length(particles))
        rand2 = rand(Float32, length(particles))
    end
    kernel!(particles, θ_updates, λ, dt, v, Lx, Ly, rand1, rand2; ndrange=length(particles))
    KernelAbstractions.synchronize(backend)
end #function

@kernel function update_particles_kernel!(particles, @Const(θ_updates), λ, dt, v, Lx, Ly, @Const(rand1), @Const(rand2))
    i = @index(Global, Linear)
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
end #function

function get_new_r(r, θ, dt, v, Lx, Ly)
    return @SVector [
        mod(r[1] + dt * v * cos(θ), Lx),
        mod(r[2] + dt * v * sin(θ), Ly)
    ]
end #function

