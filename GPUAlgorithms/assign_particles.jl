@warn "assign_particles! workgroup_size, num_workgroups hard-coded at 256"

# --------- Locate particles to cells (Algorithm 2) ---------
function assign_particles!(occupied_cells_particle_IDs, occupied_cells_particle_rs, occupied_cells_particle_θs, cell_address_list, cell_num_particles_list, particles, cell_list_params, num_particles)
    cell_num_particles_list .= Int32(0)

    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups

    kernel! = assign_particles_kernel!(CUDABackend())
    kernel!(occupied_cells_particle_IDs, occupied_cells_particle_rs, occupied_cells_particle_θs, cell_address_list, cell_num_particles_list, particles, cell_list_params, num_particles; ndrange=total_num_threads)
    KernelAbstractions.synchronize(CUDABackend())
end

@kernel function assign_particles_kernel!(occupied_cells_particle_IDs, occupied_cells_particle_rs, occupied_cells_particle_θs, @Const(cell_address_list), cell_num_particles_list, @Const(particles), cell_list_params, num_particles)
    I = @index(Global, Linear)
    stride = @ndrange()

    for i = I:stride:num_particles
        particle_i = particles[i]
        r_i = particle_i.r
        θ_i = particle_i.θ
        cell_ID = get_cell_ID(r_i, cell_list_params)
        old_num_particles = CUDA.atomic_add!(pointer(cell_num_particles_list, cell_ID), Int32(1))
        particle_address = cell_address_list[cell_ID]

        occupied_cells_particle_IDs[particle_address, old_num_particles+1] = Int32(i)
        occupied_cells_particle_rs[particle_address, old_num_particles+1] = r_i
        occupied_cells_particle_θs[particle_address, old_num_particles+1] = θ_i
    end #for i
end #function


