# --------- Locate particles to cells (Algorithm 2) ---------

#NOTE:
#TODO:
#WARN: IVE changed the code from passing each element of cell_list_params individually to just passing the whole struct as an argument --> check this still works
function assign_particles!(occupied_cells_particle_IDs, occupied_cells_particle_rs, occupied_cells_particle_θs, cell_address_list, cell_num_particles_list, particles, cell_list_params, backend)
    cell_num_particles_list .= Int32(0)
    KernelAbstractions.synchronize(backend)
    ##-----------------
    kernel! = assign_particles_kernel!(backend)
    kernel!(occupied_cells_particle_IDs, occupied_cells_particle_rs, occupied_cells_particle_θs, cell_address_list, cell_num_particles_list, particles, cell_list_params; ndrange=length(particles))
    KernelAbstractions.synchronize(backend)
end

@kernel function assign_particles_kernel!(occupied_cells_particle_IDs, occupied_cells_particle_rs, occupied_cells_particle_θs, @Const(cell_address_list), cell_num_particles_list, @Const(particles), cell_list_params)
    i = @index(Global, Linear)

    particle_i = particles[i]
    r_i = particle_i.r
    θ_i = particle_i.θ
    cell_ID = get_cell_ID(r_i, cell_list_params)
    old_num_particles = CUDA.atomic_add!(pointer(cell_num_particles_list, cell_ID), Int32(1))
    particle_address = cell_address_list[cell_ID]

    occupied_cells_particle_IDs[particle_address, old_num_particles+1] = Int32(i)
    occupied_cells_particle_rs[particle_address, old_num_particles+1] = r_i
    occupied_cells_particle_θs[particle_address, old_num_particles+1] = θ_i
end #function


