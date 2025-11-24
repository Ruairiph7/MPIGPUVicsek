#NOTE:
#TODO:
#WARN: NEED TO ENSURE THAT WE ONLY CALCULATE Θ UPDATES FOR cells within our domain, not the ghost particle cells. --> possibly implement by restricting occupied_cells_lists and num_occupied_cells to number of occupied cells within our domain. Then also change argument num_cells to number of cells within the domain (i.e. cell_list_params.num_boxes - 2*num_boxes_y?)
#^^ ACTUALLY I'VE changed it to just calculate the updates for all particles, then later neglect the ones not in our local domain

# --------- Calculate θ_updates (Algorithm 3) ---------
# NOTE: Tumbling comes later in the kernel for updating particles.

function calculate_θ_updates!(θ_updates, cell_neighbours_list, cell_address_list, cell_num_particles_list, occupied_cells_particle_IDs, occupied_cells_particle_rs, occupied_cells_particle_θs, occupied_cells_ID_list, γ, dt, R², γn, Rn², Lx, Ly, cell_width, num_occupied_cells, num_cells, backend)
    #NOTE:
    #TODO:
    #WARN:CONFIRM KERNEL CONFIGURATION HERE IS CORRECT, then apply to others too
    workgroup_size = 640
    total_num_threads = workgroup_size * num_cells
    kernel! = calculate_θ_updates_kernel!(backend, workgroup_size, total_num_threads)
    kernel!(θ_updates, cell_neighbours_list, cell_address_list, cell_num_particles_list, occupied_cells_particle_IDs, occupied_cells_particle_rs, occupied_cells_particle_θs, occupied_cells_ID_list, num_occupied_cells, γ, dt, R², γn, Rn², Lx, Ly, cell_width; ndrange=total_num_threads)
    KernelAbstractions.synchronize(backend)
end #function

@kernel function calculate_θ_updates_kernel!(θ_updates, @Const(cell_neighbours_list), @Const(cell_address_list), @Const(cell_num_particles_list), @Const(occupied_cells_particle_IDs), @Const(occupied_cells_particle_rs), @Const(occupied_cells_particle_θs), @Const(occupied_cells_ID_list), @Const(num_occupied_cells), γ, dt, R², γn, Rn², Lx, Ly, cell_width)
    gi = @index(Group, Linear)
    li = @index(Local, Linear)

    if gi <= num_occupied_cells[]
        this_cell_neighbours = @localmem SVector{8,Int32} 1
        this_cell_address = @localmem Int32 1
        this_cell_num_particles = @localmem Int32 1

        neighbour_cell_address = @localmem Int32 1
        neighbour_cell_num_particles = @localmem Int32 1

        self_rs = @localmem SVector{2,Float32} 640
        self_θs = @localmem Float32 640
        neighbour_rs = @localmem SVector{2,Float32} 640
        neighbour_θs = @localmem Float32 640

        cell_index = @uniform gi
        cell_ID = occupied_cells_ID_list[cell_index]
        this_cell_neighbours = cell_neighbours_list[cell_ID]
        this_cell_address = cell_address_list[cell_ID]
        this_cell_num_particles = cell_num_particles_list[cell_ID]
        @synchronize()

        self_num_particles = @uniform this_cell_num_particles
        num_neighbours = @uniform Int32(8)
        F_sum = @uniform 0.0f0
        Fn_sum = @uniform 0.0f0
        n = @uniform Int32(1)

        if li <= self_num_particles
            #NOTE:
            #TODO:
            #WARN: May be better to switch indices so li is iterating over rows
            particle_ID = @uniform occupied_cells_particle_IDs[cell_index, li]
            self_rs[li] = occupied_cells_particle_rs[cell_index, li]
            self_θs[li] = occupied_cells_particle_θs[cell_index, li]
        end #if
        @synchronize()

        if li <= self_num_particles
            for j = 1:self_num_particles
                if li != j
                    r_ij² = sum((self_rs[j] .- self_rs[li]) .^ 2)
                    if r_ij² < R²
                        θ_ij = self_θs[j] - self_θs[li]
                        F_sum += F(θ_ij, R²)
                        n += Int32(1)
                    end #if
                    if r_ij² < Rn²
                        θ_ij = self_θs[j] - self_θs[li]
                        Fn_sum += Fn(θ_ij, Rn²)
                    end #if
                end #if
            end #for j
        end #if

        for neighbour_num = 1:num_neighbours
            neighbour_cell_ID = this_cell_neighbours[neighbour_num]
            #NOTE: The original algorithm used the if statement version
            neighbour_cell_num_particles = cell_num_particles_list[neighbour_cell_ID]
            neighbour_cell_address = cell_address_list[neighbour_cell_ID]
            @synchronize()

            neighbour_num_particles = @uniform neighbour_cell_num_particles
            if li <= neighbour_num_particles
                neighbour_cell_index = neighbour_cell_address
                # neighbour_rs[li] = occupied_cells_list[neighbour_cell_index].rs[li]
                neighbour_rs[li] = occupied_cells_particle_rs[neighbour_cell_index, li]
                neighbour_θs[li] = occupied_cells_particle_θs[neighbour_cell_index, li]
            end #if
            @synchronize

            if neighbour_num_particles > 0
                if li <= self_num_particles
                    for j = 1:neighbour_num_particles
                        Δx, Δy = self_rs[li] .- neighbour_rs[j]
                        Δx > cell_width && (Δx -= Lx)
                        Δx < -cell_width && (Δx += Lx)
                        Δy > cell_width && (Δy -= Ly)
                        Δy < -cell_width && (Δy += Ly)
                        r_ij² = Δx^2 + Δy^2
                        if r_ij² < R²
                            θ_ij = neighbour_θs[j] - self_θs[li]
                            F_sum += F(θ_ij, R²)
                            n += Int32(1)
                        end #if
                        if r_ij² < Rn²
                            θ_ij = neighbour_θs[j] - self_θs[li]
                            Fn_sum += Fn(θ_ij, Rn²)
                        end #if
                    end #for j
                end #if
            end #if
            @synchronize
        end #for neighbour_num

        if li <= self_num_particles
            θ_updates[particle_ID] = γ * F_sum * dt / n + γn * Fn_sum * dt
        end #if
    end #if gi

end #function

