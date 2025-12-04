#NOTE:
#TODO:
#WARN: Fix need to make whole new files to change max_particles_in_cell --> make it assigned by building the kernels within this function

# --------- Perform simulation ---------

function run_simulation(N_total, max_steps;
    comm=MPI.COMM_WORLD,
    dt::Float32=0.1f0,
    R::Float32=Float32(1 / sqrt(π)),
    Rn::Float32=Float32(1 / sqrt(π)),
    γ::Float32=0.5f0,
    γn::Float32=0.0f0,
    λ::Float32=0.08f0,
    Lx::Int32=Int32(10),
    Ly::Int32=Lx,
    v::Float32=Float32(1 / sqrt(π)),
    max_num_occupied_cells::Union{Int32,Nothing}=nothing,
    max_particles_per_rank::Union{Int32,Nothing}=nothing,
    max_sendrecv_particles::Union{Int32,Nothing}=nothing,
    max_particles_in_cell::Int=512,
    ArrayType=CuArray,
    steps_to_save=100,
    save_outputs=true,
    save_OPs=true,
    steps_to_new_OP_file::Int=500000,
    markersize=0.5,
    input_files::Union{Nothing,NTuple{3,String}}=nothing,
    file_name_addon::String="",
    save_snapshots=true,
    write_final_coords=true,
    saving_coords_on_the_go=false,
    steps_to_save_on_the_go=10,
    steps_to_log=maximum((max_steps ÷ 10, 1)),
    steps_to_shrink_buffers=maximum((max_steps ÷ 10, 100000))
)


    #Store correct backend
    if ArrayType == CuArray
        backend = CUDABackend()
    else
        error("Code only set up to work for CuArrays")
        # backend = KernelAbstractions.CPU()
    end

    # ----- Prepare for MPI -----
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if rank == 0
        @warn "assign_particles! workgroup_size, num_workgroups hard-coded at 256"
        @warn "prep_cell_lists! workgroup_size, num_workgroups hard-coded at 256"
        @warn "update_particles! workgroup_size, num_workgroups hard-coded at 256"
        @warn "extract_ghosts workgroup_size, num_workgroups hard-coded at 256"
        @warn "extract_migrants workgroup_size, num_workgroups hard-coded at 256"
        @warn "(de)serialize_kernel workgroup_size, num_workgroups hard-coded at 256"
        @warn "HAVE calculate_θ_updates workgroup_size AND max_particles_in_cell HARD CODED AT 512"
    end #if (rank == 0)

    #TODO: CHECK THIS
    # # set CUDA device using local rank mapping to avoid colliding GPUs across nodes
    # Use shared communicator to get local rank on node
    local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
    local_rank = MPI.Comm_rank(local_comm)
    # Bind to local GPU slot
    CUDA.device!(local_rank)

    # Characterise local domain
    Lx_local = Lx / nprocs
    x_min = rank * Lx_local
    x_max = (rank + 1) * Lx_local

    #Initialise cell list parameters
    cell_width = maximum([R, Rn])
    cell_list_params = CellListParams(x_min - cell_width, Lx_local + 2 * cell_width, Ly, cell_width)

    #Set max_num_occupied_cells
    if isnothing(max_num_occupied_cells)
        max_num_occupied_cells = ceil(Int32, 4 * cell_list_params.num_boxes / 7)
        @show rank, max_num_occupied_cells
    end #if isnothing()

    #Initialise dynamic cell lists data structures
    num_occupied_cells = ArrayType([Int32(0)])
    (
        cell_neighbours_list,
        cell_address_list,
        cell_num_particles_list,
        occupied_cells_particle_IDs,
        occupied_cells_particle_rs,
        occupied_cells_particle_θs,
        occupied_cells_ID_list
    ) = initialise_data_structures(cell_list_params, max_num_occupied_cells, max_particles_in_cell, ArrayType)

    #Set max_particles_per_rank
    if isnothing(max_particles_per_rank)
        max_particles_per_rank = maximum((ceil(Int32, 2 * N_total / nprocs), Int32(10000)))
        @show rank, max_particles_per_rank
    end #if isnothing()

    #Set max_sendrecv_particles - i.e. maximum ghosts/migrants in a given direction
    if isnothing(max_sendrecv_particles)
        max_sendrecv_particles = ceil(Int32, max_particles_in_cell * cell_list_params.num_boxes_y)
        @show rank, max_sendrecv_particles
    end #if isnothing()
    sendrecv_buf_length = 4 * max_sendrecv_particles #(Buffers are serialised)

    #Initialse buffers to track ghost and migrant particles
    ghost_bufs = GhostBuffers(max_sendrecv_particles)
    migrant_bufs = MigrantBuffers(max_particles_per_rank, max_sendrecv_particles)

    #Initialise buffers to sendrecv ghost and migrant particles
    sendrecv_bufs = SendRecvBuffers(sendrecv_buf_length)

    #Initialise array to store particles, the first num_local_particles entries corresponding to those in our local domain
    particles, num_local_particles = initialise_particles(max_particles_per_rank, x_min, x_max, N_total, Lx, Ly, input_files, rank, comm)

    #Get a view to local_particles from the larger array
    local_particles = view(particles, 1:num_local_particles)

    #Prepare array to store θ_updates, calculate for all particles then later only use ones in our domain
    #(greatly simplifies code)
    θ_updates = initialise_θ_updates(max_particles_per_rank, ArrayType=ArrayType)

    #Open file if saving order parameter - will all be handled by rank 0
    OP_m_file = nothing
    OP_S_file = nothing
    if rank == 0
        if save_snapshots
            plots_dir = "plots/"
            mkpath(plots_dir)
        end #if save_snapshots
        if save_OPs
            OP_dir = "OPs/"
            mkpath(OP_dir)
            OP_file_number = 1
            OP_m_file = open(OP_dir * "OP_m_" * file_name_addon * "_1.txt", "w")
            OP_S_file = open(OP_dir * "OP_S_" * file_name_addon * "_1.txt", "w")
        end #if save_OPs
    end #if

    #Perform simulation
    R² = R^2
    Rn² = Rn^2

    ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    ##NOTE: For benchmarking:
    #@warn "DOING BENCHMARKING...."
    #local_times = zeros(max_steps)
    ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

    for time_step = 1:max_steps

        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
        ##NOTE: For benchmarking:
        #MPI.Barrier(comm)
        #start_time = MPI.Wtime()
        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

        if rank == 0 && time_step % steps_to_log == 0
            println("Step: " * string(time_step))
        end #if

        #Ghost particle exchange to get all interacting particles
        #---------------------------------------------#

        #Exchange ghosts serialized into buffers
        recv_left_buf, recv_right_buf = exchange_ghosts!(sendrecv_bufs, local_particles, comm, rank, nprocs, x_min, x_max, R, ghost_bufs)

        #Check if we need to raise max_particles_per_rank (locally on just this rank)
        n_left = length(recv_left_buf) ÷ 4
        n_right = length(recv_right_buf) ÷ 4
        num_particles = num_local_particles + n_left + n_right
        if num_particles > max_particles_per_rank
            max_particles_per_rank = ceil(Int32, num_particles * 1.1) #Raise maximum
            println("Rank " * string(rank) * ": Rasing max_particles_per_rank to " * string(max_particles_per_rank))
            local_particles_cpu = Array(local_particles) #Store local particles
            particles = CuArray{Particle}(undef, max_particles_per_rank) #Reallocate particles
            particles[1:num_local_particles] .= CuArray(local_particles_cpu) #Retrieve local particles
            local_particles = view(particles, 1:num_local_particles) #Reset local_particles
            θ_updates = initialise_θ_updates(max_particles_per_rank) #Reinitialise θ_updates
        #Else: try to lower maximum every steps_to_shrink_buffers steps
        elseif time_step % steps_to_shrink_buffers == 0 && max_particles_per_rank > 1.7 * num_particles
            max_particles_per_rank = maximum((ceil(Int32, num_particles * 1.7), Int32(10000))) #Lower maximum
            println("Rank " * string(rank) * ": Lowering max_particles_per_rank to " * string(max_particles_per_rank))
            local_particles_cpu = Array(local_particles) #Store local particles
            particles = CuArray{Particle}(undef, max_particles_per_rank) #Reallocate particles
            particles[1:num_local_particles] .= CuArray(local_particles_cpu) #Retrieve local particles
            local_particles = view(particles, 1:num_local_particles) #Reset local_particles
            θ_updates = initialise_θ_updates(max_particles_per_rank) #Reinitialise θ_updates
        end #if

        #Deserialize ghosts and add into particles after local_particles
        unpack_f32_to_particles!(particles, num_local_particles, recv_left_buf, recv_right_buf)
        #---------------------------------------------#

        if num_local_particles != 0

            # Carry out cell lists algorithms
            #---------------------------------------------#
            #Algorithm 1
            prep_cell_lists!(cell_address_list, cell_num_particles_list, occupied_cells_ID_list,
                num_occupied_cells, view(particles, 1:num_particles), cell_list_params, num_particles)

            #Check if we can lower max_num_occupied_cells at regular intervals
            if time_step % steps_to_shrink_buffers == 0
                lower_max_num_occupied_cells = lower_max_num_occupied_cells_check(num_occupied_cells, occupied_cells_particle_IDs, cell_list_params)
                if lower_max_num_occupied_cells != false
                    new_max, max_particles_in_cell = lower_max_num_occupied_cells
                    println("Rank " * string(rank) * ": Lowering max_num_occupied_cells to " * string(new_max))
                    (
                        occupied_cells_particle_IDs,
                        occupied_cells_particle_rs,
                        occupied_cells_particle_θs,
                    ) = reallocate_occupied_cells_lists(new_max, max_particles_in_cell, ArrayType)
                    KernelAbstractions.synchronize(backend)
                end #if lower_max_num_occupied_cells
            end #if time_step

            #Check if we need to update max_num_occupied_cells
            update_max_num_occupied_cells = check_num_occupied_cells(num_occupied_cells,
                occupied_cells_particle_IDs, cell_list_params)
            if update_max_num_occupied_cells != false
                new_max, max_particles_in_cell = update_max_num_occupied_cells
                println("Rank " * string(rank) * ": Raising max_num_occupied_cells to " * string(new_max))
                (
                    occupied_cells_particle_IDs,
                    occupied_cells_particle_rs,
                    occupied_cells_particle_θs,
                ) = reallocate_occupied_cells_lists(new_max, max_particles_in_cell, ArrayType)
                KernelAbstractions.synchronize(backend)
            end #if

            #Algorithm 2
            assign_particles!(occupied_cells_particle_IDs, occupied_cells_particle_rs, occupied_cells_particle_θs, cell_address_list, cell_num_particles_list, view(particles, 1:num_particles), cell_list_params, num_particles)

            #Algorithm 3
            calculate_θ_updates!(θ_updates, cell_neighbours_list, cell_address_list, cell_num_particles_list, occupied_cells_particle_IDs, occupied_cells_particle_rs, occupied_cells_particle_θs, occupied_cells_ID_list, γ, dt, R², γn, Rn², Lx, Ly, cell_width, num_occupied_cells, cell_list_params.num_boxes)

            #---------------------------------------------#

            #Update local particles only
            update_particles!(local_particles, θ_updates, λ, dt, v, Lx, Ly)

        else # -> num_local_particles = 0
            # @show rank, "no local particles"
        end #if num_local_particles != 0


        #Migrate particles that have moved domains
        #---------------------------------------------#
        #Find stayers; exchange migrants serialized into buffers
        stayers, recv_left_buf, recv_right_buf = exchange_migrants!(sendrecv_bufs, local_particles, comm, rank, nprocs, x_min, x_max, cell_width, migrant_bufs)

        #Check if we need to raise max_particles_per_rank (locally on just this rank)
        n_stay = length(stayers)
        n_left = length(recv_left_buf) ÷ 4
        n_right = length(recv_right_buf) ÷ 4
        num_local_particles = n_stay + n_left + n_right
        if num_local_particles > max_particles_per_rank
            max_particles_per_rank = num_particles * 1.1 #Raise maximum
            println("Rank " * string(rank) * ": Rasing max_particles_per_rank to " * string(max_particles_per_rank))
            particles = CuArray{Particle}(undef, max_particles_per_rank) #Reallocate particles
            θ_updates = initialise_θ_updates(max_particles_per_rank) #Reinitialise θ_updates
        end #if

        #Load stayers into the beginning of particles
        particles[1:n_stay] .= stayers
        #Deserialize migrants and add into particles after stayers
        unpack_f32_to_particles!(particles, n_stay, recv_left_buf, recv_right_buf)
        #Reset local_particles
        local_particles = view(particles, 1:num_local_particles)
        #---------------------------------------------#


        #Deal with outputs
        #---------------------------------------------#
        if saving_coords_on_the_go
            save_coords(time_step, steps_to_save_on_the_go, file_name_addon, local_particles, rank, comm)
        end #if

        if save_outputs
            save_plots_and_OPs(time_step, steps_to_save, local_particles, save_snapshots, save_OPs, file_name_addon, markersize, Lx, Ly, dt, OP_m_file, OP_S_file, rank, comm)
        end #if

        if rank == 0 && time_step % steps_to_new_OP_file == 0
            OP_file_number = OP_file_number + 1
            close(OP_m_file)
            close(OP_S_file)
            OP_m_file = open(OP_dir * "OP_m_" * file_name_addon * "_" * string(OP_file_number) * ".txt", "w")
            OP_S_file = open(OP_dir * "OP_S_" * file_name_addon * "_" * string(OP_file_number) * ".txt", "w")
        end #if time_step
        #---------------------------------------------#

        KernelAbstractions.synchronize(backend)

        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
        ##NOTE: For benchmarking:
        #MPI.Barrier(comm)
        #local_times[time_step] = MPI.Wtime() - start_time
        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

    end #for time_step

    #Close file if saving order parameter
    if rank == 0
        if save_OPs
            close(OP_m_file)
            close(OP_S_file)
        end #if save_OPs
        #WARN: Below fails as I need to first collect particles onto rank 0 (-> fix if needed)
        # if write_final_coords
        #     rs, θs, uids = unpack_coords(Array(particles))
        #     writedlm("final_xs_" * file_name_addon * ".txt", [rs[i][1] for i = 1:N])
        #     writedlm("final_ys_" * file_name_addon * ".txt", [rs[i][2] for i = 1:N])
        #     writedlm("final_thetas_" * file_name_addon * ".txt", θs)
        # end #if

    end #if (rank == 0)

    ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    ###NOTE: For benchmarking:
    #writedlm("times_rank"*string(rank)*"_"*file_name_addon*".txt",local_times)
    ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

    MPI.Barrier(comm)
    return nothing
end #function


