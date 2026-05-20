function run_simulation(N_total, max_steps;
    input_files::Union{Nothing,NTuple{3,String}}=nothing,
    dt::Float32=0.1f0,
    R::Float32=Float32(1 / sqrt(π)),
    γ::Float32=0.5f0,
    λ::Float32=0.08f0,
    Lx::Int32=Int32(10),
    Ly::Int32=Lx,
    v::Float32=Float32(1 / sqrt(π)),
    max_particles_per_rank::Union{Int32,Nothing}=nothing,
    max_sendrecv_particles::Union{Int32,Nothing}=nothing,
    steps_to_shrink_buffers=maximum((max_steps ÷ 10, 100000)),
    save_OPs=true,
    save_plots=true,
    save_coords=false,
    steps_to_save_OPs=100,
    steps_to_save_plots=100,
    steps_to_save_coords=10,
    steps_to_new_OP_file::Int=500000,
    file_name_addon::String="",
    markersize=0.5,
    steps_to_log=maximum((max_steps ÷ 10, 1)),
    ASYNC_SAVES::Union{Bool,Nothing}=nothing
)

    # --------- Prepare for MPI --------- #

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    mpi_params = (; comm, rank, nprocs)

    local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
    local_rank = MPI.Comm_rank(local_comm)
    local_num_devices = CUDA.ndevices()
    if local_rank >= local_num_devices
        error("Rank $rank: local_rank $local_rank exceeds available GPUs ($local_num_devices) on this node")
    end #if
    CUDA.device!(local_rank)

    # Flag if we only have a single MPI rank -- avoid communication 
    SINGLE_RANK = nprocs == 1
    if SINGLE_RANK
        println("NOTE: Running on a single MPI rank.")
        max_particles_per_rank = N_total
        max_sendrecv_particles = 0
    end #if

    # Report configuration from each rank
    for r in 0:nprocs-1
        if rank == r
            CPU_affinity = chomp(read(`taskset -cp $(getpid())`, String))
            CPU_affinity = CPU_affinity[findfirst(==(':'), CPU_affinity)+2:end]
            println("""
                    Rank $rank/$(nprocs-1):
                              GPU : $(CUDA.name(CUDA.device())) (device $local_rank)
                    Julia threads : $(Threads.nthreads())
                     CPU affinity : $(CPU_affinity)
                    """)
            flush(stdout)
        end
        MPI.Barrier(comm)
    end

    # --------- Store parameters --------- #

    # Characterise local domain
    Lx_local = Lx / nprocs
    x_min_local = rank * Lx_local
    x_max_local = (rank + 1) * Lx_local

    #Store numerical parameters
    R² = R^2
    inv_πR² = Float32(1.0f0 / (π * R²))
    numerical_params = (; N_total,
        dt, R, R², inv_πR², γ, λ, v,
        Lx, Ly, Lx_local,
        x_min_local, x_max_local)

    #Store output parameters
    output_params = (; save_OPs,
        save_plots,
        save_coords,
        steps_to_save_OPs,
        steps_to_save_plots,
        steps_to_save_coords,
        steps_to_new_OP_file,
        file_name_addon,
        markersize)

    #Set max_particles_per_rank
    if isnothing(max_particles_per_rank)
        max_particles_per_rank = maximum((ceil(Int32, 2 * N_total / nprocs), Int32(10000)))
        rank == 0 && @show max_particles_per_rank
    end #if

    #Set max_sendrecv_particles - i.e. maximum ghosts/migrants in a given direction
    if isnothing(max_sendrecv_particles)
        max_sendrecv_particles = maximum(ceil(Int32, 2 * R_max * N_total / Lx), Int32(1000))
        rank == 0 && @show max_sendrecv_particles
    end #if


    # --------- Prepare for saving outputs --------- #

    #Open file if saving order parameter - will all be handled by rank 0
    OP_m_file = nothing
    if rank == 0
        if save_plots
            plots_dir = "plots"
            mkpath(plots_dir)
        end #if save_snapshots
        if save_OPs
            OP_dir = "OPs"
            mkpath(OP_dir)
            OP_file_number = 1
            OP_m_file = open("$OP_dir/OP_m_$(file_name_addon)_1.txt", "w")
        end #if save_OPs
    end #if

    #Struct to aid in transferring/writing particles to disk
    if isnothing(ASYNC_SAVES)
        ASYNC_SAVES = Threads.nthreads() > 1
    end #if
    println("Rank $rank: Asynchronous saving set to '$ASYNC_SAVES'")
    save_bufs = SaveBuffers(max_particles_per_rank, ASYNC_SAVES=ASYNC_SAVES)

    # --------- Initialise data structures --------- #

    #Initialise cell lists
    cell_list_params = CellListParams(numerical_params, SINGLE_RANK=SINGLE_RANK)
    cells_data = CellList(cell_list_params, max_particles_per_rank)

    #Initialse MPI buffers
    ghost_bufs = GhostBuffers(max_sendrecv_particles)
    migrant_bufs = MigrantBuffers(max_particles_per_rank, max_sendrecv_particles)
    sendrecv_bufs = SendRecvBuffers(max_sendrecv_particles)

    #Initialise array to store particles, the first num_local_particles entries corresponding to those in our local domain
    particles, num_local_particles = initialise_particles(
        max_particles_per_rank,
        input_files,
        numerical_params,
        mpi_params)

    #Get a view to local_particles from the larger array
    local_particles = view(particles, 1:num_local_particles)

    #Prepare array to store θ_updates, will calculate for all particles then later only use ones in our domain
    θ_updates = initialise_θ_updates(max_particles_per_rank)

    #Initialise buffers to store random numbers for particle updates
    rand_bufs = initialise_rand_bufs(max_particles_per_rank)



    TI = time() ###########################################################################################################################################


    # --------- Perform simulation --------- #

    rank == 0 && println("Starting simulation...")
    for time_step = 1:max_steps

        if rank == 0 && time_step % steps_to_log == 0
            println("Step: $time_step")
        end #if

        # T1 = time() ###########################################################################################################################################

        # --------- Ghost particle exchange --------- #

        #Exchange ghosts serialized into buffers
        recv_left_buf, recv_right_buf = exchange_ghosts!(
            sendrecv_bufs,
            local_particles,
            ghost_bufs,
            numerical_params,
            mpi_params,
            SINGLE_RANK=SINGLE_RANK)

        #Check if we need to raise max_particles_per_rank (locally on just this rank)
        n_left = length(recv_left_buf) ÷ 4
        n_right = length(recv_right_buf) ÷ 4
        extended_num_local_particles = num_local_particles + n_left + n_right

        if extended_num_local_particles > max_particles_per_rank
            max_particles_per_rank = ceil(Int32, extended_num_local_particles * 1.1) #Raise maximum
            println("Rank $rank: Rasing max_particles_per_rank to $max_particles_per_rank")

            # Reset particles array (stash on CPU to avoid excess GPU memory use)
            local_particles_cpu = Array(local_particles)
            particles = CuVector{Particle}(undef, max_particles_per_rank)
            particles[1:num_local_particles] .= CuArray(local_particles_cpu)
            local_particles = view(particles, 1:num_local_particles)

            #Reinitialise relevant data structures
            θ_updates = initialise_θ_updates(max_particles_per_rank)
            rand_bufs = initialise_rand_bufs(max_particles_per_rank)
            cells_data = CellList(cell_list_params, max_particles_per_rank)

            reallocate_save_bufs!(save_bufs, max_particles_per_rank)

            #Else: try to lower maximum every steps_to_shrink_buffers steps
        elseif time_step % steps_to_shrink_buffers == 0 && max_particles_per_rank > 1.7 * extended_num_local_particles
            max_particles_per_rank = maximum((ceil(Int32, extended_num_local_particles * 1.7), Int32(10000)))
            println("Rank $rank: Lowering max_particles_per_rank to $max_particles_per_rank")

            # Reset particles array (stash on CPU to avoid excess GPU memory use)
            local_particles_cpu = Array(local_particles)
            particles = CuVector{Particle}(undef, max_particles_per_rank)
            particles[1:num_local_particles] .= CuArray(local_particles_cpu)
            local_particles = view(particles, 1:num_local_particles)

            #Reinitialise relevant data structures
            θ_updates = initialise_θ_updates(max_particles_per_rank)
            rand_bufs = initialise_rand_bufs(max_particles_per_rank)
            cells_data = CellList(cell_list_params, max_particles_per_rank)

            reallocate_save_bufs!(save_bufs, max_particles_per_rank)
        end #if

        #Deserialize ghosts and add into particles after local_particles
        unpack_particles!(particles, num_local_particles, recv_left_buf, recv_right_buf)


        # --------- Update particles --------- #
        if num_local_particles != 0
            #Updates based on local particles + ghosts
            get_updates!(
                θ_updates,
                view(particles, 1:extended_num_local_particles),
                cells_data,
                cell_list_params,
                extended_num_local_particles,
                numerical_params)

            #Update local particles only
            update_particles!(
                local_particles,
                θ_updates,
                numerical_params,
                rand_bufs)
        end #if


        # --------- Migrant particle exchange --------- #

        #Find stayers; exchange migrants serialized into buffers
        stayers, recv_left_buf, recv_right_buf = exchange_migrants!(
            sendrecv_bufs,
            local_particles,
            migrant_bufs,
            numerical_params,
            mpi_params,
            SINGLE_RANK=SINGLE_RANK)

        #Check if we need to raise max_particles_per_rank (locally on just this rank)
        n_stay = length(stayers)
        n_left = length(recv_left_buf) ÷ 4
        n_right = length(recv_right_buf) ÷ 4
        num_local_particles = n_stay + n_left + n_right

        if num_local_particles > max_particles_per_rank
            max_particles_per_rank = ceil(Int32, num_local_particles * 1.1) #Raise maximum
            println("Rank $rank: Rasing max_particles_per_rank to $max_particles_per_rank")

            #Reinitialise relevant data structures
            particles = CuVector{Particle}(undef, max_particles_per_rank)
            θ_updates = initialise_θ_updates(max_particles_per_rank)
            rand_bufs = initialise_rand_bufs(max_particles_per_rank)
            cells_data = CellList(cell_list_params, max_particles_per_rank)

            reallocate_save_bufs!(save_bufs, max_particles_per_rank)
        end #if

        #Load stayers into the beginning of particles
        particles[1:n_stay] .= stayers
        #Deserialize migrants and add into particles after stayers
        unpack_particles!(particles, n_stay, recv_left_buf, recv_right_buf)
        #Reset local_particles
        local_particles = view(particles, 1:num_local_particles)


        # T2 = time() ###########################################################################################################################################
        # println("Rank $(mpi_params.rank): step $time_step compute time: $(round(T2 - T1, digits=3))s")

        # --------- Write outputs --------- #

        if save_coords
            _save_coords(
                time_step, local_particles, num_local_particles,
                save_bufs, output_params, mpi_params)
        end #if

        if save_plots
            _save_plots(
                time_step, local_particles,
                output_params, numerical_params, mpi_params)
        end #if

        if save_OPs
            _save_OPs(
                time_step, local_particles, OP_m_file,
                output_params, numerical_params, mpi_params)
        end #if

        if rank == 0 && time_step % steps_to_new_OP_file == 0
            OP_file_number = OP_file_number + 1
            close(OP_m_file)
            OP_m_file = open(OP_dir * "OP_m_" * file_name_addon * "_$OP_file_number.txt", "w")
        end #if time_step

        KernelAbstractions.synchronize(CUDABackend())

        # T3 = time() ###########################################################################################################################################
        # println("Rank $(mpi_params.rank): step $time_step outputs time: $(round(T3 - T2, digits=3))s")
        # println("Rank $(mpi_params.rank): step $time_step wall time: $(round(T3 - T1, digits=3))s")


    end #for time_step


    #Close file if saving order parameter
    if rank == 0
        if save_OPs
            close(OP_m_file)
        end #if save_OPs
    end #if (rank == 0)

    #Wait for final saves to finish
    if save_bufs.ASYNC_SAVES && !isnothing(save_bufs.save_task)
        wait(save_bufs.save_task)
    end #if
    save_bufs.pinned_buf = Vector{Particle}(undef, 0) #Drop reference to pinned buffer
    GC.gc() #Encourage GC to collect old pinned buffer


    TF = time() ###########################################################################################################################################
    println("Rank $(mpi_params.rank): Total time: $(round(TF - TI, digits=3))s")


    MPI.Barrier(comm)
    return nothing
end #function


