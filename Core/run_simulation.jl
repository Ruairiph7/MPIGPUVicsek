#NOTE:
#TODO:
#WARN: Fix need to make whole new files to change max_particles_in_cell --> make it assigned by building the kernels within this function

# --------- Perform simulation ---------

function run_simulation(N_total, max_steps;
    input_files::Union{Nothing,NTuple{3,String}}=nothing,
    dt::Float32=0.1f0,
    R::Float32=Float32(1 / sqrt(π)),
    Rn::Float32=Float32(0),
    γ::Float32=0.5f0,
    γn::Float32=0.0f0,
    λ::Float32=0.08f0,
    Lx::Int32=Int32(10),
    Ly::Int32=Lx,
    v::Float32=Float32(1 / sqrt(π)),
    max_particles_per_rank::Union{Int32,Nothing}=nothing,
    max_sendrecv_particles::Union{Int32,Nothing}=nothing,
    max_particles_in_cell::Int=512,
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
)

    # ----- Prepare for MPI -----
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    mpi_params = (; comm, rank, nprocs)

    #TODO: CHECK THIS
    # # set CUDA device using local rank mapping to avoid colliding GPUs across nodes
    # Use shared communicator to get local rank on node
    local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
    local_rank = MPI.Comm_rank(local_comm)
    # Bind to local GPU slot
    CUDA.device!(local_rank)

    # Flag if we only have a single MPI rank -- avoid communication 
    SINGLE_RANK = nprocs == 1
    if SINGLE_RANK
        println("NOTE: Running on a single MPI rank.")
        max_particles_per_rank = N_total
        max_sendrecv_particles = 0
    end #if

    #Set max_particles_per_rank
    if isnothing(max_particles_per_rank)
        max_particles_per_rank = maximum((ceil(Int32, 2 * N_total / nprocs), Int32(10000)))
        rank == 0 && @show max_particles_per_rank
    end #if isnothing()

    # Characterise local domain
    Lx_local = Lx / nprocs
    x_min_local = rank * Lx_local
    x_max_local = (rank + 1) * Lx_local

    #Store numerical parameters
    R² = R^2
    Rn² = Rn^2
    R_max = maximum([R, Rn])
    numerical_params = (; N_total,
        dt, R, Rn, R², Rn², R_max,
        γ, γn, λ, Lx, Ly, v, Lx_local,
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

    #Initialise cell lists
    cell_list_params = CellListParams(numerical_params, SINGLE_RANK=SINGLE_RANK)
    cells_data = CellList(cell_list_params, max_particles_per_rank)

    #Set max_sendrecv_particles - i.e. maximum ghosts/migrants in a given direction
    if isnothing(max_sendrecv_particles)
        max_sendrecv_particles = ceil(Int32, max_particles_in_cell * cell_list_params.num_cells_y)
        rank == 0 && @show max_sendrecv_particles
    end #if isnothing()
    sendrecv_buf_length = 4 * max_sendrecv_particles #(Buffers are serialised)
    @warn "max_sendrecv_particles could cause issues ---investigate."
    #WARN: May choose to remove max_particles_in_cell as an argument, so max_sendrecv_particles will need to be set differently. Check whether this can cause out of bounds errors (previously max_particles_in_cell would have been exceeded first), and find how to solve.

    #Initialse buffers to track ghost and migrant particles
    ghost_bufs = GhostBuffers(max_sendrecv_particles)
    migrant_bufs = MigrantBuffers(max_particles_per_rank, max_sendrecv_particles)

    #Initialise buffers to sendrecv ghost and migrant particles
    sendrecv_bufs = SendRecvBuffers(sendrecv_buf_length)

    #Initialise array to store particles, the first num_local_particles entries corresponding to those in our local domain
    particles, num_local_particles = initialise_particles(
        max_particles_per_rank,
        input_files,
        numerical_params,
        mpi_params)

    #Get a view to local_particles from the larger array
    local_particles = view(particles, 1:num_local_particles)

    #Prepare array to store θ_updates, calculate for all particles then later only use ones in our domain
    #(greatly simplifies code)
    θ_updates = initialise_θ_updates(max_particles_per_rank)

    #Initialise buffers to store random numbers for particle updates
    rand_bufs = initialise_rand_bufs(max_particles_per_rank)

    #Open file if saving order parameter - will all be handled by rank 0
    OP_m_file = nothing
    OP_S_file = nothing
    if rank == 0
        if save_plots
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
    for time_step = 1:max_steps

        if rank == 0 && time_step % steps_to_log == 0
            println("Step: " * string(time_step))
        end #if

        #Ghost particle exchange to get all interacting particles
        #---------------------------------------------#
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
            println("Rank " * string(rank) * ": Rasing max_particles_per_rank to " * string(max_particles_per_rank))

            local_particles_cpu = Array(local_particles) #Store local particles
            particles = CuArray{Particle}(undef, max_particles_per_rank) #Reallocate particles
            particles[1:num_local_particles] .= CuArray(local_particles_cpu) #Retrieve local particles
            local_particles = view(particles, 1:num_local_particles) #Reset local_particles
            θ_updates = initialise_θ_updates(max_particles_per_rank) #Reinitialise θ_updates
            rand_bufs = initialise_rand_bufs(max_particles_per_rank) #Reinitialse rand_bufs
            cells_data = CellList(cell_list_params, max_particles_per_rank) #Reintialise cells_data

        #Else: try to lower maximum every steps_to_shrink_buffers steps
        elseif time_step % steps_to_shrink_buffers == 0 && max_particles_per_rank > 1.7 * extended_num_local_particles
            max_particles_per_rank = maximum((ceil(Int32, extended_num_local_particles * 1.7), Int32(10000))) #Lower maximum
            println("Rank " * string(rank) * ": Lowering max_particles_per_rank to " * string(max_particles_per_rank))

            local_particles_cpu = Array(local_particles) #Store local particles
            particles = CuArray{Particle}(undef, max_particles_per_rank) #Reallocate particles
            particles[1:num_local_particles] .= CuArray(local_particles_cpu) #Retrieve local particles
            local_particles = view(particles, 1:num_local_particles) #Reset local_particles
            θ_updates = initialise_θ_updates(max_particles_per_rank) #Reinitialise θ_updates
            rand_bufs = initialise_rand_bufs(max_particles_per_rank) #Reinitialse rand_bufs
            cells_data = CellList(cell_list_params, max_particles_per_rank) #Reintialise cells_data
        end #if

        #Deserialize ghosts and add into particles after local_particles
        unpack_f32_to_particles!(particles, num_local_particles, recv_left_buf, recv_right_buf)
        #---------------------------------------------#


        #Update particle coordinates
        #---------------------------------------------#
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

        end #if num_local_particles != 0
        #---------------------------------------------#

        #Migrate particles that have moved domains
        #---------------------------------------------#
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
            println("Rank " * string(rank) * ": Rasing max_particles_per_rank to " * string(max_particles_per_rank))
            particles = CuArray{Particle}(undef, max_particles_per_rank) #Reallocate particles
            θ_updates = initialise_θ_updates(max_particles_per_rank) #Reinitialise θ_updates
            rand_bufs = initialise_rand_bufs(max_particles_per_rank) #Reinitialse rand_bufs
            cells_data = CellList(cell_list_params, max_particles_per_rank) #Reintialise cells_data
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
        if save_coords
            write_coords(time_step, local_particles, output_params, mpi_params)
        end #if

        if save_plots || save_OPs
            save_plots_and_OPs(
                time_step,
                local_particles,
                OP_m_file,
                OP_S_file,
                output_params,
                numerical_params,
                mpi_params)
        end #if

        if rank == 0 && time_step % steps_to_new_OP_file == 0
            OP_file_number = OP_file_number + 1
            close(OP_m_file)
            close(OP_S_file)
            OP_m_file = open(OP_dir * "OP_m_" * file_name_addon * "_" * string(OP_file_number) * ".txt", "w")
            OP_S_file = open(OP_dir * "OP_S_" * file_name_addon * "_" * string(OP_file_number) * ".txt", "w")
        end #if time_step
        #---------------------------------------------#

        KernelAbstractions.synchronize(CUDABackend())
    end #for time_step

    #Close file if saving order parameter
    if rank == 0
        if save_OPs
            close(OP_m_file)
            close(OP_S_file)
        end #if save_OPs
    end #if (rank == 0)

    MPI.Barrier(comm)
    return nothing
end #function


