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
    max_particles_in_cell::Int=64,
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
    steps_to_save_on_the_go=10)

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
        max_particles_per_rank = ceil(Int32, 1.5 * N_total / nprocs)
    end #if isnothing()

    #Initialse buffers to track ghost and migrant particles
    ghost_bufs = GhostBuffers(max_particles_per_rank)
    migrant_bufs = MigrantBuffers(max_particles_per_rank)

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

    steps_to_log = maximum((max_steps ÷ 10, 1))
    steps_to_lower_max_num_occupied_cells = maximum(max_steps ÷ 10, 100000)
    for time_step = 1:max_steps
        if rank == 0 && time_step % steps_to_log == 0
            println("Step: " * string(time_step))
        end #if

        #Ghost particle exchange to get all interacting particles, store in CuArray "particles"
        ghost_particles = exchange_ghosts(local_particles, comm, rank, nprocs, x_min, x_max, R, ghost_bufs)

        num_ghosts = length(ghost_particles)
        num_particles = num_local_particles + num_ghosts
        particles[num_local_particles+1:num_particles] .= ghost_particles

        if num_local_particles != 0

            # Carry out cell lists algorithms
            #---------------------------------------------#
            #Algorithm 1
            prep_cell_lists!(cell_address_list, cell_num_particles_list, occupied_cells_ID_list,
                num_occupied_cells, view(particles, 1:num_particles), cell_list_params, num_particles)

            #Check if we can lower max_num_occupied_cells at regular intervals
            if time_step % steps_to_lower_max_num_occupied_cells == 0
                lower_max_num_occupied_cells = lower_max_num_occupied_cells(num_occupied_cells, occupied_cells_particle_IDs, cell_list_params)
                if lower_max_num_occupied_cells != false
                    new_max, max_particles_in_cell = lower_max_num_occupied_cells
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
        new_local_particles = exchange_migrants!(local_particles, comm, rank, nprocs, x_min, x_max, cell_width, migrant_bufs)
        num_local_particles = length(new_local_particles)

        particles[1:num_local_particles] .= new_local_particles
        local_particles = view(particles, 1:num_local_particles)

        if saving_coords_on_the_go
            save_coords(time_step, steps_to_save_on_the_go, file_name_addon, local_particles, rank)
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

        KernelAbstractions.synchronize(backend)

        # # NOTE: FOR BENCHMARKING
        # end #begin

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

    ##NOTE: FOR BENCHMARKING
    #return times

    MPI.Barrier(comm)
    return nothing
end #function


