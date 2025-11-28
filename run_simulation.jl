#NOTE:
#TODO:
#WARN: Fix need to make whole new files to change max_particles_in_cell --> make it assigned by building the kernels within this function
#ALSO need to fix clash of naming with IDs from cell lists algorithm and ids for particles

# --------- Perform simulation ---------

function run_simulation(N_total, max_steps;
        comm=MPI.COMM_WORLD,
        dt::Float32=0.1f0,
        R::Float32=Float32(1 / sqrt(π)),
        Rn::Float32=Float32(1 / sqrt(π)),
        γ::Float32=0.5f0,
        γn::Float32=0.5f0,
        λ::Float32=0.08f0,
        Lx::Int32=Int32(10),
        Ly::Int32=Lx,
        v::Float32=Float32(1 / sqrt(π)),
        max_num_occupied_cells::Union{Int32,Nothing}=nothing,
        max_particles_in_cell::Int=640,
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
    # set_cuda_device_for_rank(comm)
    # function set_cuda_device_for_rank(comm)
    # Use shared communicator to get local rank on node
    local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
    local_rank = MPI.Comm_rank(local_comm)
    # Bind to local GPU slot
    CUDA.device!(local_rank)
    # return local_rank
    # end


    #Initialise particles on rank 0 and broadcast to others
    # rs_all, rs_all_flat, θs_all = nothing, nothing, nothing
    rs_all = Vector{SVector{2,Float32}}(undef,N_total)
    rs_all_flat = Vector{Float32}(undef,2*N_total)
    θs_all = Vector{Float32}(undef,N_total)
    if rank == 0
        rs_all, θs_all = initialise_coords(N_total, Lx, Ly, input_files=input_files)

        rs_all_flat = Float32[x for r in rs_all for x in r]
    end

    # rs_all_flat = MPI.Bcast(rs_all_flat, 0, comm)
    MPI.Bcast!(rs_all_flat, 0, comm)
    rs_all = [SVector{2,Float32}(rs_all_flat[2i-1:2i]) for i in 1:(length(rs_all_flat) ÷ 2)]

    # θs_all = MPI.Bcast(θs_all, 0, comm)
    MPI.Bcast!(θs_all, 0, comm)

    # Characterise local domain
    Lx_local = Lx / nprocs
    x_min = rank * Lx_local
    x_max = (rank + 1) * Lx_local

    # Get particles in local domain
    function in_local_domain(r)
        return x_min <= r[1] < x_max
    end #function
    local_particle_idxs = findall(in_local_domain, rs_all)
    rs_filtered = rs_all[local_particle_idxs]
    θs_filtered = θs_all[local_particle_idxs]

    N_local = length(rs_filtered)
    N_offset = MPI.Exscan(N_local, +, comm)
    rank == 0 && (N_offset = 0)

    # Create local particles on device, using N_offset to assign unique IDs that hold globally
    local_particles_gpu = CuArray([
                                   Particle(r, θ, Int32(N_offset + i))
                                   for (i, (r, θ)) in enumerate(zip(rs_filtered, θs_filtered))
                                  ])

    #Initialise cell list parameters
    cell_width = maximum([R, Rn])
    #NOTE:
    #TODO:
    # WARN: I need to check this works correctly with periodic boundaries - maybe its dealt with by wraps later (I THINK THIS IS THE CASE)
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

    #Open file if saving order parameter - will all be handled by rank 0
    if rank == 0
        if save_snapshots
            plots_dir = "plots/"
            mkpath(plots_dir)
        end #if save_snapshots
        if save_OPs
            OP_dir = "OPs/"
            mkpath(OP_dir)
            OP_file_number = 1
            OP_m_file = open("OP_m_" * file_name_addon * "_1.txt", "w")
            OP_S_file = open("OP_S_" * file_name_addon * "_1.txt", "w")
        end #if save_OPs
    end #if

    #Perform simulation
    R² = R^2
    Rn² = Rn^2

    ##NOTE: FOR BENCHMARKING
    #times = zeros(max_steps)

    for time_step = 1:max_steps

        ##NOTE: FOR BENCHMARKING
        #times[time_step] = @elapsed begin

        if rank == 0 && time_step % 1 == 0
            println("Step: " * string(time_step))
        end #if

        @show rank, time_step, local_particles_gpu, x_min, x_max

        #Ghost particle exchange to get all interacting particles, store in CuArray "particles"
        ghost_particles_gpu = exchange_ghosts(local_particles_gpu, comm, rank, nprocs, x_min, x_max, R)
        particles = vcat(local_particles_gpu, ghost_particles_gpu)

        if length(local_particles_gpu) != 0

            #Prepare array to store θ_updates, calculate for all particles then later only use ones in our domain
            #(greatly simplifies code)
            θ_updates = initialise_θ_updates(length(particles), ArrayType=ArrayType)

            # Carry out cell lists algorithms
            #---------------------------------------------#
            #Algorithm 1
            prep_cell_lists!(cell_address_list, cell_num_particles_list, occupied_cells_ID_list,
                             num_occupied_cells, particles, cell_list_params, backend)

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
            assign_particles!(occupied_cells_particle_IDs, occupied_cells_particle_rs, occupied_cells_particle_θs, cell_address_list, cell_num_particles_list, particles, cell_list_params, backend)

            #Algorithm 3
            calculate_θ_updates!(θ_updates, cell_neighbours_list, cell_address_list, cell_num_particles_list, occupied_cells_particle_IDs, occupied_cells_particle_rs, occupied_cells_particle_θs, occupied_cells_ID_list, γ, dt, R², γn, Rn², Lx, Ly, cell_width, num_occupied_cells, cell_list_params.num_boxes, backend)

            #---------------------------------------------#

            #NOTE:
            #TODO:
            #WARN: CHECK this does correctly still allocate into the particles array:
            #
            #Update local particles only
            update_particles!(view(particles, 1:length(local_particles_gpu)), θ_updates, λ, dt, v, Lx, Ly, CUDABackend())

        else # -> length(local_particles_gpu) = 0
            # @show rank, "no local particles"
        end #if length(local_particles_gpu) != 0


        #NOTE:
        #TODO:
        #WARN: CHECK THAT this is correct, I have commented out the version CHATGPT wanted:
        #  
        #Migrate particles that have moved domains
        # local_particles_gpu = exchange_migrants!(local_particles_gpu, comm, rank, nprocs, x_min, x_max)
        local_particles_gpu = exchange_migrants!(view(particles, 1:length(local_particles_gpu)), comm, rank, nprocs, x_min, x_max, cell_width)


        #NOTE:
        #TODO:
        #WARN:NEED TO MAKE PLOTTING WORK WITH MPI -- SEND ALL PARTICLES TO ONE RANK OR MAYBE DO STUFF LOCALLY, WRITE COORDS LOCALLY (with particle ids) WITHOUT COMMINICATING
        #Could be better to do this, or to put stuff on CPU then move particles to one rank in order to ensure it can handle the memory cost (GPU has less memory than CPU?)
        if saving_coords_on_the_go
            save_coords(time_step, saving_coords_on_the_go, steps_to_save_on_the_go, file_name_addon, local_particles_gpu, rank)
        end #if

        if save_outputs
            save_plots_and_OPs(time_step, save_outputs, steps_to_save, local_particles_gpu, save_snapshots, save_OPs, file_name_addon, markersize, OP_m_file, OP_S_file, rank, comm)
        end #if

        if rank == 0 && time_step % steps_to_new_OP_file == 0
            OP_file_number = OP_file_number + 1
            close(OP_m_file)
            close(OP_S_file)
            OP_m_file = open("OP_m_" * file_name_addon * "_" * string(OP_file_number) * ".txt", "w")
            OP_S_file = open("OP_S_" * file_name_addon * "_" * string(OP_file_number) * ".txt", "w")
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
        #     rs, θs = unpack_coords(Array(particles))
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


