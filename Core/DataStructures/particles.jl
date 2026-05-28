# --------- Particle data structures --------- #

struct Particle
    x::Float32
    y::Float32
    θ::Float32
    uid::Int32 # "Unique id"
end #struct


# --------- Initialise structures for particle updates --------- #

function initialise_θ_updates(N)
    return CUDA.zeros(Float32, N)
end #function

function initialise_rand_bufs(N)
    rand1 = CUDA.zeros(Float32, N)
    rand2 = CUDA.zeros(Float32, N)
    return (; rand1, rand2)
end #function


# --------- Initialise particles from scratch or from txt files --------- #

function initialise_coords(N, Lx, Ly=Lx; input_files::Union{Nothing,NTuple{3,String}}=nothing)
    if isnothing(input_files)
        xs = [Lx * rand(Float32) for i in 1:N]
        ys = [Ly * rand(Float32) for i in 1:N]
        θs = [Float32(2π * rand()) for i = 1:N]
        uids = [Int32(i) for i = 1:N]
    else
        # Assume input_files=("input_xs.txt","input_ys.txt","input_thetas.txt")
        xs = Float32.(vec(readdlm(input_files[1])))
        ys = Float32.(vec(readdlm(input_files[2])))
        θs = Float32.(vec(readdlm(input_files[3])))
        length(xs) != N && error("Wrong number of particles in x input file")
        length(ys) != N && error("Wrong number of particles in y input file")
        length(θs) != N && error("Wrong number of particles in theta input file")
        uids = [Int32(i) for i = 1:N]
    end #if
    return xs, ys, θs, uids
end #function

function initialise_particles(max_particles_per_rank, input_files, numerical_params, mpi_params)
    N_total = numerical_params.N_total
    x_min_local = numerical_params.x_min_local
    x_max_local = numerical_params.x_max_local
    Lx = numerical_params.Lx
    Ly = numerical_params.Ly
    rank = mpi_params.rank
    comm = mpi_params.comm

    #Initialise particles on rank 0 and broadcast to others
    xs_all = Vector{Float32}(undef, N_total)
    ys_all = Vector{Float32}(undef, N_total)
    θs_all = Vector{Float32}(undef, N_total)
    uids_all = Vector{Int32}(undef, N_total)
    if rank == 0
        xs_all, ys_all, θs_all, uids_all = initialise_coords(N_total, Lx, Ly, input_files=input_files)
    end #if
    MPI.Bcast!(xs_all, 0, comm)
    MPI.Bcast!(ys_all, 0, comm)
    MPI.Bcast!(θs_all, 0, comm)
    MPI.Bcast!(uids_all, 0, comm)

    # Get particles in local domain
    in_local_domain(x) = (x_min_local <= x < x_max_local)
    local_particle_idxs = findall(in_local_domain, xs_all)
    xs_filtered = xs_all[local_particle_idxs]
    ys_filtered = ys_all[local_particle_idxs]
    θs_filtered = θs_all[local_particle_idxs]
    uids_filtered = uids_all[local_particle_idxs]

    # Create local particles on CPU
    local_particles_cpu = [Particle(x, y, θ, uid) for (x, y, θ, uid) in zip(xs_filtered, ys_filtered, θs_filtered, uids_filtered)]
    num_local_particles = length(local_particles_cpu)
    num_local_particles > max_particles_per_rank && error("Too many particles on rank " * string(rank))

    # Upload particles to GPU
    particles_gpu = CuArray{Particle}(undef, max_particles_per_rank)
    copyto!(particles_gpu, 1, CuArray(local_particles_cpu), 1, num_local_particles)

    return particles_gpu, num_local_particles
end #function


# --------- Load particles from simulation outputs --------- #

#The directory inputs_dir should contain the set of jld2 files for each rank from a single timestep
function load_particles(max_particles_per_rank, inputs_dir, numerical_params, mpi_params)
    N_total = numerical_params.N_total
    x_min_local = numerical_params.x_min_local
    x_max_local = numerical_params.x_max_local
    Lx = numerical_params.Lx
    Ly = numerical_params.Ly
    rank = mpi_params.rank
    comm = mpi_params.comm
    nprocs = mpi_params.nprocs

    #Load particles from files
    all_files = readdir(inputs_dir, join=true)
    if rank == 0
        length(all_files) != nprocs && error("Mismatch in number of ranks")
    end #if rank
    MPI.Barrier(comm)

    local_file = all_files[rank+1]
    local_dict = load(local_file)
    local_particles_cpu = local_dict["particles"]
    num_local_particles = length(local_particles_cpu)

    #Check configuration is valid
    in_local_domain(p::Particle) = (x_min_local <= p.x < x_max_local)
    in_global_domain(p::Particle) = (0 <= p.x <= Lx) && (0 <= p.y <= Ly)
    any(!in_local_domain, local_particles_cpu) && error("Rank $rank: Not all particles in local domain")
    any(!in_global_domain, local_particles_cpu) && error("Rank $rank: Not all particles in global domain")

    num_global_particles = MPI.Allreduce(num_local_particles, +, comm)
    if rank == 0
        num_global_particles != N_total && error("Mismatch in N_total")
    end #if rank
    MPI.Barrier(comm)

    #Upload particles to GPU
    particles_gpu = CuArray{Particle}(undef, max_particles_per_rank)
    copyto!(particles_gpu, 1, CuArray(local_particles_cpu), 1, num_local_particles)

    return particles_gpu, num_local_particles
end #function



# --------- Unpack particles into coordinates --------- #

function unpack_coords(particles_array::Array{Particle})
    xs = zeros(Float32, length(particles_array))
    ys = zeros(Float32, length(particles_array))
    θs = zeros(Float32, length(particles_array))
    uids = zeros(Int32, length(particles_array))
    @inbounds for i = 1:length(particles_array)
        particle_i = particles_array[i]
        xs[i] = particle_i.x
        ys[i] = particle_i.y
        θs[i] = particle_i.θ
        uids[i] = particle_i.uid
    end #for i
    return xs, ys, θs, uids
end #function

