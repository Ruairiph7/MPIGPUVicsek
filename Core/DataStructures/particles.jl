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


# --------- Initialise particles --------- #

function initialise_coords(N, Lx, Ly=Lx; input_files::Union{Nothing,NTuple{3,String}}=nothing)
    if isnothing(input_files)
        xs = [Lx * rand(Float32) for i in 1:N]
        ys = [Ly * rand(Float32) for i in 1:N]
        θs = [Float32(2π * rand()) for i = 1:N]
    else
        xs = Float32.(vec(readdlm(input_files[1])))
        ys = Float32.(vec(readdlm(input_files[2])))
        θs = Float32.(vec(readdlm(input_files[3])))
        length(xs) != N && error("Wrong number of particles in x input file")
        length(ys) != N && error("Wrong number of particles in y input file")
        length(θs) != N && error("Wrong number of particles in theta input file")
    end #if
    return xs, ys, θs
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
    if rank == 0
        xs_all, ys_all, θs_all = initialise_coords(N_total, Lx, Ly, input_files=input_files)
    end #if
    MPI.Bcast!(xs_all, 0, comm)
    MPI.Bcast!(ys_all, 0, comm)
    MPI.Bcast!(θs_all, 0, comm)

    # Get particles in local domain
    function in_local_domain(x)
        return x_min_local <= x < x_max_local
    end #function
    local_particle_idxs = findall(in_local_domain, xs_all)
    xs_filtered = xs_all[local_particle_idxs]
    ys_filtered = ys_all[local_particle_idxs]
    θs_filtered = θs_all[local_particle_idxs]

    N_local = length(xs_filtered)
    N_offset = MPI.Exscan(N_local, +, comm)
    rank == 0 && (N_offset = 0)

    # Create local particles on CPU, using N_offset to assign unique IDs that hold globally
    local_particles_cpu = [
        Particle(x, y, θ, Int32(N_offset + i))
        for (i, (x, y, θ)) in enumerate(zip(xs_filtered, ys_filtered, θs_filtered))
    ]

    # Initialise particles array on CPU, using the first num_local_particles entries for the ones in our domain
    num_local_particles = length(local_particles_cpu)
    num_local_particles > max_particles_per_rank && error("Too many particles on rank " * string(rank))

    particles_cpu = Vector{Particle}(undef, max_particles_per_rank)
    particles_cpu[1:num_local_particles] .= local_particles_cpu

    # Load onto the GPU and return with num_local_particles
    return CuArray(particles_cpu), num_local_particles
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

