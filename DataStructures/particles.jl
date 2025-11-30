using StaticArrays

# --------- Data structures ---------

struct Particle
    r::SVector{2,Float32}
    θ::Float32
    uid::Int32 #"Unique id"
end

# --------- Initialise particles ---------

# Initialise just coordinates - used for MPI version
function initialise_coords(N, Lx, Ly=Lx; input_files::Union{Nothing,NTuple{3,String}}=nothing)
    if isnothing(input_files)
        rs = [(Lx, Ly) .* @SVector(rand(Float32, 2)) for i = 1:N]
        θs = [Float32(2π * rand()) for i = 1:N]
    else
        xs = Float32.(vec(readdlm(input_files[1])))
        ys = Float32.(vec(readdlm(input_files[2])))
        θs = Float32.(vec(readdlm(input_files[3])))
        length(xs) != N && error("Wrong number of particles in x input file")
        length(ys) != N && error("Wrong number of particles in y input file")
        length(θs) != N && error("Wrong number of particles in theta input file")
        rs = [SVector{2,Float32}([xs[i], ys[i]]) for i = 1:N]
    end
    return rs, θs
end #function

function initialise_particles(max_particles_per_rank, x_min, x_max, N_total, Lx, Ly, input_files, rank, comm)

    #Initialise particles on rank 0 and broadcast to others
    rs_all = Vector{SVector{2,Float32}}(undef, N_total)
    rs_all_flat = Vector{Float32}(undef, 2 * N_total)
    θs_all = Vector{Float32}(undef, N_total)
    if rank == 0
        rs_all, θs_all = initialise_coords(N_total, Lx, Ly, input_files=input_files)
        rs_all_flat = Float32[x for r in rs_all for x in r]
    end
    MPI.Bcast!(rs_all_flat, 0, comm)
    rs_all = [SVector{2,Float32}(rs_all_flat[2i-1:2i]) for i in 1:(length(rs_all_flat)÷2)]
    MPI.Bcast!(θs_all, 0, comm)

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

    # Create local particles on CPU, using N_offset to assign unique IDs that hold globally
    local_particles_cpu = [
        Particle(r, θ, Int32(N_offset + i))
        for (i, (r, θ)) in enumerate(zip(rs_filtered, θs_filtered))
    ]

    # Initialise particles array on CPU, using the first num_local_particles entries for the ones in our domain
    num_local_particles = length(local_particles_cpu)
    num_local_particles > max_particles_per_rank && error("Too many particles on rank " * string(rank))

    particles_cpu = Vector{Particle}(undef, max_particles_per_rank)
    particles_cpu[1:num_local_particles] .= local_particles_cpu

    # Load onto the GPU and return with num_local_particles
    return CuArray(particles_cpu), num_local_particles
end #function

# --------- Unpack particles into coordinates ---------

function unpack_coords(particles_array::Array{Particle})
    rs = zeros(SVector{2,Float32}, length(particles_array))
    θs = zeros(Float32, length(particles_array))
    uids = zeros(Int32, length(particles_array))
    @inbounds for i = 1:length(particles_array)
        particle_i = particles_array[i]
        rs[i] = particle_i.r
        θs[i] = particle_i.θ
        uids[i] = particle_i.uid
    end #for i
    return rs, θs, uids
end #function

