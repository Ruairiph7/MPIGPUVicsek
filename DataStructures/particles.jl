using StaticArrays
using CUDA

# --------- Data structures ---------

struct Particle
    r::SVector{2,Float32}
    θ::Float32
    uid::Int32 #"Unique id"
end
Particle(r, θ, uid) = Particle(r, θ, uid)

# --------- Initialise particles ---------

function initialise_θ_updates(N; ArrayType=CuArray)
    return ArrayType(zeros(Float32, N))
end #function

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


# --------- Unpack particles into coordinates ---------

function unpack_coords(particles_array::Array{Particle})
    rs = zeros(SVector{2,Float32}, length(particles_array))
    θs = zeros(Float32, length(particles_array))
    @inbounds for i = 1:length(particles_array)
        particle_i = particles_array[i]
        rs[i] = particle_i.r
        θs[i] = particle_i.θ
    end #for i
    return rs, θs
end #function

