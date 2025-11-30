include("./particles.jl")

# ------------------- Ghost particles ----------------------
mutable struct GhostBuffers
    flags::CuArray{Int32}
    lefts::CuArray{Particle}
    rights::CuArray{Particle}
    counters::CuArray{Int32}
end #struct
GhostBuffers(max_particles::Int) = GhostBuffers(
    CuArray{Int32}(undef, max_particles),
    CuArray{Particle}(undef, max_particles),
    CuArray{Particle}(undef, max_particles),
    CuArray(zeros(Float32, 2))
)

# ------------------- Migrant particles ----------------------
mutable struct MigrantBuffers
    stayers::CuArray{Particle}
    lefts::CuArray{Particle}
    rights::CuArray{Particle}
    counters::CuArray{Int32}
end #struct
MigrantBuffers(max_particles::Int) = MigrantBuffers(
    CuArray{Particle}(undef, max_particles),
    CuArray{Particle}(undef, max_particles),
    CuArray{Particle}(undef, max_particles),
    CuArray(zeros(Float32, 3))
)

