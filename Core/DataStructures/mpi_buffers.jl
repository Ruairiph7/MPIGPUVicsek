# --------- Ghost particles --------- #

mutable struct GhostBuffers
    lefts::CuVector{Particle}
    rights::CuVector{Particle}
    counters::CuVector{Int64}
    overflow_flag::CuVector{Int64}
    buf_lengths::Int64
end #struct
GhostBuffers(max_particles::Union{Int64,Int64}) = GhostBuffers(
    CuVector{Particle}(undef, max_particles),
    CuVector{Particle}(undef, max_particles),
    CUDA.zeros(Int64, 2),
    CUDA.zeros(Int64, 1),
    Int64(max_particles)
)


# --------- Migrant particles --------- #

mutable struct MigrantBuffers
    stayers::CuVector{Particle}
    lefts::CuVector{Particle}
    rights::CuVector{Particle}
    counters::CuVector{Int64}
    overflow_flag::CuVector{Int64}
    buf_lengths::Int64
    stayer_overflow_flag::CuVector{Int64}
    stayer_buf_length::Int64
end #struct
MigrantBuffers(
    max_particles_per_rank::Union{Int64,Int64},
    max_sendrecv_particles::Union{Int64,Int64}
) = MigrantBuffers(
    CuVector{Particle}(undef, max_particles_per_rank),
    CuVector{Particle}(undef, max_sendrecv_particles),
    CuVector{Particle}(undef, max_sendrecv_particles),
    CUDA.zeros(Int64, 3),
    CUDA.zeros(Int64, 1),
    Int64(max_sendrecv_particles),
    CUDA.zeros(Int64, 1),
    Int64(max_particles_per_rank)
)


# --------- MPI Send/Recv Buffers --------- #

mutable struct SendRecvBuffers
    send_left::CuVector{UInt64}
    send_right::CuVector{UInt64}
    recv_left::CuVector{UInt64}
    recv_right::CuVector{UInt64}
    buf_lengths::Int64
end #struct
SendRecvBuffers(max_particles::Union{Int64,Int64}) = SendRecvBuffers(
    CuVector{UInt64}(undef, 4 * max_particles),
    CuVector{UInt64}(undef, 4 * max_particles),
    CuVector{UInt64}(undef, 4 * max_particles),
    CuVector{UInt64}(undef, 4 * max_particles),
    Int64(4 * max_particles)
)
