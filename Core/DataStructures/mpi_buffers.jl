# --------- Ghost particles --------- #

mutable struct GhostBuffers
    lefts::CuVector{Particle}
    rights::CuVector{Particle}
    counters::CuVector{Int32}
    overflow_flag::CuVector{Int32}
    buf_lengths::Int32
end #struct
GhostBuffers(max_particles::Union{Int32,Int64}) = GhostBuffers(
    CuVector{Particle}(undef, max_particles),
    CuVector{Particle}(undef, max_particles),
    CUDA.zeros(Int32, 2),
    CUDA.zeros(Int32, 1),
    Int32(max_particles)
)


# --------- Migrant particles --------- #

mutable struct MigrantBuffers
    stayers::CuVector{Particle}
    lefts::CuVector{Particle}
    rights::CuVector{Particle}
    counters::CuVector{Int32}
    overflow_flag::CuVector{Int32}
    buf_lengths::Int32
    stayer_overflow_flag::CuVector{Int32}
    stayer_buf_length::Int32
end #struct
MigrantBuffers(
    max_particles_per_rank::Union{Int32,Int64},
    max_sendrecv_particles::Union{Int32,Int64}
) = MigrantBuffers(
    CuVector{Particle}(undef, max_particles_per_rank),
    CuVector{Particle}(undef, max_sendrecv_particles),
    CuVector{Particle}(undef, max_sendrecv_particles),
    CUDA.zeros(Int32, 3),
    CUDA.zeros(Int32, 1),
    Int32(max_sendrecv_particles),
    CUDA.zeros(Int32, 1),
    Int32(max_particles_per_rank)
)


# --------- MPI Send/Recv Buffers --------- #

mutable struct SendRecvBuffers
    send_left::CuVector{UInt32}
    send_right::CuVector{UInt32}
    recv_left::CuVector{UInt32}
    recv_right::CuVector{UInt32}
    buf_lengths::Int32
end #struct
SendRecvBuffers(max_particles::Union{Int32,Int64}) = SendRecvBuffers(
    CuVector{UInt32}(undef, 4 * max_particles),
    CuVector{UInt32}(undef, 4 * max_particles),
    CuVector{UInt32}(undef, 4 * max_particles),
    CuVector{UInt32}(undef, 4 * max_particles),
    Int32(4 * max_particles)
)
