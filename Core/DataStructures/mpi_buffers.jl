include("./particles.jl")

# ------------------- Ghost particles ----------------------
mutable struct GhostBuffers
    lefts::CuArray{Particle}
    rights::CuArray{Particle}
    counters::CuArray{Int32}
end #struct
GhostBuffers(max_particles::Union{Int32,Int64}) = GhostBuffers(
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
MigrantBuffers(max_particles_on_rank::Union{Int32,Int64}, max_sendrecv_particles::Union{Int32,Int64}) = MigrantBuffers(
    CuArray{Particle}(undef, max_particles_on_rank),
    CuArray{Particle}(undef, max_sendrecv_particles),
    CuArray{Particle}(undef, max_sendrecv_particles),
    CuArray(zeros(Float32, 3))
)

# ------------------- MPI Send/Recv Buffers ----------------------
mutable struct SendRecvBuffers
    send_left_buf::CuArray{Float32}
    send_right_buf::CuArray{Float32}
    recv_left_buf::CuArray{Float32}
    recv_right_buf::CuArray{Float32}
    buf_lengths::Union{Int32,Int64}
end #struct
SendRecvBuffers(buf_lengths::Union{Int32,Int64}) = SendRecvBuffers(
    CuArray{Float32}(undef, buf_lengths),
    CuArray{Float32}(undef, buf_lengths),
    CuArray{Float32}(undef, buf_lengths),
    CuArray{Float32}(undef, buf_lengths),
    buf_lengths
)
