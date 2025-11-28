using MPI
using StaticArrays
using CUDA

# Initialize MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_ranks = MPI.Comm_size(comm)

# Number of particles
N = 10

# -------------------------
# Step 1: Create data on rank 0
# -------------------------
rs_all = Vector{SVector{2,Float32}}(undef, N)
if rank == 0
    for i in 1:N
        rs_all[i] = SVector(rand(Float32), rand(Float32))
    end
end

# -------------------------
# Step 2: Flatten to contiguous Array{Float32,1}
# -------------------------
rs_all_flat = zeros(Float32, 2*N)
if rank == 0
    for i in 1:N
        rs_all_flat[2i-1:2i] .= rs_all[i]
    end
end

# -------------------------
# Step 3: Broadcast the flat array (CPU)
# -------------------------
MPI.Bcast!(rs_all_flat, 0, comm)

# -------------------------
# Step 4: Reconstruct SVectors after broadcast
# -------------------------
rs_all = [SVector{2,Float32}(rs_all_flat[2i-1:2i]) for i in 1:N]

# -------------------------
# Optional: Move to GPU
# -------------------------
# This will create a CuArray of size (2, N)
rs_gpu = CuArray(reshape(rs_all_flat, 2, N))

# Simple check
if rank == 0
    println("Broadcast successful! First particle: ", rs_all[1])
end

# Finalize MPI
MPI.Barrier(comm)
MPI.Finalize()
