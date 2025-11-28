using MPI
using CUDA

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

 # Each rank selects its own GPU
 CUDA.device!(rank)
 println("Rank $rank using GPU $(CUDA.device())")

 # Example: create a GPU array
 N = 10_000
 x = CUDA.zeros(Float32, N)

 # Fill with some values
 x .= rank + 1

 # Allocate receive buffer on GPU for rank 0
 y = rank == 0 ? CUDA.zeros(Float32, N*nprocs) : nothing

 # Gather GPU arrays to rank 0 (CUDA-aware)
 MPI.Gather!(x, y, 0, comm)

 if rank == 0
     println("Gathered array size on GPU: ", size(y))
 end

 MPI.Barrier(comm)
 MPI.Finalize()
