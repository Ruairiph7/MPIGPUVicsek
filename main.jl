const STD_WORKGROUP_SIZE = 256
const STD_NUM_WORKGROUPS = 512

include("./Core/DataStructures/datastructures.jl")
include("./Core/GPUAlgorithms/gpualgorithms.jl")
include("./Core/MPI/mpi.jl")
include("./Core/Outputs/outputs.jl")

include("./Core/run_simulation.jl")

