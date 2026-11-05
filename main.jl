# const TILE_SIZE = 128
const TILE_SIZE = 32
# const TILE_SIZE = 16
const STD_WORKGROUP_SIZE = 256
const STD_NUM_WORKGROUPS = 512

const DENSE_THRESHOLD = 16

include("./Core/DataStructures/datastructures.jl")
include("./Core/GPUAlgorithms/gpualgorithms.jl")
include("./Core/MPI/mpi.jl")
include("./Core/Outputs/outputs.jl")

include("./Core/run_simulation.jl")

