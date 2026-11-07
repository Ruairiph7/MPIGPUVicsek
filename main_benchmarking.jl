#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#NOTE: For benchmarking:
import_time_start = time()
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

!@isdefined TILE_SIZE && (const TILE_SIZE = 32)
!@isdefined DENSE_THRESHOLD && (const DENSE_THRESHOLD = 16)
!@isdefined STD_WORKGROUP_SIZE && (const STD_WORKGROUP_SIZE = 256)
!@isdefined STD_NUM_WORKGROUPS && (const STD_NUM_WORKGROUPS = 512)

include("./Core/DataStructures/datastructures.jl")
include("./Core/GPUAlgorithms/gpualgorithms.jl")
include("./Core/MPI/mpi.jl")
include("./Core/Outputs/outputs.jl")

include("./Core/run_simulation_benchmarking.jl")

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#NOTE: For benchmarking:
import_time = time() - import_time_start
writedlm("import_time.txt", import_time)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#


