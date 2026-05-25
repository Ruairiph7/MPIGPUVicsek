#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#NOTE: For benchmarking:
import_time_start = time()
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

const TILE_SIZE = 128
const STD_WORKGROUP_SIZE = 256
const STD_NUM_WORKGROUPS = 512

include("./Core/DataStructures/datastructures.jl")
include("./Core/GPUAlgorithms/gpualgorithms.jl")
include("./Core/MPI/mpi.jl")
include("./Core/Outputs/outputs.jl")

include("./Core/run_simulation_benchmarking.jl")

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#NOTE: For benchmarking:
import_time = time() - import_time_start
writedlm("import_time.txt", import_time)

compilation_time = @elapsed begin
    MPI.Init()
    run_simulation(1,1,save_OPs=false,save_plots=false,save_coords=false)
    MPI.Finalize()
end #begin
writedlm("compilation_time.txt", compilation_time)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#


