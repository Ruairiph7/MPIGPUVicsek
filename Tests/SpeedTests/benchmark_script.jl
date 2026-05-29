include("/storage/datastore-personal/s2563493/git/MPIGPUVicsek/main_benchmarking.jl")

const Ls = [2^n for n = 1:12]
const Ns = 2 .* Ls .^ 2

MPI.Init()

# for i = eachindex(Ls)
for i = 1:10
# for i in vcat(collect(1:10),[12])
    @show i
    run_simulation(Ns[i],100,Lx=Int32(Ls[i]),save_plots=false,save_OPs=false,save_coords=false,file_name_addon=string(i))
end

MPI.Finalize()
