include("/work/tc060/tc060/ruairiph_tc060/MPIGPUVicsek/main.jl")

const γ = 0.513
const N1 = 1
const N2 = 2
const N3 = 500
const λ1 = 0
const λ2 = 0
const λs3 = [0.01,0.041,0.05,0.1]
const Lx1 = 5
const Lx2 = 5
const Lx3 = 20

MPI.Init()

# Simulation with zero noise, to check exact coords - 1 particle
run_simulation(N1, 1000, γ=Float32(γ), λ=Float32(λ1), Lx=Int32(Lx1), save_outputs=false, save_OPs=false, file_name_addon="test1", input_files=("../Baselines_2/inputXs1.txt", "../Baselines_2/inputYs1.txt", "../Baselines_2/inputThetas1.txt"), saving_coords_on_the_go=true, steps_to_save_on_the_go=1)

# Simulation with zero noise, to check exact coords - 2 particles
run_simulation(N2, 1000, γ=Float32(γ), λ=Float32(λ2), Lx=Int32(Lx2), save_outputs=false, save_OPs=false, file_name_addon="test2", input_files=("../Baselines_2/inputXs2.txt", "../Baselines_2/inputYs2.txt", "../Baselines_2/inputThetas2.txt"), saving_coords_on_the_go=true, steps_to_save_on_the_go=1)

# Simulation with noise, to check OP - range of lambads, long time
for idx = eachindex(λs3)
    run_simulation(N3, 100000, γ=Float32(γ), λ=Float32(λs3[idx]), Lx=Int32(Lx3), save_figures=true, save_snapshots=false, save_density=false, save_OP=true, steps_to_save=100, write_final_coords=false, file_name_addon="test3_"*string(idx), input_files=("../Baselines_2/inputXs3.txt", "../Baselines_2/inputYs3.txt", "../Baselines_2/inputThetas3.txt"))
end #for idx


for idx = eachindex(λs3)
    run_simulation(N3, 100000, γ=Float32(γ), λ=Float32(λs3[idx]), Lx=Int32(Lx3), save_outputs=true, save_snapshots=false, save_OPs=true, steps_to_save=100, file_name_addon="test3_"*string(idx), input_files=("../Baselines_2/inputXs3.txt", "../Baselines_2/inputYs3.txt", "../Baselines_2/inputThetas3.txt"))
end #for idx

MPI.Finalize()
