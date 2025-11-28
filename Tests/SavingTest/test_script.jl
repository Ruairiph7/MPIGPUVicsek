include("/work/tc060/tc060/ruairiph_tc060/MPIGPUVicsek/main.jl")

MPI.Init()

run_simulation(1,100,λ=0.0f0,max_particles_in_cell=64,steps_to_save=10,steps_to_save_on_the_go=1,save_outputs=false,save_OPs=false,save_snapshots=false,saving_coords_on_the_go=true)

MPI.Finalize()

