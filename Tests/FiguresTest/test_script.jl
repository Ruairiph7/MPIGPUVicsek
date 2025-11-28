include("/work/tc060/tc060/ruairiph_tc060/MPIGPUVicsek/main.jl")

MPI.Init()

run_simulation(20,2,λ=0.0f0,max_particles_in_cell=64,steps_to_save=1,steps_to_save_on_the_go=1,save_outputs=true,save_OPs=false,save_snapshots=true,saving_coords_on_the_go=false,markersize=10)

MPI.Finalize()

