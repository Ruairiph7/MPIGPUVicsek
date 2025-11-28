include("../main.jl")

MPI.Init()

run_simulation(1,10,max_particles_in_cell=64,steps_to_save=10,steps_to_save_on_the_go=10,save_outputs=false,save_OPs=false,save_snapshots=false,saving_coords_on_the_go=false)

MPI.Finalize()

