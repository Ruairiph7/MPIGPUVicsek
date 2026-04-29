include("../../main.jl")

MPI.Init()

run_simulation(20,4,λ=0.0f0,max_particles_in_cell=64,steps_to_save=1,steps_to_save_on_the_go=1,save_outputs=true,save_OPs=true,save_snapshots=false,saving_coords_on_the_go=false,steps_to_new_OP_file=2)

MPI.Finalize()

