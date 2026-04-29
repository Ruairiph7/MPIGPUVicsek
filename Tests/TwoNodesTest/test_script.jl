include("/mnt/lustre/e1000/home/tc060/tc060/ruairiph_tc060/MPIGPUVicsek/main.jl")

MPI.Init()

run_simulation(100,10,max_particles_in_cell=64,steps_to_save=1,steps_to_save_on_the_go=1,save_outputs=true,save_OPs=true,save_snapshots=true,saving_coords_on_the_go=true,markersize=10,steps_to_new_OP_file=2)

MPI.Finalize()

