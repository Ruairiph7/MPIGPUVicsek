include("/mnt/lustre/e1000/home/tc060/tc060/ruairiph_tc060/MPIGPUVicsek/main.jl")

MPI.Init()

run_simulation(1,10,max_particles_in_cell=64,steps_to_save=10,steps_to_save_on_the_go=10,save_outputs=false,save_OPs=false,save_snapshots=false,saving_coords_on_the_go=false, max_num_occupied_cells=Int32(20000),max_particles_per_rank = Int32(20000), steps_to_shrink_buffers=4)

MPI.Finalize()

