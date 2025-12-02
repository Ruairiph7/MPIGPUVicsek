# ---------------------------------------- #
# Load dependencies here incase this file is used separately
# ---------------------------------------- #
using JLD2
using StaticArrays
using DelimitedFiles
# include("../DataStructures/particles.jl")

struct Particle
    r::SVector{2,Float32}
    θ::Float32
    uid::Int32 #"Unique id"
end

function unpack_coords(particles_array::Array{Particle})
    rs = zeros(SVector{2,Float32}, length(particles_array))
    θs = zeros(Float32, length(particles_array))
    uids = zeros(Int32, length(particles_array))
    @inbounds for i = 1:length(particles_array)
        particle_i = particles_array[i]
        rs[i] = particle_i.r
        θs[i] = particle_i.θ
        uids[i] = particle_i.uid
    end #for i
    return rs, θs, uids
end #function
# ---------------------------------------- #

function get_timestep_dirs(;base_dir::String="./",outputs_dir_name::String="outputs/")
    return readdir(base_dir * outputs_dir_name, join=true)
end #function


function load_coords_from_timestep(timestep_dir::String; nprocs::Union{Nothing,Int}=nothing)
    timestep_str = match(r"([^_]+$)",timestep_dir).match
    time_step = parse(Int,timestep_str)
    rank_files = readdir(timestep_dir, join=true)
    if !isnothing(nprocs)
        length(rank_files) != nprocs && error("Mismatch in number of ranks")
    else
        nprocs = length(rank_files)
    end #if !isnothing(nprocs)

    particle_array_dicts = load.(rank_files)
    particle_arrays = [particle_array_dicts[rank+1]["particles"] for rank in 0:nprocs-1]
    N_total = sum(length.(particle_arrays))
    coord_arrays = unpack_coords.(particle_arrays)

    xs = Vector{Float32}(undef,N_total)
    ys = similar(xs)
    θs = similar(xs)
    uids = Vector{Int32}(undef,N_total)

    N_so_far = 0
    for proc_idx = 1:nprocs
        these_rs, these_θs, these_uids = coord_arrays[proc_idx]
        N_local = length(these_rs)
        for local_idx = 1:N_local
            global_idx = N_so_far + local_idx
            xs[global_idx], ys[global_idx] = these_rs[local_idx]
            θs[global_idx] = these_θs[local_idx]
            uids[global_idx] = these_uids[local_idx]
        end #for local_idx
        N_so_far += N_local
    end #for proc_idx

    particle_order = sortperm(uids)
    xs = xs[particle_order]
    ys = ys[particle_order]
    θs = θs[particle_order]
    uids = uids[particle_order]
    uids != 1:N_total && error("Error in particle order") 
        
    return time_step, xs, ys, θs
end #function

function get_full_trajectories(;base_dir::String="./",outputs_dir_name::String="outputs/")
    timestep_dirs = get_timestep_dirs(base_dir=base_dir,outputs_dir_name=outputs_dir_name)
    num_steps = length(timestep_dirs)
    time_steps = Vector{Int}(undef,num_steps)
    xs = Vector{Vector{Float32}}(undef,num_steps)
    ys = Vector{Vector{Float32}}(undef,num_steps)
    θs = Vector{Vector{Float32}}(undef,num_steps)

    for tidx = 1:num_steps
        timestep_dir = timestep_dirs[tidx]
        time_steps[tidx], xs[tidx], ys[tidx], θs[tidx] = load_coords_from_timestep(timestep_dir)
    end #for tidx

    return time_steps, xs, ys, θs
end #function


function load_coords_and_ranks_from_timestep(timestep_dir::String; nprocs::Union{Nothing,Int}=nothing)
    timestep_str = match(r"([^_]+$)",timestep_dir).match
    time_step = parse(Int,timestep_str)
    rank_files = readdir(timestep_dir, join=true)
    if !isnothing(nprocs)
        length(rank_files) != nprocs && error("Mismatch in number of ranks")
    else
        nprocs = length(rank_files)
    end #if !isnothing(nprocs)

    particle_array_dicts = load.(rank_files)
    particle_arrays = [particle_array_dicts[rank+1]["particles"] for rank in 0:nprocs-1]
    N_total = sum(length.(particle_arrays))
    coord_arrays = unpack_coords.(particle_arrays)

    xs = Vector{Float32}(undef,N_total)
    ys = similar(xs)
    θs = similar(xs)
    uids = Vector{Int32}(undef,N_total)
    ranks = similar(uids)

    N_so_far = 0
    for proc_idx = 1:nprocs
        these_rs, these_θs, these_uids = coord_arrays[proc_idx]
        N_local = length(these_rs)
        for local_idx = 1:N_local
            global_idx = N_so_far + local_idx
            xs[global_idx], ys[global_idx] = these_rs[local_idx]
            θs[global_idx] = these_θs[local_idx]
            uids[global_idx] = these_uids[local_idx]
            ranks[global_idx] = proc_idx - 1
        end #for local_idx
        N_so_far += N_local
    end #for proc_idx

    particle_order = sortperm(uids)
    xs = xs[particle_order]
    ys = ys[particle_order]
    θs = θs[particle_order]
    ranks = ranks[particle_order]

    uids = uids[particle_order]
    uids != 1:N_total && error("Error in particle order") 
        
    return time_step, xs, ys, θs, ranks
end #function


function get_full_trajectories_and_ranks(;base_dir::String="./",outputs_dir_name::String="outputs/")
    timestep_dirs = get_timestep_dirs(base_dir=base_dir,outputs_dir_name=outputs_dir_name)
    num_steps = length(timestep_dirs)
    time_steps = Vector{Int}(undef,num_steps)
    xs = Vector{Vector{Float32}}(undef,num_steps)
    ys = Vector{Vector{Float32}}(undef,num_steps)
    θs = Vector{Vector{Float32}}(undef,num_steps)
    ranks = Vector{Vector{Int32}}(undef,num_steps)

    for tidx = 1:num_steps
        timestep_dir = timestep_dirs[tidx]
        time_steps[tidx], xs[tidx], ys[tidx], θs[tidx], ranks[tidx] = load_coords_and_ranks_from_timestep(timestep_dir)
    end #for tidx

    return time_steps, xs, ys, θs, ranks
end #function


function store_trajectories(;base_dir::String="./",outputs_dir_name::String="outputs/")
    time_steps, xs, ys, θs = get_full_trajectories(base_dir=base_dir,outputs_dir_name=outputs_dir_name)
    writedlm(base_dir * outputs_dir_name * "time_steps.txt",time_steps) 
    writedlm(base_dir * outputs_dir_name * "xs.txt",xs) 
    writedlm(base_dir * outputs_dir_name * "ys.txt",ys) 
    writedlm(base_dir * outputs_dir_name * "thetas.txt",θs) 
end #function
