using MPI
using StaticArrays

struct Particle
    r::SVector{2,Float32}
    θ::Float32
    uid::Int32 #"Unique id"
end
Particle(r, θ, uid) = Particle(r, θ, uid)

MPI.Init()

comm=MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

# local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
# local_rank = MPI.Comm_rank(local_comm)
# CUDA.device!(local_rank)

function log_rss(tag, initial, last)
    rss_kb = parse(Int, chomp(read(`ps -o rss= -p $(getpid())`, String)))
    delta_to_initial = rss_kb - initial
    delta_to_last = rss_kb - last
    println("rank $rank: $tag RSS_kB=$rss_kb,   diff_0=$delta_to_initial,   diff=$delta_to_last")
    flush(stdout)
    return rss_kb
end

initial = log_rss("initial", 0, 0)

N_total = 2
rs_all = Vector{SVector{2,Float32}}(undef,N_total)
rs_all_flat = Vector{Float32}(undef,2*N_total)
θs_all = Vector{Float32}(undef,N_total)
if rank == 0
    # rs_all = [(10,10) .* @SVector(rand(Float32, 2)) for i = 1:N_total]
    # θs_all = [Float32(2π * rand()) for i = 1:N_total]
    
    rs_all = [@SVector([1.0f0,1.0f0]), @SVector([7.0f0,7.0f0])]
    θs_all = [0.0f0, 0.5f0]

    rs_all_flat = Float32[x for r in rs_all for x in r]
end

last = log_rss("Before bcast", initial, initial)

# rs_all_flat = MPI.Bcast(rs_all_flat, 0, comm)
MPI.Bcast!(rs_all_flat, 0, comm)
rs_all = [SVector{2,Float32}(rs_all_flat[2i-1:2i]) for i in 1:(length(rs_all_flat) ÷ 2)]

# θs_all = MPI.Bcast(θs_all, 0, comm)
MPI.Bcast!(θs_all, 0, comm)

# Characterise local domain
Lx_local = 10 / nprocs
x_min = rank * Lx_local
x_max = (rank + 1) * Lx_local

last = log_rss("Before filtering", initial, last)

# Get particles in local domain
function in_local_domain(r)
    return x_min <= r[1] < x_max
end #function
local_particle_idxs = findall(in_local_domain, rs_all)
rs_filtered = rs_all[local_particle_idxs]
θs_filtered = θs_all[local_particle_idxs]

last = log_rss("Before exscan", initial, last)

N_local = length(rs_filtered)
N_offset = MPI.Exscan(N_local, +, comm)
rank == 0 && (N_offset = 0)

@show rank, local_particle_idxs, N_local, N_offset

last = log_rss("Before tuple test", initial, last)

test1 = (rs_filtered[1], θs_filtered[1], 1)

last = log_rss("Before particle test", initial, last)

Test2 = Particle(rs_filtered[1], θs_filtered[1], Int32(1))

last = log_rss("Final", initial, last)

MPI.Barrier(comm)
MPI.Finalize()
