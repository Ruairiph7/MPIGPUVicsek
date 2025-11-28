# # ------------------------------------------------------------
# # Safe device-aware MPI send/recv wrappers
# # ------------------------------------------------------------
#
# function try_isend(buf, comm, dest, tag)
#     try
#         return MPI.isend(buf, comm, dest=dest, tag=tag)  # works if MPI is CUDA-aware
#     catch
#         # fallback: host bounce buffer
#         @warn "FALLING BACK TO NON CUDA-AWARE MPI"
#         h = Array(buf)
#         return MPI.isend(h, comm, dest=dest, tag=tag)
#     end
# end
#
# function try_Irecv!(buf, comm, src, tag)
#     try
#         return MPI.Irecv!(buf, comm, source=src, tag=tag)  # works if MPI is CUDA-aware
#     catch
#         # fallback: host buffer then device copy
#         @warn "FALLING BACK TO NON CUDA-AWARE MPI"
#         h = similar(Array(buf))
#         req = MPI.Irecv!(h, comm, source=src, tag=tag)
#         MPI.Wait!(req)
#         copyto!(buf, h)
#         return req
#     end
# end


###############################################
# Packing particles → Float32 buffer on GPU
###############################################

@kernel function serialize_kernel!(out, @Const(particles))
    i = @index(Global)
    p = particles[i]
    base = 4 * (i - 1)
    out[base+1] = p.r[1]
    out[base+2] = p.r[2]
    out[base+3] = p.θ
    out[base+4] = Float32(p.uid)
end

function pack_particles_to_f32(particles::CuArray{Particle})
    n = length(particles)
    n == 0 && return CuArray{Float32}(undef,0)

    out = CuArray{Float32}(undef, 4n)
    kernel! = serialize_kernel!(CUDABackend())
    kernel!(out, particles; ndrange=n)
    KernelAbstractions.synchronize(CUDABackend())
    return out
end

###############################################
# Unpacking Float32 buffer → particles on GPU
###############################################

@kernel function deserialize_kernel!(out, @Const(buf), n)
    i = @index(Global)
    base = 4 * (i - 1)
    x = buf[base+1]
    y = buf[base+2]
    θ = buf[base+3]
    uid = Int32(buf[base+4])
    out[i] = Particle(SVector{2,Float32}(x, y), θ, uid)
end

function unpack_f32_to_particles(buf::CuArray{Float32})
    n = length(buf) ÷ 4
    n == 0 && return CuArray{Particle}(undef,0)

    out = CuArray{Particle}(undef, n)
    kernel! = deserialize_kernel!(CUDABackend())
    kernel!(out, buf, n; ndrange=n)
    KernelAbstractions.synchronize(CUDABackend())
    return out
end


