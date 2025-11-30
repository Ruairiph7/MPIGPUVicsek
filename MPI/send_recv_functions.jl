@warn "(de)serialize_kernel workgroup_size, num_workgroups hard-coded at 256"

###############################################
# Packing particles → Float32 buffer on GPU
###############################################

@kernel function serialize_kernel!(out, @Const(particles), size)
    I = @index(Global)
    stride = @ndrange()

    for i = I:stride:size
        p = particles[i]
        base = 4 * (i - 1)
        out[base+1] = p.r[1]
        out[base+2] = p.r[2]
        out[base+3] = p.θ
        out[base+4] = Float32(p.uid)
    end #for i
end

function pack_particles_to_f32(particles::CuArray{Particle})
    n = length(particles)
    n == 0 && return CuArray{Float32}(undef, 0)
    out = CuArray{Float32}(undef, 4n)

    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups

    kernel! = serialize_kernel!(CUDABackend())
    kernel!(out, particles, 4n; ndrange=total_num_threads)
    KernelAbstractions.synchronize(CUDABackend())
    return out
end

###############################################
# Unpacking Float32 buffer → particles on GPU
###############################################

@kernel function deserialize_kernel!(out, @Const(buf), size)
    I = @index(Global)
    stride = @ndrange()

    for i = I:stride:size
        base = 4 * (i - 1)
        x = buf[base+1]
        y = buf[base+2]
        θ = buf[base+3]
        uid = Int32(buf[base+4])
        out[i] = Particle(SVector{2,Float32}(x, y), θ, uid)
    end #for i
end

function unpack_f32_to_particles(buf::CuArray{Float32})
    n = length(buf) ÷ 4
    n == 0 && return CuArray{Particle}(undef, 0)
    out = CuArray{Particle}(undef, n)

    workgroup_size = 256
    num_workgroups = 256
    total_num_threads = workgroup_size * num_workgroups

    kernel! = deserialize_kernel!(CUDABackend())
    kernel!(out, buf, n; ndrange=total_num_threads)
    KernelAbstractions.synchronize(CUDABackend())
    return out
end


