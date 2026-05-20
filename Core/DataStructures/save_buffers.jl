mutable struct SaveBuffers
    pinned_buf::Vector{Particle} #Pinned buffer to copy particles from GPU

    ASYNC_SAVES::Bool #Flag for if we are using asynchronous coordinate writes
    save_task::Union{Nothing,Task} #Track whether asynchronous write has completed
end #struct

function SaveBuffers(max_particles; ASYNC_SAVES=false)
    pinned_buf = CUDA.pin(Vector{Particle}(undef, max_particles))
    save_task = nothing
    return SaveBuffers(pinned_buf, ASYNC_SAVES, save_task)
end #function


function reallocate_save_bufs!(save_bufs, new_max_particles)
    !isnothing(save_bufs.save_task) && wait(save_bufs.save_task)
    save_bufs.save_task = nothing

    CUDA.unpin(save_bufs.pinned_buf)
    save_bufs.pinned_buf = CUDA.pin(Vector{Particle}(undef, new_max_particles))
    return nothing
end #function
