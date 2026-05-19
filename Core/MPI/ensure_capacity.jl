function ensure_send_capacity!(sendrecv_bufs, max_num, rank)
    length_needed = 4 * max_num
    sendrecv_bufs.buf_lengths >= length_needed && return nothing

    # (No additional buffer over length_needed as this is already applied
    # by extract_ghosts!)
    sendrecv_bufs.send_left_buf = CuVector{UInt32}(undef, length_needed)
    sendrecv_bufs.send_right_buf = CuVector{UInt32}(undef, length_needed)
    sendrecv_bufs.recv_left_buf = CuVector{UInt32}(undef, length_needed)
    sendrecv_bufs.recv_right_buf = CuVector{UInt32}(undef, length_needed)
    sendrecv_bufs.buf_lengths = length_needed
    println("Rank $rank raising sendrecv buffer size to $length_needed")
end #function

function ensure_recv_capacity!(sendrecv_bufs, left_size, right_size, rank)
    length_needed = maximum(left_size, right_size)
    sendrecv_bufs.buf_lengths >= length_needed && return nothing

    new_max = Ceil(Int32, length_needed * 1.5f0)
    sendrecv_bufs.send_left_buf = CuVector{UInt32}(undef, new_max)
    sendrecv_bufs.send_right_buf = CuVector{UInt32}(undef, new_max)
    sendrecv_bufs.recv_left_buf = CuVector{UInt32}(undef, new_max)
    sendrecv_bufs.recv_right_buf = CuVector{UInt32}(undef, new_max)
    sendrecv_bufs.buf_lengths = new_max
    println("Rank $rank raising sendrecv buffer size to $new_max")
end #function
