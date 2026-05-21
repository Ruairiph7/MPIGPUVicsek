# --------- Hilbert curve encoding --------- #

function hilbert_rotate(n, x, y, rx, ry)
    if ry == UInt32(0)
        if rx == UInt32(1)
            x = n - UInt32(1) - x
            y = n - UInt32(1) - y
        end
        x, y = y, x
    end
    return x, y
end

#Convert from 2d coords (cx, cy) on a 2^order by 2^order grid to a 1d hilbert index.
function hilbert_encode_2d(cx::UInt32, cy::UInt32, order::Int)::UInt32
    d = UInt32(0)
    n = UInt32(1) << (order - 1)   # start at highest bit
    x = cx
    y = cy
    while n > UInt32(0)
        rx = (x & n) > UInt32(0) ? UInt32(1) : UInt32(0)
        ry = (y & n) > UInt32(0) ? UInt32(1) : UInt32(0)
        d += n * n * (UInt32(3) * rx ⊻ ry)
        x, y = hilbert_rotate(n, x, y, rx, ry)
        n >>= UInt32(1)
    end
    return d
end

function compute_hilbert_ordering(ncx::Int32, ncy::Int32)
    ncells = Int(ncx * ncy)

    # Find the order needed; smallest n>=1 such that 2^n >= max(ncx, ncy)
    order = max(1, ceil(Int, log2(max(ncx, ncy))))
    @assert (1 << order) >= max(ncx, ncy) "Error in hilbert order - order: $order, max grid size: $(max(ncx,ncy))"

    # Compute Hilbert code for each valid (cx, cy) pair
    # Pairs outside the actual ncx × ncy grid are excluded automatically
    codes = Vector{Tuple{UInt32, Int32}}(undef, ncells)
    for cy in Int32(0):ncy - Int32(1)
        for cx in Int32(0):ncx - Int32(1)
            rm_idx = cy * ncx + cx + Int32(1)   # 1-based row-major
            code   = hilbert_encode_2d(UInt32(cx), UInt32(cy), order)
            codes[rm_idx] = (code, rm_idx)
        end
    end

    # Sort by Hilbert code — gives spatial ordering
    sort!(codes, by = first)

    # Build bidirectional lookup tables
    row_major_to_hilbert = Vector{Int32}(undef, ncells)
    hilbert_to_row_major = Vector{Int32}(undef, ncells)

    for (hilbert_idx, (_, rm_idx)) in enumerate(codes)
        row_major_to_hilbert[rm_idx]      = Int32(hilbert_idx)
        hilbert_to_row_major[hilbert_idx] = rm_idx
    end

    return row_major_to_hilbert, hilbert_to_row_major
end
