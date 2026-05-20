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

    # Find the order needed: smallest n such that 2^n >= max(ncx, ncy)
    # This ensures both cx and cy fit within the Hilbert grid
    order = Int(ceil(log2(max(ncx, ncy))))
    order = max(order, 1)   # minimum order 1 (2×2 Hilbert grid)

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


function print_hilbert_ordering(ncx::Int32, ncy::Int32,
                                 hilbert_to_row_major::Vector{Int32})
    # Print the Hilbert visit order as a grid
    grid = zeros(Int32, ncy, ncx)
    for (h_idx, rm_idx) in enumerate(hilbert_to_row_major)
        rm_0 = rm_idx - 1
        cx   = rm_0 % ncx + 1   # 1-based for display
        cy   = rm_0 ÷ ncx + 1
        grid[cy, cx] = Int32(h_idx)
    end

    println("Hilbert visit order for $(ncx)×$(ncy) grid:")
    for cy in ncy:-1:1   # print top row first
        println(join(lpad(grid[cy, cx], 4) for cx in 1:ncx))
    end
end
