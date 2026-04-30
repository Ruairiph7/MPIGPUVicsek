# --------- Vicsek Model Functions ---------

@inline function get_magnetisation(θs::AbstractVector)
    vector_OP = zeros(2)
    N = length(θs)
    for i = 1:N
        vector_OP = vector_OP .+ [cos(θs[i]), sin(θs[i])]
    end
    vector_OP = vector_OP / N
    return norm(vector_OP)
end

@inline function get_nematic_S(θs::AbstractVector)
    vector_OP = zeros(2)
    N = length(θs)
    for i = 1:N
        vector_OP = vector_OP .+ [cos(2 * θs[i]), sin(2 * θs[i])]
    end
    vector_OP = vector_OP / N
    return norm(vector_OP)
end

function F(θ::Float32, R²::Float32)
    return sin(θ) / (π * R²)
end

function Fn(θ::Float32, R²::Float32)
    return sin(2 * θ) / (π * R²)
end

