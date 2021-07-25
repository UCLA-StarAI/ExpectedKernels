export linear, gaussian, polynomial, set_gamma, GAMMA

GAMMA = 0.0
linear(x::Vector, y::Vector) = (x' * y)[1]
function gaussian(x::Vector, y::Vector)
    # println(GAMMA)
    exp(-GAMMA * (norm(x - y) ^ 2))
end

polynomial(x::Vector, y::Vector, d::Real=2) = (x' * y + 1)[1] ^ d

function set_gamma(gamma::Real)
    global GAMMA = gamma
end