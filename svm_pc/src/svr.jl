export SVR, train, evaluate, evaluate_pc, fast_evaluate

struct SVR
    kernel::Function
    bias::Float64
    weights::Vector
    support_vectors::Matrix
    support_vector_labels::Vector
    c::Float64
    epsilon::Float64
end

function predict(svr::SVR, x::Vector)
    res = svr.bias
    n_sv = size(svr.weights)[1]
    for i = 1 : n_sv
        res += svr.weights[i] * svr.kernel(svr.support_vectors[i, :], x)
    end
    res
end

function evaluate(svr::SVR, X::Matrix)
    n = size(X)[1]
    res = zeros(n)
    for i in 1 : n
        res[i] = predict(svr, X[i, :])
    end
    res
end

function fast_evaluate(svr::SVR, X::Matrix, gamma::Float64)
    norm2 = [sum(abs2, svr.support_vectors[i:i, :] .- X, dims=2) for i = 1 : size(svr.weights, 1)]
    norm2 = hcat(norm2...)'
    res = svr.weights' * exp.(-gamma .* norm2) .+ svr.bias
    vec(res)
end

function evaluate_pc(svr::SVR, X::DataFrame, pc::ProbCircuit)
    n = size(X)[1]
    res = svr.bias * ones(n)
    n_sv = size(svr.weights)[1]
    data1 = X
    for i = 1 : n_sv
        data2 = DataFrame(reshape(svr.support_vectors[i, :], 1, length(svr.support_vectors[i, :])))
        res += svr.weights[i] .* SingleExpectation(pc, data1, data2, svr.kernel)
    end
    res
end

function predict_bias(kernel::Function, weights::Vector, support_vectors::Matrix, x::Vector)
    res = 0
    n_sv = size(weights)[1]
    for i = 1 : n_sv
        res += weights[i] * kernel(support_vectors[i, :], x)
    end
    res
end

function train(kernel::Function, c::Float64, epsilon::Float64, thres::Float64, X::Matrix, y::Vector)
    n = size(X)[1]

    lagrange_multipliers = compute_multipliers(kernel, c, epsilon, X, y)

    lagrange_multipliers = vec(lagrange_multipliers)
    lagrange_multipliers = lagrange_multipliers[1 : n] - lagrange_multipliers[n + 1 : 2 * n]
    support_vector_indices = abs.(lagrange_multipliers) .> thres

    # display(lagrange_multipliers)
    # display(support_vector_indices)
    # display(X)
    # display(y)

    support_multipliers = lagrange_multipliers[support_vector_indices]
    support_vectors = X[support_vector_indices, :]
    support_vector_labels = y[support_vector_indices]

    display(support_multipliers)
    # display(support_vectors)

    bias = 0.0
    n_sv = size(support_vector_labels)[1]
    for i = 1 : n_sv
        bias += (support_vector_labels[i] - epsilon - predict_bias(kernel, support_multipliers, support_vectors, support_vectors[i, :]))
    end
    bias /= n_sv

    display(bias)

    SVR(kernel, bias, support_multipliers, support_vectors, support_vector_labels, c, epsilon)
end

function gram_matrix(kernel::Function, X::Matrix)
    # n = size(X)[1]
    # K1 = zeros(n, n)
    # for i = 1 : n
    #     for j = 1 : n
    #         K1[i, j] = kernel(X[i, :], X[j, :])
    #     end
    # end
    # display(K1)
    if kernel == gaussian
        X_prod = X * X'
        X_diag = diag(X_prod)
        K2 = X_diag .+ X_diag' .- (2 * X_prod)
        println("GAMMA: $(GAMMA)")
        K2 = exp.(-GAMMA * K2)
    else
        @assert false "Not implemented fast operations for other kernels!"
    end
    display(K2)
    # display(norm(K1 - K2))
    # @assert K1 ≈ K2 "Gram matrix error"
    K2
end

function compute_multipliers(kernel::Function, c::Float64, epsilon::Float64, X::Matrix, y::Vector)
    n, d = size(X)
    K = gram_matrix(kernel, X)
    P = [K -K; -K K]
    q = zeros(2 * n)
    for i = 1 : n
        q[i] = -y[n] + epsilon
        q[i + n] = y[n] + epsilon
    end

    G_std = Diagonal(-ones(2 * n))
    h_std = zeros(2 * n)

    G_slack = Diagonal(ones(2 * n))
    h_slack = fill(c, 2 * n)

    G = vcat(G_std, G_slack)
    h = vcat(h_std, h_slack)

    A = ones(2 * n)
    for i = n + 1 : 2 * n
        A[i] = -1
    end
    A = A'
    b = [0.0]
    
    initvals = fill(1 / (2 * n), 2 * n)
    options = Dict("initvals" => initvals, "show_progress" => true)
    sol = CVXOPT.qp(P, q, G, h, A=A, b=b, options=options)

    @assert sol["status"] == "optimal" "optimization failed!"
    
    x = sol["x"]

end
