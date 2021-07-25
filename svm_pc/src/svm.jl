export SVM, train, evaluate, evaluate_pc

struct SVM
    kernel::Function
    bias::Float64
    weights::Vector
    support_vectors::Matrix
    support_vector_labels::Vector
    c::Float64
end

function predict(svm::SVM, x::Vector)
    res = svm.bias
    n_sv = size(svm.weights)[1]
    for i = 1 : n_sv
        res += svm.weights[i] * svm.support_vector_labels[i] * svm.kernel(svm.support_vectors[i, :], x)
    end
    sign(res)
end

function evaluate(svm::SVM, X::Matrix)
    n = size(X)[1]
    res = zeros(n)
    for i in 1 : n
        res[i] = predict(svm, X[i, :])
    end
    res
end

function evaluate_pc(svm::SVM, X::DataFrame, pc::ProbCircuit)
    n = size(X)[1]
    res = svm.bias * ones(n)
    n_sv = size(svm.weights)[1]
    data1 = X
    for i = 1 : n_sv
        data2 = DataFrame(reshape(svm.support_vectors[i, :], 1, length(svm.support_vectors[i, :])))
        res += svm.weights[i] .* svm.support_vector_labels[i] .* SingleExpectation(pc, data1, data2, svm.kernel)
    end
    res = sign.(res)
    res
end

function predict_bias(kernel::Function, weights::Vector, support_vectors::Matrix, support_vector_labels::Vector, x::Vector)
    res = 0
    n_sv = size(weights)[1]
    for i = 1 : n_sv
        res += weights[i] * support_vector_labels[i] * kernel(support_vectors[i, :], x)
    end
    res
end

function train(kernel::Function, c::Float64, X::Matrix, y::Vector)
    lagrange_multipliers = compute_multipliers(kernel, c, X, y)
    lagrange_multipliers = vec(lagrange_multipliers)
    support_vector_indices = lagrange_multipliers .> 1e-5

    # display(support_vector_indices)
    # display(X)
    # display(y)

    support_multipliers = lagrange_multipliers[support_vector_indices]
    support_vectors = X[support_vector_indices, :]
    support_vector_labels = y[support_vector_indices]

    bias = 0.0
    n_sv = size(support_vector_labels)[1]
    for i = 1 : n_sv
        bias += (support_vector_labels[i] - predict_bias(kernel, support_multipliers, support_vectors, support_vector_labels, support_vectors[i, :]))
    end
    bias /= n_sv

    SVM(kernel, bias, support_multipliers, support_vectors, support_vector_labels, c)
end

function gram_matrix(kernel::Function, X::Matrix)
    n = size(X)[1]
    K = zeros(n, n)
    for i = 1 : n
        for j = 1 : n
            K[i, j] = kernel(X[i, :], X[j, :])
        end
    end
    K
end

function compute_multipliers(kernel::Function, c::Float64, X::Matrix, y::Vector)
    n, d = size(X)
    K = gram_matrix(kernel, X)
    P = (y * y') .* K
    q = -ones(n)

    G_std = Diagonal(-ones(n))
    h_std = zeros(n)

    G_slack = Diagonal(ones(n))
    h_slack = fill(c, n)

    G = vcat(G_std, G_slack)
    h = vcat(h_std, h_slack)

    A = y'
    b = [0.0]
    
    initvals = fill(1 / n, n)
    options = Dict("initvals" => initvals, "show_progress" => true)
    sol = CVXOPT.qp(P, q, G, h, A=A, b=b, options=options)

    @assert sol["status"] == "optimal" "optimization failed!"
    
    x = sol["x"]

end
