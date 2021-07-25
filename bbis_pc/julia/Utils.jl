module Utils

# using LightGraphs
# using MetaGraphs
using LogicCircuits
using ProbabilisticCircuits
using CVXOPT
using LinearAlgebra
using Distributions
using Random
using FactorGenerator

export print_clt, print_circuit_dfs, generate_samples, mask_samples, cvxopt

# include("consts.jl")
using Consts: THRESHOLD, BURN_IN, INTERVAL

# function sample_1_d(rng, x, i, pc)
#     @assert pc !== nothing "pc is nothing!"
#     x0 = copy(x)
#     x1 = copy(x)
#     x0[i] = 0
#     x1[i] = 1
#     x0 = XData(Bool.(reshape(x0, 1, :)))
#     x1 = XData(Bool.(reshape(x1, 1, :)))
#     p_0 = exp(log_proba(pc, x0)[1])
#     p_1 = exp(log_proba(pc, x1)[1])
#     p = p_1 / (p_0 + p_1)
#     Bool.(rand(rng, Binomial(1, p)))
# end

function sample_1_d(rng, factor, x, i)
    f, g, gsize = factor["f"], factor["g"], factor["grid_size"]
    column = Int((i - 1) % gsize[2]) + 1
    row = Int((i - column) / gsize[2]) + 1
    size = gsize[2]
    # println(i, " ", column, " ", row)
    p_0 = 1.0
    p_1 = 1.0
    if row > 1 # upward factor
        p_0 *= g[row - 1, column, x[i - size] << 1 + 0 + 1]
        p_1 *= g[row - 1, column, x[i - size] << 1 + 1 + 1]
    end
    if row < gsize[1] # downward factor
        p_0 *= g[row, column, 0 << 1 + x[i + size] + 1]
        p_1 *= g[row, column, 1 << 1 + x[i + size] + 1]
    end
    if column > 1 # leftward factor
        p_0 *= f[row, column - 1, x[i - 1] << 1 + 0 + 1]
        p_1 *= f[row, column - 1, x[i - 1] << 1 + 1 + 1]
    end
    if column < gsize[2] # rightward factor
        p_0 *= f[row, column, 0 << 1 + x[i + 1] + 1]
        p_1 *= f[row, column, 1 << 1 + x[i + 1] + 1]
    end
    p = p_1 / (p_0 + p_1)
    Bool.(rand(rng, Binomial(1, p)))
end

function generate_samples(rng, n_samples, proposal, var_num, pc=nothing, uai_folder_dir=nothing)
    if proposal == "uniform"
		samples = rand(rng, Bool, n_samples, var_num)
    elseif proposal == "exact"
        samples = zeros(Bool, n_samples, var_num)
        for i = 1 : n_samples
            samples[i, :] = ProbabilisticCircuits.sample(pc)
        end
    elseif proposal == "gibbs"
        @assert uai_folder_dir !== nothing
        uai_filename = split(uai_folder_dir, '/')
        len = length(uai_filename)
        uai_filename = uai_filename[len]
        uai_dir = "$(uai_folder_dir)/$(uai_filename).uai"

        factor = parse_uai_to_grid(uai_dir)

        x = rand(rng, Bool, 1, var_num)
        samples = []
        i = 0
        while size(samples)[1] < n_samples
            dim = i % var_num + 1
            # x[1, dim] = sample_1_d(rng, x, dim, pc)
            x[1, dim] = sample_1_d(rng, factor, x, dim)
            i += 1
            if (i > BURN_IN) && (i % INTERVAL == 0)
                push!(samples, copy(x))
            end
        end
        samples = vcat([reshape(s, 1, :) for s in samples]...)
    else
        error("Proposal $(proposal) not implemented!")
    end
    samples = Int8.(samples)
end

function mask_samples(rng, samples, mask_num)
    masked_samples = copy(samples)
    
    var_num = size(samples, 2)
    # for i = 1 : size(samples, 1)
    #     offset = i % var_num + 1
    #     if offset + mask_num - 1 <= var_num
    #         masked_samples[i, offset : (offset + mask_num - 1)] .= -1
    #     else
    #         masked_samples[i, offset : var_num] .= -1
    #         masked_samples[i, 1 : (mask_num - (var_num - offset + 1))] .= -1
    #     end
    # end
    for i = 1 : size(samples, 1)
        ids = randperm(rng, var_num)[1 : mask_num]
        masked_samples[i, ids] .= -1
    end
	
	masked_samples
end

function cvxopt(kernel_matrix; threshold=THRESHOLD)
    @assert minimum(eigvals(kernel_matrix)) > -1e-4 "kernel matrix is not positive definite: $(minimum(eigvals(kernel_matrix)))"
    evs = eigvals(kernel_matrix)    
    sample_size = size(kernel_matrix, 1)
    if minimum(abs.(kernel_matrix)) > 1e3
        kernel_matrix /= minimum(abs.(kernel_matrix))
    end
    # epsilon = (maximum(evs) / mean(evs))^2  # TODO: how to choose epsilon
    # P = kernel_matrix + epsilon * Matrix(I, sample_size, sample_size)
    # q = ones(sample_size) * (-2 * epsilon / sample_size)
    # G = -1.0 * Matrix(I, sample_size, sample_size)
    # G = [G; -1.0 * G]    
    # h = -threshold * ones(sample_size)
    # h = [h; ones(sample_size)]
    P = kernel_matrix
    q = zeros(sample_size)
    G = -1.0 * Matrix(I, sample_size, sample_size)
    h = zeros(sample_size)
    A = ones(1, sample_size)
    b = ones(1)
    initvals = 1 / sample_size * ones(1, sample_size)
    options = Dict("initvals" => initvals, "show_progress" => false)

    sol = CVXOPT.qp(P, q, G, h, A=A, b=b, options=options)
    # sol = CVXOPT.qp(P, q, G, h, options=options)
    @assert sol["status"] == "optimal" "optimization failed!"
    x = sol["x"] / sum(sol["x"])    
    x, sol["primal objective"]
end

# function print_clt(clt::CLT)
#     for e in edges(clt) print(e); print(" ");end
#     if clt isa SimpleDiGraph
#         for v in vertices(clt) print(v); print(" "); end
#     end
#     println()
#     if clt isa MetaDiGraph
#         for v in vertices(clt) print(v); print(" "); println(props(clt, v)) end
#     end
# end

function print_circuit_dfs(node::ProbΔNode, tab=0, weight=0)
    for i = 1:tab
        print("> ")
    end
    if node isa ProbLiteral
        print("Literal ", lit2var(literal(node)), " ", positive(node), " ")
        if weight > 0
            print("(", weight, ")")
        end
        print("\n")
    elseif node isa Prob⋀
        print("AND ")
        if weight > 0
            print("(", weight, ")")
        end
        print("\n")
        print(variables(node))
        print("\n")
        for i in 1:length(node.children)
            print_circuit_dfs(node.children[i], tab + 1)
        end
    elseif node isa Prob⋁
        print("OR ")
        if weight > 0
            print("(", weight, ")")
        end
        print("\n")
        p_thetas = [exp(node.log_thetas[i]) for i in 1:length(node.children)]
        for i in 1:length(node.children)
            print_circuit_dfs(node.children[i], tab + 1, p_thetas[i])
        end
    else
        println("ERROR ", node)
    end
end

end

