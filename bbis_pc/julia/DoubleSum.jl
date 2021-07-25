push!(LOAD_PATH, ".")

module DoubleSum

using ProbabilisticCircuits
using LogicCircuits
using LinearAlgebra
using StatsFuns

# include("consts.jl")
using Consts: KERNEL
# include("ThreadSafeDicts.jl")
using ThreadSafeDicts


export double_sum, log_double_sum, kernel, GlobalSumCache

function kernel(x, y, gsize)
    # # println(x, y)
    h = prod(gsize)
    if KERNEL == "hamming"
        exp(-norm(x - y) ^ 2 / h)
    elseif KERNEL == "score"
        x == y
    else
        error("kernel $(KERNEL) not implemented!")
    end
end

SumCache = Dict{Pair{ProbΔNode, ProbΔNode}, Array{Float64, 1}}

# cache = SumCache()

# tab = 0

function print_tab()
    for i in 1 : tab
        # print("  ")
    end
end

function double_sum(x1::XData{Int8}, x2::XData{Int8}, pc1::ProbΔ, pc2::ProbΔ, gsize, flip)
    # TODO: support batch data
    # global cache
    cache = SumCache()
    # get!(cache, Pair(pc1[end], pc2[end]), 0)
    # @assert False
    @assert num_examples(x1) == num_examples(x2) "x1 and x2 have different number of examples!"
    sum_node(x1, x2, pc1[end], pc2[end], cache, gsize, flip)
end

function sum_node(x1::XData{Int8}, x2::XData{Int8}, n1::Prob⋁, n2::Prob⋁, cache::SumCache, gsize, flip)
    # global tab
    # tab += 1
    # @inbounds ret = get!(cache, Pair(n1, n2)) do
    #     value = 0
    #     thetas_1 = [exp(n1.log_thetas[i]) for i in 1 : length(n1.children)]
    #     thetas_2 = [exp(n2.log_thetas[i]) for i in 1 : length(n2.children)]
    #     # print_tab()
    #     # println("or - or")
    #     # print_tab()
    #     # println("thetas_1: ", thetas_1)
    #     # print_tab()
    #     # println("thetas_2: ", thetas_2)
    #     @fastmath @simd for i in 1 : length(n1.children)
    #         for j in 1 : length(n2.children)
    #             value += thetas_1[i] * thetas_2[j] * sum_node(x1, x2, n1.children[i], n2.children[j], cache, gsize, flip)
    #         end
    #     end
    #     # print_tab()
    #     # println("value: ", value)
    #     return value
    # end
    if haskey(cache, Pair(n1, n2))
        ret = cache[Pair(n1, n2)]
        # println("hit!")
    else
        value = zeros(num_examples(x1))
        thetas_1 = [exp(n1.log_thetas[i]) for i in 1 : length(n1.children)]
        thetas_2 = [exp(n2.log_thetas[i]) for i in 1 : length(n2.children)]
        # print_tab()
        # println("or - or")
        # print_tab()
        # println("thetas_1: ", thetas_1)
        # print_tab()
        # println("thetas_2: ", thetas_2)
        @fastmath @simd for i in 1 : length(n1.children)
            for j in 1 : length(n2.children)
                value .+= (thetas_1[i] .* thetas_2[j] .* sum_node(x1, x2, n1.children[i], n2.children[j], cache, gsize, flip))
            end
        end
        cache[Pair(n1, n2)] = value
        ret = value
    end
    # tab -= 1
    return ret
end

function sum_node(x1::XData{Int8}, x2::XData{Int8}, n1::Prob⋀, n2::Prob⋀, cache::SumCache, gsize, flip)
    # global tab
    # tab += 1
    # @inbounds ret = get!(cache, Pair(n1, n2)) do
    #     value = 1
    #     @assert length(n1.children) == length(n2.children) "Circuits should share the same vtree!"
    #     # print_tab()
    #     # println("and - and")
    #     @fastmath @simd for i in 1 : length(n1.children)
    #         value *= sum_node(x1, x2, n1.children[i], n2.children[i], cache, gsize, flip)
    #     end
    #     # print_tab()
    #     # println("value: ", value)
    #     return value
    # end
    if haskey(cache, Pair(n1, n2))
        ret = cache[Pair(n1, n2)]
        # println("hit!")
    else
        value = ones(num_examples(x1))
        @assert length(n1.children) == length(n2.children) "Circuits should share the same vtree!"
        # print_tab()
        # println("and - and")
        @fastmath @simd for i in 1 : length(n1.children)
            value .*= sum_node(x1, x2, n1.children[i], n2.children[i], cache, gsize, flip)
        end
        # print_tab()
        # println("value: ", value)
        cache[Pair(n1, n2)] = value
        ret = value
    end

    # tab -= 1
    return ret
end

function sum_node(x1::XData{Int8}, x2::XData{Int8}, n1::Prob⋁, n2::ProbLiteral, cache::SumCache, gsize, flip)
    # global tab
    # tab += 1
    # @inbounds ret = get!(cache, Pair(n1, n2)) do
    #     value = 0
    #     thetas_1 = [exp(n1.log_thetas[i]) for i in 1 : length(n1.children)]
    #     # print_tab()
    #     # println("or - literal")
    #     # print_tab()
    #     # println("thetas_1: ", thetas_1)
    #     @fastmath @simd for i in 1 : length(n1.children)
    #         value += thetas_1[i] * sum_node(x1, x2, n1.children[i], n2, cache, gsize, flip)
    #     end
    #     # print_tab()
    #     # println("value: ", value)
    #     return value
    # end
    if haskey(cache, Pair(n1, n2))
        ret = cache[Pair(n1, n2)]
        # println("hit!")
    else
        value = zeros(num_examples(x1))
        thetas_1 = [exp(n1.log_thetas[i]) for i in 1 : length(n1.children)]
        # print_tab()
        # println("or - literal")
        # print_tab()
        # println("thetas_1: ", thetas_1)
        @fastmath @simd for i in 1 : length(n1.children)
            value .+= (thetas_1[i] .* sum_node(x1, x2, n1.children[i], n2, cache, gsize, flip))
        end
        # print_tab()
        # println("value: ", value)
        cache[Pair(n1, n2)] = value
        ret = value
    end
    # tab -= 1
    return ret
end 

function sum_node(x1::XData{Int8}, x2::XData{Int8}, n1::ProbLiteral, n2::Prob⋁, cache::SumCache, gsize, flip)
    # global tab
    # tab += 1
    # @inbounds ret = get!(cache, Pair(n1, n2)) do
    #     value = 0
    #     thetas_2 = [exp(n2.log_thetas[i]) for i in 1 : length(n2.children)]
    #     # print_tab()
    #     # println("literal - or")
    #     # print_tab()
    #     # println("thetas_2: ", thetas_2)
    #     @fastmath @simd for j in 1 : length(n2.children)
    #         value += thetas_2[j] * sum_node(x1, x2, n1, n2.children[j], cache, gsize, flip)
    #     end
    #     # print_tab()
    #     # println("value: ", value)
    #     return value
    # end
    if haskey(cache, Pair(n1, n2))
        ret = cache[Pair(n1, n2)]
        # println("hit!")
    else
        value = zeros(num_examples(x1))
        thetas_2 = [exp(n2.log_thetas[i]) for i in 1 : length(n2.children)]
        # print_tab()
        # println("literal - or")
        # print_tab()
        # println("thetas_2: ", thetas_2)
        @fastmath @simd for j in 1 : length(n2.children)
            value .+= (thetas_2[j] .* sum_node(x1, x2, n1, n2.children[j], cache, gsize, flip))
        end
        # print_tab()
        # println("value: ", value)
        cache[Pair(n1, n2)] = value
        ret = value
    end
    # tab -= 1
    return ret
end 

function sum_node(x1::XData{Int8}, x2::XData{Int8}, n1::ProbLiteral, n2::ProbLiteral, cache::SumCache, gsize, flip)
    # global tab
    # tab += 1
    # @inbounds ret = get!(cache, Pair(n1, n2)) do
    #     value = 0
    #     @assert lit2var(literal(n1)) == lit2var(literal(n2)) "Circuits should share the same vtree!"
    #     idx = lit2var(literal(n1))
    #     # print_tab()
    #     # println("literal: $(lit2var(literal(n1)))($(positive(n1))) $(lit2var(literal(n2)))($(positive(n2))) x1=$(x1) x2=$(x2)")
    #     # # println("end")
    #     p1 = feature_matrix(x1)[idx]
    #     p2 = feature_matrix(x2)[idx]
    #     if idx != flip
    #         if p1 == -1 && p2 == -1
    #             value = kernel(positive(n1), positive(n2), gsize)
    #         elseif p1 == -1
    #             value = (p2 == positive(n2)) * kernel(positive(n1), positive(n2), gsize)
    #         elseif p2 == -1
    #             value = (p1 == positive(n1)) * kernel(positive(n1), positive(n2), gsize)
    #         else
    #             value = (p1 == positive(n1)) * (p2 == positive(n2)) * kernel(positive(n1), positive(n2), gsize)
    #         end
    #     else
    #         if p1 == -1 && p2 == -1
    #             value = kernel(negative(n1), negative(n2), gsize)
    #         elseif p1 == -1
    #             value = (p2 == positive(n2)) * kernel(negative(n1), negative(n2), gsize)
    #         elseif p2 == -1
    #             value = (p1 == positive(n1)) * kernel(negative(n1), negative(n2), gsize)
    #         else
    #             value = (p1 == positive(n1)) * (p2 == positive(n2)) * kernel(negative(n1), negative(n2), gsize)
    #         end
    #     end

    #     # print_tab()
    #     # println("value: ", value)
    #     return value
    # end
    if haskey(cache, Pair(n1, n2))
        ret = cache[Pair(n1, n2)]
        # println("hit!")
    else
        value = zeros(num_examples(x1))
        @assert lit2var(literal(n1)) == lit2var(literal(n2)) "Circuits should share the same vtree!"
        idx = lit2var(literal(n1))
        # print_tab()
        # println("literal: $(lit2var(literal(n1)))($(positive(n1))) $(lit2var(literal(n2)))($(positive(n2))) x1=$(x1) x2=$(x2)")
        # # println("end")
        p1 = feature_matrix(x1)[:, idx]
        p2 = feature_matrix(x2)[:, idx]
        # display(p1)
        # display(p2)
        if idx != flip
            # if p1 == -1 && p2 == -1
            #     value = kernel(positive(n1), positive(n2), gsize)
            # elseif p1 == -1
            #     value = (p2 == positive(n2)) * kernel(positive(n1), positive(n2), gsize)
            # elseif p2 == -1
            #     value = (p1 == positive(n1)) * kernel(positive(n1), positive(n2), gsize)
            # else
            #     value = (p1 == positive(n1)) * (p2 == positive(n2)) * kernel(positive(n1), positive(n2), gsize)
            # end
            value .+= ((p1 .== -1) .* (p2 .== -1)) .* kernel(positive(n1), positive(n2), gsize)
            value .+= ((p1 .== -1) .* (p2 .!= -1)) .* (p2 .== positive(n2)) .* kernel(positive(n1), positive(n2), gsize)
            value .+= ((p1 .!= -1) .* (p2 .== -1)) .* (p1 .== positive(n1)) .* kernel(positive(n1), positive(n2), gsize)
            value .+= ((p1 .!= -1) .* (p2 .!= -1)) .* (p1 .== positive(n1)) .* (p2 .== positive(n2)) .* kernel(positive(n1), positive(n2), gsize)
        else
            # if p1 == -1 && p2 == -1
            #     value = kernel(negative(n1), negative(n2), gsize)
            # elseif p1 == -1
            #     value = (p2 == positive(n2)) * kernel(negative(n1), negative(n2), gsize)
            # elseif p2 == -1
            #     value = (p1 == positive(n1)) * kernel(negative(n1), negative(n2), gsize)
            # else
            #     value = (p1 == positive(n1)) * (p2 == positive(n2)) * kernel(negative(n1), negative(n2), gsize)
            # end
            value .+= ((p1 .== -1) .* (p2 .== -1)) .* kernel(negative(n1), negative(n2), gsize)
            value .+= ((p1 .== -1) .* (p2 .!= -1)) .* (p2 .== positive(n2)) .* kernel(negative(n1), negative(n2), gsize)
            value .+= ((p1 .!= -1) .* (p2 .== -1)) .* (p1 .== positive(n1)) .* kernel(negative(n1), negative(n2), gsize)
            value .+= ((p1 .!= -1) .* (p2 .!= -1)) .* (p1 .== positive(n1)) .* (p2 .== positive(n2)) .* kernel(negative(n1), negative(n2), gsize)
        end

        # print_tab()
        # println("value: ", value)
        cache[Pair(n1, n2)] = value
        ret = value
    end
    # tab -= 1
    # print('*')
    return ret
end


NodeDataPair = Pair{Int64, Array{Int8, 2}}
GlobalSumCache = Dict{Pair{NodeDataPair, NodeDataPair}, Array{Float64, 1}}

function log_double_sum(x1::Array{Int8, 2}, x2::Array{Int8, 2}, pc1::ProbΔ, pc2::ProbΔ, scope1::Dict, scope2::Dict, node2id1::Dict, node2id2::Dict, cache::GlobalSumCache, gsize, flip)
    # TODO: support batch data
    # global cache
    # cache = GlobalSumCache()
    # get!(cache, Pair(pc1[end], pc2[end]), 0)
    # @assert False
    @assert size(x1) == size(x2) "x1 and x2 have different number of examples!"
    # scope1 = variable_scopes(pc1)
    # scope2 = variable_scopes(pc2)
    # display(scope1)
    # display(scope2)
    # node2id1 = Dict(pc1[i] => i for i in 1 : length(pc1))
    # node2id2 = Dict(pc2[i] => i for i in 1 : length(pc2))
    log_sum_node(x1, x2, pc1[end], pc2[end], scope1, scope2, node2id1, node2id2, cache, gsize, flip)
end

function log_sum_node(x1::Array{Int8, 2}, x2::Array{Int8, 2}, n1::Prob⋁, n2::Prob⋁, scope1::Dict, scope2::Dict, node2id1::Dict, node2id2::Dict, cache::GlobalSumCache, gsize, flip)
    id1 = scope1[n1]
    id2 = scope2[n2]
    id1 = [id for id in id1]
    id2 = [id for id in id2]
    m1 = @view x1[:, id1]
    m2 = @view x2[:, id2]
    pair = ((node2id1[n1] => m1) => (node2id2[n2] => m2))
    if haskey(cache, pair)
        ret = cache[pair]
        # println("hit!")
    else
        # print("%")
        value = zeros(size(x1, 1))
        value .= -Inf
        log_thetas_1 = [n1.log_thetas[i] for i in 1 : length(n1.children)]
        log_thetas_2 = [n2.log_thetas[i] for i in 1 : length(n2.children)]
        # print_tab()
        # println("or - or")
        # print_tab()
        # println("thetas_1: ", thetas_1)
        # print_tab()
        # println("thetas_2: ", thetas_2)
        @fastmath @simd for i in 1 : length(n1.children)
            for j in 1 : length(n2.children)
                # value .+= (thetas_1[i] .* thetas_2[j] .* log_sum_node(x1, x2, n1.children[i], n2.children[j], cache, gsize, flip))
                value = logaddexp.(value, log_thetas_1[i] .+ log_thetas_2[j] .+ log_sum_node(x1, x2, n1.children[i], n2.children[j], scope1, scope2, node2id1, node2id2, cache, gsize, flip))
            end
        end
        cache[pair] = value
        ret = value
    end
    return ret
end

function log_sum_node(x1::Array{Int8, 2}, x2::Array{Int8, 2}, n1::Prob⋀, n2::Prob⋀, scope1::Dict, scope2::Dict, node2id1::Dict, node2id2::Dict, cache::GlobalSumCache, gsize, flip)
    id1 = scope1[n1]
    id2 = scope2[n2]
    id1 = [id for id in id1]
    id2 = [id for id in id2]
    m1 = @view x1[:, id1]
    m2 = @view x2[:, id2]
    pair = ((node2id1[n1] => m1) => (node2id2[n2] => m2))
    if haskey(cache, pair)
        ret = cache[pair]
        # println("hit!")
    else
        value = zeros(size(x1, 1))
        @assert length(n1.children) == length(n2.children) "Circuits should share the same vtree!"
        # print_tab()
        # println("and - and")
        @fastmath @simd for i in 1 : length(n1.children)
            # value .*= log_sum_node(x1, x2, n1.children[i], n2.children[i], cache, gsize, flip)
            value .+= log_sum_node(x1, x2, n1.children[i], n2.children[i], scope1, scope2, node2id1, node2id2, cache, gsize, flip)
        end
        # print_tab()
        # println("value: ", value)
        cache[pair] = value
        ret = value
    end

    return ret
end

function log_sum_node(x1::Array{Int8, 2}, x2::Array{Int8, 2}, n1::Prob⋁, n2::ProbLiteral, scope1::Dict, scope2::Dict, node2id1::Dict, node2id2::Dict, cache::GlobalSumCache, gsize, flip)
    id1 = scope1[n1]
    id2 = scope2[n2]
    id1 = [id for id in id1]
    id2 = [id for id in id2]
    m1 = @view x1[:, id1]
    m2 = @view x2[:, id2]
    pair = ((node2id1[n1] => m1) => (node2id2[n2] => m2))
    if haskey(cache, pair)
        ret = cache[pair]
        # println("hit!")
    else
        value = zeros(size(x1, 1))
        value .= -Inf
        log_thetas_1 = [n1.log_thetas[i] for i in 1 : length(n1.children)]
        # print_tab()
        # println("or - literal")
        # print_tab()
        # println("thetas_1: ", thetas_1)
        @fastmath @simd for i in 1 : length(n1.children)
            # value .+= (thetas_1[i] .* log_sum_node(x1, x2, n1.children[i], n2, cache, gsize, flip))
            value = logaddexp.(value, log_thetas_1[i] .+ log_sum_node(x1, x2, n1.children[i], n2, scope1, scope2, node2id1, node2id2, cache, gsize, flip))
        end
        # print_tab()
        # println("value: ", value)
        cache[pair] = value
        ret = value
    end
    return ret
end 

function log_sum_node(x1::Array{Int8, 2}, x2::Array{Int8, 2}, n1::ProbLiteral, n2::Prob⋁, scope1::Dict, scope2::Dict, node2id1::Dict, node2id2::Dict, cache::GlobalSumCache, gsize, flip)
    id1 = scope1[n1]
    id2 = scope2[n2]
    id1 = [id for id in id1]
    id2 = [id for id in id2]
    m1 = @view x1[:, id1]
    m2 = @view x2[:, id2]
    pair = ((node2id1[n1] => m1) => (node2id2[n2] => m2))
    if haskey(cache, pair)
        ret = cache[pair]
        # println("hit!")
    else
        value = zeros(size(x1, 1))
        value .= -Inf
        log_thetas_2 = [n2.log_thetas[i] for i in 1 : length(n2.children)]
        # print_tab()
        # println("literal - or")
        # print_tab()
        # println("thetas_2: ", thetas_2)
        @fastmath @simd for j in 1 : length(n2.children)
            # value .+= (thetas_2[j] .* log_sum_node(x1, x2, n1, n2.children[j], cache, gsize, flip))
            value = logaddexp.(value, log_thetas_2[j] .+ log_sum_node(x1, x2, n1, n2.children[j], scope1, scope2, node2id1, node2id2, cache, gsize, flip))
        end
        # print_tab()
        # println("value: ", value)
        cache[pair] = value
        ret = value
    end
    return ret
end 

function log_sum_node(x1::Array{Int8, 2}, x2::Array{Int8, 2}, n1::ProbLiteral, n2::ProbLiteral, scope1::Dict, scope2::Dict, node2id1::Dict, node2id2::Dict, cache::GlobalSumCache, gsize, flip)
    id1 = scope1[n1]
    id2 = scope2[n2]
    id1 = [id for id in id1]
    id2 = [id for id in id2]
    m1 = @view x1[:, id1]
    m2 = @view x2[:, id2]
    pair = ((node2id1[n1] => m1) => (node2id2[n2] => m2))
    if haskey(cache, pair)
        ret = cache[pair]
        # println("hit!")
    else
        value = zeros(size(x1, 1))
        @assert lit2var(literal(n1)) == lit2var(literal(n2)) "Circuits should share the same vtree!"
        idx = lit2var(literal(n1))
        # print_tab()
        # println("literal: $(lit2var(literal(n1)))($(positive(n1))) $(lit2var(literal(n2)))($(positive(n2))) x1=$(x1) x2=$(x2)")
        # # println("end")
        p1 = x1[:, idx]
        p2 = x2[:, idx]
        # display(p1)
        # display(p2)
        if idx != flip
            # if p1 == -1 && p2 == -1
            #     value = kernel(positive(n1), positive(n2), gsize)
            # elseif p1 == -1
            #     value = (p2 == positive(n2)) * kernel(positive(n1), positive(n2), gsize)
            # elseif p2 == -1
            #     value = (p1 == positive(n1)) * kernel(positive(n1), positive(n2), gsize)
            # else
            #     value = (p1 == positive(n1)) * (p2 == positive(n2)) * kernel(positive(n1), positive(n2), gsize)
            # end
            value .+= ((p1 .== -1) .* (p2 .== -1)) .* kernel(positive(n1), positive(n2), gsize)
            value .+= ((p1 .== -1) .* (p2 .!= -1)) .* (p2 .== positive(n2)) .* kernel(positive(n1), positive(n2), gsize)
            value .+= ((p1 .!= -1) .* (p2 .== -1)) .* (p1 .== positive(n1)) .* kernel(positive(n1), positive(n2), gsize)
            value .+= ((p1 .!= -1) .* (p2 .!= -1)) .* (p1 .== positive(n1)) .* (p2 .== positive(n2)) .* kernel(positive(n1), positive(n2), gsize)
        else
            # if p1 == -1 && p2 == -1
            #     value = kernel(negative(n1), negative(n2), gsize)
            # elseif p1 == -1
            #     value = (p2 == positive(n2)) * kernel(negative(n1), negative(n2), gsize)
            # elseif p2 == -1
            #     value = (p1 == positive(n1)) * kernel(negative(n1), negative(n2), gsize)
            # else
            #     value = (p1 == positive(n1)) * (p2 == positive(n2)) * kernel(negative(n1), negative(n2), gsize)
            # end
            value .+= ((p1 .== -1) .* (p2 .== -1)) .* kernel(negative(n1), negative(n2), gsize)
            value .+= ((p1 .== -1) .* (p2 .!= -1)) .* (p2 .== positive(n2)) .* kernel(negative(n1), negative(n2), gsize)
            value .+= ((p1 .!= -1) .* (p2 .== -1)) .* (p1 .== positive(n1)) .* kernel(negative(n1), negative(n2), gsize)
            value .+= ((p1 .!= -1) .* (p2 .!= -1)) .* (p1 .== positive(n1)) .* (p2 .== positive(n2)) .* kernel(negative(n1), negative(n2), gsize)
        end

        # print_tab()
        # println("value: ", value)
        value = log.(value)
        cache[pair] = value
        ret = value
        # println(pair)
    end
    # print('*')
    return ret
end

function test()
    # pc, vtree = load_struct_prob_circuit("../circuits/toy.psdd", "../circuits/toy.vtree")
    pc, vtree = load_struct_prob_circuit("../uai/test.uai.psdd", "../uai/test.uai.vtree")
    s = [
        -1, 1, 1,
        0, 0, 0,
        1, 1, 0,
        1, 0, 0
    ]
    s_0 = [
        0, 1, 1,
        0, 0, 0,
        1, 1, 0,
        1, 0, 0
    ]
    ss = [
        1, -1, 1,
        0, 0, 0,
        1, 1, 0,
        1, 0, 0
    ]
    ss_0 = [
        0, 1, 1,
        0, 0, 0,
        1, 1, 0,
        1, 0, 0
    ]
    # sample = [s; s_0]
    # sample_0 = [s; s]
    # x1 = XData(Int8.(reshape(sample, 2, :)))
    # x2 = XData(Int8.(reshape(sample_0, 2, :)))
    # println(x1)
    # println(x2)
    gsize = 12
    sample = zeros(Int8, 2, gsize)
    sample_0 = zeros(Int8, 2, gsize)
    sample[1, :] = s
    sample[2, :] = ss_0
    sample_0[1, :] = s_0
    sample_0[2, :] = ss
    # display(sample)
    # display(sample_0)
    x1 = XData(sample)
    x2 = XData(sample_0)
    display(x1)
    display(x2)
    @time ret = double_sum(x1, x2, pc, pc, gsize, 1)
    vec1 = [[0 1 1 0 0 0 1 1 0 1 0 0], [1 1 1 0 0 0 1 1 0 1 0 0]]
    vec2 = [[0 1 1 0 0 0 1 1 0 1 0 0]]
    check = 0
    @time for v1 in vec1
        for v2 in vec2
            # println(v1, v2)
            v1_f = copy(v1)
            v1_f[1] = 1 - v1[1]
            v2_f = copy(v2)
            v2_f[1] = 1 - v2[1]
            try
                log1 = log_proba(pc, XData(Bool.(v1)))[1]
                log2 = log_proba(pc, XData(Bool.(v2)))[1]
                check += exp(log1 + log2) * kernel(v1_f, v2_f, gsize)
            catch err
                # zero probability
            end
        end
    end
    # @assert ret == check "Double sum goes wrong!"
    # println("recursive: $(ret)")
    # println("brute-force: $(check)")
    display(ret)
    display(check)
end

function evaluate_time()
    pc, vtree = load_struct_prob_circuit("../circuits/toy.psdd", "../circuits/toy.vtree")
    pc1, vtree = load_struct_prob_circuit("../uai/frustrated_rand_12.uai.1.psdd", "../uai/frustrated_rand_12.uai.1.vtree")
    pc2, vtree = load_struct_prob_circuit("../uai/frustrated_rand_12.uai.1.psdd", "../uai/frustrated_rand_12.uai.1.vtree")
    gsize = 144
    # pc1, vtree = load_struct_prob_circuit("../uai/small.uai.1.psdd", "../uai/small.uai.1.vtree")
    # pc2, vtree = load_struct_prob_circuit("../uai/small.uai.1.psdd", "../uai/small.uai.1.vtree")
    # gsize = 12
    # pc1, vtree = load_struct_prob_circuit("../uai/tiny.uai.1.psdd", "../uai/tiny.uai.1.vtree")
    # pc2, vtree = load_struct_prob_circuit("../uai/tiny.uai.1.psdd", "../uai/tiny.uai.1.vtree")
    # gsize = 4
    x1 = zeros(Int8, 2, gsize)
    x2 = zeros(Int8, 2, gsize)
    # x1 = XData(sample_1)
    # x2 = XData(sample_2)
    # display(x1)
    # display(x2)
    cache = GlobalSumCache()

    scope1 = variable_scopes(pc1)
    scope2 = variable_scopes(pc2)
    node2id1 = Dict(pc1[i] => i for i in 1 : length(pc1))
    node2id2 = Dict(pc2[i] => i for i in 1 : length(pc2))

    # id1 = scope1[pc1[end]]
    # id2 = scope2[pc2[end]]
    # id1 = [id for id in id1]
    # id2 = [id for id in id2]
    # display(id1)
    # display(id2)
    # m1 = @view x1[:, id1]
    # m2 = @view x2[:, id2]
    # pair = ((node2id1[pc1[end]] => m1) => (node2id2[pc2[end]] => m2))
    # display(pair)
    println("first time")
    @time ret = log_double_sum(x1, x2, pc1, pc2, scope1, scope2, node2id1, node2id2, cache, gsize, 1)
    println()
    display(ret)
    
    # for (key, value) in cache
    #     display(key)
    #     display(value)
    # end
    display(cache)
    
    # @assert haskey(cache, pair)
    println("second time")
    @time ret = log_double_sum(x1, x2, pc1, pc2, scope1, scope2, node2id1, node2id2, cache, gsize, 1)
    println()
    display(ret)
end

end

# DoubleSum.test()
# DoubleSum.evaluate_time()
# using PProf
# @pprof DoubleSum.evaluate_time()