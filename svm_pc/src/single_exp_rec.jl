export SingleExpectation

ExpCacheDict = Dict{ProbCircuit, Array{ExpFloat, 1}}

function SingleExpectation(pc::ProbCircuit, data1::DataFrame, data2::DataFrame, kernel::Function)
    log_likelihoods = marginal(pc, data1)
    p_observed = exp.(log_likelihoods)

    cache = ExpCacheDict()
    results_unnormalized = single_exp(pc, data1, data2, kernel, cache)
    # println("results_unnormalized: $(results_unnormalized)")

    results = results_unnormalized ./ p_observed

    # display(cache)
    results
end

function single_exp(n::Union{PlainSumNode, StructSumNode}, data1::DataFrame, data2::DataFrame, kernel::Function, cache::ExpCacheDict)
    @inbounds get!(cache, n) do 
        value = zeros(ExpFloat, num_examples(data1))
        @inline pthetas(i) = ExpFloat.(exp(n.log_probs[i]))
        @fastmath @simd for i in 1 : num_children(n)
            value .+= pthetas(i) .* single_exp(children(n)[i], data1, data2, kernel, cache)
        end
        # println("Find non-zero $(value)")
        # println("pthetas: $(ExpFloat.(exp.(n.log_probs)))")
        return value
    end
end

function single_exp(n::Union{PlainMulNode, StructMulNode}, data1::DataFrame, data2::DataFrame, kernel::Function, cache::ExpCacheDict)
    @inbounds get!(cache, n) do 
        value = ones(ExpFloat, num_examples(data1))
        @fastmath @simd for i in 1 : num_children(n)
            value .*= single_exp(children(n)[i], data1, data2, kernel, cache)
        end
        # println("Find non-zero $(value)")
        return value
    end
end

function single_exp(n::Union{PlainProbLiteralNode, StructProbLiteralNode}, data1::DataFrame, data2::DataFrame, kernel::Function, cache::ExpCacheDict)
    @inbounds get!(cache, n) do 
        value = zeros(ExpFloat, num_examples(data1))
        idx = lit2var(literal(n))
        p1 = data1[:, idx]
        p2 = data2[:, idx]
        replace!(p1, missing => -1)
        n_vec = ispositive(n) .* ones(size(p2))
        value .+= (p1 .== -1) .* kernel(n_vec, p2)
        value .+= (p1 .!= -1) .* (p1 .== ispositive(n)) .* kernel(n_vec, p2)
        # println("Find non-zero $(value)")
        return value
    end
end