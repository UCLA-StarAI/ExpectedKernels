export DoubleExpectation

ExpCacheDict = Dict{Pair{ProbCircuit, ProbCircuit}, Array{ExpFloat, 2}}

function DoubleExpectation(pc1::ProbCircuit, pc2::ProbCircuit, data1, data2, kernel)
    log_likelihoods1 = marginal(pc1, data1)
    p_observed1 = exp.(log_likelihoods1)
    log_likelihoods2 = marginal(pc2, data2)
    p_observed2 = exp.(log_likelihoods2)

    cache = ExpCacheDict()
    results_unnormalized = double_exp(pc1, pc2, data1, data2, kernel, cache)

    results = results_unnormalized ./ (p_observed1 .* p_observed2)

    results
end

function double_exp(n1::Union{PlainSumNode, StructSumNode}, n2::Union{PlainSumNode, StructSumNode}, data1, data2, kernel, cache::ExpCacheDict)
    @inbounds get!(cache, Pair(n1, n2)) do 
        value = zeros(ExpFloat, num_examples(data1))
        @inline pthetas_1(i) = ExpFloat.(exp(n1.log_probs[i]))
        @inline pthetas_2(i) = ExpFloat.(exp(n2.log_probs[i]))
        @fastmath @simd for i in 1 : num_children(n1)
            for j in 1 : num_children(n2)
                value .+= (pthetas_1(i) .* pthetas_2(j) .* double_exp(children(n1)[i], children(n2)[j], data1, data2, kernel, cache))
            end
        end
        return value
    end
end

function double_exp(n1::Union{PlainMulNode, StructMulNode}, n2::Union{PlainMulNode, StructMulNode}, data1, data2, kernel, cache::ExpCacheDict)
    @inbounds get!(cache, Pair(n1, n2)) do 
        value = ones(ExpFloat, num_examples(data1))
        @assert num_children(n1) == num_children(n2) "Circuits should share the same vtree!"
        @fastmath @simd for i in 1 : num_children(n1)
            value .*= double_exp(children(n1)[i], children(n2)[i], data1, data2, kernel, cache)
        end

        return value
    end
end

function double_exp(n1::Union{PlainSumNode, StructSumNode}, n2::Union{PlainProbLiteralNode, StructProbLiteralNode}, data1, data2, kernel, cache::ExpCacheDict)
    @inbounds get!(cache, Pair(n1, n2)) do 
        value = zeros(ExpFloat, num_examples(data1))
        @inline pthetas_1(i) = ExpFloat.(exp(n1.log_probs[i]))
        @fastmath @simd for i in 1 : num_children(n1)
            value .+= (pthetas_1(i) .* double_exp(children(n1)[i], n2, data1, data2, kernel, cache))
        end
        return value
    end
end

function double_exp(n1::Union{PlainProbLiteralNode, StructProbLiteralNode}, n2::Union{PlainSumNode, StructSumNode}, data1, data2, kernel, cache::ExpCacheDict)
    @inbounds get!(cache, Pair(n1, n2)) do 
        value = zeros(ExpFloat, num_examples(data1))
        @inline pthetas_2(i) = ExpFloat.(exp(n2.log_probs[i]))
        @fastmath @simd for i in 1 : num_children(n2)
            value .+= (pthetas_2(i) .* double_exp(n1, children(n2)[i], data1, data2, kernel, cache))
        end
        return value
    end
end

function double_exp(n1::Union{PlainProbLiteralNode, StructProbLiteralNode}, n2::Union{PlainProbLiteralNode, StructProbLiteralNode}, data1, data2, kernel, cache::ExpCacheDict)
    @inbounds get!(cache, Pair(n1, n2)) do 
        value = zeros(ExpFloat, num_examples(data1))
        @assert lit2var(literal(n1)) == lit2var(literal(n2)) "Circuits should share the same vtree!"
        idx = lit2var(literal(n1))
        p1 = data1[:, idx]
        p2 = data2[:, idx]
        value .+= ((p1 .== -1) .* (p2 .== -1)) .* kernel(ispositive(n1), ispositive(n2))
        value .+= ((p1 .== -1) .* (p2 .!= -1)) .* (p2 .== ispositive(n2)) .* kernel(ispositive(n1), ispositive(n2))
        value .+= ((p1 .!= -1) .* (p2 .== -1)) .* (p1 .== ispositive(n1)) .* kernel(ispositive(n1), ispositive(n2))
        value .+= ((p1 .!= -1) .* (p2 .!= -1)) .* (p1 .== ispositive(n1)) .* (p2 .== ispositive(n2)) .* kernel(ispositive(n1), ispositive(n2))
        return value
    end
end