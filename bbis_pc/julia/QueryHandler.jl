module QueryHandler

using ProbabilisticCircuits
using LogicCircuits
using StatsFuns

export evidence_query, log_evidence_query, marginal_query, log_marginal_query, marginal_per_var, get_sample_marginals, get_sample_log_marginals

# function evidence_query(x, factor)
#     """
#     parameters:
#         x: instantiation of all variables
#         factor: factors of the grid
#     return:
#         p(x)
#     """
#     f, g, gsize = factor["f"], factor["g"], factor["grid_size"]    
#     @assert length(x) == prod(gsize)

#     evi = 1.0
#     for i = 1:gsize[1]
#         for j = 1:gsize[2] - 1            
#             evi *= f[i, j, x[(i - 1) * gsize[2] + j] << 1 + x[(i - 1) * gsize[2] + j + 1] + 1]            
#         end
#     end
#     for i = 1:gsize[1] - 1
#         for j = 1:gsize[2]
#             evi *= g[i, j, x[(i - 1) * gsize[2] + j] << 1 + x[i * gsize[2] + j] + 1]
#         end
#     end
#     evi
# end

# function log_evidence_query(x, factor)
#     """
#     parameters:
#         x: instantiation of all variables
#         factor: factors of the grid
#     return:
#         log p(x)
#     """
#     f, g, gsize = factor["f"], factor["g"], factor["grid_size"]    
#     @assert length(x) == prod(gsize)

#     log_evi = 0.0
#     for i = 1:gsize[1]
#         for j = 1:gsize[2] - 1            
#             log_evi += log(f[i, j, x[(i - 1) * gsize[2] + j] << 1 + x[(i - 1) * gsize[2] + j + 1] + 1])            
#         end
#     end
#     for i = 1:gsize[1] - 1
#         for j = 1:gsize[2]
#             log_evi += log(g[i, j, x[(i - 1) * gsize[2] + j] << 1 + x[i * gsize[2] + j] + 1])
#         end
#     end

#     log_evi
# end

# function marginal_query(x, factor, pc=nothing)
#     """
#     parameters:
#         x: instantiation of all variables / variables except first line
#         factor: factors of the grid
#         pc: PC on first line
#     return:
#         p(x)
#     """
#     gsize = factor["grid_size"]

#     if length(x) == prod(gsize)
#         mar = evidence_query(x, factor)
#     elseif length(x) == (gsize[1] - 1) * gsize[2]
#         if pc === nothing
#             mar = 0
#             for i = 0:2^gsize[2] - 1
#                 xc = digits(i, base=2, pad=gsize[2])
#                 xc = reshape(xc, 1, gsize[2])
#                 xc_x = hcat(xc, x)
#                 mar += evidence_query(xc_x, factor)
#             end
#         else
#             pad = -1 * ones(1, gsize[2])
#             x_data = XData(Int8.(hcat(pad, x[:, 1:gsize[2]])))
#             fs = Dict("grid_size" => (gsize[1] - 1, gsize[2]), "f" => factor["f"][2:end, :, :], "g" => factor["g"][2:end, :, :])
#             mar = exp.(log_proba(pc, x_data))[1] * evidence_query(x, fs)
#         end        
#     else
#         error("x of size $(size(x)), factor size $(factor["grid_size"]), pc $(pc) Not implemented!")
#     end
#     mar    
# end

# function log_marginal_query(x, factor, pc=nothing)
#     """
#     parameters:
#         x: instantiation of all variables / variables except first line
#         factor: factors of the grid
#         pc: PC on first line
#     return:
#         log p(x)
#     """
#     gsize = factor["grid_size"]

#     if length(x) == prod(gsize)
#         log_mar = log_evidence_query(x, factor)
#     elseif length(x) == (gsize[1] - 1) * gsize[2]
#         if pc === nothing
#             evi = []
#             for i = 0:2^gsize[2] - 1
#                 xc = digits(i, base=2, pad=gsize[2])
#                 xc = reshape(xc, 1, gsize[2])
#                 xc_x = hcat(xc, x)
#                 tmp = [evi; log_evidence_query(xc_x, factor)]
#             end
#             log_mar = logsumexp(evi)
#         else
#             pad = -1 * ones(1, gsize[2])
#             x_data = XData(Int8.(hcat(pad, x[:, 1:gsize[2]])))
#             fs = Dict("grid_size" => (gsize[1] - 1, gsize[2]), "f" => factor["f"][2:end, :, :], "g" => factor["g"][2:end, :, :])
#             log_mar = log_proba(pc, x_data)[1] + log_evidence_query(x, fs)
#         end        
#     else
#         error("x of size $(size(x)), factor size $(factor["grid_size"]), pc $(pc) Not implemented!")
#     end
#     log_mar    
# end

# function marginal_per_var(factor)
#     """
#     parameters:
#         factor: factors of the grid
#     return:
#         marginals of every variable
#     """
#     gsize = factor["grid_size"]
#     dim = prod(gsize)
  
#     p = zeros(2, dim)
#     for i = 0:2^dim - 1
#         x = digits(i, base=2, pad=dim)
#         evi = exp(log_marginal_query(x, factor))
#         # evi_test = marginal_query(x, factor)
#         # @assert evi ≈ evi_test
#         for j = 1:dim
#             p[x[j] + 1, j] += evi
#         end
#     end
#   # println(p)
#   # println(sum(p, dims=1))
#     p = p ./ sum(p, dims=1)
#     p
# end

function marginal_per_var(pc, var_num)
    """
    parameters:
        pc: a structured probabilistic circuit
        var_num: number of variables
    return:
        marginals of every variable
    """
    p = zeros(2, var_num)
    for i = 1 : var_num
        x = -1 * ones(1, var_num)
        x[i] = 0
        x = XData(Int8.(x))
        p[1, i] = exp(log_proba(pc, x)[1])
        x = -1 * ones(1, var_num)
        x[i] = 1
        x = XData(Int8.(x))
        p[2, i] = exp(log_proba(pc, x)[1])
    end
    println(p)
    # p = p ./ sum(p, dims=1)
    p
end

# function get_sample_marginals(factor, x, pc = nothing)
#     x_mars = Dict("p_xs" => [], "f_xs" => [])
#     sample_dim = length(x)
#     x = reshape(x, 1, sample_dim)
#     f, g, gsize = factor["f"], factor["g"], factor["grid_size"]    
#     if !(pc === nothing)        
#         fs = Dict("grid_size" => (gsize[1] - 1, gsize[2]), "f" => factor["f"][2 : end, :, :], "g" => factor["g"][2 : end, :, :])
#     end
    
#     for i in 1 : sample_dim
#         x_flip = copy(x)
#         x_flip[i] = !x_flip[i]
#         push!(x_mars["p_xs"], exp(log_marginal_query(x_flip, factor, pc)))
#         if !(pc === nothing)
#             push!(x_mars["f_xs"], exp(log_marginal_query(x_flip, fs)))
#         end
#     end
#     push!(x_mars["p_xs"], exp(log_marginal_query(x, factor, pc)))
#     if !(pc === nothing)
#         push!(x_mars["f_xs"], exp(log_marginal_query(x, fs)))
#     end
#     x_mars
# end

# function get_sample_log_marginals(factor, x, pc = nothing)
#     x_mars = Dict("p_xs" => [], "f_xs" => [])
#     sample_dim = length(x)
#     x = reshape(x, 1, sample_dim)
#     f, g, gsize = factor["f"], factor["g"], factor["grid_size"]    
#     if !(pc === nothing)        
#         fs = Dict("grid_size" => (gsize[1] - 1, gsize[2]), "f" => factor["f"][2 : end, :, :], "g" => factor["g"][2 : end, :, :])
#     end
    
#     for i in 1 : sample_dim
#         x_flip = copy(x)
#         x_flip[i] = !x_flip[i]
#         push!(x_mars["p_xs"], log_marginal_query(x_flip, factor, pc))
#         if !(pc === nothing)
#             push!(x_mars["f_xs"], log_marginal_query(x_flip, fs))
#         end
#     end
#     push!(x_mars["p_xs"], log_marginal_query(x, factor, pc))
#     if !(pc === nothing)
#         push!(x_mars["f_xs"], log_marginal_query(x, fs))
#     end
#     x_mars
# end

end