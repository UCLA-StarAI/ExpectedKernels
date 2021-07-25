using Revise
push!(LOAD_PATH, ".")

module MaskGridSeq

ENV["MPLBACKEND"] = "Agg"
using PyPlot
using Random
using LogicCircuits
using ProbabilisticCircuits
using Statistics
using Logging
using Dates
using BSON
using Distributions
using LinearAlgebra
using ThreadPools

using QueryHandler
using Utils
using DoubleSum
using Consts: TOLERANCE, KERNEL, RANDOMSEED
using FactorGenerator
using Mask

flag = "run-seq"

function compute_k_matrix_entry(x1, x2, pc, gsize)
    # @info x1 x2 gsize

    # print_circuit_dfs(pc1[end])
    # print_circuit_dfs(pc2[end])
    x1 = reshape(x1, 1, :)
    x2 = reshape(x2, 1, :)

    @assert size(x1) == size(x2) "x1 and x2 have different size!"
    n, var_num = size(x1)

    value = zeros(n)
    # @assert length(x1) == length(x2)
    # var_num = length(x1)

    p_s1 = exp.(log_proba(pc, XData(Int8.(x1))))
    p_s2 = exp.(log_proba(pc, XData(Int8.(x2))))
    for i = 1 : var_num
        x1_flip = copy(x1)
        x1_flip[:, i] = 1 .- x1_flip[:, i]
        x2_flip = copy(x2)
        x2_flip[:, i] = 1 .- x2_flip[:, i]
        # display(XData(Int8.(x1)))
        # display(XData(Int8.(x2)))
        # display(XData(Int8.(x1_flip)))
        # display(XData(Int8.(x2_flip)))
        sums = [
            double_sum(XData(Int8.(x1)), XData(Int8.(x2)), pc, pc, gsize, i),
            double_sum(XData(Int8.(x1_flip)), XData(Int8.(x2)), pc, pc, gsize, i),
            double_sum(XData(Int8.(x1)), XData(Int8.(x2_flip)), pc, pc, gsize, i),
            double_sum(XData(Int8.(x1_flip)), XData(Int8.(x2_flip)), pc, pc, gsize, i)
        ]
        tmp = sums[1] .- sums[2] .- sums[3] .+ sums[4]
        # display(value)
        # display(double_sum(XData(Int8.(x1)), XData(Int8.(x2)), pc, pc, gsize, i))
        # display(tmp)
        # display((x1[:, i] .!= -1) .* (x2[:, i] .!= -1))
        value .+= (x1[:, i] .!= -1) .* (x2[:, i] .!= -1) .* tmp
        # print('>')
    end

    value ./= (p_s1 .* p_s2)

    value
end

function compute_k_matrix_entry(x1, x2, pc, pc1, pc2, markov1, markov2, scope1, scope2, node2id1, node2id2, gsize)
    # @info x1 x2 gsize

    # print_circuit_dfs(pc1[end])
    # print_circuit_dfs(pc2[end])

    # @assert size(x1) == size(x2) "x1 and x2 have different size!"
    # n, var_num = size(x1)

    value = 0.
    @assert length(x1) == length(x2)
    var_num = length(x1)

    x1 = reshape(x1, 1, :)
    x2 = reshape(x2, 1, :)
    # display(x1)
    # display(x2)

    # @time p_f1 = exp.(pc1[2]) * exp.(log_factor_product(x1, markov1))
    # @time p_f2 = exp.(pc2[2]) * exp.(log_factor_product(x2, markov2))
    log_p_s1 = log_proba(pc1[1], XData(Int8.(x1)))[1]
    log_p_s2 = log_proba(pc2[1], XData(Int8.(x2)))[1]
    log_p_f1 = log_factor_product(x1, markov1)
    log_p_f2 = log_factor_product(x2, markov2)

    cache = GlobalSumCache()
    for i = 1 : var_num
        if x1[i] != -1 && x2[i] != -1
            # display(cache)
            x1_flip = copy(x1)
            x1_flip[i] = 1 - x1_flip[i]
            x2_flip = copy(x2)
            x2_flip[i] = 1 - x2_flip[i]
            log_p_f1_flip = log_factor_product(x1_flip, markov1)
            log_p_f2_flip = log_factor_product(x2_flip, markov2)
            log_sums = [
                log_double_sum(x1, x2, pc1[1], pc2[1], scope1, scope2, node2id1, node2id2, cache, gsize, i)[1],
                log_double_sum(x1_flip, x2, pc1[1], pc2[1], scope1, scope2, node2id1, node2id2, cache, gsize, i)[1],
                log_double_sum(x1, x2_flip, pc1[1], pc2[1], scope1, scope2, node2id1, node2id2, cache, gsize, i)[1],
                log_double_sum(x1_flip, x2_flip, pc1[1], pc2[1], scope1, scope2, node2id1, node2id2, cache, gsize, i)[1]
            ]
            coefficients = [1, -1, -1, 1]
            log_marginals = [
                log_p_f1 + log_p_f2,
                log_p_f1_flip + log_p_f2,
                log_p_f1 + log_p_f2_flip,
                log_p_f1_flip + log_p_f2_flip
            ]
            log_marginals .-= (log_p_s1 + log_p_s2 + log_p_f1 + log_p_f2)
            sums = exp.(log_sums .+ log_marginals)
            value += sum(sums .* coefficients)
            
        end
    end

    # cache = nothing
    # println('*')
    value
end

function compute_k_matrix_entry(x1, x2, markov1, markov2, gsize)
    # non-collapsed

    # @info x1 x2 gsize

    # print_circuit_dfs(pc1[end])
    # print_circuit_dfs(pc2[end])

    # @assert size(x1) == size(x2) "x1 and x2 have different size!"
    # n, var_num = size(x1)

    value = 0.
    @assert length(x1) == length(x2)
    var_num = length(x1)

    x1 = reshape(x1, 1, :)
    x2 = reshape(x2, 1, :)
    # display(x1)
    # display(x2)

    log_p_f1 = log_factor_product(x1, markov1)
    log_p_f2 = log_factor_product(x2, markov2)

    for i = 1 : var_num
        # display(cache)
        x1_flip = copy(x1)
        x1_flip[i] = 1 - x1_flip[i]
        x2_flip = copy(x2)
        x2_flip[i] = 1 - x2_flip[i]
        log_p_f1_flip = log_factor_product(x1_flip, markov1)
        log_p_f2_flip = log_factor_product(x2_flip, markov2)
        
        coefficients = [1, -1, -1, 1]
        log_ps = [
            log_p_f1 + log_p_f2,
            log_p_f1_flip + log_p_f2,
            log_p_f1 + log_p_f2_flip,
            log_p_f1_flip + log_p_f2_flip
        ]
        log_ps .-= (log_p_f1 + log_p_f2)
        kernels = [
            kernel(x1_flip, x2_flip, gsize),
            kernel(x1, x2_flip, gsize),
            kernel(x1_flip, x2, gsize),
            kernel(x1, x2, gsize)
        ]
        value += sum(exp.(log_ps) .* coefficients .* kernels)

    end

    # cache = nothing
    # println('*')
    value
end

function compute_k_row(samples, pc, uai_folder_dir, gsize, collapsed)
    # pcs: ([pc, log_partition], [markov])
    n_samples = size(samples, 1)
    row = zeros(n_samples)
    vec = []
    for i = 1 : n_samples
        push!(vec, (i => n_samples))
    end
    @qbthreads for (i, j) in vec
        if collapsed && (uai_folder_dir !== nothing)
            uai_filename = split(uai_folder_dir, '/')
            len = length(uai_filename)
            uai_filename = uai_filename[len]
            uai_dir = "$(uai_folder_dir)/$(uai_filename).uai"
            # lock(mutex)
            # println("loading arg1")
            dir1 = "$(uai_dir).$(i)"
            pc1, _ = load_struct_prob_circuit("$(dir1).psdd", "$(dir1).vtree")
            lines = readlines("$(dir1).partition")
            partition1 = parse(Float64, lines[1])
            markov1 = parse_uai_to_markov("$(dir1).markov")
            scopes1 = variable_scopes(pc1)
            node2id1 = Dict(pc1[k] => k for k in 1 : length(pc1))
            dir2 = "$(uai_dir).$(j)"
            pc2, _ = load_struct_prob_circuit("$(dir2).psdd", "$(dir2).vtree")                
            lines = readlines("$(dir2).partition")
            partition2 = parse(Float64, lines[1])
            markov2 = parse_uai_to_markov("$(dir2).markov")
            scopes2 = variable_scopes(pc2)                
            node2id2 = Dict(pc2[k] => k for k in 1 : length(pc2))
            # println("arg1 loaded")
            # unlock(mutex)
            tmp = compute_k_matrix_entry(samples[i, :], samples[j, :], pc, (pc1, partition1), (pc2, partition2), markov1, markov2, scopes1, scopes2, node2id1, node2id2, gsize)
        elseif uai_folder_dir !== nothing
            # non-collapsed
            uai_filename = split(uai_folder_dir, '/')
            len = length(uai_filename)
            uai_filename = uai_filename[len]
            uai_dir = "$(uai_folder_dir)/$(uai_filename).uai"
            # lock(mutex)
            # println("loading arg1")
            whole_markov = parse_uai_to_markov("$(uai_dir)")
            # println("arg1 loaded")
            # unlock(mutex)
            tmp = compute_k_matrix_entry(samples[i, :], samples[j, :], whole_markov, whole_markov, gsize)         
        else
            println("should assert False!")
            @assert False "not implemented!"
        end
        # lock(mutex)
        row[i] = tmp
        # unlock(mutex)
        print('>')
    end
    println()
    row
end

function inference(samples, k_p, weighted)
    weights, loss = nothing, nothing
    if weighted
        weights, loss = cvxopt(k_p)
    end
    res = Dict("samples" => samples, "weights" => weights, "loss" => loss)
end

function distance(samples, exact_mar, weights, pc, uai_folder_dir, type, collapsed)
    n_samples = size(samples, 1)
    if weights === nothing
        weights = ones(n_samples) ./ n_samples
    end
    var_num = size(samples, 2)
    mar = zeros(2, var_num)
    for i = 1 : n_samples
        sample = samples[i, :]
        sample = reshape(sample, 1, :)
        if collapsed && uai_folder_dir !== nothing
            uai_filename = split(uai_folder_dir, '/')
            len = length(uai_filename)
            uai_filename = uai_filename[len]
            uai_dir = "$(uai_folder_dir)/$(uai_filename).uai"
            output_dir = "$(uai_dir).$(i)"
            norm_pc, _ = load_struct_prob_circuit("$(output_dir).psdd", "$(output_dir).vtree")
        else
            norm_pc = pc
        end
        for j in 1 : var_num
            if sample[j] == -1
                norm = exp(log_proba(norm_pc, XData(Int8.(sample)))[1])
                for v in [0, 1]
                    x = copy(sample)
                    x[j] = v
                    p = exp(log_proba(norm_pc, XData(Int8.(x)))[1]) / norm
                    mar[v + 1, j] += weights[i] * p
                end
            else
                mar[sample[j] + 1, j] += weights[i]
            end
        end
    end
    for k = 1 : var_num
        if isapprox(mar[1, k], 0.0; atol=TOLERANCE, rtol=0)
            mar[1, k], mar[2, k] = 0.0, 1.0
        elseif isapprox(mar[1, k], 1.0; atol=TOLERANCE, rtol=0)
            mar[1, k], mar[2, k] = 1.0, 0.0
        end
    end
    # display(mar)
    @assert sum(mar, dims=1) ≈ ones(1, var_num)
    if type == "hellinger"
        diff = [norm(sqrt.(exact_mar[:, k]) - sqrt.(mar[:, k])) / sqrt(2) for k = 1 : var_num]
    elseif type == "kl"
        diff = [sum(mar[:, k] .* log.(mar[:, k] ./ exact_mar[:, k])) for k = 1 : var_num]
    end
    mean(diff)
end

function run_seq(pc_dir, vtree_dir, gsize, tries, max_sample, step, mask_num; gt_dir=nothing, uai_dir=nothing, distance_type="hellinger")
    timestamp = now()
    io = open("../log/$(timestamp)-$(gsize)-$(flag).log", "w+")
    # logger = SimpleLogger(io)
    logger = SimpleLogger()
    global_logger(logger)
    @info "configuration" gsize tries max_sample step KERNEL RANDOMSEED

    if pc_dir !== nothing
        pc, vtree = load_struct_prob_circuit(pc_dir, vtree_dir)
    else
        pc = nothing
    end

    if gt_dir === nothing
        @assert uai_dir !== nothing "no uai_dir for marginals!"
        marginal_path = "$(uai_dir)/marginals.bson"
        if isfile(marginal_path)
            print("file ", marginal_path, " found!\n")
            exact_mar = BSON.load(marginal_path)["marginals"]
        else
            exact_mar = marginal_per_var(pc, prod(gsize))
            bson(marginal_path, Dict("marginals" => exact_mar))
        end
    else
        exact_mar = parse_uai_ground_truth(gt_dir)
    end
    @info exact_mar
    flush(io)
    
    rng = MersenneTwister(RANDOMSEED)
    # configs = vec(collect.(Iterators.product([true, false], [true, false], ["uniform", "exact"])))
    configs = vec(collect.(Iterators.product([false], [true], ["uniform"])))
    results = Dict((c => [[] for _ = 1 : tries] for c in configs))
    opt_loss = Dict((c => [[] for _ = 1 : tries] for c in configs))

    @fastmath @simd for round = 1 : tries
        @info "ROUND $(round)"

        println("sampling time")
        
        samples_path = "$(uai_dir)/ROUND-$(round)-NUM-$(max_sample)-UNIFORM-SAMPLE.bson"
        if isfile(samples_path)
            print("file ", samples_path, " found!\n")
            all_samples = BSON.load(samples_path)["all_samples"]
        else
            @time all_samples = generate_samples(rng, max_sample, "uniform", prod(gsize), pc)
            # all_samples = mask_samples(all_samples, gsize, mask_num)
            # if uai_dir !== nothing
            #     # pcs = generate_masked_circuits(gsize, all_samples, uai_dir, vtree_dir)
            #     generate_masked_circuits(gsize, all_samples, uai_dir, vtree_dir)
            # end
            bson(samples_path, Dict("all_samples" => all_samples))
        end
        

        for c in configs
            collapsed, weighted, proposal = c
            @info "RUN collapsed = $(collapsed), weighted = $(weighted), proposal = $(proposal)"
            println("config: $(c)")
            
            # pcs = nothing
            
            k_p = zeros(max_sample, max_sample)
            for i = 1 : max_sample
                sub_samples = all_samples[1 : i, :]
                computation_path = "$(uai_dir)/ROUND-$(round)-NUM-$(max_sample)-KP.bson"
                # weighted
                row = compute_k_row(sub_samples, pc, uai_dir, gsize, collapsed)
                # unweighted
                for j = 1 : i
                    k_p[j, i] = row[j]
                    k_p[i, j] = row[j]
                end
                sub_k_p = k_p[1 : i, 1 : i]
                computation = Dict("sub_k_p" => sub_k_p, "sub_samples" => sub_samples)
                bson(computation_path, computation)
                res_weighted = inference(sub_samples, sub_k_p, true)
                weights_weighted = res_weighted["weights"]
                err_weighted = distance(sub_samples, exact_mar, weights_weighted, pc, uai_dir, distance_type, collapsed)
                err_weighted = log(err_weighted)
                res_unweighted = inference(sub_samples, nothing, false)
                weights_unweighted = res_unweighted["weights"]
                err_unweighted = distance(sub_samples, exact_mar, weights_unweighted, pc, uai_dir, distance_type, collapsed)
                err_unweighted = log(err_unweighted)
                println("num=$(i) err_weighted=$(err_weighted) err_unweighted=$(err_unweighted)")
            end

            flush(io)
        end        
    end

    close(io)
end

end

using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "--pc"
        help = "pc file directory"
        arg_type = String
        default = nothing
    "--vtree"
        help = "vtree file directory"
        arg_type = String
    "--grid1"
        help = "the first dimension of the grid"
        arg_type = Int
    "--grid2"
        help = "the second dimension of the grid"
        arg_type = Int
    "--tries"
        help = "number of tries for one grid"
        arg_type = Int
        default = 1
    "--max-samples", "-s"
        help = "the maximum number of samples"
        arg_type = Int
    "--step"
        help = "step of samples"
        arg_type = Int
        default = 5
    "--mask", "-m"
        help = "the number of variables in the mask"
        arg_type = Int
    "--uai"
        help = "path to the uai file"
        arg_type = String
    "--gt"
        help = "ground truth file"
        arg_type = String    
        default = nothing
    "--dist"
        help = "distance type: kl/hellinger"
        arg_type = String
end

pa = parse_args(s)

@time MaskGridSeq.run_seq(pa["pc"], pa["vtree"], (pa["grid1"], pa["grid2"]), pa["tries"], pa["max-samples"], pa["step"], pa["mask"]; gt_dir=pa["gt"], uai_dir=pa["uai"], distance_type=pa["dist"])
