
module Mask

using DataStructures
using LogicCircuits
using ProbabilisticCircuits

# include("gen_grid.jl")
using FactorGenerator
# include("utils.jl")
using Utils

export generate_masked_circuits, log_factor_product

function partition(gsize, mask)
    """
    parameters:
        gsize: (m, n)
        mask: bitmap for mask
    return:
        clusters of unmasked points
    """
    @assert size(mask) == gsize
    m, n = gsize
    observed = copy(mask)
    clusters = []
    for i in 1 : m
        for j in 1 : n
            if observed[i, j] == 1
                continue
            end
            # push!(clusters, bfs4tree(i, j, gsize, observed, factors))
            push!(cluster, bfs(i, j, gsize, observed))
        end
    end
    clusters
end

function bfs(i, j, gsize, observed)
    q = Queue{Tuple}()
    visited = []
    enqueue!(q, (i, j))
    observed[i, j] = 1
    while length(q) > 0
        x, y = dequeue!(q)
        push!(visited, (x, y))
        if in_bound(x - 1, y, gsize) && observed[x - 1, y] == 0
            enqueue!(q, (x - 1, y))
            observed[x - 1, y] = 1
        end
        if in_bound(x + 1, y, gsize) && observed[x + 1, y] == 0
            enqueue!(q, (x + 1, y))
            observed[x + 1, y] = 1
        end
        if in_bound(x, y - 1, gsize) && observed[x, y - 1] == 0
            enqueue!(q, (x, y - 1))
            observed[x, y - 1] = 1
        end
        if in_bound(x, y + 1, gsize) && observed[x, y + 1] == 0
            enqueue!(q, (x, y + 1))
            observed[x, y + 1] = 1
        end
        # display(q)
    end
    visited
end

function bfs4tree(i, j, gsize, observed, factors)
    f, g = factors["f"], factors["g"]
    visited = []
    cpt = Dict(0=>1.0, 1=>1.0)
    push!(visited, (i, j, cpt))
    observed[i, j] = 1
    head = 0
    tail = 1
    while tail > head
        head += 1
        x, y, _ = visited[head]
        if in_bound(x - 1, y, gsize) && observed[x - 1, y] == 0
            factor = g[x - 1, y, :]
            cpt = Dict((0, 0)=>factor[1], (1, 0)=>factor[2], (0, 1)=>factor[3], (1, 1)=>factor[4])
            push!(visited, (x - 1, y, cpt))
            observed[x - 1, y] = 1
            tail += 1
        end
        if in_bound(x + 1, y, gsize) && observed[x + 1, y] == 0
            factor = g[x, y, :]
            cpt = Dict((0, 0)=>factor[1], (1, 0)=>factor[2], (0, 1)=>factor[3], (1, 1)=>factor[4])
            push!(visited, (x + 1, y, cpt))
            observed[x + 1, y] = 1
            tail += 1
        end
        if in_bound(x, y - 1, gsize) && observed[x, y - 1] == 0
            factor = f[x, y - 1, :]
            cpt = Dict((0, 0)=>factor[1], (1, 0)=>factor[2], (0, 1)=>factor[3], (1, 1)=>factor[4])
            push!(visited, (x, y - 1, cpt))
            observed[x, y - 1] = 1
            tail += 1
        end
        if in_bound(x, y + 1, gsize) && observed[x, y + 1] == 0
            factor = f[x, y, :]
            cpt = Dict((0, 0)=>factor[1], (1, 0)=>factor[2], (0, 1)=>factor[3], (1, 1)=>factor[4])
            push!(visited, (x, y + 1, cpt))
            observed[x, y + 1] = 1
            tail += 1
        end
        # display(q)
    end
    visited
end

function in_bound(i, j, gsize)
    m, n = gsize
    return (i >= 1) && (i <= m) && (j >= 1) && (j <= n)
end

function simplify(gsize, sample, markov::MarkovNet)
    """
    parameters:
        gsize: (m, n)
        sample: [1, mn] -1 for collapsed variables, 0/1 for sampled variables
        markov: original MarkovNet
    return:
        simplified MarkovNet
    """
    simplified_markov = MarkovNet()
    var_num = prod(gsize)
    @assert var_num == markov.var_num "MarkovNet has wrong number of variables!"
    simplified_markov.var_num = var_num
    simplified_markov.cardinalities = markov.cardinalities

    sample_markov = MarkovNet()
    sample_markov.var_num = var_num
    sample_markov.cardinalities = markov.cardinalities
    # for i in 1 : var_num
    #     if sample[i] != -1
    #         clique = Clique()
    #         clique.var_num = 1
    #         push!(clique.vars, i)
    #         if sample[i] == 1
    #             clique.cpt = Dict(0 => 0.0, 1 => 1.0)
    #         else
    #             clique.cpt = Dict(0 => 1.0, 1 => 0.0)
    #         end
    #         push!(simplified_markov.cliques, clique)
    #     #     push!(simplified_markov.cardinalities, 1)
    #     # else
    #     #     push!(simplified_markov.cardinalities, 2)
    #     end
    # end
    for clique in markov.cliques
        if clique.var_num == 1
            # i = clique.vars[1]
            # if sample[i] == -1
            #     push!(simplified_markov.cliques, clique)
            # end
            error("Unexpected node potiential!")
        elseif clique.var_num == 2
            i = clique.vars[1]
            j = clique.vars[2]
            if sample[i] == -1 && sample[j] == -1
                push!(simplified_markov.cliques, clique)
            elseif sample[i] == -1
                # cpt = clique.cpt
                # potentials = Dict(0 => cpt[(0, sample[j])], 1 => cpt[(1, sample[j])])
                # simplified_clique = Clique()
                # simplified_clique.var_num = 1
                # push!(simplified_clique.vars, i)
                # simplified_clique.cpt = potentials
                # push!(simplified_markov.cliques, simplified_clique)
                push!(simplified_markov.cliques, clique)
            elseif sample[j] == -1
                # cpt = clique.cpt
                # potentials = Dict(0 => cpt[(sample[i], 0)], 1 => cpt[(sample[i], 1)])
                # simplified_clique = Clique()
                # simplified_clique.var_num = 1
                # push!(simplified_clique.vars, j)
                # simplified_clique.cpt = potentials
                # push!(simplified_markov.cliques, simplified_clique)
                push!(simplified_markov.cliques, clique)
            else
                # cpt = clique.cpt
                # if sample[i] == 0
                #     potentials = Dict(0 => cpt[(0, sample[j])], 1 => 0.0)
                # else
                #     potentials = Dict(0 => 0, 1 => cpt[(1, sample[j])])
                # end
                # simplified_clique = Clique()
                # simplified_clique.var_num = 1
                # push!(simplified_clique.vars, i)
                # simplified_clique.cpt = potentials
                # push!(simplified_markov.cliques, simplified_clique)
                push!(sample_markov.cliques, clique)
            end
        else
        end
    end
    # for clique in markov.cliques
    #     if clique.var_num == 1
    #         error("Unexpected node potiential!")
    #     elseif clique.var_num == 2
    #         i = clique.vars[1]
    #         j = clique.vars[2]
    #         simplified_clique = Clique()
    #         simplified_clique.var_num = 2
    #         push!(simplified_clique.vars, i, j)            
    #         if sample[i] == -1 && sample[j] == -1
    #             push!(simplified_markov.cliques, clique)
    #         elseif sample[i] == -1
    #             cpt = clique.cpt
    #             potentials = Dict((0, sample[j]) => cpt[(0, sample[j])], (1, sample[j]) => cpt[(1, sample[j])])
    #             simplified_clique.cpt = potentials
    #             push!(simplified_markov.cliques, simplified_clique)
    #         elseif sample[j] == -1
    #             cpt = clique.cpt
    #             potentials = Dict((sample[i], 0) => cpt[(sample[i], 0)], (sample[i], 1) => cpt[(sample[i], 1)])
    #             simplified_clique.cpt = potentials
    #             push!(simplified_markov.cliques, simplified_clique)
    #         else
    #             cpt = clique.cpt
    #             potentials = Dict((sample[i], sample[j]) => cpt[(sample[i], sample[j])])
    #             simplified_clique.cpt = potentials
    #             push!(simplified_markov.cliques, simplified_clique)
    #         end
    #     else
    #         error("Unexpected clique with 2+ variables!")
    #     end
    # end
    simplified_markov, sample_markov
end

function generate_masked_circuits(gsize, samples, uai_folder_dir, vtree_dir)
    n_samples = size(samples, 1)
    # pcs = Vector()
    sample_markovs = Vector()
    for i in 1 : n_samples
        uai_filename = split(uai_folder_dir, '/')
        len = length(uai_filename)
        uai_filename = uai_filename[len]
        uai_dir = "$(uai_folder_dir)/$(uai_filename).uai"

        markov = parse_uai_to_markov(uai_dir)
        simplified_markov, sample_markov = simplify(gsize, samples[i, :], markov)

        output_dir = "$(uai_dir).$(i)"
        parse_markov_to_uai(output_dir, simplified_markov)
        run(`../uai/uai_compiler $(output_dir) $(vtree_dir)`)
        # pc, vtree = load_struct_prob_circuit("$(output_dir).psdd", "$(output_dir).vtree")
        # save_as_dot(pc, "$(output_dir).dot")
        # lines = readlines("$(output_dir).partition")
        # log_partition = parse(Float64, lines[1])
        # push!(pcs, (pc, log_partition))

        # output_dir = "$(uai_dir).$(i).sample"
        # if length(sample_markov.cliques) > 0
        #     parse_markov_to_uai(output_dir, sample_markov)
        #     run(`../uai/uai_compiler $(output_dir) $(vtree_dir)`)
        #     pc, vtree = load_struct_prob_circuit("$(output_dir).psdd", "$(output_dir).vtree")
        #     save_as_dot(pc, "$(output_dir).dot")
        # else
        #     pc = nothing
        # end
        # push!(sample_markovs, sample_markov)
        parse_markov_to_uai("$(output_dir).markov", sample_markov)
    end
    # pcs, sample_markovs
end

function log_factor_product(sample, markov::MarkovNet)
    res = 0
    for clique in markov.cliques
        res += log(clique.cpt[sample[clique.vars[1]], sample[clique.vars[2]]])
    end
    res
end

function test()
    gsize = (4, 3)
    uai_dir = "../uai/small.uai"
    markov = parse_uai_to_markov(uai_dir)
    # display(markov)
    sample = [
        -1, 1, 1,
        0, 0, 0,
        1, 1, 0,
        1, 0, 0
    ]
    simplified_markov = simplify(gsize, sample, markov)
    # display(simplified_markov)
    output_dir = "../uai/test.uai"
    vtree_dir = "../uai/small.uai.vtree"
    parse_markov_to_uai(output_dir, simplified_markov)
    run(`../uai/uai_compiler $(output_dir) $(vtree_dir)`)
    pc, vtree = load_struct_prob_circuit("$(output_dir).psdd", "$(output_dir).vtree")
    check_pc, check_vtree = load_struct_prob_circuit("$(uai_dir).psdd", "$(uai_dir).vtree")
    sample_1 = [
        0, 1, 1,
        0, 0, 0,
        1, 1, 0,
        1, 0, 0
    ]
    sample_2 = [
        1, 1, 1,
        0, 0, 0,
        1, 1, 0,
        1, 0, 0
    ]
    x1 = reshape(sample_1, 1, :)
    x2 = reshape(sample_2, 1, :)
    xs = reshape(sample, 1, :)
    x1 = XData(Bool.(x1))
    x2 = XData(Bool.(x2))
    xs = XData(Int8.(xs))
    display(log_proba(pc, x1) .+ log_proba(check_pc, xs))
    display(log_proba(check_pc, x1))
    display(log_proba(pc, x2) .+ log_proba(check_pc, xs))
    display(log_proba(check_pc, x2))
end

end

# Mask.test()