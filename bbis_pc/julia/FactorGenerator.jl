push!(LOAD_PATH, ".")
module FactorGenerator

using Random
# include("consts.jl")
using Consts: RANDOMSEED

export Clique, MarkovNet, generate_grid, parse_uai_to_grid, parse_uai_ground_truth, parse_uai_to_markov, parse_markov_to_uai

mutable struct Clique
    var_num::Int
    vars::Vector{Int}
    cpt::Dict
    function Clique()
        new(0, Vector{Int}(), Dict())
    end
end

mutable struct MarkovNet
    # type = "MARKOV"
    var_num::Int
    cardinalities::Vector{Int}
    cliques::Vector{Clique}
    function MarkovNet()
        new(0, Vector{Int}(), Vector{Clique}())
    end
end

function generate_grid(gsize; rng = nothing, gen = "random")
    @assert length(gsize) == 2
    if rng === nothing
        rng = MersenneTwister(RANDOMSEED)
    end

    if gen == "random"        
        f = rand(rng, Float64, (gsize[1], gsize[2] - 1, 4))
        g = rand(rng, Float64, (gsize[1] - 1, gsize[2], 4))

        for i in 1 : gsize[1] - 1
            for j in 1 : gsize[2] - 1
                f[i, j, 1] = 0.95
                f[i, j, 3] = 0.05
                f[i, j, 2] = 0.95
                f[i, j, 4] = 0.05
            end
        end

    elseif gen == "uniform"
        f = ones(Float64, (gsize[1], gsize[2] - 1, 4)) * 0.5
        g = ones(Float64, (gsize[1] - 1, gsize[2], 4)) * 0.5    

    else
        error("Not implemented!")
    end

    factor = Dict("grid_size" => gsize, "f" => f, "g" => g)
    factor
end

function parse_uai_to_grid(filename)
    model = readlines(filename)
    @assert string(model[1]) == "MARKOV" "Network Type Not Supported"
    filter!(x -> x!="", model)
    n_variables = parse(Int, model[2])
    
    cards = split(model[3], " ")
    for i in 1 : n_variables
        card = parse(Int, cards[i])
        @assert card == 2 "Boolean Variables Only"
    end
    
    n_factors = parse(Int, model[4])
    potentials = Dict()
    flag = 4 + n_factors + 1
    for i in 1 : n_factors
        indices = split(model[4+i], " ")
        n_fvars = parse(Int, indices[1])
        # n_pots = parse(Int, model[flag])

        if n_fvars == 1
            xi = parse(Int, indices[2]) + 1
            
            pots = filter!(x->x!="", split(model[flag+1], " "))
            pots = (parse(Float64, pots[1]), parse(Float64, pots[2]))
            potentials[xi] = Dict(0 => pots[1], 1 => pots[2])            
            
            # println("node potential $(xi) added")
            # println("with potentials: $(potentials[xi])")
        
        elseif n_fvars == 2
            xi, xj = parse(Int, indices[2]) + 1, parse(Int, indices[3]) + 1
            
            pots = filter!(x->x!="", split(model[flag+1], " "))            
            pots = (parse(Float64, pots[1]), parse(Float64, pots[2]), parse(Float64, pots[3]), parse(Float64, pots[4]))
            potentials[(xi, xj)] = Dict((0, 0) => pots[1], (0, 1) => pots[2], (1, 0) => pots[3], (1, 1) => pots[4])
            
            # println("edge potential $(xi) --- $(xj) added")
            # println("with potentials: $(potentials[(xi, xj)])")
        
        else
            error("$(n_fvars)-factor Not Implemented!")
        end
        flag = flag + 2
    end

    # turn into factors amenable to current alg
    gsize = (Int(sqrt(n_variables)), Int(sqrt(n_variables)))
    f = ones(Float64, (gsize[1], gsize[2] - 1, 4))
    g = ones(Float64, (gsize[1] - 1, gsize[2], 4))
    for i in 1 : gsize[1]
        for j in 1 : gsize[2] - 1
            xi, xj = (i - 1) * gsize[2] + j, (i - 1) * gsize[2] + j + 1
            f[i, j, 1] = potentials[(xi, xj)][0, 0]
            f[i, j, 2] = potentials[(xi, xj)][0, 1]
            f[i, j, 3] = potentials[(xi, xj)][1, 0]
            f[i, j, 4] = potentials[(xi, xj)][1, 1]
        end
    end

    for i in 1 : gsize[1] - 1
        for j in 1 : gsize[2]
            xi, xj = (i - 1) * gsize[2] + j, i * gsize[2] + j
            g[i, j, 1] = potentials[(xi, xj)][0, 0]
            g[i, j, 2] = potentials[(xi, xj)][0, 1]
            g[i, j, 3] = potentials[(xi, xj)][1, 0]
            g[i, j, 4] = potentials[(xi, xj)][1, 1]
        end
    end
    factor = Dict("grid_size" => gsize, "f" => f, "g" => g)
    factor
end

function parse_uai_to_markov(filename)
    markov = MarkovNet()

    model = readlines(filename)
    @assert string(model[1]) == "MARKOV" "Network Type Not Supported"
    filter!(x -> x!="", model)
    n_variables = parse(Int, model[2])
    markov.var_num = n_variables
    
    cards = split(model[3], " ")
    for i in 1 : n_variables
        card = parse(Int, cards[i])
        push!(markov.cardinalities, card)
        @assert card == 2 "Boolean Variables Only"
    end
    
    n_factors = parse(Int, model[4])
    potentials = Dict()
    flag = 4 + n_factors + 1
    for i in 1 : n_factors
        indices = split(model[4+i], " ")
        n_fvars = parse(Int, indices[1])
        # n_pots = parse(Int, model[flag])

        if n_fvars == 1
            xi = parse(Int, indices[2]) + 1
            
            pots = filter!(x->x!="", split(model[flag+1], " "))
            pots = (parse(Float64, pots[1]), parse(Float64, pots[2]))
            potentials = Dict(0 => pots[1], 1 => pots[2])            

            clique = Clique()
            clique.var_num = 1
            push!(clique.vars, xi)
            clique.cpt = potentials
            push!(markov.cliques, clique)
            
            # println("node potential $(xi) added")
            # println("with potentials: $(potentials)")
        
        elseif n_fvars == 2
            xi, xj = parse(Int, indices[2]) + 1, parse(Int, indices[3]) + 1
            
            pots = filter!(x->x!="", split(model[flag+1], " "))            
            pots = (parse(Float64, pots[1]), parse(Float64, pots[2]), parse(Float64, pots[3]), parse(Float64, pots[4]))
            potentials = Dict((0, 0) => pots[1], (0, 1) => pots[2], (1, 0) => pots[3], (1, 1) => pots[4])

            clique = Clique()
            clique.var_num = 2
            push!(clique.vars, xi, xj)
            clique.cpt = potentials
            push!(markov.cliques, clique)
            
            # println("edge potential $(xi) --- $(xj) added")
            # println("with potentials: $(potentials)")
        
        else
            error("$(n_fvars)-factor Not Implemented!")
        end
        flag = flag + 2
    end

    markov
end

function parse_markov_to_uai(filename, markov::MarkovNet)
    open(filename, "w") do io
        write(io, "MARKOV\n")
        write(io, "$(markov.var_num)\n")
        for card in markov.cardinalities
            @assert card == 2 "Boolean Variables Only"
            write(io, "$(card) ")
        end
        write(io, "\n")
        write(io, "$(length(markov.cliques))\n")
        for clique in markov.cliques
            if clique.var_num == 1
                write(io, "1 $(clique.vars[1] - 1)\n")
            elseif clique.var_num == 2
                write(io, "2 $(clique.vars[1] - 1) $(clique.vars[2] - 1)\n")
            else
                error("Unexpected numbers of variables in clique!")
            end
        end
        write(io, "\n")
        cards = markov.cardinalities
        for clique in markov.cliques
            write(io, "$(2 ^ clique.var_num)\n")
            write(io, " ")
            if clique.var_num == 1
                write(io, " $(clique.cpt[0]) $(clique.cpt[1])\n")
            elseif clique.var_num == 2
                write(io, " $(clique.cpt[0, 0]) $(clique.cpt[0, 1]) $(clique.cpt[1, 0]) $(clique.cpt[1, 1])\n")
            else
                error("Unexpected numbers of variables in clique!")
            end
            # for (k, v) in clique.cpt
            #     write(io, " $(v)")
            # end
            # write(io, "\n")
        end
    end
end

function parse_uai_ground_truth(filename)
    data = readlines(filename)
    all_mars = Dict()
    for line in data
        if length(line) < 4 || line[1:4] != "Pr(x"
            continue
        end
        xi = parse(Int, line[5:14]) + 1
        mars = replace.(line[21:end], ['[',']','=', ' '] => "")
        mars = split(mars, ",")
        all_mars[xi] = (parse(Float64, mars[1]), parse(Float64, mars[2]))        
    end
    n_vars = length(keys(all_mars))
    p = zeros(2, n_vars)
    for i in 1 : n_vars
        p[1, i] = all_mars[i][1]
        p[2, i] = all_mars[i][2]
    end
    p
end
end

# uai_dir = "../uai/small.uai"
# FactorGenerator.parse_uai_to_markov(uai_dir)
# filename = "../uai/frustrated_easy_12.uai"
# open(filename, "w") do io
#     write(io, "MARKOV\n")
#     write(io, "144\n")
#     for i in 1 : 144
#         write(io, "2 ")
#     end
#     write(io, "\n")
#     write(io, "144\n")
#     for i in 1 : 144
#         write(io, "1 $(i - 1)\n")
#     end
#     write(io, "\n")
#     for i in 1 : 144
#         write(io, "2\n")
#         write(io, " ")
#         write(io, " 0.3 0.7\n")
#     end
# end
