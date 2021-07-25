using PyPlot
using BSON
using Statistics
using LinearAlgebra
using ProbabilisticCircuits
using LogicCircuits

UAI_PATH = "../uai/frus_4_4_id0"
OUTPUT_PATH = "../plot/frus_4_4_id0"
ROUND = 1
MAX_SAMPLE = 200
MASK = 8
STEP = 10
VAR_NUM = 16

fontsize = 32
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["xtick.labelsize"] = fontsize - 4
rcParams["ytick.labelsize"] = fontsize - 4
rcParams["axes.labelsize"] = fontsize
rcParams["axes.titlesize"] = fontsize
rcParams["legend.fontsize"] = fontsize - 6
rcParams["figure.autolayout"] = true
rcParams["pdf.fonttype"] = 42

function distance(var_num, mar, exact_mar, type)
    
    @assert sum(mar, dims=1) ≈ ones(1, var_num)
    if type == "hellinger"
        diff = [norm(sqrt.(exact_mar[:, k]) - sqrt.(mar[:, k])) / sqrt(2) for k = 1 : var_num]
    elseif type == "kl"
        diff = [sum(exact_mar[:, k] .* log.(exact_mar[:, k] ./ mar[:, k])) for k = 1 : var_num]
    end
    mean(diff)
end

function compute_mar(samples, weights, uai_folder_dir)
    n_samples = size(samples, 1)
    if weights === nothing
        weights = ones(n_samples) ./ n_samples
    end

    # smooth
    weights .+= 1e-2
    weights /= sum(weights)

    var_num = size(samples, 2)
    mar = zeros(2, var_num)
    for i = 1 : n_samples
        sample = samples[i, :]
        sample = reshape(sample, 1, :)

        uai_filename = split(uai_folder_dir, '/')
        len = length(uai_filename)
        uai_filename = uai_filename[len]
        uai_dir = "$(uai_folder_dir)/$(uai_filename).uai"
        output_dir = "$(uai_dir)"
        norm_pc, _ = load_struct_prob_circuit("$(output_dir).psdd", "$(output_dir).vtree")

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
    mar
end

function seperate_plot(collapsed_uniform_weighted, collapsed_uniform_unweighted, uniform_weighted, uniform_unweighted, collapsed_exact_weighted, collapsed_exact_unweighted, exact_weighted, exact_unweighted, title, type)
    axis = STEP : STEP : MAX_SAMPLE

    clf()

    f, ax = subplots(2, 2, figsize=(18, 12))
    f.suptitle(title)
    linewidth = 3.5

    # figure 11: collapsed samples initiated with uniform distribution
    if collapsed_uniform_unweighted !== nothing && collapsed_uniform_weighted !== nothing
        ax[1, 1].set_title("CBBIS(uniform)")
        for i = 1 : ROUND
            ax[1, 1].plot(axis, vec(collapsed_uniform_unweighted[i, :]), color="skyblue")
        end
        mean_unweighted = mean(collapsed_uniform_unweighted, dims = 1)
        ax[1, 1].plot(axis, vec(mean_unweighted), color="deepskyblue", label="unweighted", linewidth=linewidth)

        for i = 1 : ROUND
            ax[1, 1].plot(axis, vec(collapsed_uniform_weighted[i, :]), color="pink")
        end
        mean_weighted = mean(collapsed_uniform_weighted, dims = 1)
        ax[1, 1].plot(axis, vec(mean_weighted), color="deeppink", label="weighted", linewidth=linewidth)

        ax[1, 1].set_xlabel("sample size")
        ax[1, 1].set_ylabel("mean dist.")
        ax[1, 1].legend(loc=1)
    end

    # figure 12: collapsed samples initiated with gibbs
    if collapsed_exact_unweighted !== nothing && collapsed_exact_weighted !== nothing
        ax[1, 2].set_title("CBBIS(gibbs)")
        for i = 1 : ROUND
            ax[1, 2].plot(axis, vec(collapsed_exact_unweighted[i, :]), color="skyblue")
        end
        mean_unweighted = mean(collapsed_exact_unweighted, dims = 1)
        ax[1, 2].plot(axis, vec(mean_unweighted), color="deepskyblue", label="unweighted", linewidth=linewidth)

        for i = 1 : ROUND
            ax[1, 2].plot(axis, vec(collapsed_exact_weighted[i, :]), color="pink")
        end
        mean_weighted = mean(collapsed_exact_weighted, dims = 1)
        ax[1, 2].plot(axis, vec(mean_weighted), color="deeppink", label="weighted", linewidth=linewidth)

        ax[1, 2].set_xlabel("sample size")
        ax[1, 2].set_ylabel("mean dist.")
        ax[1, 2].legend(loc=1)
    end

    # figure 21: samples initiated with uniform distribution
    if uniform_unweighted !== nothing && uniform_weighted !== nothing
        ax[2, 1].set_title("BBIS(uniform)")
        for i = 1 : ROUND
            ax[2, 1].plot(axis, vec(uniform_unweighted[i, :]), color="skyblue")
        end
        mean_unweighted = mean(uniform_unweighted, dims = 1)
        ax[2, 1].plot(axis, vec(mean_unweighted), color="deepskyblue", label="unweighted", linewidth=linewidth)

        for i = 1 : ROUND
            ax[2, 1].plot(axis, vec(uniform_weighted[i, :]), color="pink")
        end
        mean_weighted = mean(uniform_weighted, dims = 1)
        ax[2, 1].plot(axis, vec(mean_weighted), color="deeppink", label="weighted", linewidth=linewidth)

        ax[2, 1].set_xlabel("sample size")
        ax[2, 1].set_ylabel("mean dist.")
        ax[2, 1].legend(loc=1)
    end
    
    # figure 22: samples initiated with pc
    if exact_unweighted !== nothing && exact_weighted !== nothing
        ax[2, 2].set_title("BBIS(gibbs)")
        for i = 1 : ROUND
            ax[2, 2].plot(axis, vec(exact_unweighted[i, :]), color="skyblue")
        end
        mean_unweighted = mean(exact_unweighted, dims = 1)
        ax[2, 2].plot(axis, vec(mean_unweighted), color="deepskyblue", label="unweighted", linewidth=linewidth)

        for i = 1 : ROUND
            ax[2, 2].plot(axis, vec(exact_weighted[i, :]), color="pink")
        end
        mean_weighted = mean(exact_weighted, dims = 1)
        ax[2, 2].plot(axis, vec(mean_weighted), color="deeppink", label="weighted", linewidth=linewidth)

        ax[2, 2].set_xlabel("sample size")
        ax[2, 2].set_ylabel("mean dist.")
        ax[2, 2].legend(loc=1)
    end

    filename = "$(OUTPUT_PATH)_separated_$(type).pdf"
    savefig(filename)
end

function unified_plot(collapsed_uniform_weighted, collapsed_uniform_unweighted, uniform_weighted, uniform_unweighted, collapsed_exact_weighted, collapsed_exact_unweighted, exact_weighted, exact_unweighted, title, type)
    cuw = (collapsed_uniform_weighted === nothing) ? nothing : mean(collapsed_uniform_weighted, dims = 1)
    cuu = (collapsed_uniform_unweighted === nothing) ? nothing : mean(collapsed_uniform_unweighted, dims = 1)
    uw = (uniform_weighted === nothing) ? nothing : mean(uniform_weighted, dims = 1)
    uu = (uniform_unweighted === nothing) ? nothing : mean(uniform_unweighted, dims = 1)
    cew = (collapsed_exact_weighted === nothing) ? nothing : mean(collapsed_exact_weighted, dims = 1)
    ceu = (collapsed_exact_unweighted === nothing) ? nothing : mean(collapsed_exact_unweighted, dims = 1)
    ew = (exact_weighted === nothing) ? nothing : mean(exact_weighted, dims = 1)
    eu = (exact_unweighted === nothing) ? nothing : mean(exact_unweighted, dims = 1)

    axis = STEP : STEP : MAX_SAMPLE

    clf()

    PyPlot.title("comparison")
    if cuw !== nothing
        plot(axis, vec(cuw), label="CBBIS(uniform)")
    end
    if cuu !== nothing
        plot(axis, vec(cuu), label="Collapse(uniform)")
    end
    if uw !== nothing
        plot(axis, vec(uw), label="BBIS(uniform)")
    end
    if uu !== nothing
        plot(axis, vec(uu), label="Non-collapse(uniform)")
    end
    if cew !== nothing
        plot(axis, vec(cew), label="CBBIS(gibbs)")
    end
    if ceu !== nothing
        plot(axis, vec(ceu), label="Collapse(gibbs)")
    end
    if ew !== nothing
        plot(axis, vec(ew), label="BBIS(gibbs)")
    end
    if eu !== nothing
        plot(axis, vec(eu), label="Non-collapse(gibbs)")
    end
    
    xlabel("sample size")
    ylabel("mean dist.")
    legend(loc=1)
    
    filename = "$(OUTPUT_PATH)_unified_$(type).pdf"
    savefig(filename)
end

function final_plot(collapsed_uniform_weighted, collapsed_uniform_unweighted, uniform_weighted, uniform_unweighted, collapsed_exact_weighted, collapsed_exact_unweighted, exact_weighted, exact_unweighted, proposal, type)

    cuw = (collapsed_uniform_weighted === nothing) ? nothing : mean(collapsed_uniform_weighted, dims = 1)
    cuu = (collapsed_uniform_unweighted === nothing) ? nothing : mean(collapsed_uniform_unweighted, dims = 1)
    uw = (uniform_weighted === nothing) ? nothing : mean(uniform_weighted, dims = 1)
    uu = (uniform_unweighted === nothing) ? nothing : mean(uniform_unweighted, dims = 1)
    cew = (collapsed_exact_weighted === nothing) ? nothing : mean(collapsed_exact_weighted, dims = 1)
    ceu = (collapsed_exact_unweighted === nothing) ? nothing : mean(collapsed_exact_unweighted, dims = 1)
    ew = (exact_weighted === nothing) ? nothing : mean(exact_weighted, dims = 1)
    eu = (exact_unweighted === nothing) ? nothing : mean(exact_unweighted, dims = 1)

    axis = STEP : STEP : MAX_SAMPLE

    clf()
    fig = PyPlot.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    ax.set_title("frus 4x4 (\$q\$ $(proposal))")
    # ax.set_aspect(1.5)
    linewidth = 3.5

    if collapsed_uniform_unweighted !== nothing && collapsed_uniform_weighted !== nothing
        for i = 1 : ROUND
            ax.plot(axis, vec(collapsed_uniform_unweighted[i, :]), color="skyblue")
        end
        ax.plot(axis, vec(cuu), color="deepskyblue", label="CVS", linewidth=linewidth)

        for i = 1 : ROUND
            ax.plot(axis, vec(collapsed_uniform_weighted[i, :]), color="pink")
        end
        ax.plot(axis, vec(cuw), color="deeppink", label="CBBIS", linewidth=linewidth)
    end

    if uniform_unweighted !== nothing && uniform_weighted !== nothing
        for i = 1 : ROUND
            ax.plot(axis, vec(uniform_unweighted[i, :]), color="yellowgreen")
        end
        ax.plot(axis, vec(uu), color="darkolivergreen", label="VS", linewidth=linewidth)

        for i = 1 : ROUND
            ax.plot(axis, vec(uniform_weighted[i, :]), color="bisque")
        end
        ax.plot(axis, vec(uw), color="darkorange", label="BBIS", linewidth=linewidth)
    end

    if collapsed_exact_unweighted !== nothing && collapsed_exact_weighted !== nothing
        for i = 1 : ROUND
            ax.plot(axis, vec(collapsed_exact_unweighted[i, :]), color="skyblue")
        end
        ax.plot(axis, vec(ceu), color="deepskyblue", label="CVS", linewidth=linewidth)

        for i = 1 : ROUND
            ax.plot(axis, vec(collapsed_exact_weighted[i, :]), color="pink")
        end
        ax.plot(axis, vec(cew), color="deeppink", label="CBBIS", linewidth=linewidth)
    end

    if exact_unweighted !== nothing && exact_weighted !== nothing
        for i = 1 : ROUND
            ax.plot(axis, vec(exact_unweighted[i, :]), color="yellowgreen")
        end
        ax.plot(axis, vec(eu), color="darkolivegreen", label="VS", linewidth=linewidth)

        for i = 1 : ROUND
            ax.plot(axis, vec(exact_weighted[i, :]), color="bisque")
        end
        ax.plot(axis, vec(ew), color="darkorange", label="BBIS", linewidth=linewidth)
    end

    ax.grid()
    ax.set_xlabel("\$N\$ (# samples)")
    ax.set_ylabel("avg marginal $(type) dist.")
    ax.legend(loc=1)

    filename = "$(OUTPUT_PATH)_final_$(proposal)_$(type).pdf"
    savefig(filename)
end

# configs = vec(collect.(Iterators.product([false, true], [true, false], ["uniform"])))
configs = vec(collect.(Iterators.product([false, true], [true, false], ["gibbs"])))
exact_mar = BSON.load("$(UAI_PATH)/marginals.bson")["marginals"]
display(exact_mar)
println()

results_kl = Dict((c => [[] for _ = 1 : ROUND] for c in configs))
results_hellinger = Dict((c => [[] for _ = 1 : ROUND] for c in configs))

for round = 1 : ROUND
    for c in configs
        collapsed, weighted, proposal = c
        for num = STEP : STEP : MAX_SAMPLE
            res_path = "$(UAI_PATH)/RES-$(c)-NUM=$(num)-MASK=$(MASK)-ROUND=$(round).bson"
            res = BSON.load(res_path)
            # display(res)
            # println()
            samples, weights = res["samples"], res["weights"]
            mar = compute_mar(samples, weights, UAI_PATH)
            # display(mar)
            # println()
            # display(res["mar"])
            # println()
            # @assert mar ≈ res["mar"] "compute results should be equal"
            # mar = res["mar"]
            err = distance(VAR_NUM, mar, exact_mar, "kl")
            push!(results_kl[c][round], err)
            err = distance(VAR_NUM, mar, exact_mar, "hellinger")
            push!(results_hellinger[c][round], err)
            print('>')
        end
        println()
    end
end

plot_results = Dict()
for c in configs
    plot_results[c] = vcat([reshape(log.(r), 1, :) for r in results_kl[c]]...)
end
display(plot_results)
println()

# collapsed_uniform_weighted = plot_results[[true, true, "uniform"]]
# collapsed_uniform_unweighted = plot_results[[true, false, "uniform"]]
# uniform_weighted = plot_results[[false, true, "uniform"]]
# uniform_unweighted = plot_results[[false, false, "uniform"]]
collapsed_exact_weighted = plot_results[[true, true, "gibbs"]]
collapsed_exact_unweighted = plot_results[[true, false, "gibbs"]]
exact_weighted = plot_results[[false, true, "gibbs"]]
exact_unweighted = plot_results[[false, false, "gibbs"]]

collapsed_uniform_weighted = nothing
collapsed_uniform_unweighted = nothing
uniform_weighted = nothing
uniform_unweighted = nothing
# collapsed_exact_weighted = nothing
# collapsed_exact_unweighted = nothing
# exact_weighted = nothing
# exact_unweighted = nothing

title = "uai=$(UAI_PATH)-mask=$(MASK)-distance=kl"

# seperate_plot(collapsed_uniform_weighted, collapsed_uniform_unweighted, uniform_weighted, uniform_unweighted, collapsed_exact_weighted, collapsed_exact_unweighted, exact_weighted, exact_unweighted, title, DISTANCE_TYPE)
# unified_plot(collapsed_uniform_weighted, collapsed_uniform_unweighted, uniform_weighted, uniform_unweighted, collapsed_exact_weighted, collapsed_exact_unweighted, exact_weighted, exact_unweighted, title, DISTANCE_TYPE)
final_plot(collapsed_uniform_weighted, collapsed_uniform_unweighted, uniform_weighted, uniform_unweighted, collapsed_exact_weighted, collapsed_exact_unweighted, exact_weighted, exact_unweighted, "gibbs", "kl")

plot_results = Dict()
for c in configs
    plot_results[c] = vcat([reshape(log.(r), 1, :) for r in results_hellinger[c]]...)
end
display(plot_results)
println()

# collapsed_uniform_weighted = plot_results[[true, true, "uniform"]]
# collapsed_uniform_unweighted = plot_results[[true, false, "uniform"]]
# uniform_weighted = plot_results[[false, true, "uniform"]]
# uniform_unweighted = plot_results[[false, false, "uniform"]]
collapsed_exact_weighted = plot_results[[true, true, "gibbs"]]
collapsed_exact_unweighted = plot_results[[true, false, "gibbs"]]
exact_weighted = plot_results[[false, true, "gibbs"]]
exact_unweighted = plot_results[[false, false, "gibbs"]]

collapsed_uniform_weighted = nothing
collapsed_uniform_unweighted = nothing
uniform_weighted = nothing
uniform_unweighted = nothing
# collapsed_exact_weighted = nothing
# collapsed_exact_unweighted = nothing
# exact_weighted = nothing
# exact_unweighted = nothing

title = "uai=$(UAI_PATH)-mask=$(MASK)-distance=hellinger"

# seperate_plot(collapsed_uniform_weighted, collapsed_uniform_unweighted, uniform_weighted, uniform_unweighted, collapsed_exact_weighted, collapsed_exact_unweighted, exact_weighted, exact_unweighted, title, DISTANCE_TYPE)
# unified_plot(collapsed_uniform_weighted, collapsed_uniform_unweighted, uniform_weighted, uniform_unweighted, collapsed_exact_weighted, collapsed_exact_unweighted, exact_weighted, exact_unweighted, title, DISTANCE_TYPE)
final_plot(collapsed_uniform_weighted, collapsed_uniform_unweighted, uniform_weighted, uniform_unweighted, collapsed_exact_weighted, collapsed_exact_unweighted, exact_weighted, exact_unweighted, "gibbs", "hellinger")
