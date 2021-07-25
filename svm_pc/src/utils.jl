export learn_circuits, plot_rmse, ExpFloat

const ExpFloat = Float64

function learn_circuits(data)
    pc, vtree = learn_chow_liu_tree_circuit(data)
    println("Initial circuits has $(num_nodes(pc)) nodes, $(num_parameters(pc)) parameters.")
    println("Training set log-likelihood is ", log_likelihood_avg(pc, data))

    loss(circuit) = ProbabilisticCircuits.heuristic_loss(circuit, data)
    pc = struct_learn(pc; 
        primitives=[split_step], 
        kwargs=Dict(split_step=>(loss=loss,)),
        maxiter=10)
    estimate_parameters(pc, data; pseudocount=1.0)
    println("Circuits has $(num_nodes(pc)) nodes, $(num_parameters(pc)) parameters.")
    println("Training set log-likelihood is ", log_likelihood_avg(pc, data))
    pc
end

fontsize = 52
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["xtick.labelsize"] = fontsize
rcParams["ytick.labelsize"] = fontsize
rcParams["axes.labelsize"] = fontsize
rcParams["axes.titlesize"] = fontsize
rcParams["legend.fontsize"] = fontsize - 8
rcParams["figure.autolayout"] = true
rcParams["pdf.fonttype"] = 42

function plot_rmse(x, y_pc, y_mode, y_map, y_sample, title, ylabel, legend)
    clf()
    fig = PyPlot.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    linewidth = 4.5
    ax.set_title("$(title)")
    
    round = size(y_pc, 1)

    mean_mode = mean(y_mode, dims=1)
    std_mode = std(y_mode, dims=1)
    ax.plot(x, vec(mean_mode), color="deepskyblue", label="Median Imputation", linewidth=linewidth, marker="o", markersize=20)
    ax.fill_between(x, vec(mean_mode - std_mode), vec(mean_mode + std_mode), alpha=0.3, color="deepskyblue")

    mean_map = mean(y_map, dims=1)
    std_map = std(y_map, dims=1)
    ax.plot(x, vec(mean_map), color="darkolivegreen", label="MAP", linewidth=linewidth, marker="*", markersize=20)
    ax.fill_between(x, vec(mean_map - std_map), vec(mean_map + std_map), alpha=0.3, color="darkolivegreen")

    # mean_sample = mean(y_sample, dims=1)
    # std_sample = std(y_sample, dims=1)
    # ax.plot(x, vec(mean_sample), color="darkorange", label="Sample", linewidth=linewidth, marker="^", markersize=20)
    # ax.fill_between(x, vec(mean_sample - std_sample), vec(mean_sample + std_sample), alpha=0.3, color="darkorange")

    mean_pc = mean(y_pc, dims=1)
    std_pc = std(y_pc, dims=1)
    ax.plot(x, vec(mean_pc), color="deeppink", label="Expected Prediction", linewidth=linewidth, marker="s", markersize=20)
    ax.fill_between(x, vec(mean_pc - std_pc), vec(mean_pc + std_pc), alpha=0.3, color="deeppink")

    ax.grid()
    ax.set_xlabel("Missing Probability")
    if ylabel
        ax.set_ylabel("RMSE")
    end
    if legend
        ax.legend(loc=2)
    end
    savefig("rmse_$(title).pdf")
    PyPlot.close()
end