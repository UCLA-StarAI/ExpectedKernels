module Plot

    include("svm_pc.jl")
    using .SVMPC
    using BSON
    
    DATASET = ["abalone", "delta-ailerons", "elevators", "insurance"]
    miss_probs = [.9, .8, .7, .6, .5, .4, .3, .2, .1]

    for dataset in DATASET
        rmse_dict = BSON.load("/path/to/repo/svm_pc/output/rmse_$(dataset).bson")
        rmse_pc = rmse_dict["pc"]
        rmse_mode = rmse_dict["mode"]
        rmse_map = rmse_dict["map"]
        rmse_sample = rmse_dict["sample"]
        if dataset == "delta-ailerons"
            plot_rmse(miss_probs, rmse_pc, rmse_mode, rmse_map, rmse_sample, dataset, true, true)
        else
            plot_rmse(miss_probs, rmse_pc, rmse_mode, rmse_map, rmse_sample, dataset, false, false)
        end
    end
end