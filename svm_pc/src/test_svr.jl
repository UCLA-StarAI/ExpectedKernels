module TestSVR
    include("svm_pc.jl")
    using .SVMPC

    using RDatasets
    using CSV
    using DataFrames
    using LogicCircuits
    using ProbabilisticCircuits
    using Impute: Substitute, impute, SRS
    using StatsBase
    using Statistics
    using JSON
    using BSON
    using ThreadPools

    DATASET = ["delta-ailerons", "abalone", "insurance", "elevators"]
    # DATASET = ["elevators"]

    for dataset in DATASET
        X_train = Matrix(DataFrame(CSV.File("/path/to/repo/svm_pc/data/$(dataset)/$(dataset)_train_x.csv")))
        y_train = vec(Matrix(DataFrame(CSV.File("/path/to/repo/svm_pc/data/$(dataset)/$(dataset)_train_y.csv"))))
        X_valid = Matrix(DataFrame(CSV.File("/path/to/repo/svm_pc/data/$(dataset)/$(dataset)_valid_x.csv")))
        y_valid = vec(Matrix(DataFrame(CSV.File("/path/to/repo/svm_pc/data/$(dataset)/$(dataset)_valid_y.csv"))))

        # X_train = X_train[1 : 5000, :]
        # y_train = y_train[1 : 5000]
        # X_test = X_test[1 : 10, :]
        # y_test = y_test[1 : 10]

        pc_data = DataFrame(BitArray(X_train))
        # display(pc_data)
        pc = learn_circuits(pc_data)
        # pc = load_prob_circuit(zoo_psdd_file("insurance.psdd"))

        dict = JSON.parsefile("/path/to/repo/svm_pc/params/$(dataset)")
        # display(dict)

        c = dict["C"]
        epsilon = dict["epsilon"]
        gamma = dict["gamma"]
        coef = Vector(Float64.(dict["coef"][1]))
        b = Float64(dict["b"][1])
        sv = hcat(dict["sv"]...)'

        # display(c)
        # display(epsilon)
        # display(gamma)
        # display(coef)
        # display(b)
        # display(sv)
        set_gamma(gamma)
        svr = SVR(gaussian, b, coef, sv, coef, c, epsilon)
        
        y_predict = evaluate(svr, X_valid)
        display(y_predict[1:10])
        rmse = rmsd(y_predict, Float64.(y_valid))
        println("RMSE(fully observed): $rmse")

        keep_probs = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
        rmse_pcs = zeros(5, 9)
        rmse_modes = zeros(5, 9)
        rmse_means = zeros(5, 9)
        rmse_samples = zeros(5, 9)
        rmse_maps = zeros(5, 9)

        tuples = []
        for t = 1 : 5
            for i = 1 : 9
                push!(tuples, (t => i))
            end
        end

        @qbthreads for (t, i) in tuples
            X_test_missing = make_missing_mcar(DataFrame(X_valid); keep_prob=keep_probs[i])
            y_predict_pc = evaluate_pc(svr, X_test_missing, deepcopy(pc))
            rmse_pc = rmsd(y_predict_pc, Float64.(y_valid))
            rmse_pcs[t, i] = rmse_pc
            println("RMSE(pc): $rmse_pc")

            X_test_mode = Matrix(X_test_missing)
            Union
            X_test_mode = impute(X_test_mode, Substitute(; statistic=mode); dims=:cols)
            y_predict_mode = evaluate(svr, X_test_mode)
            rmse_mode = rmsd(y_predict_mode, Float64.(y_valid))
            rmse_modes[t, i] = rmse_mode
            println("RMSE(mode): $rmse_mode")

            X_test_missing_bit = make_missing_mcar(DataFrame(BitArray(X_valid)); keep_prob=keep_probs[i])
            multiple_size = 5
            X_test_sample, _ = ProbabilisticCircuits.sample(deepcopy(pc), multiple_size, X_test_missing_bit)
            X_test_sample = reshape(X_test_sample, (size(X_test_sample, 1) * size(X_test_sample, 2), size(X_test_sample, 3)))
            # display(X_test_sample)
            # @assert false
            # X_test_sample = X_test_sample[1, :, :]
            # y_predict_sample = evaluate(svr, X_test_sample)
            y_predict_sample_multiple = fast_evaluate(svr, X_test_sample, gamma)
            y_predict_sample = zeros(size(y_valid, 1))
            for i = 1 : size(y_valid, 1)
                y_predict_sample[i] = mean(y_predict_sample_multiple[(i - 1) * multiple_size + 1 : i * multiple_size])
            end
            # display(y_predict_sample)
            # display(y_valid)
            rmse_sample = rmsd(y_predict_sample, Float64.(y_valid))
            rmse_samples[t, i] = rmse_sample
            println("RMSE(sample): $rmse_sample")

            X_test_missing_bit = make_missing_mcar(DataFrame(BitArray(X_valid)); keep_prob=keep_probs[i])
            X_test_map, _ = MAP(deepcopy(pc), X_test_missing_bit)
            y_predict_map = fast_evaluate(svr, Matrix(X_test_map), gamma)
            rmse_map = rmsd(y_predict_map, Float64.(y_valid))
            rmse_maps[t, i] = rmse_map
            println("RMSE(map): $rmse_map")
        end
        
        rmse_dict = Dict("pc" => rmse_pcs, "mode" => rmse_modes, "map" => rmse_maps, "sample" => rmse_samples)
        bson("/path/to/repo/svm_pc/output/rmse_$(dataset).bson", rmse_dict)
        
        # plot_rmse(keep_probs, rmse_pcs, rmse_modes, dataset)

    end
    
end



