module SVMPC

    using LinearAlgebra
    using CVXOPT
    using Statistics
    using LogicCircuits
    using ProbabilisticCircuits
    using DataFrames
    using PyPlot

    include("kernels.jl")
    include("svm.jl")
    include("svr.jl")
    include("utils.jl")
    include("double_exp_rec.jl")
    include("single_exp_rec.jl")

end