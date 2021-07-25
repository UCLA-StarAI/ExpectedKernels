module Consts
export THRESHOLD, RANDMOSEED, KERNEL, TOLERANCE

# const SAMPLE_SIZE = 10
const MAX_SAMPLE = 300
const STEP = 20
const TRIES = 5
const GRID_DIM = 4
const GRID_SIZE = (3, GRID_DIM)
const BURN_IN = 2000
const INTERVAL = 1000
# const BANDWIDTH = GRID_SIZE ^ 2
const LONG_GIBBS_STEP = 1000
const THRESHOLD = 0
const RANDOMSEED = 2345
const RES_DIR = "../experiments/"
const KERNEL = "hamming"
# const KERNEL = "score"
const PYTHON = "python3"
const TOLERANCE = 1e-7
end