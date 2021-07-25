THREADS = 40
FILENAME = "rand_4_4_id0"
M = 4
N = 4
TRIES = 5
SAMPLES = 200
STEP = 10
MASK = 8
DIST = "hellinger"

println("julia -t $(THREADS) MaskGridSeq.jl --pc ../uai/$(FILENAME)/$(FILENAME).uai.psdd --vtree ../uai/$(FILENAME)/$(FILENAME).uai.vtree --grid1 $(M) --grid2 $(N) --tries $(TRIES) --max-samples $(SAMPLES) --step $(STEP) --mask $(MASK) --uai ../uai/$(FILENAME) --dist $(DIST)")