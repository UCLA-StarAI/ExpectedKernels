TRIES = 1
GRID1 = 8
GRID2 = 8
DIR = "../uai/"
POT="RANDOM"

if POT == "RANDOM":
        PREFIX = "rand"
elif POT == "RANDOM_FRUSTRATED":
        PREFIX = "frus"
else:
        raise NotImplementedError

for i in range(TRIES):
        print(f"python grid.py -m{GRID1} -n{GRID2} --pot={POT} {DIR}{PREFIX}_{GRID1}_{GRID2}_id{i}.uai")
