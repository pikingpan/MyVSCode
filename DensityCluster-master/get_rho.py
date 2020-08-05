import numpy as np
import math


class GET_RHO(object):
    __nfunc_ = -1
    __rho_ = [0.01, 0.01, 0.01, 0.01, 0.5, 0.5, 0.2, 0.5, 0.2, 0.01, 
			0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 ]
    __nopt_ = [2, 5, 1, 4, 2, 18, 36, 81, 216, 12, 6, 8, 6, 6, 8, 6, 8, 6, 8, 8 ]
    __maxfes_ = [50000, 50000, 50000, 50000, 50000, 200000, 200000, 400000, 400000, 200000, 
			200000, 200000, 200000, 400000, 400000, 400000, 400000, 400000, 400000, 400000 ]
    __dimensions_ = [1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 5, 5, 10, 10, 20]

    def __init__(self, nofunc):
        assert (nofunc > 0 and nofunc <= 20)
        self.__nfunc_ = nofunc

    def get_rho(self):
        return self.__rho_[self.__nfunc_-1]   

def how_many_goptima(pop, fits, accuracy):
    # pop: NP, D
	NP, D = pop.shape[0], pop.shape[1]
	# Evaluate population
	# Descenting sorting
	order = np.argsort(fits)[::-1]
	print(order)
	# Sort population based on its fitness values
	sorted_pop = pop[order,:]
	spopfits = fits[order]	
	# find seeds in the temp population (indices!)
	seeds_idx = find_seeds_indices(sorted_pop, fits.get_rho() )	
	count = 0
	goidx = []
	for idx in seeds_idx:
		# evaluate seed
		seed_fitness = spopfits[idx] #f.evaluate(sorted_pop[idx])
		# |F_seed - F_goptimum| <= accuracy
		if math.fabs( seed_fitness - fits.get_fitness_goptima() ) <= accuracy:
			count = count + 1
			goidx.append(idx)
		# save time
		if count == fits.get_no_goptima():
			break
	# gather seeds
	seeds = sorted_pop[goidx]
	return count, seeds

def find_seeds_indices(sorted_pop, radius):
    seeds = []
    seeds_idx = []
    # Determine the species seeds: iterate through sorted population 
    for i, x in enumerate(sorted_pop):
        found = False 
		# Iterate seeds
        for j, sx in enumerate(seeds):
			# Calculate distance from seeds
            dist = math.sqrt( sum( (x - sx)**2 ) )

			# If the Euclidean distance is less than the radius
            if dist <= radius:
                found = True
                break
        if not found:
            seeds.append(x)
            seeds_idx.append(i)
    return seeds_idx   