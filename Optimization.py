# Import necessary libraries
import numpy as np
from numpy.random import randint, rand
import matplotlib.pyplot as plt

# From Slide 11, step 1 is to Choose size of population, prob of mutation and crossover_rate
# Parameters of the genetic algorithm
bounds = [[-3, 3], [-3, 3]]
iteration = 500 
bits = 20 # number of bits for each variable
pop_size = 100 # size of the population
crossover_rate = 0.8 # probability of crossover
mutation_rate = 0.1 # probability of mutation

# Step 2: Define the fitness function
def objective_function(I): 
    x = I[0]
    y = I[1]
    return (1 - x)**2 * np.exp(-x**2 - (y + 1)**2)-(x - x**3 - y**3) * np.exp(-x**2 - y**2)

#-----------------------------------------------------------------------------
# Now start the genetic algorithm optimization 
#-----------------------------------------------------------------------------

# Step 3: Randomly initialize the population
pop = [randint(0, 2, bits * len(bounds)).tolist() for _ in range(pop_size)] 

# Cross over operator
# This function performs crossover between pairs of pa	rents in the population
def crossover(pop, crossover_rate):
	offspring = list()
	for i in range(int(len(pop)/2)):
		# select parents
		p1 = pop[2*i-1].copy() # parent 1
		p2 = pop[2*i].copy() # parent 2
		# ensure parents are different
		# check for crossover
		if rand() < crossover_rate:
			# select crossover point
			cp = randint(1, len(p1)-1, size=2) # two random cutting points
			while cp[0] == cp[1]: # ensure they are different
				cp = randint(1, len(p1)-1, size=2)   # two random cutting points
			cp = sorted(cp) # sort the cutting points
			
			c1 = p1[:cp[0]] + p2[cp[0]:cp[1]] + p1[cp[1]:]
			c2 = p2[:cp[0]] + p1[cp[0]:cp[1]] + p2[cp[1]:]
			offspring.append(c1)
			offspring.append(c2)
		else:
			# no crossover, just copy parents
			offspring.append(p1)
			offspring.append(p2)

	return offspring

# This function performs mutation on the population
# It randomly flips bits in the chromosomes based on the mutation rate 
def mutation(pop, mutation_rate):
	offspring = list()
	for i in range(int(len(pop))):
		p1 = pop[i].copy()
		if rand() < mutation_rate:
			cp = randint(0, len(p1)) #random gene
			c1 = p1
			if c1[cp] == 1:
				c1[cp] = 0  # flip the bit
			else:
				c1[cp] = 1
				
			offspring.append(c1)
		else:
			offspring.append(p1)
	return offspring

# This function decodes the binary chromosome to real values based on the bounds
# It converts each segment of the binary string to an integer and then maps it to the corresponding real value
def decoding(bounds, bits, chromosome):
	real_chromosome = list()
	for i in range(len(bounds)):
		start, end = i * bits, (i * bits) + bits  # extract the chromosome
		sub = chromosome[start:end]  # extract the sub-chromosome
		chars = ''.join([str(s) for s in sub])  # convert to string
		integer = int(chars, 2)  # convert to integer
		real_value = bounds[i][0] + (integer / (2 ** bits)) * (bounds[i][1] - bounds[i][0])  # convert to real value
		real_chromosome.append(real_value)  # add to the real chromosome
	return real_chromosome

# roulette wheel selection
def selection(pop, fitness, pop_size):
	# Convert fitness to positive values for roulette wheel
	min_f = min(fitness)
	adjusted_fitness = [f - min_f + 1e-8 for f in fitness]  # Shift to positive

	#(best solution)
	next_generation = []
	elite = np.argmax(fitness)  # index of the best candidate
	next_generation.append(pop[elite])  # add the best candidate to the next generation

 	# Roulette wheel probabilities
	p = [f / sum(adjusted_fitness) for f in adjusted_fitness]  # probabilities of selection

	# Select new population (elite + roulette selections)
	index = list(range(len(pop)))
	index_selected = np.random.choice(index, size=pop_size - 1, replace=False, p=p)
	s = 0
	for j in range(pop_size - 1):
		next_generation.append(pop[index_selected[s]])  # add the selected candidates to the next generation
		s += 1
	return next_generation

# main program
best_fitness = []
for gen in range(iteration):
	# calculate the fitness of the population
	offspring = crossover(pop, crossover_rate) # perform crossover
	offspring = mutation(offspring, mutation_rate) # perform mutation
	# offspring is a list of new chromosomes created from the population
	
	for s in offspring:
		pop.append(s)  # combine the population with the offspring
		
	real_chromosomes = [decoding(bounds, bits, p) for p in pop]  # decode the chromosomes
	fitness = [objective_function(d) for d in real_chromosomes]  # calculate the fitness of the population

	index = np.argmax(fitness)  # index of the best candidate
	current_best = pop[index]  # best candidate
	best_fitness.append(max(fitness))  # store the best fitness of the generation
	pop = selection(pop, fitness, pop_size) # select the next generation based on fitness

# Plotting the results
fig = plt.figure()
fig.suptitle('Genetic Algorithm Optimization')
plt.plot(best_fitness, label='Best Fitness')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.legend()
plt.show()

# Output the results
print('Maximum objective function value: ', max(best_fitness))
print('Optimal solution: ', decoding(bounds, bits, current_best))