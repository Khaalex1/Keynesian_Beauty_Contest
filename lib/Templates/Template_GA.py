"""
Template functions for the Genetic Algorithm. Paradigm : Maximisation

Inspired by "Sean Luke. Essentials of Metaheuristics. Lulu, second edition, 2013.", URL :  http://cs.gmu.edu/∼sean/book/metaheuristics/
and "Clever algorithms: nature-inspired programming recipes", URL : https://github.com/clever-algorithms/CleverAlgorithms
"""

import numpy as np
import copy
from Keynesian_Beauty_Contest.lib.Templates.Template_Benchmarks import *


# ========================================================================
# GA functions

def initialise(bound_min, bound_max, pop_shape, integer_val = False):
    """
    Uniformly initialise population
    :param bound_min: minimal bound of initialisation
    :param bound_max: maximal bound of initialisation
    :param pop_shape: population shape
    :param integer_val: if True, population is initialised with integer values. Else, real-valued
    :return: initialised population as an array of shape pop_shape
    """
    if integer_val:
        return np.random.randint(bound_min, bound_max, pop_shape)
    return np.random.uniform(bound_min, bound_max, pop_shape)


# =========================================================================
# selection functions

def wheel(A, fitness):
    """
    roulette wheel by cumsum
    :param A: All individuals
    :param fitness: fitness function
    :param func: evaluation function
    :return: Selected individual according to roulette
    """
    nb_indiv = A.shape[1]
    fit_eval = fitness(A)
    norm_fit = fit_eval / (np.sum(fit_eval) + 1e-8)
    cumul_proba = np.cumsum(norm_fit)
    cumul_proba[-1] = 1
    return cumul_proba


def wheel_selection(A, iter, fitness):
    """
    selecting fitted individuals by wheel roulette
    :param A: all individuals, array (n, nb_indiv)
    :param iter: nb individuals to select
    :param fitness: the fitness function
    :return: (sorted) array of selected individuals
    """
    cumul_proba = wheel(A, fitness)
    # finding 'iter' number between 0 and 1
    r = np.random.uniform(0, 1, iter)
    # extracting the corresponding indexes in cumul_proba
    select_idx = cumul_proba.searchsorted(r)
    # extracting population corresponding to the indexes
    selected_pop = A[:, select_idx]
    fit_eval = fitness(selected_pop)
    return selected_pop[:, (fit_eval).argsort()[::-1]]


def tournament_selection(A, iter, fitness, tourn_size=3):
    """
    :param A: all individuals, array (n, nb_indiv)
    :param iter: nb individuals to select
    :param fitness: fitness function evaluating populations
    :param tourn_size: size of a tournament
    :return: array of selected individuals
    """
    chrom_size = A.shape[0]
    nb_chrom = A.shape[1]
    winners = np.ones((chrom_size, 1))
    for i in range(iter):
        tourn_idxs = np.random.choice(nb_chrom, tourn_size, replace=False)
        competitors = A[:, tourn_idxs]
        fit_vector = fitness(competitors)
        winner = competitors[:, fit_vector.argmax()][:, None]
        winners = np.hstack((winners, winner))
    return winners[:, 1:]


# =========================================================================
# crossover functions

def uniform_crossover(x, y, crossover_rate=1):
    """
    uniform crossover
    :param x: parent x (numpy : shape = (n,1))
    :param y: parent y (numpy : shape = (n,1))
    :param crossover_rate: crossover rate <=1
    :return: crossover children
    """
    crossover_event = (np.random.uniform(0, 1) < crossover_rate)
    if crossover_event:
        nb_points = x.shape[0]
        choices = np.array([0, 1])
        weights = np.array([0.5, 0.5])
        random_factor = np.random.choice(choices, size=(nb_points, 1), p=weights)
        child1 = random_factor * x + (1 - random_factor) * y
        child2 = random_factor * y + (1 - random_factor) * x
    else:
        child1 = x
        child2 = y
    return np.hstack((child1, child2))


def middle_point_crossover(x, y, crossover_rate=1):
    """
    middle-point crossover
    :param x: parent x (numpy : shape = (n,1))
    :param y: parent y (numpy : shape = (n,1))
    :param crossover_rate: crossover rate <=1
    :return: crossover children
    """
    crossover_event = (np.random.uniform(0, 1) < crossover_rate)
    if crossover_event:
        nb_points = x.shape[0]
        genes_fact = np.zeros((nb_points, 1))
        genes_fact[nb_points // 2:] = 1
        child1 = genes_fact * x + (1 - genes_fact) * y
        child2 = genes_fact * y + (1 - genes_fact) * x
        return np.hstack((child1, child2))
    else:
        child1 = x
        child2 = y
    return np.hstack((child1, child2))


def crossing(S, fitness, crossover, nb_child=100, crossover_rate=1):
    """
    Effective crossing of selected population
    :param S: selected population to apply crossover on
    :param fitness: fitness function that evaluates individuals
    :param nb_child: nb children to produce by crossover
    :param crossover_rate: crossover rate <=1
    :return: children produced by crossover in selected population
    """
    nb_points = S.shape[0]
    pop = S.shape[1]
    A = np.zeros((nb_points, 1))
    while A.shape[1] < nb_child + 1:
        i = np.random.randint(pop)
        j = np.random.randint(pop)
        while i == j:
            j = np.random.randint(pop)
        par1 = S[:, i][:, None]
        par2 = S[:, j][:, None]
        children = crossover(par1, par2, crossover_rate=crossover_rate)
        # print(children.shape)
        A = np.hstack((A, children))
    A = A[:, 1:]
    cross_score = fitness(A)
    sorted_A = A[:, (cross_score).argsort()]
    return sorted_A


# =========================================================================
# mutation functions

#second order gaussian shrink function, allowing to parametrise the normalising factor of the Gaussian std
def GA_second_order_shrink(norm):
    """
    Second order Gaussian shrink parametrising the normalslising factor of std
    :param norm: normalslising factor of std
    :return: mutation function that behaves like gaussian_shrink_mutation without having to specify norm
    """
    def evaluate(pop, mut_rate, bound_min=0, bound_max=0):
        return GA_shrink_mutation(pop, mut_rate, bound_min, bound_max, norm)
    return evaluate


def GA_shrink_mutation(pop, mut_rate, bound_min=0, bound_max=0, norm=1):
    """
    mutation of pop by gaussian shrink
    :param pop: population of antibodies, array (ab_size, pop_size)
    :param mut_rate: mutation rate / probability
    :param bound_min: min bound of mutation
    :param bound_max: max bound of mutation
    :param norm: normalising factor of std
    :return: mutated pop
    """
    nb_points = pop.shape[0]
    nb_indiv = pop.shape[1]
    choices = np.array([0, 1])
    weights = np.array([1 - mut_rate, mut_rate])
    # Components that will be mutated
    bin_factor = np.random.choice(choices, size=pop.shape, p=weights)
    adder = np.random.normal(loc = 0, scale = np.abs(np.max(pop)-np.min(pop))/norm, size=pop.shape)
    if bound_min<bound_max:
        mutated = pop + bin_factor*adder
        exceed_min_row, exceed_min_col = np.where(mutated<bound_min)
        mutated[exceed_min_row, exceed_min_col] = pop[exceed_min_row, exceed_min_col]
        exceed_max_row, exceed_max_col = np.where(mutated > bound_max)
        mutated[exceed_max_row, exceed_max_col] = pop[exceed_max_row, exceed_max_col]
    else:
        mutated = pop + bin_factor * adder
    return mutated

def GA_uniform_mutation(pop, mut_rate, bound_min=-1, bound_max=1):
    """
    random uniformly distributed mutation within specified bounds
    :param pop: pop, array (n, nb_indiv)
    :param mut_rate: mutation rate / probability
    :param bound_min: minimal bound of mutation
    :param bound_max: maximal bound of mutation
    :return: mutated pop
    """
    nb_points = pop.shape[0]
    nb_indiv = pop.shape[1]
    choices = np.array([0, 1])
    weights = np.array([1 - mut_rate, mut_rate])
    # Components that will be mutated
    bin_factor = np.random.choice(choices, size=pop.shape, p=weights)
    random_mutations = np.random.uniform(bound_min, bound_max, pop.shape)
    mutated = (1-bin_factor)*pop + bin_factor*random_mutations
    return mutated

def GA_integer_mutation(pop, mut_rate, bound_min=0, bound_max=101):
    """
    random uniformly distributed mutation of INTEGER value within specified bounds
    :param pop: pop, array (n, nb_indiv)
    :param mut_rate: mutation rate / probability
    :param bound_min: minimal bound of mutation
    :param bound_max: maximal bound of mutation
    :return: mutated pop
    """
    nb_points = pop.shape[0]
    nb_indiv = pop.shape[1]
    choices = np.array([0, 1])
    weights = np.array([1 - mut_rate, mut_rate])
    # Components that will be mutated
    bin_factor = np.random.choice(choices, size=pop.shape, p=weights)
    random_mutations = np.random.randint(bound_min, bound_max, pop.shape)
    mutated = (1-bin_factor)*pop + bin_factor*random_mutations
    return mutated


# =========================================================================
# elitism

def elitism(prev_gen, new_gen, fitness_func, number=1):
    """
    introduce elites to new gen
    :param prev_gen: previous generation of indiv (n, nb_indiv)
    :param new_gen: actual generation on which elitism is applied
    :param fitness_func: fitness
    :param number: number of elites to transfer from previous gen to new one
    :return: new_gen with elites
    """
    fit_vector = fitness_func(prev_gen)
    best_idxs = fit_vector.argsort()[::-1][:number]
    transferable = prev_gen[:, best_idxs]
    if len(transferable.shape) < 2:
        transferable = transferable[:, None]
    updated_gen = np.hstack((new_gen, transferable))
    return updated_gen


def culling(A, fitness_func, number=1, bound_min=-1, bound_max=1, integer_val = False):
    """
    Eliminate least fitted individuals and replace them
    :param A: total pop (n, nb_indiv)
    :param fitness_func: the fitness
    :param number: number of indiv to eliminate
    :return: pop replacing weakest indiv
    """
    pop = copy.deepcopy(A)
    fit_eval = fitness_func(pop)
    pop = pop[:, fit_eval.argsort()[::-1]]
    if number == 0:
        return pop
    new_pop = pop[:, :(-number)]
    diversity = initialise(bound_min, bound_max, (pop.shape[0], number), integer_val)
    return np.hstack((new_pop, diversity))


# =============================================================================
# run functions

def GA_run(fitness, pop_shape, bound_min, bound_max, selection, select_size, crossover, nb_child,
           crossover_rate, mut_function, gen_mut_rate, nb_elites, cull, nb_gen, print_bool=True, integer_val = False):
    """
    Iterate GA during nb_gen generations
    :param fitness: fitness function
    :param pop_shape: pop shape conserved during all the run (n, nb_indiv)
    :param bound_min:
    :param bound_max:
    :param selection: selection function
    :param select_size:
    :param crossover: crossover function
    :param nb_child:
    :param crossover_rate:
    :param mut_function: mutation function
    :param gen_mut_rate: mutation rate
    :param nb_elites:
    :param cull: nb indiv to replace
    :param nb_gen:
    :param print_bool: if True, print fitness results every 10 gen.
    :param integer_val: if True, genes are integer-valued. Else, real-valued
    :return: final childen pop (n, nb_indiv) and its fitness vector
    """
    parents = initialise(bound_min, bound_max, pop_shape, integer_val)
    fit_vector = fitness(parents)
    best_fitness = fit_vector.max()
    best_index = fit_vector.argmax()
    if print_bool:
        print("Generation ", 0)
        print("Best individual = ", parents[:, best_index])
        print("Best fitness = ", best_fitness)
        print("==============================")
    for i in range(1, nb_gen + 1):
        selected_parents = selection(parents, select_size, fitness)
        selected_parents = elitism(parents, selected_parents, fitness, number=1)
        # all crossed children
        children = crossing(selected_parents, fitness, crossover, nb_child, crossover_rate)
        # mutation
        children = mut_function(children, gen_mut_rate, bound_min, bound_max)
        # elitism & culling
        children = elitism(selected_parents, children, fitness, number=nb_elites)
        children = culling(children, fitness, cull, bound_min, bound_max, integer_val)
        fit_vector = fitness(children)
        best_fitness = fit_vector.max()
        best_index = fit_vector.argmax()
        if i % 10 == 0 and print_bool:
            # print("---------------------------------------------")
            print("Generation ", i)
            print("Best individual = ", children[:, best_index])
            print("Best fitness = ", best_fitness)
            print("====================================================")
        parents = children
        # if (i % (nb_gen // 4) == 2 and self.fitness.value(parents).min() > thresh_reset):
        #    parents = np.random.uniform(3 * self.bound_min / 4, 3 * self.bound_max / 4, parents.shape)
    print("Final best fitness = ", best_fitness)
    print("Final best individual = ", children[:, best_index])
    print("====================================================")
    return children, fit_vector


def GA_multiple_runs(fitness, pop_shape, bound_min, bound_max, selection, select_size, crossover, nb_child,
           crossover_rate, mut_function, gen_mut_rate, nb_elites, cull, nb_gen, nb_runs=3, print_bool=False, integer_val = False):
    """
    Compute fitness convergence statistics on multiple runs
    :param fitness: fitness function evaluating pop
    :param pop_shape: int tuple
    :param bound_min: float
    :param bound_max: float
    :param selection: selection function
    :param select_size: int
    :param crossover: cross. function
    :param nb_child: int < pop_shape[1]
    :param crossover_rate: float <1
    :param mut_function: mutation function
    :param gen_mut_rate: float <1
    :param nb_elites: int < pop_shape[1]
    :param cull: nb replaced, int < pop_shape[1]
    :param nb_gen: int
    :param nb_runs: int (preferably odd)
    :param print_bool: bool
    :param integer_val: bool
    :return: pop array of median fitness, median fitness vector, fitness vector of each run best individual
    """
    if (nb_runs%2) == 0:
        nb_runs += 1
    total_select = np.zeros((3, pop_shape[0], pop_shape[1] + nb_elites))
    total_fit = np.zeros((3, pop_shape[1]+ nb_elites))
    for i in range(nb_runs):
        select_pop, fit_vector = GA_run(fitness, pop_shape, bound_min, bound_max, selection, select_size, crossover, nb_child,
           crossover_rate, mut_function, gen_mut_rate, nb_elites, cull, nb_gen, print_bool, integer_val)
        fit_sorted_index = np.argsort(fit_vector)[::-1]
        total_fit[i, :] = np.sort(fit_vector)[::-1]
        total_select[i, :, :] = select_pop[:, fit_sorted_index]

    med_fitness = np.median(total_fit, axis = 0)
    med_index = np.where(total_fit == med_fitness)

    unique_col_index, ind = np.unique(med_index[1], return_index=True)
    row_index = med_index[0][ind]

    final_select = total_select[row_index, :, unique_col_index].T

    #final_select = total_select[med_index[0], :, med_index[1]].T
    # sort array
    final_select = final_select[:, np.argsort(med_fitness)[::-1]]
    med_fitness = np.sort(med_fitness)[::-1]

    #stats
    avg_f = np.mean(total_fit[:, 0])
    std_f = np.std(total_fit[:, 0])
    min_f = np.min(total_fit[:, 0])
    max_f = np.max(total_fit[:, 0])
    print("----------------------------------------------------")
    print("Fitness Avg ({} runs) = {}".format(nb_runs, avg_f))
    print("Fitness Std ({} runs) = {}".format(nb_runs, std_f))
    print("Fitness Min ({} runs) = {}".format(nb_runs, min_f))
    print("Fitness Max ({} runs) = {}".format(nb_runs, max_f))
    print("----------------------------------------------------")
    return final_select, med_fitness, total_fit[:, 0]
