"""
Template functions for the Clonal Selection Algorithm (CLONALG). Paradigm : Maximisation

Inspired by "L. N. De Castro and F. J. Von Zuben. Learning and optimization using the clonal
selection principle"
URL : https://dx.doi.org/10.1109/tevc.2002.1011539
"""


import numpy as np
from Keynesian_Beauty_Contest.lib.Templates.Template_Benchmarks import *


#========================================================================
# Clonal Selection functions

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


def select(pop, fitness, select_size):
    """
    Select best antibodies based on fitness
    :param pop: population of antibodies
    :param fitness: fitness function
    :param select_size: number of individuals to select
    :return: selected population (by descending order) and sorted fitness vector
    """
    fit = fitness(pop)
    best_indices = np.argsort(fit)[::-1][:select_size]
    return pop[:, best_indices], np.sort(fit)[::-1][:select_size]


def clone(pop, clone_rate):
    """
    Clone population
    :param pop: 2D array (ab_size, pop_size) of antibodies
    :param clone_rate: clone rate < 1
    :return: population of clones (ab_size, Nc*pop_size)
    """
    N = pop.shape[1]
    Nc = round(clone_rate*N)
    return np.repeat(pop, N*[Nc], axis=1)


def hypermutate(pop, fitness, mut_function, mut_factor, bound_min=0, bound_max=0):
    """
    hypermutation of clone population
    :param pop: D array (ab_size, pop_size) of antibodies
    :param fitness: fitness function that evaluates pop
    :param mut_function: function of mutation type
    :param bound_min: minimal bound of mutation
    :param bound_max: maximal bound of mutation
    :param mut_factor: mutation factor beta
    :return: Hypermutated array of same shape as pop
    """
    fit = fitness(pop)

    #attributing proba of mutation for each individual (high when 'bad' fitness, low when 'good' fitness)
    mut_prob_array = np.exp(-abs(mut_factor)*fit)
    mut_prob_weights = np.tile(mut_prob_array, (pop.shape[0],1))

    # mutating concerned individuals
    mut_pop = mut_function(pop, mut_prob_weights, bound_min, bound_max)
    return mut_pop, mut_prob_array


"""
Mutation functions
"""

def second_order_shrink(norm):
    """
    Second order Gaussian shrink parametrising the normalslising factor of std
    :param norm: normalslising factor of std
    :return: mutation function that behaves like gaussian_shrink_mutation without having to specify norm
    """
    def evaluate(pop, mut_rate, bound_min=0, bound_max=0):
        return gaussian_shrink_mutation(pop, mut_rate, bound_min, bound_max, norm)
    return evaluate

def gaussian_shrink_mutation(pop, mut_prob_weights = 0, bound_min=0, bound_max=0, norm=1):
    """
    Mutation of pop by gaussian shrink
    :param pop: population of antibodies, array (ab_size, pop_size)
    :param mut_prob_weights: mutation rate of each coordinate of any antibody
    :param bound_min: min bound of mutation
    :param bound_max: max bound of mutation
    :param norm: normalising factor of std
    :return: mutated pop of same shape as pop
    """
    nb_points = pop.shape[0]
    nb_indiv = pop.shape[1]
    weights = mut_prob_weights.flatten()
    # Components that will be mutated
    bin_factor = np.array([np.random.choice([0, 1], 1, p=[1 - w, w])[0] for w in weights]).reshape(pop.shape)
    adder = np.random.normal(loc = 0, scale = np.abs(np.max(pop)-np.min(pop))/norm, size=pop.shape)
    #verify if antibodies are within bounds
    if bound_min<bound_max:
        mutated = pop + bin_factor*adder
        exceed_min_row, exceed_min_col = np.where(mutated<bound_min)
        mutated[exceed_min_row, exceed_min_col] = pop[exceed_min_row, exceed_min_col]
        exceed_max_row, exceed_max_col = np.where(mutated > bound_max)
        mutated[exceed_max_row, exceed_max_col] = pop[exceed_max_row, exceed_max_col]
    else:
        mutated = pop + bin_factor * adder
    return mutated

def uniform_mutation(pop, mut_prob_weights = 0, bound_min=-1, bound_max=1):
    """
    random uniformly distributed mutation within specified bounds
    :param pop: population array, array (ab_size, pop_size)
    :param mut_prob_weights: probability weights for each component (coordinate)
    :param bound_min: minimal bound of mutation
    :param bound_max: maximal bound of mutation
    :return: mutated pop of same shape as pop
    """
    nb_points = pop.shape[0]
    nb_indiv = pop.shape[1]
    weights = mut_prob_weights.flatten()
    # Components that will be mutated
    bin_factor = np.array([np.random.choice([0, 1], 1, p=[1 - w, w])[0] for w in weights]).reshape(pop.shape)
    random_mutations = np.random.uniform(bound_min, bound_max, pop.shape)
    mutated = (1-bin_factor)*pop + bin_factor*random_mutations
    return mutated

def integer_mutation(pop, mut_prob_weights = 0, bound_min=0, bound_max=101):
    """
    Random uniformly distributed mutation on integer range
    :param pop: population array, array (ab_size, pop_size)
    :param mut_prob_weights: probability weights for each component (coordinate)
    :param bound_min: minimal bound of mutation
    :param bound_max: maximal bound of mutation
    :return: mutated pop of same shape as pop
    """
    nb_points = pop.shape[0]
    nb_indiv = pop.shape[1]
    weights = mut_prob_weights.flatten()
    # Components that will be mutated
    bin_factor = np.array([np.random.choice([0, 1], 1, p=[1 - w, w])[0] for w in weights]).reshape(pop.shape)
    random_mutations = np.random.randint(bound_min, bound_max, pop.shape)
    mutated = (1-bin_factor)*pop + bin_factor*random_mutations
    return mutated


def replace(ord_pop, nb_replaced, bound_min, bound_max, integer_val = False):
    """
    Replace less fitted individuals by random ones
    :param ord_pop: pop (array (ab_size, pop_size)), sorted by descending fitness
    :param nb_replaced: nb of indiv to be replaced
    :param bound_min: minimal bound of incoming individuals
    :param bound_max: maximal bound of incoming individuals
    :return: pop with replaced individuals
    """
    ord_pop[:, -nb_replaced:] = initialise(bound_min, bound_max, (ord_pop.shape[0], nb_replaced), integer_val)
    return ord_pop


#=========================================================================================================================================
#Run functions

def CLONALG_run(fitness, pop_shape, bound_min, bound_max, select_size, clone_rate, mut_function,
        mut_factor, nb_replaced, nb_gen, print_bool = True, integer_val=False):
    """
    run CSA algorithm during nb_gen generations
    :param fitness: fitness function to evaluate populations
    :param pop_shape: shape of population
    :param bound_min: min bound of population (and mutation)
    :param bound_max: max bound of population (and mutation)
    :param select_size: nb antibodies selected
    :param clone_rate: clone rate <1
    :param mut_function: function of mutation type
    :param mut_factor: mutation factor beta
    :param nb_replaced: nb of individuals to be replaced
    :param nb_gen: nb of gen during which CLONALG is run
    :param print_bool: print fitness results every 10 gen.
    :param integer_val: boolean to indicate if pop values are integer or not
    :return: resulting pop and its fitness vector
    """
    pop = initialise(bound_min, bound_max, pop_shape, integer_val)
    select_pop, fit_vector = select(pop, fitness, select_size)
    for i in range(0, nb_gen):
        if (i%10==0 and print_bool == True):
            print("Generation ", i)
            print("Best individual = ", select_pop[:, 0])
            print("Best fitness = ", fit_vector[0])
            print("====================================================")
        clones = clone(select_pop, clone_rate)
        clones, mut_prob_array = hypermutate(clones, fitness, mut_function, mut_factor, bound_min, bound_max)
        pop, _ = select(np.hstack((pop, clones)), fitness, pop.shape[1])
        pop = replace(pop, nb_replaced, bound_min, bound_max, integer_val)
        select_pop, fit_vector = select(pop, fitness, select_size)
    print("Final best fitness = ", fit_vector[0])
    print("Final best individual = ", select_pop[:, 0])
    print("====================================================")
    return select_pop, fit_vector


def CLONALG_multiple_runs(fitness, pop_shape, bound_min, bound_max, select_size, clone_rate, mut_function,
        mut_factor, nb_replaced, nb_gen, nb_runs =3, print_bool = False, integer_val=False):
    """
    Compute fitness convergence statistics on multiple runs
    :param fitness: fitness function evaluating pop
    :param pop_shape: population shape, int tuple
    :param bound_min: float
    :param bound_max: float
    :param select_size: int
    :param clone_rate: float<1
    :param mut_function: mutation type function
    :param mut_factor: float
    :param nb_replaced: int
    :param nb_gen: int
    :param nb_runs: int (odd preferaby)
    :param print_bool: bool
    :param integer_val: bool
    :return: pop array of median fitness, median fitness vector, list of stats (mean, std, min, max fitness)
    """
    if (nb_runs%2) == 0:
        nb_runs += 1
    total_select = np.zeros((3, pop_shape[0], select_size))
    total_fit = np.zeros((3, select_size))
    for i in range(nb_runs):
        select_pop, fit_vector = CLONALG_run(fitness, pop_shape, bound_min, bound_max, select_size, clone_rate, mut_function,
        mut_factor, nb_replaced, nb_gen, print_bool , integer_val)
        total_select[i, :, :] = select_pop
        total_fit[i, :] = fit_vector
    med_fitness = np.median(total_fit, axis = 0)
    med_index = np.where(total_fit == med_fitness)

    unique_col_index, ind = np.unique(med_index[1], return_index = True)
    row_index = med_index[0][ind]

    final_select = total_select[row_index,:,  unique_col_index].T
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
    return final_select, med_fitness, [avg_f, std_f, min_f, max_f]















