from Keynesian_Beauty_Contest.lib.Templates.Template_CLONALG import *
from Keynesian_Beauty_Contest.lib.Templates.Template_GA import *
from Keynesian_Beauty_Contest.lib.Players.fitness_KBC import *

"""
Train a single agent (generalised by grid search)
"""

if __name__ == "__main__":

    """Train RegNet2, optimising L2 with CLONALG"""
    """Max fitness = 2 800 000"""

    #"""
    memory = 2
    fit_wc = False
    fitness = compatible_RegNet_fitness(memory, fit_wc)

    ab_size = 151
    pop_size = 10
    pop_shape = (ab_size, pop_size)
    bound_min = -1
    bound_max = 1
    select_size = int(0.5 * pop_size)
    clone_rate = 0.5
    # possible mutation functions : gaussian_shrink_mutation, uniform_mutation, integer_mutation
    mut_function = uniform_mutation
    mut_factor = 2*1e-6
    nb_replaced = ceil(0.15*pop_size)
    nb_gen = 501
    printer = 1
    integer_val = False

    select_pop, fit_vector = CLONALG_run(fitness, pop_shape, bound_min, bound_max, select_size, clone_rate,
                                         mut_function,
                                         mut_factor, nb_replaced, nb_gen, printer, integer_val)

    best_config, _ = robust_eval(select_pop, fitness, nb_trials=10, print_bool=False)
    #"""
    #==============================================================================================

    """Train Dict-opt, optimising WC with GA"""
    """Max fitness = 350"""

    """
    fit_wc = True
    fitness = compatible_dict_opt_fitness(fit_wc)

    chrom_size = 101
    pop_size = 10
    pop_shape = (chrom_size, pop_size)
    bound_min = 0
    bound_max = 101
    # possible selection functions : wheel_selection, tournament_selection
    selection = wheel_selection
    select_size = int(0.25 * pop_size)
    # possible crossover functions : uniform_crossover, middle_point_crossover
    crossover = middle_point_crossover
    crossover_rate = 1
    nb_child = pop_size
    # possible mutation functions : GA_shrink_mutation, GA_uniform_mutation, GA_integer_mutation
    mut_function = GA_integer_mutation
    gen_mut_rate = 0.05
    nb_elites = ceil(0.15 * pop_size)
    cull = ceil(0.15 * pop_size)
    nb_gen = 501
    printer = 1
    integer_val = True

    children, fit_vector = GA_run(fitness, pop_shape, bound_min, bound_max, selection, select_size, crossover,
                                  nb_child,
                                  crossover_rate, mut_function, gen_mut_rate, nb_elites, cull, nb_gen, printer,
                                  integer_val)
    best_config, _ = robust_eval(select_pop, fitness, nb_trials=10, print_bool=False)
    #"""