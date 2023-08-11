from Keynesian_Beauty_Contest.lib.Templates.Template_CLONALG import *
from Keynesian_Beauty_Contest.lib.Templates.Template_GA import *
from Keynesian_Beauty_Contest.lib.Players.fitness_KBC import *
import pandas as pd

"""
Train a single agent (generalised by grid search)
"""

if __name__ == "__main__":
    """Training RegNet2, optimising WC with CLONALG"""
    """Max fitness = 350"""
    #"""
    memory = 2
    # optimising the win count ?
    fit_wc = True
    fitness = compatible_RegNet_fitness(memory, fit_wc)

    pop_size = 20
    ab_size = 151
    pop_shape = (ab_size, pop_size)
    bound_min = -1
    bound_max = 1
    
    select_rate = 0.5
    select_size = int(select_rate * pop_size)

    clone_rate = 0.5

    # mutation factor beta
    mut_factor = 1e-2
    # possible mutation functions : gaussian_shrink_mutation, uniform_mutation, integer_mutation
    mut_function = uniform_mutation
    # mutated values are interger or not
    integer_val = False

    # ratio of replaced population
    rep_rate = 0.15
    nb_replaced = ceil(rep_rate * pop_size)

    nb_gen = 501
    nb_runs = 1
    printer = True

    final_select, med_fitness, best_fit_vector  = CLONALG_multiple_runs(fitness, pop_shape, bound_min,
                                        bound_max, select_size, clone_rate, mut_function,
                                        mut_factor, nb_replaced, nb_gen, nb_runs, printer, integer_val)

    #"""
    #==============================================================================================

    """Training Dict-opt, optimising WC with GA"""
    """Max fitness = 350"""

    """
    # optimising the win count ?
    fit_wc = True
    fitness = compatible_dict_opt_fitness(fit_wc)
    

    pop_size = 20
    chrom_size = 101
    pop_shape = (chrom_size, pop_size)
    bound_min = 0
    bound_max = 101

    # possible selection functions : wheel_selection, tournament_selection
    selection = wheel_selection
    select_rate = 0.5
    select_size = int(select_rate* pop_size)

    # possible crossover functions : uniform_crossover, middle_point_crossover
    crossover = middle_point_crossover
    crossover_rate = 1
    nb_child = pop_size

    # possible mutation functions : GA_shrink_mutation, GA_uniform_mutation, GA_integer_mutation
    mut_function = GA_integer_mutation
    # mutated values are integer or not
    integer_val = True
    mut_rate = 0.05

    # ratio of elites who are reintroduced in population. Also ratio of replaced population
    elite_rate = 0.15
    nb_elites = ceil(elite_rate * pop_size)
    cull = ceil(elite_rate * pop_size)

    nb_gen = 501
    nb_runs = 1
    printer = True

    final_select, med_fitness, best_fit_vector  = GA_multiple_runs(fitness, pop_shape, bound_min,
                                bound_max, selection, select_size, crossover,nb_child, crossover_rate, mut_function,
                                mut_rate, nb_elites, cull, nb_gen, nb_runs, printer,integer_val)

    #"""