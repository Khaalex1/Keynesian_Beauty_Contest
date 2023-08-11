from Keynesian_Beauty_Contest.lib.Templates.Template_CLONALG import *
from Keynesian_Beauty_Contest.lib.Templates.Template_GA import *

"""
Testing CLONALG and/or GA on benchmark functions 
in Keynesian_Beauty_Contest.lib.Templates.Template_Benchmarks
Refer to this file to have the ground truth argmin.
 
NOTE : Paradigm is always maximisation. A fitness of 100 000 may be 
equivalent to a min value of 0
"""

if __name__ == "__main__":

    """RUNNING CLONALG (SHIFTED SPHERE BENCHMARK)"""
    #"""
    #possible fitness functions : sphere, shifted_sphere, double_sum, rastrigin, rosenbrock, griewank, schwefel
    #refer to Keynesian_Beauty_Contest.lib.Templates.Template_Benchmarks for ground truth solutions
    fitness = shifted_sphere
    #to maximise fitness, takes the opposite
    fitness = opposite_fitness(fitness)
    # to make opposite fitness positive
    constant = 1e5
    fitness = adder(fitness, constant)

    ab_size = 10
    pop_size = 40
    pop_shape = (ab_size, pop_size)
    bound_min = -100
    bound_max = 100
    select_size = 10
    clone_rate = 0.25
    #possible mutation functions : gaussian_shrink_mutation, uniform_mutation, integer_mutation
    mut_function = uniform_mutation
    mut_factor = 2.5e-5
    nb_replaced = 3
    nb_gen = 1001
    nb_runs = 1
    print_bool = True
    integer_val = False

    print("=========================================")
    print("CLONALG RUN (RASTRIGIN BENCHMARK)")
    select_pop, fit_vector, _ = CLONALG_multiple_runs(fitness, pop_shape, bound_min, bound_max, select_size, clone_rate,
                                                      mut_function,
                                                      mut_factor, nb_replaced, nb_gen, nb_runs, print_bool, integer_val)
    #"""


    """RUNNING GA"""

    #"""
    # possible fitness functions : sphere, double_sum, rastrigin, rosenbrock, griewank, schwefel
    #refer to Keynesian_Beauty_Contest.lib.Templates.Template_Benchmarks for ground truth solutions
    fitness = rastrigin
    # but no need to make opposite_fitness positive for GA
    # to maximise fitness
    fitness = opposite_fitness(fitness)

    chrom_size = 10
    pop_size = 20
    pop_shape = (chrom_size, pop_size)
    bound_min = -100
    bound_max = 100
    # possible selection functions : wheel_selection, naive_selection, random_selection
    selection = tournament_selection
    select_size = 10
    # possible crossover functions : uniform_crossover, k_points_crossover, blx_alpha_crossover, sbx_alpha_crossover
    crossover = middle_point_crossover
    crossover_rate = 1
    nb_child = pop_size
    # possible mutation functions : GA_shrink_mutation, GA_uniform_mutation, GA_integer_mutation
    norm = 3
    mut_function = GA_second_order_shrink(norm)
    gen_mut_rate = 0.05
    nb_elites = 3
    #nb replaced
    cull = 2
    nb_gen = 1001
    nb_runs = 1
    print_bool = True
    integer_val = False

    print("=========================================")
    print("GA RUN")

    select_pop, fit_vector, _ = GA_multiple_runs(fitness, pop_shape, bound_min, bound_max, selection, select_size, crossover, nb_child,
                                   crossover_rate, mut_function, gen_mut_rate, nb_elites, cull, nb_gen, nb_runs, print_bool,integer_val)

    #"""
