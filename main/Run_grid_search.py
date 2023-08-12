from Keynesian_Beauty_Contest.lib.Grid_search.grid_search_CLONALG import *
from Keynesian_Beauty_Contest.lib.Grid_search.grid_search_GA import *


"""
WARNING : RUNNING THIS FILE TAKES DAYS TO TERMINATE (DURATION DEPENDING ON HARDWARE)

Note that it is also possible to test only ONE set of hyperparameters and save it in a file e.g. "filename.csv".
TO DO SO : for example, testing CLONALG with pop_size = 40, sel_rate= 0.5, clone_rate = 0.25, beta_mag = 2, rep_rate = 0.05
-> comb = [[40], [0.5], [0.25], [2], 0.05]]
-> CLONALG_grid_search(comb, ... filename = "filename.csv")
"""

if __name__ == "__main__":

    """CLONALG GRID SEARCH"""
    #"""
    pop_sizes = [10, 20, 40]
    sel_rates = [0.25, 0.5]
    crs = [0.5, 0.25]
    beta_mag = [0.5, 1, 2]
    rep_rates = [0.05, 0.15]
    # regrouping into one list of lists
    all_lists = [pop_sizes, sel_rates, crs, beta_mag, rep_rates]
    #all possible combinations of hyperparam
    comb = list(itertools.product(*all_lists))

    # Grid search for RegNet1 optimising L2 Loss with AIS/CLONALG
    memory= 1
    fit_wc = False
    layers = [memory, 10, 10, 1]
    agent = memory_RegNet(layers)
    fitness = compatible_RegNet_fitness(memory, fit_wc)
    bound_min = -1
    bound_max = 1
    integer_val = False
    filename = "test_AIS_L2_m1.csv"
    df = CLONALG_grid_search(comb, fitness, agent, bound_min, bound_max, memory, integer_val, fit_wc, filename)
    #"""

    #====================================================================================================================
    """GA GRID SEARCH"""
    #"""
    pop_sizes = [10, 20, 40]
    select_rates = [0.25, 0.5]
    #cross_type 0 is Uniform Crossover, 1 is Middle-point Crossover
    cross_types = [0, 1]
    mut_rates = [0.05, 0.15]
    elite_rates = [0.05, 0.15]
    
    #gather all lists of hyperpar. into one list of list
    all_lists = [pop_sizes, select_rates, cross_types, mut_rates, elite_rates]
    #all possible combinations of hyperparam
    comb = list(itertools.product(*all_lists))

    # Grid search for dict-opt optimising Win Count with GA
    memory = 1
    fit_wc = True
    agent = dict_opt_agent
    fitness = compatible_dict_opt_fitness(fit_wc)
    bound_min = 0
    bound_max = 101
    integer_val = True
    filename = "test_GA_WC.csv"
    df = GA_grid_search(comb, fitness, agent, bound_min, bound_max, memory, integer_val, fit_wc, filename)
    #"""