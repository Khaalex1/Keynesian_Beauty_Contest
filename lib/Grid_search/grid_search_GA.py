from Keynesian_Beauty_Contest.lib.Players.fitness_KBC import *
from Keynesian_Beauty_Contest.lib.Templates.Template_Benchmarks import *
from Keynesian_Beauty_Contest.lib.Templates.Template_GA import *
from math import *
import pandas as pd



"""
Note that it is also possible to test only ONE set of hyperparameters and save it in a file e.g. "filename.csv".
TO DO SO : for example, testing GA with pop_size = 40, sel_rate= 0.5, cross_type = 0, mut_rate = 0.05, elite_rate = 0.05
-> comb = [[40], [0.5], [0], [0.05], 0.05]]
-> GA_grid_search(comb, ... filename = "filename.csv")
"""

def GA_grid_search(all_comb, fitness, agent, bound_min=-1, bound_max=1, memory=1, integer_val=False, fit_wc=False,
                        filename='test.csv'):
    """
    Run GA grid search
    :param all_comb: combinated list of all hyperparameter sub lists as [pop_sizes, sel_rates, cross_types, mut_rates, elite_rates]
    :param fitness: function computing indiv/pop fitness
    :param agent: agent function
    :param bound_min: float
    :param bound_max: float
    :param memory: number of memories of agent, 1 or 2
    :param integer_val: if True, individuals are integer-valued. Else, real-valued
    :param fit_wc: if True, fitness = Win Count so the inverse magnitude is 2e-2.
    inverse magnitude is 1e-6
    :param filename: name of file (csv) where to store training results
    :return: panda dataframe of results
    """
    chrom_size = 141 if memory == 1 else 151
    chrom_size = 101 if integer_val else chrom_size
    mut_function = GA_integer_mutation if integer_val else GA_uniform_mutation

    all_levels = list(itertools.combinations_with_replacement([0, 1, 2, 3], 4))
    all_data = []
    avg_wc_list = []
    min_wc_list = []
    max_wc_list = []
    std_wc_list = []
    avg_loss_list = []
    min_loss_list = []
    max_loss_list = []
    std_loss_list = []
    best_indiv_list = []
    i = 0
    for comb in all_comb:
        print(i, comb)
        pop_size, select_rate, cross_type, mut_rate, elite_rate = comb[0], comb[1], comb[2], comb[3], comb[4]
        select_size = ceil(select_rate * pop_size)
        nb_elites = ceil(elite_rate * pop_size)
        cull = nb_elites
        pop_shape = (chrom_size, pop_size)
        select_function = wheel_selection
        # 0 is uniform cross., 1 is middle-point
        crossover = uniform_crossover if cross_type == 0 else middle_point_crossover
        select_pop, fit_vector, _ = GA_multiple_runs(fitness, pop_shape, bound_min, bound_max, select_function, select_size,
                                                     crossover, pop_size, 1, mut_function, mut_rate, nb_elites,
                                                     cull, nb_gen=101, nb_runs=3, print_bool=False,integer_val= integer_val)
        #best_config, fit_stats = robust_eval(select_pop, fitness, nb_trials=10, print_bool=False)
        max_index = np.argmax(fit_vector)
        best_config = select_pop[:, max_index]
        list_wc, list_loss = evaluation(agent, best_config, begin_measure=memory, print_bool=False, random_crowds=False)
        avg_wc, std_wc, max_wc, min_wc = np.mean(list_wc), np.std(list_wc), np.max(list_wc), np.min(list_wc)
        avg_loss, std_loss, max_loss, min_loss = np.mean(list_loss), np.std(list_loss), np.max(list_loss), np.min(
            list_loss)
        avg_wc_list.append(avg_wc)
        min_wc_list.append(min_wc)
        max_wc_list.append(max_wc)
        std_wc_list.append(std_wc)
        avg_loss_list.append(avg_loss)
        min_loss_list.append(min_loss)
        max_loss_list.append(max_loss)
        std_loss_list.append(std_loss)
        best_indiv_list.append(best_config)
        i += 1
    df = pd.DataFrame({'Hyperparameters': all_comb,
                       'Avg Win Count': avg_wc_list,
                       'Min Win Count': min_wc_list,
                       'Max Win Count': max_wc_list,
                       'Std Win Count': std_wc_list,
                       'Avg L2 Loss': avg_loss_list,
                       'Min L2 Loss': min_loss_list,
                       'Max L2 Loss': max_loss_list,
                       'Std L2 Loss': std_loss_list,
                       'Best Agent': best_indiv_list
                       })
    if filename:
        df.to_csv(filename, index=False)
    return df