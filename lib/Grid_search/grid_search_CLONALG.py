from Keynesian_Beauty_Contest.lib.Players.fitness_KBC import *
from Keynesian_Beauty_Contest.lib.Templates.Template_Benchmarks import *
from Keynesian_Beauty_Contest.lib.Templates.Template_CLONALG import *
from math import *
import pandas as pd


def CLONALG_grid_search(all_comb, fitness, agent, bound_min=-1, bound_max=1, memory=1, integer_val=False, fit_wc=False, filename='test.csv'):
    """
    Run CLONALG grid search
    :param all_comb: combinated list of all hyperparameter sub lists as [pop_sizes, sel_rates, crs, beta_mag, rep_rates]
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
    inv_mag = 1e-2 if fit_wc else 1e-6
    ab_size = 141 if memory==1 else 151
    ab_size = 101 if integer_val else ab_size
    mut_function = integer_mutation if integer_val else uniform_mutation

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
        pop_size, select_rate, clone_rate, mut_const, replaced_rate = comb[0], comb[1], comb[2], comb[3], comb[4]
        select_size = ceil(select_rate * pop_size)
        nb_replaced = ceil(replaced_rate * pop_size)
        mut_factor = mut_const * inv_mag
        pop_shape = (ab_size, pop_size)
        select_pop, fit_vector, _ = CLONALG_multiple_runs(fitness, pop_shape, bound_min, bound_max, select_size,
                                                          clone_rate, mut_function, mut_factor, nb_replaced, nb_gen=101,
                                                          nb_runs=3, print_bool=False, integer_val=integer_val)
        #best_config, fit_stats = robust_eval(select_pop, fitness, nb_trials = 10, print_bool = False)
        max_index = np.argmax(fit_vector)
        best_config = select_pop[:, max_index]
        list_wc, list_loss = evaluation(agent, best_config, begin_measure=memory, print_bool=False, random_crowds=False)
        avg_wc, std_wc, max_wc, min_wc = np.mean(list_wc), np.std(list_wc), np.max(list_wc), np.min(list_wc)
        avg_loss, std_loss, max_loss, min_loss = np.mean(list_loss), np.std(list_loss), np.max(list_loss), np.min(list_loss)
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