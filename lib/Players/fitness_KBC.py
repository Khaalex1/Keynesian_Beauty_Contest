from Keynesian_Beauty_Contest.lib.Players.crowd import *
from Keynesian_Beauty_Contest.lib.Players.RegNets import *
import itertools

"""
Fitness definitions in the KBC. PARADIGM : MAXIMISATION
"""

#======================================================================================================
#dict_opt agent fitness

def dict_opt_fitness(pop_agent, fit_wc = True):
    """
    Computes fitness of Dict-opt agent
    :param pop_agent: population of agent configurations, array (101, nb_indiv)
    :param fit_wc: if True, the fitness is the Win Count. Else, the L2-Loss
    :return: fitness vector of the agent population. TO BE MAXIMISED
    """
    all_levels_combinations = list(itertools.combinations_with_replacement([0, 1, 2, 3], 4))
    #norm_factor = len(all_levels_combinations) * 9
    fit_vector = []
    agent = dict_opt_agent
    for agent_config in pop_agent.T:
        fit_agent = 0
        for levels in all_levels_combinations:
            win_counts, loss = crowd_iterate(levels, agent, agent_config, p=2 / 3, nb_rounds=10, print_bool=0,
                                             begin_measure=1)
            if fit_wc:
                fit_agent += win_counts[-1]
            else:
                fit_agent += (90000 - loss)
        fit_vector.append(fit_agent)
    return np.array(fit_vector)

def compatible_dict_opt_fitness(fit_wc = True):
    """
    Correct signature of fitness function for Dict-opt (fit_wc embedded)
    :param fit_wc: if True, the fitness is the Win Count. Else, the L2-Loss
    :return: fitness function to be optimised by Template_GA or Template_CLONALG
    """
    def evaluate(pop_agent):
        return dict_opt_fitness(pop_agent, fit_wc)
    return evaluate


#===================================================================================================
#RegNet agent fitness

# WARNING: LOSS IS TRANSFORMED INTO FITNESS TO MAXIMISE
def fitness_RegNet(pop_agent, memory = 1, fit_wc = True):
    """
    Computes fitness of RegNet agent
    :param pop_agent: population of network weights, array (nb_weights, nb_indiv)
    :param memory: number of memory used in input -> determines to use either RegNet1 or 2
    :param fit_wc: if True, the fitness is the Win Count. Else, the L2-Loss
    :return: fitness vector of the agent (network) population. TO BE MAXIMISED
    """
    all_levels_combinations = list(itertools.combinations_with_replacement([0, 1, 2, 3], 4))
    fit_vector = []
    #norm_factor = len(all_levels_combinations) * (10 - memory)
    if memory == 1:
        agent = memory_RegNet(layers = [1, 10, 10, 1])
    else:
        agent = memory_RegNet(layers=[2, 10, 10, 1])
    for agent_config in pop_agent.T:
        fit_agent = 0
        for levels in all_levels_combinations:
            win_counts, loss = crowd_iterate(levels, agent, agent_config, p=2/3, nb_rounds=10, print_bool=0, begin_measure = memory)
            if fit_wc:
                fit_agent += win_counts[-1]
            else:
                fit_agent += (10000*(10 - memory) - loss)
        fit_vector.append(fit_agent)
    return np.array(fit_vector)


def compatible_RegNet_fitness(memory = 1, fit_wc = True):
    """
    Correct signature of fitness function for RegNet (fit_wc embedded)
    :param memory: number of memory used in input -> determines to use either RegNet1 or 2
    :param fit_wc: if True, the fitness is the Win Count. Else, the L2-Loss
    :return: fitness function to be optimised by Template_GA or Template_CLONALG
    """
    def evaluate(pop_agent):
        return fitness_RegNet(pop_agent, memory, fit_wc)
    return evaluate


def evaluation(agent, config, begin_measure=0, print_bool=False, random_crowds = False, nb_opp = 4):
    """
    Evaluate an agent (configuration) for 10 trials against all possible crowds
    :param agent: agent function
    :param config: configuration of decision process : size-101 list (Dict-opt) or weight array (1-D)
    :param begin_measure: int
    :param print_bool: bool
    :param random_crowds: if true, evaluation is done against randomised-levels crowds
    :param nb_opp: nb of opponents
    :return: list of agent win counts for 10 trials, list of loss for 10 trials
    """
    possible_levels = [0, 1, 2, 3]
    all_levels_combinations = list(itertools.combinations_with_replacement(possible_levels, nb_opp))
    list_wc = []
    list_loss = []
    printer = print_if_bool2(print_bool)
    for i in range(10):
        wc = 0
        loss = 0
        for levels in all_levels_combinations:
            if random_crowds:
                printer("RANDOM CROWD")
                win_counts, perf = random_crowd_iterate(possible_levels, agent, config, nb_players = 5, p = 2/3,  nb_rounds= 10, print_bool = print_bool, begin_measure =begin_measure)
            else:
                printer("Crowd = {}".format(levels))
                win_counts, perf = crowd_iterate(levels, agent, config, p = 2/3,  nb_rounds= 10, print_bool = print_bool, begin_measure =begin_measure)
            wc += win_counts[-1]
            loss += perf
        list_wc.append(wc)
        list_loss.append(loss)
    avg_wc, std_wc, min_wc, max_wc = np.mean(list_wc), np.std(list_wc), np.min(list_wc), np.max(list_wc)
    avg_loss, std_loss, min_loss, max_loss = np.mean(list_loss), np.std(list_loss), np.min(list_loss), np.max(list_loss)
    print("-----------------------------------------------------------------------------------------")
    print("Statistics of selected indiv. on 10 confrontations against ALL crowds :")
    print("WC avg, std, min, max = ", [avg_wc, std_wc, min_wc, max_wc])
    print("Loss avg, std, min, max = ", [ avg_loss, std_loss, min_loss, max_loss])
    print("=========================================================================================")
    print("=========================================================================================")
    return list_wc, list_loss




