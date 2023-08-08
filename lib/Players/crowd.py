import numpy as np
from math import *


#=========================================================================================
#secondary functions to print string or not

def print_if_bool(string, print_bool):
    """
    Print string only if print_bool = True
    :param string:
    :param print_bool:
    :return:
    """
    if print_bool:
        print(string)

def print_if_bool2(print_bool):
    """
    Second order version of print_if_bool, parametrising print_if_bool
    :param print_bool:
    :return: function that behaves like print_if_bool without to specify print_bool
    """
    def evaluate(string):
        return print_if_bool(string, print_bool)
    return evaluate


#========================================================================================

def simulated_player(p, k, prev_avg):
    """
    Play of a simulated player of rationality k
    :param p: parameter of beauty contest / target factor (float)
    :param k: rationality level (int)
    :param prev_avg: previous average (int)
    :return: play according to level k (int)
    """
    # first round / no memory yet
    if prev_avg == -1:
        #great noise added in first round
        play = round(p**k*50 + np.random.normal(0, 9))
        play = play if play >= 0 else 0
        play = play if play <= 100 else 100
    else:
        #smaller noise in rounds >1
        play = round(p ** k * prev_avg + np.random.normal(0, 0.25))
        play = play if play>=0 else 0
        play = play if play <= 100 else 100
    return play


def simple_agent(config, agent_memory):
    """
    Test agent -> level 0 player
    :param config: configuration of decision process / Idle here
    :param agent_memory: memory of previous rounds
    :return: level-0 play
    """
    if len(agent_memory)==0:
        play = 50
    else:
        play = agent_memory[-1]
    return play


def dict_opt_agent(config, agent_memory):
    """
    Dictionary-optimiser agent
    :param config: array of responses against memory
    :param agent_memory: previous memory
    :return: play of the dictionary against the key memory[-1]
    """
    #default play
    if len(agent_memory)==0:
        play = 50
    else:
        #the most recent memory is the key to play in response to
        prev_avg = agent_memory[-1]
        play = config[prev_avg]
    return play


def crowd_iterate(levels, agent, config, p = 2/3,  nb_rounds= 10, print_bool = 1, begin_measure =0):
    """
    iterate plays of the crowd and computing results during nb_rounds
    :param levels: list of levels of the opponents (to the agent)
    :param agent: agent function
    :param config: configuration of the agent's decision process (1-D array)
    :param p: parameter of Beauty Contest (float)
    :param nb_rounds: number of rounds (int)
    :param print_bool: if True, print results at each round (bool)
    :param begin_measure: from which round the cumulative L2 Loss is computed for the agent (int)
    :return: list of all players win count, and the agent loss
    """
    printer = print_if_bool2(print_bool)
    #general info
    nb_players = (len(levels) + 1)
    prev_avg = -1
    prev_target = -1
    #agent info
    win_counts = np.zeros(nb_players)
    agent_loss = 0
    agent_memory = []
    for i in range(nb_rounds):
        printer("Round {}".format(i))
        plays = []
        # each player chooses a number
        for j in range(len(levels)):
            k = levels[j]
            player_decision = simulated_player(p, k, prev_avg)
            plays.append(player_decision)
        agent_play = agent(config, agent_memory)
        plays.append(agent_play)
        printer("Choices = {}".format(plays))
        # update game info
        avg = np.mean(plays)
        target = round(p*avg, 2)
        diff = np.abs(np.array(plays) - target)
        idxs = np.where(diff == diff.min())[0]
        printer("Average = {}".format(avg))
        printer("Target = {}".format(target))
        printer("Winner(s) = {}".format(idxs))
        printer("--------------------------------------")
        win_counts[idxs] += 1
        prev_avg = round(avg)
        prev_target = target
        #update agent info
        agent_memory.append(prev_avg)
        if i>= begin_measure:
            agent_loss += (agent_play - target) ** 2
    printer("Win counts = {}".format(win_counts))
    printer("Agent Loss = {}".format(agent_loss))
    printer("========================================")
    return win_counts, agent_loss

def random_crowd_iterate(possible_levels, agent, config, nb_players = 5, p = 2/3,  nb_rounds= 10, print_bool = 1, begin_measure =0):
    """
    Like crowd_iterate but players change levels between each round
    :param possible_levels: list of possible rationality levels
    :param agent: agent function
    :param config: 1-D array
    :param nb_players: number of players (including agent) in the crowd (int)
    :param p: KBC factor, float
    :param nb_rounds: int
    :param print_bool: if True, print results at each round (bool)
    :param begin_measure: from which round to measure fitness, int
    :return: list of all players win count, and the agent loss
    """
    printer = print_if_bool2(print_bool)
    #general info
    prev_avg = -1
    prev_target = -1
    #agent info
    win_counts = np.zeros(nb_players)
    agent_loss = 0
    agent_memory = []
    for i in range(nb_rounds):
        printer("Round {}".format(i))
        plays = []
        # each player chooses a number
        levels = list(np.random.choice(possible_levels, nb_players-1, replace = True))
        for j in range(len(levels)):
            k = levels[j]
            player_decision = simulated_player(p, k, prev_avg)
            plays.append(player_decision)
        agent_play = agent(config, agent_memory)
        plays.append(agent_play)
        printer("Choices = {}".format(plays))
        # update game info
        avg = np.mean(plays)
        target = round(p*avg, 2)
        diff = np.abs(np.array(plays) - target)
        idxs = np.where(diff == diff.min())[0]
        printer("Average = {}".format(avg))
        printer("Target = {}".format(target))
        printer("Winner(s) = {}".format(idxs))
        printer("--------------------------------------")
        win_counts[idxs] += 1
        prev_avg = round(avg)
        prev_target = target
        #update agent info
        agent_memory.append(prev_avg)
        if i>= begin_measure:
            agent_loss += (agent_play - target) ** 2
    printer("Win counts = {}".format(win_counts))
    printer("Agent Loss = {}".format(agent_loss))
    printer("========================================")
    return win_counts, agent_loss



