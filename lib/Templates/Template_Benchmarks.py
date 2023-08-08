"""
Testing fitness functions

Origin : "Problem Definitions and Evaluation Criteria for the CEC 2005 Special Session
 on Real-Parameter Optimization"
"""

import numpy as np

#===================================================================
# Second order functions for normalising (min -> max problems)

def inv_abs_fitness(fitness):
    """
    Invert a fitness function
    :param fitness: original fitness function
    :return: second order (transformed) function that behaves like fitness
    """
    def evaluate(x):
        return (1/(1+abs(fitness(x))))
    return evaluate

def adder(fitness, C):
    """
    Add a constant to a fitness function
    :param fitness: original fitness function
    :param C: Additive constant to fitness
    :return: second order (transformed) function that behaves like fitness
    """
    def evaluate(x):
        return fitness(x) + abs(C)
    return evaluate

def opposite_fitness(fitness):
    """
    Change sign of fitness function and add a constant
    :param fitness: original fitness function
    :param C: Additive constant to fitness
    :return: second order (transformed) function that behaves like fitness
    """
    def evaluate(x):
        return -fitness(x)
    return evaluate


#=========================================================
#Actual benchmarks

def sphere(x):
    """
    Standard sphere function, min = 0, argmin = [0, 0, ..., 0]
    :param x: 1D array of size >=2 (numpy : shape = (n,1))
    :return: sphere evaluation
    """
    S = np.sum(x**2, axis = 0)
    return S

def shifted_sphere(x):
    """
    Sphere function (shifted by [0, 1, 2, ...]), min = 0, argmin = [0, 1, 2,..., n-1]
    :param x: 1D array of size >=2 (numpy : shape = (n,1))
    :return: sphere evaluation
    """
    shift = np.array([[i] for i in range(x.shape[0])])
    X = x - shift
    S = np.sum(X**2, axis = 0)
    return S

def double_sum(x):
    """
    Double sum function, min = 0, argmin = [0, 0, ..., 0]
    :param x: 1D array of size >=2 (numpy : shape = (n,1))
    :return: double-sum evaluation
    """
    s1 = np.array([np.sum(x[:(i+1),:], axis = 0) for i in range(len(x))])
    s2 = np.sum(s1**2, axis = 0)
    return s2

def rastrigin(x):
    """
    Rastrigin benchmark function, min = 0, argmin = [0, 0, ..., 0]
    :param x: 1D array (numpy : shape = (n,1))
    :return: Rastrigin function evaluation in x
    """
    A = 10
    n = len(x)
    S = np.sum(x**2 - A*np.cos(2*np.pi*x), axis = 0)
    return A*n + S

def rosenbrock(x):
    """
    Rosenbrock benchmark function, min = 0, argmin = [1, 1, ..., 1]
    :param x: 1D array of size >2 (numpy : shape = (n,1))
    :return: Rosenbrock function evaluation in x
    """
    n = len(x)
    if n<2:
        print("Array length must be > 2")
        return False
    if len(x.shape)<2:
        x = x.reshape((n, 1))
    xi = x[:-1, :]
    xp1 = x[1:, :]
    S = np.sum(100*(xp1 - xi**2)**2 + (1-xi)**2, axis = 0)
    return S


def schwefel(x):
    """
    Schwefel benchmark function, min = 0, argmin = [420.9687, 420.9687, ..., 420.9687]
    :param x: 1D array of size >2 (numpy : shape = (n,1))
    :return: Schwefel function evaluation in x
    """
    V = 418.9829
    S = V*x.shape[0] - np.sum(x*np.sin(np.sqrt(abs(x))), axis = 0)
    return S

def griewank(x):
    """
    Griewank benchmark function, min = 0, argmin = [0, 0, ..., 0]
    :param x: 1D array of size >2 (numpy : shape = (n,1))
    :return: Griewank function evaluation in x
    """
    S = 1 + np.sum((x**2)/4000, axis =0) - np.prod((np.array([[1/np.sqrt(i)] for i in range(1, x.shape[0]+1)]))*np.cos(x), axis = 0)
    return S

#================================================================================================
# Robust estimation of best individual in population for non-deterministic fitness

def robust_eval(select_pop, fitness, nb_trials=10, print_bool=False):
    """
    Robust estimation of best individual in population for non-deterministic fitness
    :param select_pop: population of individuals, array of shape ((n, nb_indiv))
    :param fitness: fitness to evaluate select_pop
    :param nb_trials: number of times to run evaluation
    :param print_bool: boolean for printing statistical results of best individual or not
    :return: best_individual array (n, ) and list of perfrmance statistics (mean, std, min fitness, max fitness)
    """
    fit_trials = np.zeros((nb_trials, select_pop.shape[1]))
    for i in range(nb_trials):
        fit_trials[i, :] = fitness(select_pop)
    avg_vector = np.mean(fit_trials, axis=0)
    std_vector = np.std(fit_trials, axis=0)
    min_vector = np.min(fit_trials, axis=0)
    max_vector = np.max(fit_trials, axis=0)

    best_avg_index = np.argmax(avg_vector)
    best_indiv = select_pop[:, best_avg_index]
    best_avg_fit = avg_vector[best_avg_index]
    best_std_fit = std_vector[best_avg_index]
    best_min_fit = min_vector[best_avg_index]
    best_max_fit = max_vector[best_avg_index]
    if print_bool:
        print("Best Indiv. on average = ", best_indiv)
        print("Avg fitness = ", best_avg_fit)
        print("Std fitness = ", best_std_fit)
        print("Min fitness = ", best_min_fit)
        print("Max fitness = ", best_max_fit)
        print("=============================================")
    return best_indiv, [best_avg_fit, best_std_fit, best_min_fit, best_max_fit]