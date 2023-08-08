import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Keynesian_Beauty_Contest.lib.Read_results.read_pandas import *


def D1_plotter(weights, p=2/3, title = "Agent response to memory of previous round", dict_opt = False):
    """
    plot response profile of memory-1 agent (play vs Avg[t-1])
    :param weights: config array of agent. Either weights for RegNet1 or response values (Dict-opt)
    :param p: parameter of KBC
    :param title: string for plot title
    :param dict_opt: if True, agent is considered as Dict-opt. Else RegNet1
    :return:
    """
    X = np.arange(0, 51, 1).reshape((1, -1))
    if dict_opt:
        Y = weights[:51]
    else:
        Y = forward(weights, X, layers = [1, 10, 10, 1])
    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.plot(X.flatten(), 'g--', label="Rationality 0")
    plt.plot(p*X.flatten(), 'b--', label = "Rationality 1" )
    plt.plot(p**2 * X.flatten(), 'm--', label="Rationality 2")
    plt.plot(p ** 3 * X.flatten(), 'k--', label="Rationality 3")
    plt.plot(0*X.flatten(),'r-', label=r'Rationality $\infty$')
    if dict_opt:
        plt.stem(Y, label="Agent (dict_opt)")
    else:
        plt.stem(Y[0], label = "Agent (RegNet1)")
    plt.xlabel(r'$Avg[t-1]$')
    plt.ylabel(r"$Response[t]$")
    #plt.xlim(X.max(), X.min())
    plt.legend()
    if title:
        plt.title(title)
    plt.show()

#============================================================
#auxiliary forward functions

def new_forward(x1, x2, weights):
    """
    Adapted forward for 2 memories, specified separately
    :param x1: Avg[t-2]
    :param x2: Avg[t-1]
    :param weights: flattened weights of ANN
    :return: prediction
    """
    X = np.array([[x1], [x2]])
    Y = forward(weights, X, layers = [2, 10, 10, 1])[0,0]
    return Y

def D2_forward(X1,X2, weights):
    """
    Adapted forward for 2 meshgrid memory array
    :param X1: array of Avg[t-2] obtained from meshgrid
    :param X2: array of Avg[t-1] obtained from meshgrid
    :param weights: flattened weight array (1-D)
    :return: array of evaluation on X1 and X2 (shape X1.shape)
    """
    Y = np.zeros((X1.shape[0], X1.shape[1]))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = new_forward(X1[i,j], X2[i,j], weights)
    return Y

#================================================================

def D2_plotter(weights, title = "Agent response to memory of previous rounds", print_plane = True):
    """
    2-D plot of RegNet2 response
    :param weights: flattened array of weights (+ biases)
    :param title: string of plot title
    :param print_plane: if True, eliminates Avg[t-1]>Avg[t-2] points (negated)
    :return:
    """
    #range of inputs
    rg = np.arange(0, 51, 1)
    X1, X2 = np.meshgrid(rg, rg)
    Y = D2_forward(X1, X2, weights)

    fig = plt.figure(figsize = (14, 5))
    ax = fig.add_subplot(1, 2, 1, projection = '3d')
    surf = ax.plot_surface(X1, X2, Y, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
    ax.set_xlabel(r'$Avg[t-2]$')
    ax.set_ylabel(r'$Avg[t-1]$')
    ax.set_zlabel("RegNet2 response")
    if title:
        ax.set_title(title)
    #contour plot
    ax = fig.add_subplot(1, 2, 2, aspect='equal')
    if print_plane:
        Y[np.where(X2>X1)[0], np.where(X2>X1)[1]]=-1
        cp = ax.contourf(X1, X2, Y)
        fig.colorbar(cp)  # Add a colorbar to a plot
        if title:
            ax.set_title('Contour Plot of the agent response \n' + r'($Avg[t-1]>Avg[t-2]$ points removed)')
        ax.set_xlabel(r'$Avg[t-2]$')
        ax.set_ylabel(r'$Avg[t-1]$')
    else:
        cp = ax.contourf(X1, X2, Y)
        fig.colorbar(cp)  # Add a colorbar to a plot
        if title:
            ax.set_title(r'Contour Plot of the agent response')
        ax.set_xlabel(r'$Avg[t-2]$')
        ax.set_ylabel(r'$Avg[t-1]$')
    plt.show()