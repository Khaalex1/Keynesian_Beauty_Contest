import numpy as np
import pandas as pd
from Keynesian_Beauty_Contest.lib.Players.fitness_KBC import *

def decoder(s):
    """
    decode a string of a float list
    :param s: string of a list of float values
    :return: list within the string s
    """
    s = s.replace("\n", "").strip()
    s = s[1:-1]
    s = [float(num) for num in s.split()]
    return np.array(s)

def extract_best_from_pandas(df):
    """
    extract best individuals based on Win Count Average from panda dataframe
    :param df: panda dataframe containing multiple individuals with their performance
    :return: config (weights or values of dict for Dict-opt) array of best agent
    """
    best_index = np.argmax(df['Avg Win Count'])
    print(df.iloc[[best_index]].T)
    best_config = df['Best Agent'][best_index]
    best_config = decoder(best_config)
    return best_config

# def hyperparameter_sensitivity(df):
#     """
#     Compute hyperparameter sensitivity
#     :param df: panada dataframe containing individuals and their performance
#     :return: WC sensitivity and L2 sensitivity
#     """
#     #wc_avg = np.mean(df["Avg Win Count"])
#     wc_std = np.std(df["Avg Win Count"])
#     #loss_avg = np.mean(df["Avg L2 Loss"])
#     loss_std = np.std(df["Avg L2 Loss"])
#     print("===========================================================")
#     print("STD of Win Count of best individuals = ", wc_std)
#     print("STD of Loss of best individuals = ", loss_std)
#     print("===========================================================")
#     return wc_std, loss_std

