from Keynesian_Beauty_Contest.lib.Read_results.response_plot import *

"""
NOTICE FOR READING RESULTS STORED IN CSV FILES:

- Result files are in ../result_files/
- Each file contains agents of a particular type (and performance) 
trained on multiple sets of hyperparameters 
- The first acronym is the metaheuristic used in training : AIS (CLONALG) or GA
- The second sequence is the agent type : RegNet or dict-opt
- L2 or WC then mean whether the L2 Loss or the Win Count has been optimised
- (only for RegNet) m1 or m2 represent the number of memories
 of RegNet (RegNet1 or RegNet2)

- Exemple : AIS_RegNet_L2_m1.csv -> file containing RegNet1 
agents optimising L2 Loss with AIS (on multiple sets of hyperparam.)
- Exemple : GA_dict_opt_WC.csv -> file containing Dict-opt 
agents optimising Win Count with GA (on multiple sets of hyperparam.)

"""

if __name__ == "__main__":

    """Plot the response profile of RegNet2, optimising Win Count with AIS"""
    regnet2_ais_wc = pd.read_csv("../result_files/AIS_RegNet_WC_m2.csv")
    weights = extract_best_from_pandas(regnet2_ais_wc)
    title = "RegNet2 response profile (AIS - Win Count)"
    # If True, nullify Avg[t-1]>Avg[t-2] in contour plot
    print_plane = True
    D2_plotter(weights, title, print_plane)


    """Plot the response profile of RegNet1, optimising Win Count with GA"""
    regnet1_ga_wc = pd.read_csv("../result_files/GA_RegNet_WC_m1.csv")
    weights = extract_best_from_pandas(regnet1_ga_wc)
    p = 2/3
    title = "RegNet1 response profile (GA - Win Count)"
    # specify if the agent is Dict-opt or not
    dict_opt = False
    D1_plotter(weights, p, title, dict_opt)
