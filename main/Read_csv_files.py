from Keynesian_Beauty_Contest.lib.Read_results.read_pandas import *
from Keynesian_Beauty_Contest.lib.Players.fitness_KBC import *

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
    # static-rationality crowds
    random_crowds = False
    #nb of opponents against agent
    nb_opp = 4

    #==============================================================================================
    """Best configuration of Dict-opt, trained with CLONALG/AIS on Win Count fitness"""

    print("Best Dict-opt info")
    df = pd.read_csv("../result_files/AIS_dict_opt_WC.csv")
    best_config = extract_best_from_pandas(df)
    print("\nBest Dict-opt config = \n", best_config)
    print("===================================================================================")

    #================================================================================================================================
    """Best configuration of RegNet1, trained with CLONALG/AIS on Win Count fitness"""

    print("Best RegNet1 info")
    df1 = pd.read_csv("../result_files/AIS_RegNet_WC_m1.csv")
    best_config1 = extract_best_from_pandas(df1)
    print("\nBest RegNet1 weights = \n", best_config1)
    print("===================================================================================")

    #=================================================================================================================================
    """Best configuration of RegNet2, trained with CLONALG/AIS on L2 fitness"""

    print("Best RegNet2 info")
    df2 = pd.read_csv("../result_files/AIS_RegNet_L2_m2.csv")
    best_config2 = extract_best_from_pandas(df2)
    print("\nBest RegNet2 weights = \n", best_config2)

    print("===================================================================================")
    #=================================================================================================================================



















    """combined weights of HybridNet"""

    # weights = np.hstack((best_config1, best_config2))
    # agent = HybridNet

    """evaluating HybridNet on 10 trials"""

    # list_wc, list_loss = evaluation(agent, weights, begin_measure=1, print_bool=True, random_crowds=random_crowds, nb_opp = nb_opp)


