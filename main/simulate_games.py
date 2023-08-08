from Keynesian_Beauty_Contest.lib.Players.crowd import *
from Keynesian_Beauty_Contest.lib.Players.RegNets import *
from Keynesian_Beauty_Contest.lib.Read_results.response_plot import *

"""
Simulate a game of an agent against a particular crowd of rationality levels
"""

if __name__ == "__main__":
    #Best configuration of RegNet1, trained with CLONALG/AIS on Win Count fitness
    df1 = pd.read_csv("../result_files/AIS_RegNet_WC_m1.csv")
    best_config1 = extract_best_from_pandas(df1)

    #Best configuration of RegNet2, trained with CLONALG/AIS on L2 fitness
    df2 = pd.read_csv("../result_files/AIS_RegNet_L2_m2.csv")
    best_config2 = extract_best_from_pandas(df2)

    #combining weights for configuring HybridNet
    weights = np.hstack((best_config1, best_config2))
    agent = HybridNet

    """Run simulation. Agent's play is the last in the list"""
    print('=======================================================')
    #specify levels of crowd
    levels = (1, 1, 1, 2)
    p = 2/3
    nb_rounds = 10
    print_bool = 1
    #win_counts, agent_loss = crowd_iterate(levels, agent, weights, p, nb_rounds, print_bool)


    #========================================================================================
    # df = pd.read_csv("../result_files/AIS_dict_opt_WC.csv")
    # config = extract_best_from_pandas(df)
    # agent = dict_opt_agent
    # levels = (0, 1, 2, 3)
    # p = 2 / 3
    # nb_rounds = 10
    # print_bool = 1
    # win_counts, agent_loss = crowd_iterate(levels, agent, config, p, nb_rounds, print_bool)

