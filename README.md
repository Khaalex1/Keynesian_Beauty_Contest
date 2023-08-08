# Dissertation_Keynes_Beauty

This Dissertation project is about optimising plays against the Level-k model in the Keynesian Beauty Contest. Agents are trained using the metaheuristics CLONALG and GA.
The project is built as follows :
- The "lib" folder contains the implementation of the project, for instance the designing of the players, the agents, the fitness definitions, read and plot functions.
- The "main" folder contains the main files, enabling to run a particular action, for example training an agent, simulating plays and extracting best agents from csv files
- "result_files" contains results gathered during the project, as csv files

Experiments show that CLONALG is always a more efficient training method than GA given the sets of hyperparameters tested. It reaches initially a peak win rate of 76% with a memory-1 Neuroevolved network (RegNet1).
An hybrid model combining plays of memory-1 and memory-2 networks, called in the project HybridNet, increases this performance to 81%, which can be considered excellent.

The reader is encouraged to refer to the "main" files to understand how the program is run and what it does.
# Keynesian_Beauty_Contest
