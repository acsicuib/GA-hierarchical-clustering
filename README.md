This repository includes the source code, the data and the plots for the paper entitled "Optimizing fog colony layout and service placement through genetic algorithms and hierarchical clustering" and submitted to Journal of Systems Architecture for evaluation.



For the execution of the optimization algorithm

```
python main-GAcolonyPartitionV4.py
```

For the analysis of the results and the generation of the plots

```
python plotsGeneration.py
```

The experiments are set up by editing the configuration file "configuration.py"

```python
appNumberRange = range(20,61,20) #number of applications
nodeNumberRange = range(100,301,100) #number of fog devices
ILP_METHOD = False #required for the use of the Greedy algorithm in the service placement phase
numberGenerations = 100 #number of generations
randomSeedGenetic = [1,11,21,31,41,51,61,71,81,91] #different random seeds for each experiment repetition
rangeOfExperimentsRepetitions = range(0,10) #number of experiment repetitions for each experimental scenario
```

Folder "results" contains data and the plots of the experimentation phase for all the repetitions of all the experiments scenarios. These results are organized in folder. Each folder contains the results of one of the 10 repetitions of one of the 9 experiment scenarios.

