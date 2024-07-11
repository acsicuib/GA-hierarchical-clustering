This repository includes the source code, the data and the plots for the paper entitled "Optimizing fog colony layout and service placement through genetic algorithms and hierarchical clustering", published in the journal "Expert systems with applications":

```
Francisco Talavera, Isaac Lera, Carlos Juiz, Carlos Guerrero, Optimizing fog colony layout and service placement through genetic algorithms and hierarchical clustering, Expert Systems with Applications, Volume 254, 2024, 124372, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2024.124372.
```

If you would consider to cite the paper, you could use this bibtex record:

```
@article{TALAVERA2024124372,
title = {Optimizing fog colony layout and service placement through genetic algorithms and hierarchical clustering},
journal = {Expert Systems with Applications},
volume = {254},
pages = {124372},
year = {2024},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2024.124372},
url = {https://www.sciencedirect.com/science/article/pii/S0957417424012387},
author = {Francisco Talavera and Isaac Lera and Carlos Juiz and Carlos Guerrero},
keywords = {Fog computing, Fog colony, Service placement, Optimization, Hybrid genetic algorithm}
}
```

**Acknowledgment**:

This research was supported by MICIU/AEI/10.13039/501100011033, Spain [grant number PID2021-128071OB-I00] and FEDER,UE, Spain [grant number PID2021-128071OB-I00].


**Execution**:

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

Folder "results" contains data and the plots of the experimentation phase for all the repetitions of all the experiments scenarios. These results are organized in folder. Each folder contains the results of one of the 10 repetitions of one of the 9 experiment scenarios. The forlder name indicates the number of repetition and the number of applications and nodes in the expriment sceniaro.
