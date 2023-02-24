Para la ejecución del código

```
python main-GAcolonyPartitionV4.py
```

Para la generación de los gráficos y el análisis de los datos

```
python plotsGeneration.py
```

Para la configuración de los experimentos antes de su ejecución editar el archivo configuration.py

```python
appNumberRange = range(20,61,20) #configuration para el experimento del articulo
nodeNumberRange = range(100,301,100) #configuracion para el experimento del articulo
ILP_METHOD = False
numberGenerations = 100 #configuracion para el experimento del articulo
randomSeedGenetic = [1,11,21,31,41,51,61,71,81,91]
rangeOfExperimentsRepetitions = range(0,10)
```
