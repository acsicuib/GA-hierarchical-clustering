import numpy


def defGeneticSeed(i):
    global randomGenetic
    randomGenetic = numpy.random.RandomState(randomSeedGenetic[i])





#appNumberRange =range(20,101,20)
appNumberRange = range(20,21,20)
appNumberRange = range(20,61,20) #configuration para el experimento del articulo
appNumberRange = range(10,21,10)


#nodeNumberRange =range(100,401,50)
nodeNumberRange = range(100,101,50)
nodeNumberRange = range(100,301,100) #configuracion para el experimento del articulo
nodeNumberRange = range(50,101,50)


experiments2execute = list()

for appNumber in appNumberRange:
    for nodeNumber in nodeNumberRange:
        oneExperiment = (appNumber,nodeNumber)
        experiments2execute.append(oneExperiment)


ILP_METHOD = False
numberGenerations = 100 #configuracion para el experimento del articulo
numberGenerations = 3



randomSeedGenetic = [1,11,21,31,41,51,61,71,81,91]
rangeOfExperimentsRepetitions = range(0,3)



global randomGenetic

