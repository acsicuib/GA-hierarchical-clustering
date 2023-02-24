import pandas as pd

import configuration
import pickle
import math
import copy
import matplotlib.pyplot as plt
import os
# Libraries for radar plots
from math import pi


def getExperimentStorageFolder(nodeNumber,appNumber,repNum):
    global storageFoldershort

    storageFoldershort = "./plots2/"
    return storageFoldershort+str(nodeNumber)+"nodes"+str(appNumber)+"apps"+str(repNum)+"repetition/"
def drawAllRadarPlots():

    # ******************************************************************************************
    #   Calculation of the generation number where the pareto set of the GA dominates the two control solutions
    # This is an independent code that can be executed isolated, since the data needed is readed from a file
    # that was generated in the whole previous execution of all this  script.
    # ******************************************************************************************

    getExperimentStorageFolder(0, 0, 0)  # dummy execution just to set the value of storageFoldershort

    with open(storageFoldershort + 'radarplot.pickle', 'rb') as f:
        radar2 = pickle.load(f)

    radarLabels = radar2['radarLabels']
    fsColonyOptTime = radar2['fsColonyOptTime']
    oneColoyOptTime = radar2['oneColoyOptTime']
    selectedFromParetoOptTime = radar2['selectedFromParetoOptTime']
    fsColonyNetTime = radar2['fsColonyNetTime']
    oneColoyNetTime = radar2['oneColoyNetTime']
    selectedFromParetoNetTime = radar2['selectedFromParetoNetTime']
    finalHyperVolume = radar2['finalHyperVolume']
    finalCoverageFSGA = radar2['finalCoverageFSGA']
    finalCoverage1CGA = radar2['finalCoverage1CGA']
    finalCoverageGAFS = radar2['finalCoverageGAFS']
    finalCoverageGA1C = radar2['finalCoverageGA1C']

    #
    # radarLabels=radar2['radarLabels']
    # fsColonyOptTime=radar2['fsColonyOptTime']
    # oneColoyOptTime=radar2['oneColoyOptTime']
    # selectedFromParetoOptTime=radar2['selectedFromParetoOptTime']
    # fsColonyNetTime=radar2['fsColonyNetTime']
    # oneColoyNetTime=radar2['oneColoyNetTime']
    # selectedFromParetoNetTime=radar2['selectedFromParetoNetTime']
    # finalHyperVolume=radar2['finalHyperVolume']
    # finalCoverageFSGA=radar2['finalCoverageFSGA']
    # finalCoverage1CGA=radar2['finalCoverage1CGA']
    # finalCoverageGAFS=radar2['finalCoverageGAFS']
    # finalCoverageGA1C=radar2['finalCoverageGA1C']
    #
    # radarLabels.pop(-1)
    # fsColonyOptTime.pop(-1)
    # oneColoyOptTime.pop(-1)
    # selectedFromParetoOptTime.pop(-1)
    # fsColonyNetTime.pop(-1)
    # oneColoyNetTime.pop(-1)
    # selectedFromParetoNetTime.pop(-1)
    # finalHyperVolume.pop(-1)
    # finalCoverageFSGA.pop(-1)
    # finalCoverage1CGA.pop(-1)
    # finalCoverageGAFS.pop(-1)
    # finalCoverageGA1C.pop(-1)

    import pandas as pd

    data4radarplots = {'radarLabels': radarLabels, 'fsColonyOptTime': fsColonyOptTime,
                       'oneColoyOptTime': oneColoyOptTime, 'selectedFromParetoOptTime': selectedFromParetoOptTime,
                       'fsColonyNetTime': fsColonyNetTime, 'oneColoyNetTime': oneColoyNetTime,
                       'selectedFromParetoNetTime': selectedFromParetoNetTime}
    df = pd.DataFrame(data4radarplots)
    aggregatedColumns = ['fsColonyOptTime', 'oneColoyOptTime', 'selectedFromParetoOptTime', 'fsColonyNetTime',
                         'oneColoyNetTime', 'selectedFromParetoNetTime']
    mean_df = df.groupby('radarLabels')[aggregatedColumns].mean()
    mean_df.reset_index(inplace=True)
    std_df = df.groupby('radarLabels')[aggregatedColumns].std()
    std_df.reset_index(inplace=True)

    # df.groupby('name')['fsColonyOptTime','oneColoyOptTime', 'selectedFromParetoOptTime', 'fsColonyNetTime', 'oneColoyNetTime', 'selectedFromParetoNetTime'].sem()

    radarLabels = mean_df["radarLabels"].values.tolist()
    fsColonyOptTime = mean_df["fsColonyOptTime"].values.tolist()
    oneColoyOptTime = mean_df["oneColoyOptTime"].values.tolist()
    selectedFromParetoOptTime = mean_df["selectedFromParetoOptTime"].values.tolist()
    fsColonyNetTime = mean_df["fsColonyNetTime"].values.tolist()
    oneColoyNetTime = mean_df["oneColoyNetTime"].values.tolist()
    selectedFromParetoNetTime = mean_df["selectedFromParetoNetTime"].values.tolist()

    std_radarLabels = std_df["radarLabels"].values.tolist()
    std_fsColonyOptTime = std_df["fsColonyOptTime"].values.tolist()
    std_oneColoyOptTime = std_df["oneColoyOptTime"].values.tolist()
    std_selectedFromParetoOptTime = std_df["selectedFromParetoOptTime"].values.tolist()
    std_fsColonyNetTime = std_df["fsColonyNetTime"].values.tolist()
    std_oneColoyNetTime = std_df["oneColoyNetTime"].values.tolist()
    std_selectedFromParetoNetTime = std_df["selectedFromParetoNetTime"].values.tolist()

    series2Plot = list()
    std_series2Plot = list()

    ind1 = copy.copy(oneColoyOptTime)
    ind1.append(oneColoyOptTime[0])
    series2Plot.append(ind1)

    std_ind1 = copy.copy(std_oneColoyOptTime)
    std_ind1.append(std_oneColoyOptTime[0])
    std_series2Plot.append(std_ind1)

    ind2 = copy.copy(fsColonyOptTime)
    ind2.append(fsColonyOptTime[0])
    series2Plot.append(ind2)

    std_ind2 = copy.copy(std_fsColonyOptTime)
    std_ind2.append(std_fsColonyOptTime[0])
    std_series2Plot.append(std_ind2)

    ind3 = copy.copy(selectedFromParetoOptTime)
    ind3.append(selectedFromParetoOptTime[0])
    series2Plot.append(ind3)

    std_ind3 = copy.copy(std_selectedFromParetoOptTime)
    std_ind3.append(std_selectedFromParetoOptTime[0])
    std_series2Plot.append(std_ind3)

    paletteValues = [0, 1, 2]
    radarTitle = 'placement_time (ms)'
    theLabels = ["one-colony", "fixed-size", "smallED-GA"]
    fileName2Store = "radarOptTime.pdf"

    plotRadarFigures(radarLabels, series2Plot, std_series2Plot, paletteValues, radarTitle, theLabels, fileName2Store)

    fileName2Store = "radarOptTime3.pdf"
    plotRadarFigures(radarLabels, series2Plot, std_series2Plot, paletteValues, radarTitle, theLabels, fileName2Store,
                     y_limit=(0, 50))

    series2Plot = list()

    ind2 = copy.copy(fsColonyOptTime)
    ind2.append(fsColonyOptTime[0])
    series2Plot.append(ind2)

    std_ind2 = copy.copy(std_fsColonyOptTime)
    std_ind2.append(std_fsColonyOptTime[0])
    std_series2Plot.append(std_ind2)

    ind3 = copy.copy(selectedFromParetoOptTime)
    ind3.append(selectedFromParetoOptTime[0])
    series2Plot.append(ind3)

    std_ind3 = copy.copy(std_selectedFromParetoOptTime)
    std_ind3.append(std_selectedFromParetoOptTime[0])
    std_series2Plot.append(std_ind3)

    paletteValues = [1, 2]
    radarTitle = 'placement_time (ms)'
    theLabels = ["one-colony", "fixed-size", "smallED-GA"]
    fileName2Store = "radarOptTime2.pdf"

    plotRadarFigures(radarLabels,series2Plot, std_series2Plot, paletteValues, radarTitle, theLabels, fileName2Store)

    series2Plot = list()

    ind1 = copy.copy(oneColoyNetTime)
    ind1.append(oneColoyNetTime[0])
    series2Plot.append(ind1)

    std_ind1 = copy.copy(std_oneColoyNetTime)
    std_ind1.append(std_oneColoyNetTime[0])
    std_series2Plot.append(std_ind1)

    ind2 = copy.copy(fsColonyNetTime)
    ind2.append(fsColonyNetTime[0])
    series2Plot.append(ind2)

    std_ind2 = copy.copy(std_fsColonyNetTime)
    std_ind2.append(std_fsColonyNetTime[0])
    std_series2Plot.append(std_ind2)

    ind3 = copy.copy(selectedFromParetoNetTime)
    ind3.append(selectedFromParetoNetTime[0])
    series2Plot.append(ind3)

    std_ind3 = copy.copy(std_selectedFromParetoNetTime)
    std_ind3.append(std_selectedFromParetoNetTime[0])
    std_series2Plot.append(std_ind3)

    paletteValues = [0, 1, 2]
    radarTitle = 'response_time (ms)'
    theLabels = ["one-colony", "fixed-size", "smallED-GA"]
    fileName2Store = "radarNetTime.pdf"

    plotRadarFigures(radarLabels, series2Plot, std_series2Plot, paletteValues, radarTitle, theLabels, fileName2Store)

    series2Plot = list()

    ind2 = copy.copy(fsColonyNetTime)
    ind2.append(fsColonyNetTime[0])
    series2Plot.append(ind2)

    std_ind2 = copy.copy(std_fsColonyNetTime)
    std_ind2.append(std_fsColonyNetTime[0])
    std_series2Plot.append(std_ind2)

    ind3 = copy.copy(selectedFromParetoNetTime)
    ind3.append(selectedFromParetoNetTime[0])
    series2Plot.append(ind3)

    std_ind3 = copy.copy(std_selectedFromParetoNetTime)
    std_ind3.append(std_selectedFromParetoNetTime[0])
    std_series2Plot.append(std_ind3)

    paletteValues = [1, 2]
    radarTitle = 'response_time (ms)'
    theLabels = ["fixed-size", "smallED-GA"]
    fileName2Store = "radarNetTime2.pdf"

    plotRadarFigures(radarLabels, series2Plot, std_series2Plot, paletteValues, radarTitle, theLabels, fileName2Store)
def calculateSmallestGenerationDomination():
    # ******************************************************************************************
    #   Calculation of the generation number where the pareto set of the GA dominates the two control solutions
    # This is an independent code that can be executed isolated, since the data needed is readed from a file
    # that was generated in the whole previous execution of all this  script.
    # ******************************************************************************************

    experimentLabels = list()
    experimentSmallestGeneration = list()

    for oneExperiment in configuration.experiments2execute:
        appNumber = oneExperiment[0]
        nodeNumber = oneExperiment[1]

        for repNum in configuration.rangeOfExperimentsRepetitions:

            storageFolder = getExperimentStorageFolder(nodeNumber, appNumber, repNum)

            with open(storageFolder + 'resultdata.pickle', 'rb') as f:
                oneIterationData = pickle.load(f)

            f.close()

            controlCases = oneIterationData['controlCases']

            smallestGenerationDominates = 1000000

            for i in range(configuration.numberGenerations):


                minX = min(oneIterationData['meanMinX'])
                minY = min(oneIterationData['meanMinY'])
                maxX = max(oneIterationData['meanMaxX'])
                maxY = max(oneIterationData['meanMaxY'])

                diffX = maxX - minX
                diffY = maxY - minY



                hyperVolume = list()
                coverage1CGA = list()
                coverageFSGA = list()
                coverageGA1C = list()
                coverageGAFS = list()



                paretoMaxX = -1.0
                paretoMaxY = -1.0
                paretoMinX = float('inf')
                paretoMinY = float('inf')

                SOLdominates1C = 0
                SOLdominatesFS = 0

                one1CdominatesSOL = 0
                FSdominatesSOL = 0

                for j in range(len(oneIterationData['all_paretox'][i])):
                    x_value = oneIterationData['all_paretox'][i][j]
                    y_value = oneIterationData['all_paretoy'][i][j]

                    if (controlCases['otCc'] <= x_value) and (controlCases['rtCc'] <= y_value):
                        FSdominatesSOL += 1

                    if (controlCases['ot1c'] <= x_value) and (controlCases['rt1c'] <= y_value):
                        one1CdominatesSOL += 1

                    if (x_value <= controlCases['otCc']) and (y_value <= controlCases['rtCc']):
                        SOLdominatesFS = 1

                    if (x_value <= controlCases['ot1c']) and (y_value <= controlCases['rt1c']):
                        SOLdominates1C = 1

                    #                    print ("otCc  "+str(otCc)+" x_value  "+str(x_value)+" rtCc "+str(rtCc)+" y_value "+str(y_value)+"  ")

                    #                ot1c,rt1c
                    #
                    #                otCc,rtCc
                    #
                    #                        #LAS X son los optimization times
                    #                        #las Y son los estimated response times

                    x_value = (x_value - minY) / diffX
                    y_value = (y_value - minY) / diffY

                    paretoMinX = min(paretoMinX, x_value)
                    paretoMinY = min(paretoMinY, y_value)

                    paretoMaxX = min(paretoMaxX, x_value)
                    paretoMaxY = min(paretoMaxY, y_value)


                value4HyperVolume = (float(paretoMaxX) - float(paretoMinX)) * (float(paretoMaxY) - float(paretoMinY))
                hyperVolume.append(value4HyperVolume)

                perc_dominanceFSGA = float(float(FSdominatesSOL) / float(len(oneIterationData['all_paretox'][i])))
                coverageFSGA.append(perc_dominanceFSGA)

                perc_dominance1CGA = float(float(one1CdominatesSOL) / float(len(oneIterationData['all_paretox'][i])))
                coverage1CGA.append(perc_dominance1CGA)

                if (SOLdominates1C == 1) and (SOLdominatesFS == 1):
                    smallestGenerationDominates = min(smallestGenerationDominates, i + 1)

                coverageGA1C.append(SOLdominates1C)
                coverageGAFS.append(SOLdominatesFS)

            experimentLabels.append(oneExperiment)
            experimentSmallestGeneration.append(smallestGenerationDominates)

    data4TableAnalysis = {'experimentLabels': experimentLabels,
                          'experimentSmallestGeneration': experimentSmallestGeneration}
    df = pd.DataFrame(data4TableAnalysis)
    aggregatedColumns = ['experimentSmallestGeneration']
    mean_df = df.groupby('experimentLabels')[aggregatedColumns].mean()
    mean_df.reset_index(inplace=True)
    std_df = df.groupby('experimentLabels')[aggregatedColumns].std()
    std_df.reset_index(inplace=True)

    print(mean_df)
    print(std_df)
def generateScatter4AllGenerations(oneIterationData,meanCloserTo00X,meanCloserTo00Y,xlimitmin,ylimitmin,xlimit,ylimit):
    #PARA poder hacer los cálculos normalizados, esta función necesita que se haya
    # calculado anteriormente:
    # minX diffX
    # minY  diffY


    #
    # ###Recorremos todas las generaciones que se han ejecutado en este experimento concreto (para un num de nodos,
    # #un numero de apps y una rep concreta.
    for i in range(len(oneIterationData['all_paretox'])):
    #
    #     minDistance = float('inf')
    #     indxMin = -1
    #
    #     paretoMaxX = -1.0
    #     paretoMaxY = -1.0
    #     paretoMinX = float('inf')
    #     paretoMinY = float('inf')
    #
    #     SOLdominates1C = 0
    #     SOLdominatesFS = 0
    #
    #     one1CdominatesSOL = 0
    #     FSdominatesSOL = 0
    #
    #     print("Generation " + str(i))
    #     print("Pareto size " + str(len(oneIterationData['all_paretox'][i])))
    #
    #
    #     #Para cada una de las generaciones de este experimento, buscamos cual es la solucion que
    #     #esta mas cerca del centro de coordenadas, es decir aplicamos una regla de seleccion
    #     #de una instruccion del parteo front y buscamos cual es la solucion que la cumple
    #     #para posteriormente poder marcar esta solucion con un * en el scatter plot
    #     for j in range(len(oneIterationData['all_paretox'][i])):
    #         x_value = oneIterationData['all_paretox'][i][j]
    #         y_value = oneIterationData['all_paretoy'][i][j]
    #
    #
    #         #TODO, hoy 24/feb he cambiado lo que creo que es un error, ya que se restaba por el minY en lugar de por el minX
    #         #x_value = (x_value - minY) / diffX
    #         x_value = (x_value - minX) / diffX
    #         y_value = (y_value - minY) / diffY
    #
    #         paretoMinX = min(paretoMinX, x_value)
    #         paretoMinY = min(paretoMinY, y_value)
    #
    #         paretoMaxX = min(paretoMaxX, x_value)
    #         paretoMaxY = min(paretoMaxY, y_value)
    #
    #         distanceToCeroCero = math.sqrt(x_value ** 2 + y_value ** 2)
    #         if distanceToCeroCero < minDistance:
    #             minDistance = distanceToCeroCero
    #             indxMin = j
    #
    #     # meanCloserTo00X.append(oneIterationData['all_paretox'][i][j])
    #     # meanCloserTo00Y.append(oneIterationData['all_paretoy'][i][j])
    #     # TODO: SEGURO QUE NO TENGO QUE ESCOGER LA QUE ES INDICE J EN LUGAR DE indxMin ????
    #     # TODO: RESUELTO, LA QUE HAY QUE COGER ES indxMin
    #     meanCloserTo00X.append(oneIterationData['all_paretox'][i][indxMin])
    #     meanCloserTo00Y.append(oneIterationData['all_paretoy'][i][indxMin])
    #     meanCloserTo00.append(minDistance)

        ax = plt.subplot()
        plt.text(0.05, 0.98, "Generation " + str(i + 1), horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, fontsize=12, fontweight="bold")

        plt.text(0.95, 0.98, str(nodeNumber) + "nodes\n" + str(appNumber) + "apps", horizontalalignment='right',
                 verticalalignment='top', transform=ax.transAxes, fontsize=12, fontweight="bold")

        plt.xlim(xlimitmin, xlimit)
        plt.ylim(ylimitmin, ylimit)

        plt.scatter(oneIterationData['all_nonparetox'][i], oneIterationData['all_nonparetoy'][i])
        plt.scatter(oneIterationData['all_paretox'][i], oneIterationData['all_paretoy'][i])
        plt.scatter(controlCases['ot1c'], controlCases['rt1c'], marker='x', color='g')
        plt.scatter(controlCases['otCc'], controlCases['rtCc'], marker='x', color='r')

        plt.scatter(meanCloserTo00X[i], meanCloserTo00Y[i],
                    marker='*',
                    color='#000000')

        plt.title('Objective space')
        plt.xlabel("placement_time (ms)")
        if configuration.ILP_METHOD:
            plt.ylabel("Free resources")
        else:
            plt.ylabel("response_time (ms)")

        # plt.xlim(min_x, max_x)
        # plt.ylim(min_y, max_y)

        plt.savefig(storageFolder + "scaled_scatterPlot_GEN" + str(i) + "_APPS" + str(appNumber) + "_NODES" + str(nodeNumber) + ".pdf", format='pdf')
        plt.close()
def getParetoComparativeMetrics(oneIterationData):
    hyperVolume = list()
    coverageFSGA = list()
    coverage1CGA = list()
    coverageGAFS = list()
    coverageGA1C = list()


    for i in range(len(oneIterationData['all_paretox'])):


        paretoMaxX = -1.0
        paretoMaxY = -1.0
        paretoMinX = float('inf')
        paretoMinY = float('inf')

        SOLdominates1C = 0
        SOLdominatesFS = 0

        one1CdominatesSOL = 0
        FSdominatesSOL = 0

        print("Generation " + str(i))
        print("Pareto size " + str(len(oneIterationData['all_paretox'][i])))

        for j in range(len(oneIterationData['all_paretox'][i])):
            x_value = oneIterationData['all_paretox'][i][j]
            y_value = oneIterationData['all_paretoy'][i][j]

            if (controlCases['otCc'] <= x_value) and (controlCases['rtCc'] <= y_value):
                FSdominatesSOL += 1

            if (controlCases['ot1c'] <= x_value) and (controlCases['rt1c'] <= y_value):
                one1CdominatesSOL += 1

            if (x_value <= controlCases['otCc']) and (y_value <= controlCases['rtCc']):
                SOLdominatesFS = 1

            if (x_value <= controlCases['ot1c']) and (y_value <= controlCases['rt1c']):
                SOLdominates1C = 1


            x_value = (x_value - minY) / diffX
            y_value = (y_value - minY) / diffY

            paretoMinX = min(paretoMinX, x_value)
            paretoMinY = min(paretoMinY, y_value)

            paretoMaxX = min(paretoMaxX, x_value)
            paretoMaxY = min(paretoMaxY, y_value)



        value4HyperVolume = (float(paretoMaxX) - float(paretoMinX)) * (float(paretoMaxY) - float(paretoMinY))
        hyperVolume.append(value4HyperVolume)

        perc_dominance = float(float(FSdominatesSOL) / float(len(oneIterationData['all_paretox'][i])))
        coverageFSGA.append(perc_dominance)

        perc_dominance = float(float(one1CdominatesSOL) / float(len(oneIterationData['all_paretox'][i])))
        coverage1CGA.append(perc_dominance)

        coverageGA1C.append(SOLdominates1C)
        coverageGAFS.append(SOLdominatesFS)

    return hyperVolume[-1], coverageFSGA[-1], coverage1CGA[-1], coverageGAFS[-1], coverageGA1C[-1]
def plotRadarFigures(radarLabels, series2Plot, std_series2Plot, paletteValues, radarTitle, theLabels, fileName2Store,y_limit=None):

    # Set data
    # df = pd.DataFrame({
    # 'group': radarLabels,
    # 'var1': fsColonyOptTime,
    # 'var2': oneColoyOptTime,
    # 'var3': selectedFromParetoOptTime
    # })
    #


    my_palette = plt.cm.get_cmap("Set2", 4)

    # ------- PART 1: Create background

    # number of variable
    categories = radarLabels
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    #fig = plt.figure(figsize=(10, 10))
    #ax = fig.add_subplot(projection='polar')

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=8)
    if y_limit!=None:
        plt.ylim(y_limit[0], y_limit[1])

    ax.tick_params(axis='x', which='major', pad=10)

    # Draw ylabels
    # ax.set_rlabel_position(0)
    # plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
    # plt.ylim(0,40)


    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1

    # for i in range(len(series2Plot)):
    #     errorSerieMax = []
    #     errorSerieMin = []
    #     for j in range(len(series2Plot[i])):
    #         errorSerieMax.append(series2Plot[i][j]*1.1)
    #         errorSerieMin.append(series2Plot[i][j]*0.9)
    #     ax.plot(angles, series2Plot[i], linewidth=1, color="black", linestyle='solid', label=theLabels[i])
    #     ax.fill(angles, series2Plot[i], color=my_palette(paletteValues[i]), alpha=0.4)
    #
    #     ax.fill_between(angles, errorSerieMax, errorSerieMin, facecolor=my_palette(paletteValues[i]), alpha=1)


    for i in range(len(series2Plot)):
        errorSerieMax = []
        errorSerieMin = []
        for j in range(len(series2Plot[i])):
#            errorSerieMax.append(series2Plot[i][j]*1.1)
#            errorSerieMin.append(series2Plot[i][j]*0.9)
            #errorSerieMax.append(series2Plot[i][j] + std_series2Plot[i][j])
            #errorSerieMin.append(series2Plot[i][j] - std_series2Plot[i][j])
            errorSerieMax.append(std_series2Plot[i][j])
            errorSerieMin.append(std_series2Plot[i][j])
        ax.plot(angles, series2Plot[i], linewidth=1, color=my_palette(paletteValues[i]), linestyle='solid', label=theLabels[i])
        ax.fill(angles, series2Plot[i], color=my_palette(paletteValues[i]), alpha=0.4)

        #ax.errorbar(angles, series2Plot[i], yerr=[errorSerieMax, errorSerieMin], xerr=0.2,
#                    fmt='o', ecolor='g', capthick=2)

        #ax.errorbar(angles, series2Plot[i], xerr=0.2, yerr=0.4)

        #ax.errorbar(angles, series2Plot[i], yerr=errorSerieAvg, fmt='o', color=my_palette(paletteValues[i]), ecolor="black", capsize=7)
        ax.errorbar(angles, series2Plot[i], yerr=[errorSerieMin, errorSerieMax], fmt='o', color=my_palette(paletteValues[i]), ecolor=my_palette(paletteValues[i]), elinewidth=3)

    # plt.title('Values of the placement_time objective for the three algorithms')

    # Add legend
    # plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))


    plt.title(radarTitle, y=1.2, fontweight="bold")

    # Add legend


    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([plt.Rectangle((0, 0), 1, 1, fc=my_palette(paletteValues[i])) for i, handle in enumerate(handles)],
               [label for i, label in enumerate(labels)],
               handlelength=0.8, handleheight=0.8, fontsize=6, loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()

    # Show the graph
    plt.savefig(storageFoldershort + fileName2Store, format='pdf')
    plt.close()
def getBestSolution4EachGeneration(oneIterationData,minX,minY,maxX,maxY):
    #PARA poder hacer los cálculos normalizados, esta función necesita que se haya
    # calculado anteriormente:
    # minX maxX diffX
    # minY maxY  diffY


    meanCloserTo00 = list()
    meanCloserTo00X = list()
    meanCloserTo00Y = list()

    diffX = maxX - minX
    diffY = maxY - minY

    ###Recorremos todas las generaciones que se han ejecutado en este experimento concreto (para un num de nodos,
    #un numero de apps y una rep concreta.
    for i in range(len(oneIterationData['all_paretox'])):

        minDistance = float('inf')
        indxMin = -1


        print("Generation " + str(i))
        print("Pareto size " + str(len(oneIterationData['all_paretox'][i])))


        #Para cada una de las generaciones de este experimento, buscamos cual es la solucion que
        #esta mas cerca del centro de coordenadas, es decir aplicamos una regla de seleccion
        #de una instruccion del parteo front y buscamos cual es la solucion que la cumple
        #para posteriormente poder marcar esta solucion con un * en el scatter plot
        for j in range(len(oneIterationData['all_paretox'][i])):
            x_value = oneIterationData['all_paretox'][i][j]
            y_value = oneIterationData['all_paretoy'][i][j]


            #TODO, hoy 24/feb he cambiado lo que creo que es un error, ya que se restaba por el minY en lugar de por el minX
            #x_value = (x_value - minY) / diffX
            x_value = (x_value - minX) / diffX
            y_value = (y_value - minY) / diffY


            distanceToCeroCero = math.sqrt(x_value ** 2 + y_value ** 2)
            if distanceToCeroCero < minDistance:
                minDistance = distanceToCeroCero
                indxMin = j

        # meanCloserTo00X.append(oneIterationData['all_paretox'][i][j])
        # meanCloserTo00Y.append(oneIterationData['all_paretoy'][i][j])
        # TODO: SEGURO QUE NO TENGO QUE ESCOGER LA QUE ES INDICE J EN LUGAR DE indxMin ????
        # TODO: RESUELTO, LA QUE HAY QUE COGER ES indxMin
        meanCloserTo00X.append(oneIterationData['all_paretox'][i][indxMin])
        meanCloserTo00Y.append(oneIterationData['all_paretoy'][i][indxMin])
        meanCloserTo00.append(minDistance)


    return meanCloserTo00, meanCloserTo00X, meanCloserTo00Y



finalHyperVolume = list()
finalCoverage1CGA = list()
finalCoverageFSGA = list()
finalCoverageGA1C = list()
finalCoverageGAFS = list()


radarLabels = list()
fsColonyNetTime = list()
oneColoyNetTime = list()
selectedFromParetoNetTime = list()

fsColonyOptTime = list()
oneColoyOptTime = list()
selectedFromParetoOptTime = list()


for oneExperiment in configuration.experiments2execute:
    appNumber = oneExperiment[0]
    nodeNumber = oneExperiment[1]

    for repNum in configuration.rangeOfExperimentsRepetitions:
        storageFolder = getExperimentStorageFolder(nodeNumber,appNumber,repNum)


        with open(storageFolder + 'alldata.pickle', 'rb') as f:
            data2Store = pickle.load(f)

        oneIterationData = data2Store['oneIterationData']
        GAstructure4partition = data2Store['GAstructure4partition']
        #domainConfiguration = data2Store['domainConfiguration']
        controlCases = data2Store['controlCases']

        # ******************************************************************************************
        #   Plots creation
        # ******************************************************************************************


        # LAS X son los optimization times
        # las Y son los estimated execution (responses) times


        # xlimit =sum(oneIterationData['meanMaxX'])/len(oneIterationData['meanMaxX'])
        # ylimit =sum(oneIterationData['meanMaxY'])/len(oneIterationData['meanMaxY'])

        xlimit =max(sum(oneIterationData['meanMaxX']) / len(oneIterationData['meanMaxX']), controlCases['otCc'])
        ylimit = max(sum(oneIterationData['meanMaxY']) / len(oneIterationData['meanMaxY']), controlCases['rtCc'])




        # xlimit =max(sum(oneIterationData['meanMaxX'])/len(oneIterationData['meanMaxX']),ot1c,otCc)
        # ylimit =max(sum(oneIterationData['meanMaxY'])/len(oneIterationData['meanMaxY']),rt1c,rtCc)


        # xlimitmin =sum(oneIterationData['meanMinX)/len(oneIterationData['meanMinX)
        # ylimitmin =sum(oneIterationData['meanMinY)/len(oneIterationData['meanMinY)

        xlimitmin = min(sum(oneIterationData['meanMinX']) / len(oneIterationData['meanMinX']), controlCases['otCc'])
        ylimitmin = min(sum(oneIterationData['meanMinY']) / len(oneIterationData['meanMinY']), controlCases['rtCc'])

        # xlimitmin =min(sum(oneIterationData['meanMinX'])/len(oneIterationData['meanMinX']),ot1c,otCc)
        # ylimitmin =min(sum(oneIterationData['meanMinY'])/len(oneIterationData['meanMinY']),rt1c,rtCc)


        # asisDiffX = (xlimit-xlimitmin)*1.0
        # asisDiffY = (ylimit-ylimitmin)*1.0
        #
        #
        # asisDiffX = 0.0
        # asisDiffY = 0.0
        #
        #
        # xlimit=int(xlimit+asisDiffX)
        # ylimit=int(ylimit+asisDiffY)
        #
        # xlimitmin=int(xlimitmin-asisDiffX)
        # ylimitmin=int(ylimitmin-asisDiffY)


        xlimit = xlimit * 1.1
        ylimit = ylimit * 1.1

        xlimitmin = xlimitmin * 0.9
        ylimitmin = ylimitmin * 0.9

        # en oneIterationData['meanMinX'], etc... tenemos, para cada una de las genercacion, el minimo y maximo de
        #las soluciones de esa generacion. Si queremos los maximos y miniimos de toda la ejecucion es simplemente
        #coger el minimio y maximo respectivamente de eseas listas.

        minX = min(oneIterationData['meanMinX'])
        minY = min(oneIterationData['meanMinY'])
        maxX = max(oneIterationData['meanMaxX'])
        maxY = max(oneIterationData['meanMaxY'])

        diffX = maxX - minX
        diffY = maxY - minY


        #Para poder dibujar los scatter plots, primero debemos de saber los valores mínimos y máximos de toda la
        # ejecución, es decir, de todas las soluciones de todas las generaciones. Así podremos normalizar para encontrar
        # la solución más cercana al (0,0). Igualmente, para mantener una escala de los scatterplots homogenea, usamos
        #esos valores para determinar el límite del gráfico en su eje X e Y.

        meanCloserTo00, meanCloserTo00X, meanCloserTo00Y =  getBestSolution4EachGeneration(oneIterationData,minX,minY,maxX,maxY)
        generateScatter4AllGenerations(oneIterationData,meanCloserTo00X, meanCloserTo00Y, xlimitmin,ylimitmin,xlimit,ylimit)


        hV,cFSGA,c1CGA,cGAFS,cGA1c = getParetoComparativeMetrics(oneIterationData)

        finalHyperVolume.append(hV)
        finalCoverageFSGA.append(cFSGA)
        finalCoverage1CGA.append(c1CGA)
        finalCoverageGAFS.append(cGAFS)
        finalCoverageGA1C.append(cGA1c)



        radarLabels.append(str(nodeNumber) + "nodes\n" + str(appNumber) + "apps")
        fsColonyOptTime.append(controlCases['otCc'])
        oneColoyOptTime.append(controlCases['ot1c'])
        selectedFromParetoOptTime.append(meanCloserTo00X[-1])

        fsColonyNetTime.append(controlCases['rtCc'])
        oneColoyNetTime.append(controlCases['rt1c'])
        selectedFromParetoNetTime.append(meanCloserTo00Y[-1])



        # ******************************************************************************************
        #   END Plots creation
        # ******************************************************************************************

        oneIterationData['controlCases'] = controlCases
        oneIterationData['GAstructure4partition'] = GAstructure4partition
        #oneIterationData['domainConfiguration'] = domainConfiguration

        file = open(storageFolder + 'resultdata.pickle', 'wb')
        print("STORED VALUES IN "+storageFolder + 'resultdata.pickle')
        pickle.dump(oneIterationData, file)
        file.close()
        # with open(storageFolder+'resultdata.pickle', 'rb') as f:
        #    oneIterationData2 = pickle.load(f)

# ******************************************************************************************
#   END Main loud execution experiments for different number of apps and nodes
# ******************************************************************************************



radar = dict()
radar['radarLabels'] = radarLabels
radar['fsColonyOptTime'] = fsColonyOptTime
radar['oneColoyOptTime'] = oneColoyOptTime
radar['selectedFromParetoOptTime'] = selectedFromParetoOptTime
radar['fsColonyNetTime'] = fsColonyNetTime
radar['oneColoyNetTime'] = oneColoyNetTime
radar['selectedFromParetoNetTime'] = selectedFromParetoNetTime
radar['finalHyperVolume'] = finalHyperVolume
radar['finalCoverageFSGA'] = finalCoverageFSGA
radar['finalCoverage1CGA'] = finalCoverage1CGA
radar['finalCoverageGAFS'] = finalCoverageGAFS
radar['finalCoverageGA1C'] = finalCoverageGA1C

for i, v in enumerate(radar['radarLabels']):
    print(
        v + "  &  " + str(radar['finalHyperVolume'][i]) + "  &  " + str(radar['finalCoverageFSGA'][i]) + "  &  " + str(
            radar['finalCoverage1CGA'][i]) + "  &  " + str(radar['finalCoverageGAFS'][i]) + "  &  " + str(
            radar['finalCoverageGA1C'][i]) + "\\\\ \midrule")

file = open(storageFoldershort + 'radarplot.pickle', 'wb')
pickle.dump(radar, file)
file.close()



drawAllRadarPlots()
calculateSmallestGenerationDomination()
