import sklearn
import datetime
import sys
import pandas as pd
import numpy as np


from bvWorldEvo import *
from bvLifeEvo import *

import string
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
#'file' + id_generator()

def makeLongEpochStats(df, saveDir, scaleFactor=10, learningCutoff=0):
    # define the stats to summarize
#    stats = ['deadWorld','wolfEn', 'wolfRe', 'wolfFa', 'rabbitEn', 'rabbitRe', 'rabbitFa',
#           'wolfNum', 'rabbitNum', 'grassNum', 'debrisNum']
    stats = ['deadWorld','SubmergedEnergy','SubmergedSprawl', 'SubmergedConsumption', 'floatingEnergy', 'floatingSprawl', 'floatingConsumption', 'aquaticEnergy','aquaticSprawl','aquaticConsumption','herbEnergy','herbSprawl','herbConsumption','woodyEnergy','woodySprawl','woodyConsumption','SubmergedNum','floatingNum','aquaticNum','herbNum','woodyNum','waterNum','debrisNum']
    # trim the data to the observations we want to use
    dfLen = df.shape[0]
    df = df.iloc[learningCutoff:dfLen]
    # create label vector
    labels = []
    for num in range(1, scaleFactor+1):
        labels += [num for x in range(int(np.ceil(dfLen/scaleFactor)))]
    # assign label vector to column
    df['labels'] = labels[:dfLen]
    # convert to long format
    df_long = pd.melt(df, id_vars=['labels'], value_vars=stats)
    # summarize each stat by label
    df_mean = df_long.groupby(['labels','variable'])['value'].mean()
    # write to csv
    pd.DataFrame(df_mean).to_csv(saveDir + '/epochStats-long.csv')

######## PARAMETERS FOR LOADING DATA AND MODELING
#'''
#simCols = ['years','firstExt', 'firstExtSTD', 'deadWorld', 'deadWorldSTD', 'id',
#      'wolfEn',
#      'wolfRe',
#      'wolfFa',
#      'rabbitEn',
#      'rabbitRe',
#      'rabbitFa',
#      'wolfNum',
#      'rabbitNum',
#      'grassNum',
#      'debrisNum']
#'''
'''
simCols = ['years','firstExt', 'firstExtSTD', 'deadWorld', 'deadWorldSTD', 'id',
      'SubmergedEnergy',
      'SubmergedSprawl',
      'SubmergedConsumption',
      'floatingEnergy',
      'floatingSprawl',
      'floatingConsumption',
      'aquaticEnergy',
      'aquaticSprawl',
      'aquaticConsumption',
      'herbEnergy',
      'herbSprawl',
      'herbConsumption',
      'woodyEnergy',
      'woodySprawl',
      'woodyConsumption',
      'SubmergedNum',
      'floatingNum',
      'aquaticNum',
      'herbNum',
      'woodyNum',
      'waterNum',
      'debrisNum']
'''
yList = ['firstExt']
#xList = ['wolfEn',
#          'wolfRe',
#          'wolfFa',
#          'rabbitEn',
#          'rabbitRe',
#          'rabbitFa',
#          'wolfNum',
#          'rabbitNum',
#          'grassNum',
#          'debrisNum']

xList = ['SubmergedEnergy',
          'SubmergedSprawl',
          'SubmergedConsumption',
          'floatingEnergy',
          'floatingSprawl',
          'floatingConsumption',
          'aquaticEnergy',
          'aquaticSprawl',
          'aquaticConsumption',
          'herbEnergy',
          'herbSprawl',
          'herbConsumption',
          'woodyEnergy',
          'woodySprawl',
          'woodyConsumption',
          'SubmergedNum',
          'floatingNum',
          'aquaticNum',
          'herbNum',
          'woodyNum',
          'waterNum',
          'debrisNum']

#def learnParamsRF(optsNum, years,
#          saveDir,
#          wolfEn,
#          wolfRe,
#          wolfFa,
#          rabbitEn,
#          rabbitRe,
#          rabbitFa,
#          wolfNum,
#          rabbitNum,
#          grassNum,
#          debrisNum,
#          xList,
#          yList):


def learnParamsRF(optsNum, years,
          saveDir,
          SubmergedEnergy,
          SubmergedSprawl,
          SubmergedConsumption,
          floatingEnergy,
          floatingSprawl,
          floatingConsumption,
          aquaticEnergy,
          aquaticSprawl,
          aquaticConsumption,
          herbEnergy,
          herbSprawl,
          herbConsumption,
          woodyEnergy,
          woodySprawl,
          woodyConsumption,
          SubmergedNum,
          floatingNum,
          aquaticNum,
          herbNum,
          woodyNum,
          waterNum,
          debrisNum,
          xList,
          yList):
    # get the latest sim data
    simDF = pd.read_csv(saveDir + '/epochStats.csv')
    # check if we've reached successful stasis (10 in a row that hit the max years)
    if min(simDF.iloc[-10:]['deadWorld']) == years:
        return ['END', simDF.shape[0]]

    # train the model
    rfModel=RandomForestRegressor(n_estimators=300,
                            max_depth=None,
                            max_features=.8,
                            min_samples_split=1,
                            #random_state=0,
                            n_jobs=-1)

    rfModel.fit(simDF[xList], np.array(simDF[yList]).ravel())

    # get options
    xs = []
#    for i in range(0,optsNum):
#        we = max(int(wolfEn + (np.random.randn(1)[0] * 10)), 100) # minimum of 100
#        wr = int(wolfRe + (np.random.randn(1)[0] * 15))
#        if wr < we * 1.1:
#            wr = we * 1.1
#        wf = max(int(wolfFa + (np.random.randn(1)[0] * 5)), 5) # minimum of 5

#        re = max(int(rabbitEn + (np.random.randn(1)[0] * 10)), 25) # minimum of 25
#        rr = int(rabbitRe + (np.random.randn(1)[0] * 10))
#        if rr < re * 1.1:
#            rr = re * 1.1
#        rf = max(int(rabbitFa + (np.random.randn(1)[0] * 5)), 5) # minimum of 5

        # minumum of 1 for each of these
#        wn = max(int(wolfNum + (np.random.randn(1)[0] * 3)), 1)
#        rn = max(int(rabbitNum + (np.random.randn(1)[0] * 5)), 1)
#        gn = max(int(grassNum + (np.random.randn(1)[0] * 10)), 1)
#        dn = max(int(debrisNum + (np.random.randn(1)[0] * 10)), 1)
#        xs.append([we, wr, wf, re, rr, rf, wn, rn, gn, dn])

    for i in range(0,optsNum):
        woody_energy = max(int(woodyEnergy + (np.random.randn(1)[0] * 10)), 100) # minimum of 100
        woody_sprawl = int(woodySprawl + (np.random.randn(1)[0] * 15))
        if woody_sprawl < woody_energy * 2:
            woody_sprawl = woody_energy * 2
        woody_energy_consumption = max(int(woodyConsumption + (np.random.randn(1)[0] * 5)), 5) # minimum of 5

        herb_energy = max(int(herbEnergy + (np.random.randn(1)[0] * 10)), 100) # minimum of 100
        herb_sprawl = int(herbSprawl + (np.random.randn(1)[0] * 15))
        if herb_sprawl < herb_energy * 1.8:
            herb_sprawl = herb_energy * 1.8
        herb_energy_consumption = max(int(herbConsumption + (np.random.randn(1)[0] * 5)), 5) # minimum of 5

        aqu_energy = max(int(aquaticEnergy + (np.random.randn(1)[0] * 10)), 25) # minimum of 25
        aqu_energy = int(aquaticSprawl + (np.random.randn(1)[0] * 10))
        if aqu_sprawl < aqu_energy * 1.6:
            aqu_sprawl = aqu_energy * 1.6
        aqu_energy_consumption = max(int(aquaticConsumption + (np.random.randn(1)[0] * 5)), 5) # minimum of 5

        flo_energy = max(int(floatingEnergy + (np.random.randn(1)[0] * 10)), 100) # minimum of 100
        flo_sprawl = int(floatingSprawl + (np.random.randn(1)[0] * 15))
        if flo_sprawl < flo_energy * 1.4:
            flo_sprawl = flo_energy * 1.4
        flo_energy_consumption = max(int(floatingConsumption + (np.random.randn(1)[0] * 5)), 5) # minimum of 5

        sub_energy = max(int(SubmergedEnergy + (np.random.randn(1)[0] * 10)), 25) # minimum of 25
        sub_sprawl = int(SubmergedSprawl + (np.random.randn(1)[0] * 10))
        if sub_sprawl < sub_energy * 1.1:
            sub_sprawl = sub_energy * 1.1
        sub_energy_consumption = max(int(SubmergedConsumption + (np.random.randn(1)[0] * 5)), 5) # minimum of 5

        # minumum of 1 for each of these
        woody = max(int(woodyNum + (np.random.randn(1)[0] * 3)), 1)
        herb = max(int(herbNum + (np.random.randn(1)[0] * 5)), 1)
        aquatic = max(int(aquaticNum + (np.random.randn(1)[0] * 6)), 1)
        floating = max(int(floatingNum + (np.random.randn(1)[0] * 8)), 1)
        Submerged = max(int(SubmergedNum + (np.random.randn(1)[0] * 10)), 1)
        WetlandorPond = max(int(waterNum + (np.random.randn(1)[0] * 10)), 1)
        dn = max(int(debrisNum + (np.random.randn(1)[0] * 10)), 1)
        xs.append([woody_energy, woody_Sprawl, woody_Consumption , herb_energy, herb_sprawl, herb_energy_consumption, aqu_energy, aqu_sprawl, aqu_energy_consumption, flo_energy, flo_sprawl, flo_energy_consumption, sub_energy, sub_sprawl, sub_energy_consumption, woody, herb, aquatic, floating, submerged, WetlandorPond, dn])

    optsDF = pd.DataFrame(xs, columns = xList)

    # predict the best of the options
    optsDF['preds'] = rfModel.predict(optsDF)
    winner = optsDF[optsDF.preds == max(optsDF.preds)]

    # returns the parameter values for the best option (as well as the predicted firstExt)
    return winner

#def learnParamsLM(saveDir,
#          years,
#          wolfEn,
#          wolfRe,
#          wolfFa,
#          rabbitEn,
#          rabbitRe,
#          rabbitFa,
#          wolfNum,
#          rabbitNum,
#          grassNum,
#          debrisNum,
#          xList,
#          yList, incremental = True):
def learnParamsLM(saveDir,
          years,
          SubmergedEnergy,
          SubmergedSprawl,
          SubmergedConsumption,
          floatingEnergy,
          floatingSprawl,
          floatingConsumption,
          aquaticEnergy,
          aquaticSprawl,
          aquaticConsumption,
          herbEnergy,
          herbSprawl,
          herbConsumption,
          woodyEnergy,
          woodySprawl,
          woodyConsumption,
          SubmergedNum,
          floatingNum,
          aquaticNum,
          herbNum,
          woodyNum,
          waterNum,
          debrisNum,
          xList,
          yList, incremental = True):
    # get the latest sim data
    simDF = pd.read_csv(saveDir + '/epochStats.csv')
    # check if we've reached successful stasis (10 in a row that hit max years)
    if min(simDF.iloc[-10:]['deadWorld']) == years:
        return ['END', simDF.shape[0]]

    # train the model
    lm = LinearRegression().fit(simDF[xList], np.array(simDF[yList]).ravel())

    # round coefficients to 1 or -1, or the nearest integer if |coef| > 1
    def pushToInt(num):
        if 0 < num <= 1:
            return int(1)
        elif 0 > num >= -1:
            return int(-1)
        else:
            return int(round(num, 0))

    def returnSign(num):
        if 0 < num:
            return int(1)
        elif 0 > num :
            return int(-1)
        else:
            return 0

    if incremental == True:
        # push to nearest int, multiply first six by 5
        adjustments = [returnSign(x)*5 for x in lm.coef_[0:6]] + [returnSign(x) for x in lm.coef_[6:]]
    else:
        # for first six, multiply by 10
        # for critter numbers just push to nearest int
        adjustments = [int(x*10) for x in lm.coef_[0:6]] + [pushToInt(x) for x in lm.coef_[6:]]

    # add this list to previous parameters to adjust each iteration
    return adjustments

#def testLife(saveDir,
#        years,
#        wolfEn,
#        wolfRe,
#        wolfFa,
#        rabbitEn,
#        rabbitRe,
#        rabbitFa,
#        wolfNum,
#        rabbitNum,
#        grassNum,
#        debrisNum,
#        endOnExtinction = True,
#        endOnOverflow = True,
#        saveParamStats = False,
#        savePlotDF = False,
#        epochNum = 1):
#    testStats = []
def testLife(saveDir,
        years,
        SubmergedEnergy,
        SubmergedSprawl,
        SubmergedConsumption,
        floatingEnergy,
        floatingSprawl,
        floatingConsumption,
        aquaticEnergy,
        aquaticSprawl,
        aquaticConsumption,
        herbEnergy,
        herbSprawl,
        herbConsumption,
        woodyEnergy,
        woodySprawl,
        woodyConsumption,
        SubmergedNum,
        floatingNum,
        aquaticNum,
        herbNum,
        woodyNum,
        waterNum,
        debrisNum,
        endOnExtinction = True,
        endOnOverflow = True,
        saveParamStats = False,
        savePlotDF = False,
        epochNum = 1):
    testStats = []
    #create an instance of World
    bigValley = World(5,saveDir)
    print(datetime.datetime.now())

    #define your life forms
#    newLife(Predator('wolf', energy = wolfEn, repro = wolfRe, fatigue = wolfFa),
#            bigValley, 'wolf') # adding in the parameters from above
#    newLife(Prey('rabbit', energy = rabbitEn, repro = rabbitRe, fatigue = rabbitFa),
#            bigValley, 'rabbit') # adding in the parameters from above
#    newLife(Plant('grass'), bigValley, 'grass')
#    newLife(Rock('debris'), bigValley, 'debris')

    #define your life forms
    newLife(Predator('floating', energy = floatingEnergy, repro = floatingSprawl, fatigue = floatingConsumption),
            bigValley, 'floating') # adding in the parameters from above
    newLife(Prey('Submerged', energy = SubmergedEnergy, repro = SubmergedSprawl, fatigue = SubmergedConsumption),
            bigValley, 'SubmergedEnergy') # adding in the parameters from above
    newLife(Prey('aquatic', energy = aquaticEnergy, repro = aquaticSprawl, fatigue = aquaticConsumption),
            bigValley, 'aquatic') # adding in the parameters from above
    newLife(Prey('herb', energy = herbEnergy, repro = herbSprawl, fatigue = herbConsumption),
            bigValley, 'herb') # adding in the parameters from above
    newLife(Prey('woody', energy = woodyEnergy, repro = woodySprawl, fatigue = woodyConsumption),
            bigValley, 'woody') # adding in the parameters from above
    newLife(Plant('water'), bigValley, 'water')
    newLife(Rock('debris'), bigValley, 'debris')


    #now populate the world
#    populate(bigValley, 'wolf', wolfNum)
#    populate(bigValley, 'rabbit', rabbitNum)
#    populate(bigValley, 'grass', grassNum)
#    populate(bigValley, 'debris', debrisNum)

    populate(bigValley, 'floating', floatingNum)
    populate(bigValley, 'Submerged', SubmergedNum)
    populate(bigValley, 'aquatic', aquaticNum)
    populate(bigValley, 'herb', herbNum)
    populate(bigValley, 'woody', woodyNum)
    populate(bigValley, 'water', waterNum)
    populate(bigValley, 'debris', debrisNum)

    #now run the test
    test = bigValley.silentTime(years,
        yearlyPrinting = True,
        endOnExtinction = endOnExtinction,
        endOnOverflow = endOnOverflow,
        saveParamStats = saveParamStats,
        savePlotDF = savePlotDF,
        epochNum = epochNum)
    testStats.append(test)
    print('testStats ::: ' + str(testStats))

    # return stats for each test
    testDF = pd.DataFrame(testStats, columns=['firstExt', 'deadWorld', 'id'])

    #return testStats
    print(testDF)
    return testDF



# function to count the lines in a file (for seeing how many epochs have been logged)
'''
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    	#return i + 1
    	print(str(fname) + ' is now ' + str(i + 1) + ' lines long.')
'''

def runSim(saveDir,
          years,
          SubmergedEnergy,
          SubmergedSprawl,
          SubmergedConsumption,
          floatingEnergy,
          floatingSprawl,
          floatingConsumption,
          aquaticEnergy,
          aquaticSprawl,
          aquaticConsumption,
          herbEnergy,
          herbSprawl,
          herbConsumption,
          woodyEnergy,
          woodySprawl,
          woodyConsumption,
          SubmergedNum,
          floatingNum,
          aquaticNum,
          herbNum,
          woodyNum,
          waterNum,
          debrisNum,
          endOnExtinction=True,
          endOnOverflow=True,
          saveParamStats=False,
          savePlotDF = False,
          epochNum = 1):
    start=datetime.datetime.now()
    #file_name = 'data/' + str(tests) + 'x' + str(years) + '-' + id_generator() + '.csv'
    testDF = testLife(saveDir,
        years,
        SubmergedEnergy,
        SubmergedSprawl,
        SubmergedConsumption,
        floatingEnergy,
        floatingSprawl,
        floatingConsumption,
        aquaticEnergy,
        aquaticSprawl,
        aquaticConsumption,
        herbEnergy,
        herbSprawl,
        herbConsumption,
        woodyEnergy,
        woodySprawl,
        woodyConsumption,
        SubmergedNum,
        floatingNum,
        aquaticNum,
        herbNum,
        woodyNum,
        waterNum,
        debrisNum,
        endOnExtinction = endOnExtinction,
        endOnOverflow = endOnOverflow,
        saveParamStats = saveParamStats,
        savePlotDF = savePlotDF,
        epochNum = epochNum)
    thisSim = [
        years,
        round(np.mean(testDF['firstExt']), 2), # first extinction
        round(np.std(testDF['firstExt']), 2),
        round(np.mean(testDF['deadWorld']), 2), # dead world
        round(np.std(testDF['deadWorld']), 2),
        ','.join(testDF['id'].tolist()),
        SubmergedEnergy,
        SubmergedSprawl,
        SubmergedConsumption,
        floatingEnergy,
        floatingSprawl,
        floatingConsumption,
        aquaticEnergy,
        aquaticSprawl,
        aquaticConsumption,
        herbEnergy,
        herbSprawl,
        herbConsumption,
        woodyEnergy,
        woodySprawl,
        woodyConsumption,
        SubmergedNum,
        floatingNum,
        aquaticNum,
        herbNum,
        woodyNum,
        waterNum,
        debrisNum]
    print(thisSim)
    #open the file
    file = open(saveDir + '/epochStats.csv', "a")
    #write the new line
    file.write(str(thisSim).strip('[]') + '\n')
    #close the file
    file.close()

    #print the number of lines logged to the file
    #epochsRun = len(open(saveDir + '/epochStats.csv').readlines())
    epochStats = pd.read_csv(saveDir + '/epochStats.csv')
    epochsRun = epochStats.shape[0]
    print("%d EPOCHS HAVE BEEN RUN SO FAR" % epochsRun)

    # if more than 20, calculate long stats
    if epochsRun > 20:
        makeLongEpochStats(epochStats, saveDir)

    print(datetime.datetime.now()-start)
    print('%%%%%%%%')

##############
def continentLife(saveDir,
        years,
        idList,
        endOnExtinction = True,
        endOnOverflow = True,
        saveParamStats = False,
        savePlotDF = False,
        epochNum = 1):
    testStats = []

    #create an instance of World
    bigValley = World(5, saveDir)
    print(datetime.datetime.now())

    #### load continent stats
    contDF = pd.read_csv('testData/continentStats.csv', index_col=0)

    if idList == 'random': #######haven't tested this #######################
        idList = np.random.choice(contDF.index, 3) ### pick three ###########

    ##### CURRENTLY ONLY WORKS FOR 3 #####
    offsets = [[-25,-25], [25,-25], [0,17]]

    # call the stats for each id and feed them in
    for i, runId in enumerate(idList):
        #define your life forms
#        '''
#        newLife(Predator('wolf', energy = contDF.loc[runId,'wolfEn'], repro = contDF.loc[runId,'wolfRe'], fatigue = contDF.loc[runId,'wolfFa']),
#                bigValley, 'wolf') # adding in the parameters from above
#        newLife(Prey('rabbit', energy = contDF.loc[runId,'rabbitEn'], repro = contDF.loc[runId,'rabbitRe'], fatigue = contDF.loc[runId,'rabbitFa']),
#                bigValley, 'rabbit') # adding in the parameters from above
#        newLife(Plant('grass'), bigValley, 'grass')
#        newLife(Rock('debris'), bigValley, 'debris')

        newLife(Predator('floating', energy = contDF.loc[runId,'floatingEnergy'], repro = contDF.loc[runId,'floatingSprawl'], fatigue = contDF.loc[runId,'floatingConsumption']),
                bigValley, 'floating') # adding in the parameters from above
        newLife(Predator('aquatic', energy = contDF.loc[runId,'aquaticEnergy'], repro = contDF.loc[runId,'aquaticSprawl'], fatigue = contDF.loc[runId,'aquaticConsumption']),
                bigValley, 'aquatic') # adding in the parameters from above
        newLife(Predator('herb', energy = contDF.loc[runId,'herbEnergy'], repro = contDF.loc[runId,'herbSprawl'], fatigue = contDF.loc[runId,'herbConsumption']),
                bigValley, 'herb') # adding in the parameters from above
        newLife(Predator('woody', energy = contDF.loc[runId,'woodyEnergy'], repro = contDF.loc[runId,'woodySprawl'], fatigue = contDF.loc[runId,'woodyConsumption']),
                bigValley, 'woody') # adding in the parameters from above
        newLife(Pray('Submerged', energy = contDF.loc[runId,'SubmergedEnergy'], repro = contDF.loc[runId,'SubmergedSprawl'], fatigue = contDF.loc[runId,'SubmergedConsumption']),
                bigValley, 'Submerged') # adding in the parameters from above
        newLife(Plant('water'), bigValley, 'water')
        newLife(Rock('debris'), bigValley, 'debris')


        #now populate the world
        populate(bigValley, 'floating', contDF.loc[runId,'floatingNum'], offset = offsets[i])
        populate(bigValley, 'Submerged', contDF.loc[runId,'SubmergedNum'], offset = offsets[i])
        populate(bigValley, 'aquatic', contDF.loc[runId,'aquatic'], offset = offsets[i])
        populate(bigValley, 'herb', contDF.loc[runId,'herbNum'], offset = offsets[i])
        populate(bigValley, 'woody', contDF.loc[runId,'woodyNum'], offset = offsets[i])
        populate(bigValley, 'water', contDF.loc[runId,'waterNum'], offset = offsets[i])
        populate(bigValley, 'debris', contDF.loc[runId,'debrisNum'], offset = offsets[i])


        '''
        newLife(Predator(runId + 'floating', energy = contDF.loc[runId,'floatingEnergy'], repro = contDF.loc[runId,'floatingSprawl'], fatigue = contDF.loc[runId,'floatingConsumption']),
                bigValley, runId + 'floating') # adding in the parameters from above
        newLife(Predator(runId + 'aquatic', energy = contDF.loc[runId,'aquaticEnergy'], repro = contDF.loc[runId,'aquaticSprawl'], fatigue = contDF.loc[runId,'aquaticConsumption']),
                bigValley, runId + 'aquatic') # adding in the parameters from above
        newLife(Predator(runId + 'herb', energy = contDF.loc[runId,'herbEnergy'], repro = contDF.loc[runId,'herbSprawl'], fatigue = contDF.loc[runId,'herbConsumption']),
                bigValley, runId + 'herb') # adding in the parameters from above
        newLife(Predator(runId + 'woody', energy = contDF.loc[runId,'woodyEnergy'], repro = contDF.loc[runId,'woodySprawl'], fatigue = contDF.loc[runId,'woodyConsumption']),
                bigValley, runId + 'woody') # adding in the parameters from above
        newLife(Prey(runId + 'submerged', energy = contDF.loc[runId,'SubmergedEnergy'], repro = contDF.loc[runId,'SubmergedSprawl'], fatigue = contDF.loc[runId,'SubmergedConsumption']),
                bigValley, runId + 'submerged') # adding in the parameters from above
        newLife(Plant('water'), bigValley, 'water')
        newLife(Rock('debris'), bigValley, 'debris')

        #now populate the world
        populate(bigValley, runId + 'floating', contDF.loc[runId,'floatingNum'], offset = offsets[i])
        populate(bigValley, runId + 'aquatic', contDF.loc[runId,'aquatic'], offset = offsets[i])
        populate(bigValley, runId + 'herb', contDF.loc[runId,'herbNum'], offset = offsets[i])
        populate(bigValley, runId + 'woody', contDF.loc[runId,'woodyNum'], offset = offsets[i])
        populate(bigValley, runId + 'submerged', contDF.loc[runId,'SubmergedNum'], offset = offsets[i])
        populate(bigValley, 'water', contDF.loc[runId,'waterNum'], offset = offsets[i])
        populate(bigValley, 'debris', contDF.loc[runId,'debrisNum'], offset = offsets[i])
        print('offset' + str(i) + str(offsets[i]))

    #now run the test
    test = bigValley.silentTime(years,
        yearlyPrinting = True,
        endOnExtinction = endOnExtinction,
        endOnOverflow = endOnOverflow,
        saveParamStats = saveParamStats,
        savePlotDF = savePlotDF,
        continents = True,
        epochNum = epochNum)
    testStats.append(test)
    print('testStats ::: ' + str(testStats))

    # return stats for each test
    testDF = pd.DataFrame(testStats, columns=['firstExt', 'deadWorld', 'id'])

    #return testStats
    print(testDF)
    return testDF



#        '''
#        newLife(Predator(runId + 'wolf', energy = contDF.loc[runId,'wolfEn'], repro = contDF.loc[runId,'wolfRe'], fatigue = contDF.loc[runId,'wolfFa']),
#                bigValley, runId + 'wolf') # adding in the parameters from above
#        newLife(Prey(runId + 'rabbit', energy = contDF.loc[runId,'rabbitEn'], repro = contDF.loc[runId,'rabbitRe'], fatigue = contDF.loc[runId,'rabbitFa']),
#                bigValley, runId + 'rabbit') # adding in the parameters from above
#        newLife(Plant('grass'), bigValley, 'grass')
#        newLife(Rock('debris'), bigValley, 'debris')

        #now populate the world
#        populate(bigValley, runId + 'wolf', contDF.loc[runId,'wolfNum'], offset = offsets[i])
#        populate(bigValley, runId + 'rabbit', contDF.loc[runId,'rabbitNum'], offset = offsets[i])
#        populate(bigValley, 'grass', contDF.loc[runId,'grassNum'], offset = offsets[i])
#        populate(bigValley, 'debris', contDF.loc[runId,'debrisNum'], offset = offsets[i])
#        print('offset' + str(i) + str(offsets[i]))
    #now run the test
#    test = bigValley.silentTime(years,
#        yearlyPrinting = True,
#        endOnExtinction = endOnExtinction,
#        endOnOverflow = endOnOverflow,
#        saveParamStats = saveParamStats,
#        savePlotDF = savePlotDF,
#        continents = True,
#        epochNum = epochNum)
#    testStats.append(test)
#    print('testStats ::: ' + str(testStats))

    # return stats for each test
#    testDF = pd.DataFrame(testStats, columns=['firstExt', 'deadWorld', 'id'])

    #return testStats
#    print(testDF)
#    return testDF



##############

def runSimLearningRF1(saveDir,
          years,
          SubmergedEnergy,
          SubmergedSprawl,
          SubmergedConsumption,
          floatingEnergy,
          floatingSprawl,
          floatingConsumption,
          aquaticEnergy,
          aquaticSprawl,
          aquaticConsumption,
          herbEnergy,
          herbSprawl,
          herbConsumption,
          woodyEnergy,
          woodySprawl,
          woodyConsumption,
          SubmergedNum,
          floatingNum,
          aquaticNum,
          herbNum,
          woodyNum,
          waterNum,
          debrisNum,
          optsNum=25,
          endOnExtinction=True,
          endOnOverflow=True,
          saveParamStats=False,
          savePlotDF = False,
          epochNum = 1):
    start=datetime.datetime.now()

    ####### DO THE LEARNING

    winner = learnParamsRF(optsNum,
          saveDir,
          SubmergedEnergy,
          SubmergedSprawl,
          SubmergedConsumption,
          floatingEnergy,
          floatingSprawl,
          floatingConsumption,
          aquaticEnergy,
          aquaticSprawl,
          aquaticConsumption,
          herbEnergy,
          herbSprawl,
          herbConsumption,
          woodyEnergy,
          woodySprawl,
          woodyConsumption,
          SubmergedNum,
          floatingNum,
          aquaticNum,
          herbNum,
          woodyNum,
          waterNum,
          debrisNum,
          xList,
          yList)

    # if we've reached successful stasis (10 in a row that hit max years)
    if isinstance(winner, (list)):
        return winner

    ####### RUN THE SIM

    testDF = testLife(saveDir,
        years,
        int(winner['SubmergedEnergy']),
        int(winner['SubmergedSprawl']),
        int(winner['SubmergedConsumption']),
        int(winner['floatingEnergy']),
        int(winner['floatingSprawl']),
        int(winner['floatingConsumption']),
        int(winner['aquaticEnergy']),
        int(winner['aquaticSprawl']),
        int(winner['aquaticConsumption']),
        int(winner['herbEnergy']),
        int(winner['herbSprawl']),
        int(winner['herbConsumption']),
        int(winner['woodyEnergy']),
        int(winner['woodySprawl']),
        int(winner['woodyConsumption']),
        int(winner['SubmergedNum']),
        int(winner['floatingNum']),
        int(winner['aquaticNum']),
        int(winner['herbNum']),
        int(winner['woodyNum']),
        int(winner['waterNum']),
        int(winner['debrisNum']),
        endOnExtinction = endOnExtinction,
        endOnOverflow = endOnOverflow,
        saveParamStats = saveParamStats,
        savePlotDF = savePlotDF,
        epochNum = epochNum)
    thisSim = [
        years,
        round(np.mean(testDF['firstExt']), 2), # first extinction
        round(np.std(testDF['firstExt']), 2),
        round(np.mean(testDF['deadWorld']), 2), # dead world
        round(np.std(testDF['deadWorld']), 2),
        ','.join(testDF['id'].tolist()),
        int(winner['SubmergedEnergy']),
        int(winner['SubmergedSprawl']),
        int(winner['SubmergedConsumption']),
        int(winner['floatingEnergy']),
        int(winner['floatingSprawl']),
        int(winner['floatingConsumption']),
        int(winner['aquaticEnergy']),
        int(winner['aquaticSprawl']),
        int(winner['aquaticConsumption']),
        int(winner['herbEnergy']),
        int(winner['herbSprawl']),
        int(winner['herbConsumption']),
        int(winner['woodyEnergy']),
        int(winner['woodySprawl']),
        int(winner['woodyConsumption']),
        int(winner['SubmergedNum']),
        int(winner['floatingNum']),
        int(winner['aquaticNum']),
        int(winner['herbNum']),
        int(winner['woodyNum']),
        int(winner['waterNum']),
        int(winner['debrisNum']),int(winner['preds'])]
    print(thisSim)
    print('$$$$$\n PREDICTED FirstExt: ' + str(int(winner['preds'])) + '\n$$$$$')
    #open the file
    file = open(saveDir + '/epochStats.csv', "a")
    #write the new line
    file.write(str(thisSim).strip('[]') + '\n')
    #print the number of lines logged to the file
    #file_len(file_name)
    print("%d lines in your choosen file" % len(open(saveDir + '/epochStats.csv').readlines()))
    ##print "%d lines in your choosen file" % len(file.readlines())

    #close the file
    file.close()
    print(datetime.datetime.now()-start)
    print('%%%%%%%%')
    return 'finished thisSim'
