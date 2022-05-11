# python bigValleyLearningD3LM.py 500 50 new plot
import sys
import os
import time
import sklearn
# print(sys.argv)
os.chdir(sys.path[0])

sys.path.append('./bvSimFiles/') # Add location of local packages to path
#sys.path.append('/Users/Seth/Documents/bigValley-Python/bvSimFiles/') # Add location of local packages to path

from bvSimLearning import *
#from bvSimArchiver import *

import string
import random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

## SET PARAMETERS

#max number of years for each epoch
#print(sys.argv)
years = int(sys.argv[1]) # default is 500
# years = sys.argv[1] # default is 500
# years = sys.argv[0].encode()


# number of simulations to run in total before the the program quits
reps = int(sys.argv[2]) # default is 5
### NOTE: IF STARTING ANEW, it will run 500 dumb reps
#### and THEN start the prescribed number of learning reps

seedReps = 25
bigRun = False # whether to run one for 10,000 years at the end
# either give it 'new' to start over or the ID code of a past trial to continue
if sys.argv[3] == 'new':
  simID = id_generator(3)
  print('STARTING ANEW ')
else:
  simID = sys.argv[3]

# either give it 'plot' to save plotData or anything else to only save epochStats
if sys.argv[4] == 'plot':
  plotting = True
  print('PLOTTING!!!')
else:
  plotting = False
  print('NOT plotting!!!')

# make directory for storing
saveDir = 'plotData/LM-' + simID
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

# write file headers for the new file
epochStats = open(saveDir + '/epochStats.csv', "w")
epochStats.write('years,firstExt,firstExtSTD,deadWorld,deadWorldSTD,id,SubmergedEnergy,SubmergedSprawl,SubmergedConsumption,floatingEnergy,floatingSprawl,floatingConsumption,aquaticEnergy,aquaticSprawl,aquaticConsumption,herbEnergy,herbSprawl,herbConsumption,woodyEnergy,woodySprawl,woodyConsumption,SubmergedNum,floatingNum,aquaticNum,herbNum,woodyNum,waterNum,debrisNum\n')
epochStats.close()
#________________________________________________
#5.woody stats
woody_energy = 2000
woody_sprawl = 2300
woody_energy_consumption = 80

#4.herb stats
herb_energy = 1000
herb_sprawl = 1300
herb_energy_consumption = 60

#3.aquatic stats
aqu_energy = 600
aqu_sprawl = 800
aqu_energy_consumption = 40

#2.floating stats
#we = 300 # energy
flo_energy = 70
#wr = 400 # repro
flo_sprawl = 100
#wf = 20 # fatigue
flo_energy_consumption = 10

#1.submerged stats
#re = 70 # energy re
sub_energy = 70
#rr = 100 # repro
sub_sprawl = 100
#rf = 10 # fatigue
sub_energy_consumption = 10

#0.numbers of each critter
woody = 50
herb = 50
aquatic = 80
floating = 10
Submerged = 20
WetlandorPond = 1000 #water NUMBER
dn = 5  # ROCK

#__________________________________________________
######## PARAMETERS FOR LOADING DATA AND MODELING

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
#################
## RUN THE SIM ##
#################

########
# IF STARTING ANEW, run 500 dumb reps before fitting the initial model
########
#if sys.argv[3] == 'new':
#    for i in range(0, seedReps):
#        # set parameters for this run
#        wolfEn = int(we + (np.random.randn(1)[0] * 10))
#        wolfRe = int(wr + (np.random.randn(1)[0] * 15))
#        if wolfRe < wolfEn * 1.1:
#          wolfRe = wolfEn * 1.1
#        wolfFa = max(int(wf + (np.random.randn(1)[0] * 5)), 5) # minimum of 5
#
#        rabbitEn = int(re + (np.random.randn(1)[0] * 10))
#        rabbitRe = int(rr + (np.random.randn(1)[0] * 10))
#        if rabbitRe < rabbitEn * 1.1:
#          rabbitRe = rabbitEn * 1.1
#        rabbitFa = max(int(rf + (np.random.randn(1)[0] * 5)), 5) # minimum of 5
#
#        # minumum of 1 for each of these
#        wolfNum = max(int(wn + (np.random.randn(1)[0] * 3)), 1)
#        rabbitNum = max(int(rn + (np.random.randn(1)[0] * 5)), 1)
#        grassNum = max(int(gn + (np.random.randn(1)[0] * 10)), 1)
#        debrisNum = max(int(dn + (np.random.randn(1)[0] * 10)), 1)

if sys.argv[3] == 'new':
    for i in range(0, seedReps):
        # set parameters for this run
        floatingEnergy = int(flo_energy + (np.random.randn(1)[0] * 10))
        floatingSprawl = int(flo_sprawl + (np.random.randn(1)[0] * 15))
        if floatingSprawl < floatingEnergy * 100:
          floatingSprawl = floatingEnergy * 100
        floatingConsumption = max(int(flo_energy_consumption + (np.random.randn(1)[0] * 5)), 5) # minimum of 5

        SubmergedEnergy = int(sub_energy + (np.random.randn(1)[0] * 10))
        SubmergedSprawl = int(sub_sprawl + (np.random.randn(1)[0] * 10))
        if SubmergedSprawl < SubmergedEnergy * 500:
          SubmergedSprawl = SubmergedEnergy * 500
        SubmergedConsumption = max(int(sub_energy_consumption + (np.random.randn(1)[0] * 5)), 5) # minimum of 5

        aquaticEnergy = int(aqu_energy + (np.random.randn(1)[0] * 10))
        aquaticSprawl = int(aqu_sprawl + (np.random.randn(1)[0] * 10))
        if aquaticSprawl < aquaticEnergy * 200:
          aquaticSprawl = aquaticEnergy * 200
        aquaticConsumption = max(int(aqu_energy_consumption + (np.random.randn(1)[0] * 5)), 5) # minimum of 5

        herbEnergy = int(herb_energy + (np.random.randn(1)[0] * 10))
        herbSprawl = int(herb_sprawl + (np.random.randn(1)[0] * 10))
        if herbSprawl < herbEnergy * 60:
          herbSprawl = herbEnergy * 60
        herbConsumption = max(int(herb_energy_consumption + (np.random.randn(1)[0] * 5)), 5) # minimum of 5

        woodyEnergy = int(woody_energy + (np.random.randn(1)[0] * 10))
        woodySprawl = int(woody_sprawl + (np.random.randn(1)[0] * 10))
        if woodySprawl < woodyEnergy * 1000:
          woodySprawl = woodyEnergy * 1000
        woodyConsumption = max(int(woody_energy_consumption + (np.random.randn(1)[0] * 5)), 5) # minimum of 5

        # minumum of 1 for each of these
        woodyNum = max(int(woody + (np.random.randn(1)[0] * 100)), 1)
        herbNum = max(int(herb + (np.random.randn(1)[0] * 100)), 1)
        aquaticNum = max(int(aquatic + (np.random.randn(1)[0] * 100)), 1)
        floatingNum = max(int(floating + (np.random.randn(1)[0] * 100)), 1)
        SubmergedNum = max(int(Submerged + (np.random.randn(1)[0] * 100)), 1)
        waterNum = max(int(WetlandorPond + (np.random.randn(1)[0] * 100)), 1)
        debrisNum = max(int(dn + (np.random.randn(1)[0] * 100)), 1)

        # RUN THE SIM
        runSim(saveDir,
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
              savePlotDF = plotting,
              saveParamStats = False,
              epochNum = i)

    newStartingParams = [woody_energy, woody_sprawl, woody_energy_consumption, herb_energy, herb_sprawl, herb_energy_consumption, aqu_energy, aqu_sprawl, aqu_energy_consumption, flo_energy, flo_sprawl, flo_energy_consumption, sub_energy, sub_sprawl, sub_energy_consumption, woody, herb, aquatic, floating, Submerged, WetlandorPond, dn]
else:
    previousDF = pd.read_csv(saveDir + '/epochStats.csv')
    newStartingParams = previousDF.iloc[-1][xList].tolist()

        # RUN THE SIM
#        runSim(saveDir,
#              years,
#              wolfEn,
#              wolfRe,
#              wolfFa,
#              rabbitEn,
#              rabbitRe,
#              rabbitFa,
#              wolfNum,
#              rabbitNum,
#              grassNum,
#              debrisNum,
#              endOnExtinction = True,
#              savePlotDF = plotting,
#              saveParamStats = False,
#              epochNum = i)
    # set starting params for learning
#    newStartingParams = [we,wr,wf,re,rr,rf,wn,rn,gn,dn,]
#else:
#    previousDF = pd.read_csv(saveDir + '/epochStats.csv')
#    newStartingParams = previousDF.iloc[-1][xList].tolist()

#########
# IF CONTINUING A PREVIOUS RUN, or once the intitial 500 have run,
# RUN THE LEARNING SIM
#########

for i in range(0, reps):
    # set previous starting params from the newStartingParams from the last run
    previousParams = newStartingParams
    # re-learn the starting parameters
    adjustments = learnParamsLM(saveDir,
        years,
        previousParams[0],
        previousParams[1],
        previousParams[2],
        previousParams[3],
        previousParams[4],
        previousParams[5],
        previousParams[6],
        previousParams[7],
        previousParams[8],
        previousParams[9],
        xList,
        yList,
        incremental = True) ### THIS IS THE MAIN DIFFERENCE BETWEEN LM1 AND LM2

    # if we've reached successful stasis (10 in a row that hit 500)
    if adjustments[0] == 'END':
        print('$$$$$$$$$\n$$$$$$$$$\nSUCCESSFUL STASIS!!!')
        print('ran for ' + str(adjustments[1]) + ' years\n$$$$$$$$$\n$$$$$$$$$')
        break

    # adjust previousParams and set as newStartingParams
    newStartingParams = np.array(previousParams) + np.array(adjustments)

    # FINISH RE-LEARNING, print note
    print('%%%%%%%%\n%%%%%%%%\nRESET STARTING PARAMETERS.\nAdjustments:')
    print(adjustments)
    #print(newStartingParams)
    print('%%%%%%%%\n%%%%%%%%\n')
    #print('continuing in ...')
    #print('5'); time.sleep(1); print('4'); time.sleep(1); print('3'); time.sleep(1); print('2'); time.sleep(1); print('1'); time.sleep(1)
#___________________________________________________________________________________
    # set parameters for this run
#    wolfEn = max(newStartingParams[0], 100) # minimum of 100
#    wolfRe = max(newStartingParams[1], round((wolfEn * 1.1), 0)) # minimum of wolfEn * 1.1
#    wolfFa = max(newStartingParams[2], 5) # minimum of 5

#    rabbitEn = max(newStartingParams[3], 25) # minimum of 25
#    rabbitRe = max(newStartingParams[4], round((rabbitEn * 1.1), 0)) # minimum of rabbitEn * 1.1
#    rabbitFa = max(newStartingParams[5], 5) # minimum of 5

    # minumum of 1 for each of these
#    wolfNum = int(max(newStartingParams[6], 1))
#    rabbitNum = int(max(newStartingParams[7], 1))
#    grassNum = int(max(newStartingParams[8], 1))
#    debrisNum = int(max(newStartingParams[9], 1))

    # set parameters for this run
    woodyEnergy = max(newStartingParams[3], 125) # minimum of 125
    woodySprawl = max(newStartingParams[4], round((woodyEnergy * 1.1), 0)) # minimum of rabbitEn * 1.1
    woodyConsumption = max(newStartingParams[5], 5) # minimum of 5

    herbEnergy = max(newStartingParams[0], 100) # minimum of 100
    herbSprawl = max(newStartingParams[1], round((herbEnergy * 1.1), 0)) # minimum of wolfEn * 1.1
    herbConsumption = max(newStartingParams[2], 5) # minimum of 5

    aquaticEnergy = max(newStartingParams[0], 75) # minimum of 75
    aquaticSprawl = max(newStartingParams[1], round((aquaticEnergy * 1.1), 0)) # minimum of wolfEn * 1.1
    aquaticConsumption = max(newStartingParams[2], 5) # minimum of 5

    floatingEnergy = max(newStartingParams[0], 50) # minimum of 50
    floatingSprawl = max(newStartingParams[1], round((floatingEnergy * 1.1), 0)) # minimum of wolfEn * 1.1
    floatingConsumption = max(newStartingParams[2], 5) # minimum of 5

    SubmergedEnergy = max(newStartingParams[0], 25) # minimum of 25
    SubmergedSprawl = max(newStartingParams[1], round((SubmergedEnergy * 1.1), 0)) # minimum of wolfEn * 1.1
    SubmergedConsumption = max(newStartingParams[2], 5) # minimum of 5

    # minumum of 1 for each of these

    woodyNum = int(max(newStartingParams[3], 1))
    herbNum = int(max(newStartingParams[4], 1))
    aquaticNum = int(max(newStartingParams[5], 1))
    floatingNum = int(max(newStartingParams[6], 1))
    SubmergedNum = int(max(newStartingParams[7], 1))
    waterNum = int(max(newStartingParams[8], 1))
    debrisNum = int(max(newStartingParams[9], 1))

    # RUN THIS ITERATION
#    runSim(saveDir,
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
#          endOnOverflow = True,
#          savePlotDF = plotting,
#          saveParamStats = False,
#          epochNum = (i + seedReps))
#    print(adjustments)

    runSim(saveDir,
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
          endOnOverflow = True,
          savePlotDF = plotting,
          saveParamStats = False,
          epochNum = (i + seedReps))
    print(adjustments)


if bigRun == True:
    # ONCE WE REACHED SUCCESS, RUN A BIG ONE
    # set parameters for this run
    woodyEnergy = max(newStartingParams[3], 125) # minimum of 25
    woodySprawl = max(newStartingParams[4], round((rabbitEn * 1.1), 0)) # minimum of rabbitEn * 1.1
    woodyConsumption = max(newStartingParams[5], 5) # minimum of 5

    herbEnergy = max(newStartingParams[3], 100) # minimum of 25
    herbSprawl = max(newStartingParams[4], round((rabbitEn * 1.1), 0)) # minimum of rabbitEn * 1.1
    herbConsumption = max(newStartingParams[5], 5) # minimum of 5

    aquaticEnergy = max(newStartingParams[3], 75) # minimum of 25
    aquaticSprawl = max(newStartingParams[4], round((rabbitEn * 1.1), 0)) # minimum of rabbitEn * 1.1
    aquaticConsumption = max(newStartingParams[5], 5) # minimum of 5

    floatingEnergy = max(newStartingParams[0], 50) # minimum of 100
    floatingSprawl = max(newStartingParams[1], round((wolfEn * 1.1), 0)) # minimum of wolfEn * 1.1
    floatingConsumption = max(newStartingParams[2], 5) # minimum of 5

    SubmergedEnergy = max(newStartingParams[0], 25) # minimum of 100
    SubmergedSprawl = max(newStartingParams[1], round((wolfEn * 1.1), 0)) # minimum of wolfEn * 1.1
    SubmergedConsumption = max(newStartingParams[2], 5) # minimum of 5

    # minumum of 1 for each of these
    SubmergedNum = max(int(max(newStartingParams[6], 1)), 1)
    floatingNum = max(int(max(newStartingParams[7], 1)), 1)
    aquaticNum = max(int(max(newStartingParams[6], 1)), 1)
    herbNum = max(int(max(newStartingParams[6], 1)), 1)
    woodyNum = max(int(max(newStartingParams[6], 1)), 1)
    waterNum = int(max(newStartingParams[8], 1))
    debrisNum = int(max(newStartingParams[9], 1))

    runSim(saveDir,
          10000,
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
          endOnOverflow = False,
          saveParamStats = True,
          savePlotDF = plotting,
          epochNum = (i + seedReps + 1))
