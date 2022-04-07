# -*- coding: utf-8 -*-
"""
RunIt Created on Thu May  6 13:08:12 2021

@author: Maher Jakob Abou Zeid

import logging as l
logger = l.getLogger()
logger.setLevel("DEBUG")
"""
# Parameters to be set
STRICT = True
favorOneself = False
numAlternatives = 5
useUniformUpdateRule = True # Should generally be True. Using the Non-Uniform variant increases update-times considerably.
relRange = (51, 101,2) # Tuple of Start, Stop (exclusive), Step
agRange = (29, 53, 2) # Tuple of Start, Stop (exclusive), Step

maxRoundsBeforeSimEnd = 40
numSimulations=1000

useQuickUpdate = False


import pathlib
path_to_here = str(pathlib.Path(__file__).parent.absolute())


import sys
sys.path.append(path_to_here+"/libraries")
import distcomputing as dists
import pandas as pd


distances = ["KS", "DP","CS"]       # Only change that if you know how to manipulate the subsequent code.
measures = [i+str(numAlternatives) for i in distances]
dMatDict = dict()


# Load or compute and save the distance-matrices.
for key in measures:
    try: 
       dMatDict[key] = pd.read_csv(path_to_here+"/DistanceMatrices/"+key+".csv", index_col=0)
       dMatDict[key].index = dMatDict[key].columns # Loading from csv truncates index-strings. This line corrects that mistake.
    except FileNotFoundError: 
        dMatDict = dists.genDistMatrices(numAlternatives) #calcdistmatrices returns a dict, in this case with only one entry
        for key in measures:
            dMatDict[key].to_csv(path_or_buf = path_to_here+"/DistanceMatrices/"+key+".csv")
        break

if useQuickUpdate: 
    import numpysimfunsQ as sf
    print("Quick update")
else: 
    if useUniformUpdateRule: import numpysimfuns as sf
    else: 
        assert numAlternatives <6, "For more than 5 alternatives you should really use the uniform update rule!" # Benchmarking with 6 alternatives resulted in an update-duration of 2 seconds with uniform,  456 seconds (Seven and a half minutes!!) with non-uniform update! (It was the strict case)
        import numpysimfunsNONuniform as sf # Use different update functions.
        # Sort the dist-DF in a very particular manner as position is not arbitrary any more.
        if numAlternatives ==3: # Sort it in exactly the way that Soroush did it.
            jav2PyIdx = {0: '000', 1: '001', 2: '010', 3: '100', 4: '011', 5: '101', 6: '110', 7: '012', 8: '021', 9: '102', 10: '201',11: '120',12: '210'}
            Py2Jav = dict(zip(jav2PyIdx.values(), jav2PyIdx.keys()))
        else: # Just sort the strict rankings to the end.
            Py2Jav = {i: str(int(int(max(list(i)))==len(i)-1))+str(i) for i in dMatDict[list(dMatDict.keys())[0]].columns}
            jav2PyIdx = dict(zip(Py2Jav.values(), Py2Jav.keys()))
        for m in dMatDict.keys():
            dMat = dMatDict[m]
            dMat = dMat.rename(columns=Py2Jav)
            dMat.index=dMat.columns
            dMat = dMat.sort_index(0).sort_index(1)
            dMat = dMat.rename(columns=jav2PyIdx)
            dMat.index=dMat.columns
            dMatDict[m] = dMat
    

# Define a small wrapper because we are going to simulate with a lot of different parameters
def main(dMat, rounds, agents, reliability, simulations, favorBool, strict):
    if strict: 
        isStrict = [int(max(list(i)))==len(i)-1 for i in dMat.index]
        dMat = dMat.loc[isStrict, isStrict]
        delOut = sf.deliberate(dMat, rounds, agents, reliability, simulations, favorBool)
        Results = delOut
    else: Results = sf.deliberate(dMat, rounds, agents, reliability, simulations, favorBool) 
    print()
    return Results, dMat

import time as t

def ETAmessage(avtime, numReliabilities, count):
    msg = "ETA is "
    ETAseconds = avtime*(numReliabilities- count) 
    ETA = t.time()+ETAseconds
    if ETAseconds < 300: msg += str(round(ETAseconds/60,1)) + " minutes"
    elif t.localtime(ETA)[:3]==t.localtime()[:3]: msg += t.strftime("%H:%M", t.localtime(ETA))
    else: msg += t.ctime(ETA)
    return msg


for numAg in range(*agRange):
    Result = dict(zip(list(dMatDict.keys()),[dict() for i in dMatDict]))
    relDurations =[]
    avtime = "nan"
    print(f"Running reliabilities with {numAg} Agents")
    for count, rel in enumerate(range(*relRange)):
        numReliabilities = len(list(range(*relRange)))
        starttime = t.time()
        if avtime != "nan":
            print("previous reliabilities took on average",round(avtime) , "seconds")  
            print(f"about {round(count/numReliabilities*100)}% of reliabilities done, {ETAmessage(avtime, numReliabilities, count)}")
    
        for key in dMatDict.keys():
            print(f"\n{key}, Rel {rel}% started")
            Result[key][rel], dict_df_dist = main(dMat=dMatDict[key], rounds=maxRoundsBeforeSimEnd, agents=numAg, reliability=rel/100, simulations=numSimulations, favorBool=favorOneself, strict=STRICT)
            
        dMatDict[key] = dict_df_dist
        relDurations.append(t.time()-starttime)
        avtime = sum(relDurations)/len(relDurations)
        
    
    import pickle
    with open(path_to_here+"/Results/"+f"{'S' if STRICT else 'W'}{numAlternatives}{ 'U' if useUniformUpdateRule else 'N'}{numAg}{'favor' if favorOneself else ''}.pkl", "wb") as f:
        pickle.dump(Result, f)






