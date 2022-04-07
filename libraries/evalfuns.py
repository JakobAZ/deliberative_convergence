import numpy as np
import itertools 
import pandas as pd
import networkx as nx
import logging as l

def singlePeakCounter(opinMatrix, mode="s"):
    """This function implements counting of the pref-profiles that are single peaked w.r.t. some ordering.
    It takes:
    - opinMatrix: a 3d-array (ranks,simulations, agents)
    - mode: one of ["s", "w", "p"] coding for strict (s) and weak single-peakedness (w) as well as single-plateauedness (p).
    """
    # 0)  We first implement this function in the most matrixy, theoretically best way possible. 
    # Afterwards we might need to change the code to sequentially compute the stuff acommodating memory-constraints.
    numAlts = opinMatrix.shape[0]# Extract the number of alternatives

    # 1) Expand further to test for the respective property by all possible rankings at once.
    # This is a $$ sim x numAg x numAlts x numAlts! $$- matrix. 
    # It should be handleable as long as sim x numAg x numAlts < numAlts! as we are dealing with a numAlts!^2-matrix when calculating the dist-matrices.
    import itertools
    allPreOrderings = [*itertools.permutations(range(numAlts))]
    l.debug(f"The preorderings are\n{allPreOrderings}")
    ordersToTest = np.concatenate([opinMatrix[tup, :,:].reshape(1, *opinMatrix.shape) for tup in allPreOrderings], axis = 0)
    
    #Strict SP allows every rank to comprise of exactly one alternative and for a given structural dimension you can only change direction once.
    #Weak SP requires only that one doesn't go towards betterness again, i.e. direction changes only once, there is only one best alternative but one can stay at a suboptimal plateau. 
    #SPl allows indifference only at the best (which is down in our case) and needs at least one strict preference.
    
    # All of the above conditions are conveniently checked when using differences.
    Diffs = -np.diff(ordersToTest, axis = 1)
    SignChanges = np.diff(Diffs > 0, axis = 1).sum(axis = 1)
    if mode == "s":
        # Prefs are strictly single peaked if one of: 
        # I) best option is at one extreme, there is no place where the sign of the rank-diff changes.
        # II) best option is somewhere in the middle, ergo there is exactly one spot where the sign of the differences changes. Also it first goes down, then up.
        isSP = ((ordersToTest.max(axis=1) == ordersToTest.shape[1]-1)     # Given that all ranks are uniquely occupied, i.e., excluding opinions like [1,0,1] (which would only be weakly SP)
            & ( (SignChanges==0 )                                           # Condition I
            | ((SignChanges==1 ) & (Diffs[:,0,:]>0 )) ) )                # Condition II (recall lower numbers are better so we have \/ which means the first of the differences must be positive.)
        l.debug(f"Strict SP, is SP shape: {isSP.shape}")
    elif mode == "p": 
        # p-prefs look like "\_", "_/", or "\_/". Call these cases L, J and U
        increasing = Diffs < 0      # Bool-array true wherever the graph is increasing (going to worseness).
        flat = Diffs == 0           # Similar but staying at a level
        decreasing = Diffs>0        # Going to betterness.
        oneDecreasing = (np.diff(decreasing, axis=1).sum(1)<=1)  # Meaning that all positive differences (if existent) follow each other and they are located not between two non-decreasing parts.
        oneFlat = (np.diff(flat, axis = 1).sum(1) <= 1)          # Similar to oneDecreasing, only that it is allowed that the flat part is either nonexistent or the whole graph is flat.
        
        isL = (decreasing[:, 0]                     # It begins decreasingly (and thus there is a decreasing part).
             & oneDecreasing                        # At most one decreasing part.
             &  oneFlat                             # There is at most one flat part (maybe none).
             & (~increasing.any(axis=1))  )         # There is no increasing part.

        isJ = ( (~decreasing.any(axis=1))           # There is no decreasing part.
              & oneFlat                             # There is at most one flat part (maybe none).
              & increasing[:,-1] )                  # It ends with the increasing part (which needs to exist for this to hold).
            
        isU = (decreasing[:, 0]                             # It begins decreasingly.
             & oneDecreasing                                # Exactly one decreasing part.
             & (np.diff(increasing, axis=1).sum(1)==1)      # Exactly one increasing part
             & increasing[:,-1] )                           # It ends with the increasing part.
        l.debug(f"isL: shape: {isL.shape}\n")
        isSP = isL|isJ|isU
        newline="\n"
        l.debug(f"For simulation 0 isSP is\n{newline.join([str(list(row)) for row in isSP[:,0]])}")
    elif mode == "w": 
        """ Idea: Implement a smarter WSP checker by looking for a unique best option. 
        Then use cumsum to make a bool-array that divides the array into before and after the peak.
        before the peak all should be weakly decreasing, after it weakly increasing"""
        stupidWSPchecker = lambda Diffs: (Diffs[(Diffs>=0).argmin():] <= 0 ).all()
        simulations, agents  = opinMatrix.shape[1:]
        for ag in range(agents):
            for sim in range(simulations):
                for preO in enumerate(allPreOrderings):
                    ordersToTest[preO[0],0,sim, ag] = stupidWSPchecker(ordersToTest[preO[0],:,sim, ag])
        isSP = ordersToTest[:,0,:, :]
    return isSP.mean(axis = 2).max(axis=0)     # Select the maximum of the ratios of SP preferences

def checkForCycles(opinMatrix, strict = True):
    """Inputs:
    - opinMatrix is, again, all opinions in all simulations before/after deliberation, i.e. a 3d-matrix of opinions.
    - strict <bool> indicates whether strict preferences are used for the cycles or not."""
    # 1) get all pairwise majorities
    numAlts, simulations, agents = opinMatrix.shape
    pairwiseComparisons = itertools.permutations(range(numAlts), 2)
    compDict={}
    for comp in pairwiseComparisons:
        if strict: compDict[comp] = ((opinMatrix[comp[0],:,:] < opinMatrix[comp[1],:,:]).mean(axis=1) > .5)
        else: compDict[comp] = ((opinMatrix[comp[0],:,:] <= opinMatrix[comp[1],:,:]).mean(axis=1) > .5)
    # Make a Graph of the preferences and check that thing for cycles
    f = lambda x: np.array(list(x))
    CyList = list()
    for sim in range(simulations):
        vertices = f(compDict.keys())[f(compDict.values())[:,sim]]
        G = nx.DiGraph(list(vertices))
        CyList.append(len(list(nx.simple_cycles(G))))
    return (np.array(CyList) > 0).astype(float) 


checkConsensus = lambda intMatrix: (intMatrix == intMatrix[:,0].reshape(-1,1)).all(1)

def distance2C(intMatrix, dist_df):
    consensualSimulations = (intMatrix == intMatrix[:,0].reshape(-1,1)).all(1)
    output = (~consensualSimulations).astype(np.float) # Writes a 0 for every simulation that is already in consensus (1 for the others)
    l.debug(f"consensualSimulations:\n{consensualSimulations}")
    if not consensualSimulations.all():
        simulations, agents = intMatrix.shape
        # The following 4 lines (including the comment) are recycled code from the update-fun:
        f = lambda n,dist_df,current,simulations: dist_df.iloc[current[:,n]].values.T.reshape(-1, current.shape[0],1)
        # Next line gets us a Datacube where for all agents in all simulations each "floor" holds the distance between a potential new opinion and the current one.
        nonzeroDistances = np.concatenate([f(agent, dist_df, intMatrix, simulations) for agent in range(agents)], axis=2).astype(np.float_)    
        l.debug(f"nonzeroDistances:\n{nonzeroDistances}")
        output = (nonzeroDistances**2).sum(2).min(0).reshape(-1,)**.5 # Fills in all correct non-zero distances to consensus
    return output

def countClusters(intMatrix):
    assert len(intMatrix.shape) == 2, "intMatrix, came in an unexpected shape!"
    return np.array([np.unique(row).shape[0] for row in intMatrix]).reshape(-1,)

def strictPrCheck(opinMatrix):
    return (opinMatrix.max(axis=0) == opinMatrix.shape[0]).mean(1).reshape(opinMatrix.shape[1],1)

def genOpMat(intMat, dist_df):
    numAlts = len(dist_df.columns[0]) # Extract the number of alternatives. 
    sim, ag = intMat.shape
    opMat = np.array(dist_df.index[intMat.reshape(-1)]).reshape(sim, ag)
    f = lambda x, i:  x.str.slice(int(0+i),int(1+i))
    LL = [(pd.DataFrame(opMat).apply(f, args = (i,))).values.reshape(1, sim, ag) for i in range(numAlts)]
    opMat = np.concatenate(LL, axis = 0)
    opMat = opMat.astype(int)      
    return opMat

def intransitiveProfiles(opinMatrix):
    """Inputs:
    - opinMatrix is, again, all opinions in all simulations before/after deliberation, i.e. a 3d-matrix of opinions.
    """
    # 1) get all pairwise majorities
    numAlts, simulations, agents = opinMatrix.shape
    pairwiseComparisons = itertools.permutations(range(numAlts), 2)
    compDict={}
    for comp in pairwiseComparisons:
        compDict[comp] = ((opinMatrix[comp[0],:,:] <= opinMatrix[comp[1],:,:]).mean(axis=1) > .5)
        
    triples = itertools.permutations(range(numAlts), 3)
    isIntrans = np.full(simulations, False)
    for triple in triples: 
        isIntrans[~isIntrans] = compDict[triple[0:2]][~isIntrans] & compDict[triple[1:3]][~isIntrans] & ~compDict[(triple[0], triple[2])][~isIntrans]
    return isIntrans.astype(int) 

def evaluate(Results, dist_df, Threshold, strictCycles= True):
    """- Threshold: the ratio of agents with SP preferences that is needed to call the profile SP
    """
    #here you should call all the evaluation functions. 
    # 3) Prettify the opinions after deliberation for the eval functions: 
    # 3.1) 
    opinsBeforeDelibInt = Results[0,:,:]
    opinsBeforeDelibOp = genOpMat(opinsBeforeDelibInt, dist_df)

    opinsAfterDelibInt = Results[(Results.shape[0]-1),:,:]
    opinsAfterDelibOp = genOpMat(opinsAfterDelibInt, dist_df)
    
    # 3.2) Expand the 2d-array of strings into a 3d-array of single integers (at first stored as strings).
    # Unfortunately we have to use pandas series to slice the stuff. Replace this way of doing things as soon as a better one is available!
    
    outDF = pd.DataFrame()
    outDF["clusters_before"] = countClusters(intMatrix = opinsBeforeDelibInt)
    outDF["clusters_after"] = countClusters(intMatrix = opinsAfterDelibInt)
    outDF["dist2C_before"] = distance2C(intMatrix = opinsBeforeDelibInt, dist_df = dist_df)
    outDF["dist2C_after"] = distance2C(intMatrix = opinsAfterDelibInt, dist_df = dist_df)
    outDF["cycles_before"] = checkForCycles(opinMatrix = opinsBeforeDelibOp, strict=strictCycles)
    outDF["cycles_after"] = checkForCycles(opinMatrix = opinsAfterDelibOp, strict=strictCycles)
    outDF["prox2sp_after"] = singlePeakCounter(opinMatrix = opinsAfterDelibOp, mode="s")
    outDF["prox2sp_before"] = singlePeakCounter(opinMatrix = opinsBeforeDelibOp, mode="s")
    outDF["prox2spl_before"] = singlePeakCounter(opinMatrix = opinsBeforeDelibOp, mode="p")
    #outDF["dist2wsp"] = 1-singlePeakCounter(opinMatrix = opinsAfterDelibOp, mode="w")"""
    outDF["prox2spl_after"] = singlePeakCounter(opinMatrix = opinsAfterDelibOp, mode="p")
    outDF["spProfile"] = outDF["prox2sp_after"] >=(Threshold)
    #outDF["wspProfile"]  = outDF["dist2wsp_after"] <=(1-Threshold)
    outDF["splProfile"] = outDF["prox2spl_after"] >= Threshold
    outDF["strictProfile"] = strictPrCheck(opinMatrix=opinsAfterDelibOp) >= Threshold
    outDF["intransitiveProfiles_before"] = intransitiveProfiles(opinsBeforeDelibOp)
    outDF["intransitiveProfiles_after"] = intransitiveProfiles(opinsAfterDelibOp)
    return outDF