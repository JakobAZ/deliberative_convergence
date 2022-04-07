import numpy as np
import pandas as pd
import networkx as nx
import time as t
import logging as l

def nAry(num, base):
    """This function is meant to return a base-ary representation of num.
    One can improve it, certainly, but until now I didn't.
    """
    assert base < 10, "For more than 9 alternatives we need to adjust the numerical system (e.g. to hex) for the nAry-function to work!"
    newNum=""
    numAlts = base
    while num > 0:
        newNum = str(num % base) + newNum
        num //= base
    newNum = "0"*(numAlts-len(newNum))+newNum
    return newNum #as a string

def validRanking(r, N):
    for n in range(1, N+1):
        if str(n) in r and not str(n-1) in r:
            return False
    return True

fak = lambda x: 1 if x<=1 else x*fak(x-1)

def weakRankings(numAlts):
    L = [nAry(i,numAlts) for i in range(numAlts**(numAlts)) if validRanking(nAry(i,numAlts), numAlts)]
    return np.array(L)


def genJ(ranking, numAlts="Not necessary anymore"):
    numAlts = len(ranking)
    Col = np.array(list(ranking)).reshape(-1,1).astype(int)
    Row = Col.T
    return np.repeat(Col, numAlts, axis = 1) <= np.repeat(Row, numAlts, axis = 0) 


def genDistMatrices(numAlts):
    s = t.time()
    RankingSpace = weakRankings(numAlts)
    l.debug(f"RankingSpace as dimensions{RankingSpace.shape}")
    # This part is used also in DP-calculation:
    L = [genJ(i).reshape(1,1,numAlts, numAlts) for i in RankingSpace]
    Row = np.concatenate(L, axis=0) # Row and Col are to be understood as if every entry in the matrix was a matrix of dim numAlts^2
    Col = np.concatenate(L, axis=1) # Row and Col are to be understood as if every entry in the matrix was a matrix of dim numAlts^2
    BoolMat = (np.repeat(Col, RankingSpace.shape[0], axis = 0) != np.repeat(Row, RankingSpace.shape[0], axis = 1))
    distMats = dict()

    # First KS distances
    distMats[f"KS{numAlts}"] = pd.DataFrame(BoolMat.sum(3).sum(2), index = RankingSpace, columns = RankingSpace).astype(np.int16)
    print(f"Computing KS{numAlts} took {round(t.time()-s, 2)} seconds!")
    # Next DP distances
    s = t.time()
    DPAM = (BoolMat.any(2).sum(2) == 1) | (BoolMat.any(3).sum(2) == 1) # DP adjacency matrix given by the insight that if the Judgement-matrices have their differences contained in one column or row, then (and only then) DP-Distance is 1.
    intSpace = np.arange(RankingSpace.shape[0]) # The integers here are the indices of elements of RankingSpace.
    edges = zip((intSpace.reshape(1,-1) * DPAM).reshape(-1), (intSpace.reshape(-1,1) * DPAM).reshape(-1))
    G=nx.Graph(edges)
    distMats[f"DP{numAlts}"] = pd.DataFrame(dict(nx.all_pairs_shortest_path_length(G))).sort_index(0).sort_index(1).astype(np.int16)
    distMats[f"DP{numAlts}"].index, distMats[f"DP{numAlts}"].columns = RankingSpace,RankingSpace
    print(f"Computing DP{numAlts} took {round(t.time()-s, 2)} seconds!")
    
    # Finally CS
    s = t.time()
    rankings = RankingSpace.copy()
    assert len(rankings.shape)==1, "Rankings-vector should come as a 1-d-array of strings!"
    f = lambda x, i:  x.str.slice(int(0+i),int(1+i))
    LL = [pd.DataFrame(rankings).apply(f, args = (i,)).values.reshape(-1, 1) for i in range(numAlts)]
    rankings = np.concatenate(LL, axis = 1).astype(np.int) +1
    l.debug(f"rankings is now:\n{rankings}")
    # The loop is going through the possible ranks that the alternatives could have been assigned
    CS_rankings = rankings.copy().astype(float)
    
    currentRank = np.repeat(1, rankings.shape[0]).reshape(-1,1)
    for i in range(1, 1+numAlts):
        l.debug(f"Rank considered is i={i}, currentRank (by opinion) is\n{currentRank}")
        numAtRank = (rankings==i).sum(axis=1).reshape(-1,1)
        curCSnums = np.concatenate([currentRank, currentRank+numAtRank-1], axis = 1).mean(axis=1).reshape(-1,1)
        CS_rankings[rankings==i] = ((rankings==i) * curCSnums)[rankings==i]
        # Correct for how many alternatives were at rank i. 
        currentRank += numAtRank
    
    L = [i.reshape(1,1,-1)  for i in CS_rankings]
    # return sum of absolute diffs aka CS-distance of CS_rankings cartesian CS_rankings
    Row = np.concatenate(L, axis=0) # Same trick as with KS to do functions on the cartesian product.
    Col = np.concatenate(L, axis=1)
    l.debug(f"Dimensions of Row and Col are {Row.shape}, {Col.shape}")
    CS_distances = (np.abs(np.repeat(Col, RankingSpace.shape[0], axis = 0) - np.repeat(Row, RankingSpace.shape[0], axis = 1)).sum(axis=-1)).astype(np.int16)
    distMats[f"CS{numAlts}"] = pd.DataFrame(CS_distances, index = RankingSpace, columns = RankingSpace)
    print(f"Computing CS{numAlts} took {round(t.time()-s, 2)} seconds!")
    return distMats