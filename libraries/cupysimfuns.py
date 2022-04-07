import cupy as np
import logging as l
import pandas as pd
l.debug("Cupysimfuns with debug-messages")
def update(current, announcement,dist_df, simulations, agents, rel, favorOneself = False):
    """This function computes the new opinions of all agents given their old opinions and an announcement.
    Doing this for all sims at once means that current is an intMatrix and announcement is an intVector.
    The outputis an intMatrix of updated opinions.
    -dist_df: a DataFrame holding the distances of all permissible opinions (weak or strict).
    -current: intMatrix of opinion INDICES
    -simulations, agents: int
    -rel: float
    """
    # 0) make sure all inputs are of expected shape:
    assert current.shape == (simulations, agents), "intMatrix current came in unexpected shape!"
    assert announcement.shape == (simulations,) , f"announcement came in unexpected shape ({announcement.shape} instead of {(simulations,)})!"
    assert type(simulations) == int and type(agents) == int and type(rel) == float, "sim, agents or rel came in unexpected datatype, should be int, int, float."
    
    # 1) get some useful variables: helper holds all opinion-indices as an r3A. 
    numOpins = dist_df.shape[0]
    helper = np.repeat(np.repeat(np.arange(numOpins).reshape(-1,1,1), current.shape[0], axis = 1 ), current.shape[1], axis = 2  ) 
    # 2) get distances of all potential opinions from announcement and current opinion
    # 2.1 dFromAnnoun
    dFromAnnoun = np.array(dist_df.iloc[announcement.get()].values.T.reshape(-1,simulations,1)).astype(np.float)
    # 2.2 dFromOp
    f = lambda n,dist_df,current,simulations: np.array(dist_df.iloc[current.get()[:,n]].values.T.reshape(-1,simulations,1))
        # Next line gets us a Datacube where for all agents in all simulations each "floor" holds the distance between a potential new opinion and the current one.
    dFromOp = np.concatenate([f(n, dist_df, current, simulations) for n in range(agents)], axis=2).astype(np.float)
    # 3) calculate score-r3A
    score = np.sqrt(rel * dFromOp**2 + (1-rel) * dFromAnnoun**2) # NB: dFromAnnoun is getting broadcasted here!
    # 4) get all optimal opinions for everyone in every simulation
    if favorOneself: # Deducts from own opinion a minusule but tie-breaking amount if we want to favour ourselves.
        score[helper==current] = (score[helper==current]-(1E-5)) #This amount should be sufficiently small that the own opinion doesn't get preferred if it is not among the truly optimal.
    optOpins = score == np.min(score, axis = 0) # Gives a bool-array True whenever an opinion is optimal.
    # 5) Assign randomly uniform values to the optimal opinions, 0 to all non-optimal opinions.
    ChoiceScore = np.random.uniform(0,1, (optOpins.shape)) * optOpins
    Choice = (ChoiceScore == ChoiceScore.max(axis=0)) * helper # Multiplies a bool-array that is True at the unique (but possibly random) choice with the array holding al opinion-indices
    newOpin = Choice.max(axis=0) # Choice is 0 everywhere other than at the unique choices, there it holds the opinion-index. Reducing with max gives the array in convenient form. 
    assert newOpin.shape ==(simulations, agents), f"new Opinion has shape {newOpin.shape}"
    l.debug("current :\n"+ str(current) +"\n")
    l.debug("Announ :\n"+ str(announcement)+"\n")
    l.debug("newOpin :\n"+ str(newOpin)+"\n")
    l.debug(f"score (shape {score.shape}) of agents 0&1 in simulation 0:\n"+ str(score[:,0,:2]))
    l.debug("num of optimal opins by agent :\n"+ str((optOpins.sum(axis=0))))
    optOpins = optOpins.astype(np.float16)
    optOpins[optOpins==0] = np.nan
    l.debug("optimal opins by agent :\n"+ str((optOpins* helper)))
    return newOpin

def deliberate(dist_df, rounds, agents, reliability, simulations, favorOneself = False, dynamicSimEnd=True):
    """This function runs the desired number of deliberations all at once using nd-arrays where n >= 3.
    Inputs: 
            - dist_df: a DataFrame giving the distances for all pairs of possible opinions
            - rounds: the number of deliberative rounds, i.e., times each agent is to speak up.
            - agents: the number of agents
            - reliability: the float-reliability that everyone attaches to themselves.
            - simulations: the number of Simulations that are to be run at once.
    """
    import time as t
    # I was tempted to have the agents announce their actual opinions. Instead, they will voice the INDEX of their opinions. 
    # The derivation is kind of lengthy but it works quickly.
    isStrict = [int(max(list(i)))==len(i)-1 for i in  dist_df.columns]
    arr = (dist_df.loc[isStrict, isStrict].index)
    Rankings = np.random.choice(a= arr.shape[0], size=((simulations*agents)))
    Rankings = arr[Rankings.get()]
    Op2Int = pd.DataFrame(dict(zip(dist_df.index, range(dist_df.shape[0]))), index = ["OpinIntegers"]).T
    Rankings = np.array(Op2Int.loc[Rankings].values.reshape(1,simulations, agents))    # ...computation of new round results
    current = Rankings.copy().reshape(simulations, agents)
    l.debug(f"Initial rankings: \n{current}")
    # Now deliberate:
    for rnum in range(rounds): 
        startTime = t.time()
        # First fix speaking order for THIS round but all simulations at once.
        # It is a matrix where a 5 at position ij means that in simulation i, agent 5 is the j'th to speak.
        order = np.array([np.random.choice(agents, (agents,), replace = False) for i in range(simulations)])
        helper = np.repeat(np.arange(agents).reshape(1,-1), simulations, axis=0)
        l.debug(f"Speaking order: \n{order}")
        # The helper-array is an array of stacked aranges which will be used to provide a bool-matrix.
        # This works because using '==' on a matrix and a vector will produce row-wise evaluated bool-rows.
        # The bool-matrix will be used to index into the current opinions.
        # Now, everyone gets to speak her opinion, the others update.
        for step in range(agents): # each round is made of agents [int] steps, because everyone gets to speak their mind once (and only once).
            announcement = current[helper == order[:,step].reshape(-1,1)]
            current = update(current, announcement, dist_df, simulations, agents, reliability, favorOneself)
        print(f"Round {rnum+1} of {rounds} completed. This round took {round(t.time()-startTime,2)} seconds", end="\n")
        Rankings = np.concatenate([Rankings, current.reshape(1, simulations, agents)], axis = 0)
        if rnum > 0 and dynamicSimEnd:
            last2roundsnochanges = ((Rankings[-1] == Rankings[-2] )& (Rankings[-1] == Rankings[-3])).all().all()
            if last2roundsnochanges: 
                print("Deliberation got boring in round", rnum, "\nThus, there are", Rankings.shape[0], "int Matrices in Rankings (one is initial rankings, the rest are updated opinions).")
                break
    return Rankings