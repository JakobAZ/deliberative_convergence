# -*- coding: utf-8 -*-
"""
EvalALL Created on Thu May  6 18:21:34 2021

@author: Maher Jakob Abou Zeid
"""
import pathlib
path_to_here = str(pathlib.Path(__file__).parent.absolute())
import sys
sys.path.append(path_to_here+"/libraries")

import pandas as pd
import pickle  
import evalfuns as ef
import os
folder, children, files =  next(os.walk(path_to_here+"/Results"))

for file in files:
    with open(folder+"/"+file, "rb") as f:
        print(file)
        Result = pickle.load(f)
    
    measures = list(Result.keys())
    reliabilities = list(Result[measures[0]].keys())
    
    dMatDict = dict(zip(measures, [i+".csv" for i in measures] ))
    for key in measures:
        dMatDict[key] = pd.read_csv(path_to_here+"/DistanceMatrices"+"/"+dMatDict[key], index_col=0)
        dMatDict[key].index = dMatDict[key].columns # Loading from csv truncates index-strings. This line corrects that mistake.
        if file[0]=="S": # Truncate the dist_df.
            isStrict = [int(max(list(i)))==len(i)-1 for i in  dMatDict[key].index]
            dMatDict[key] = dMatDict[key].loc[isStrict, isStrict]
        elif file[2] == "N": # Sort the dist_df accordingly.            print("elif triggered")       
            if file[1] =="3": # Sort it in exactly the way that Soroush did it.
                    jav2PyIdx = {0: '000', 1: '001', 2: '010', 3: '100', 4: '011', 5: '101', 6: '110', 7: '012', 8: '021', 9: '102', 10: '201',11: '120',12: '210'}
                    Py2Jav = dict(zip(jav2PyIdx.values(), jav2PyIdx.keys()))
            else: # Just sort the strict rankings to the end.
                Py2Jav = {i: str(int(int(max(list(i)))==len(i)-1))+str(i) for i in dMatDict[list(dMatDict.keys())[0]].columns}
                jav2PyIdx = dict(zip(Py2Jav.values(), Py2Jav.keys()))
                dMat = dMatDict[key]
                dMat = dMat.rename(columns=Py2Jav)
                dMat.index=dMat.columns
                dMat = dMat.sort_index(0).sort_index(1)
                dMat = dMat.rename(columns=jav2PyIdx)
                dMat.index=dMat.columns
                dMatDict[key] = dMat

    try: 
        with open(f"Results Archive/{file[:-4]} Evaluated.pkl", "rb") as f: 
            df_reduced = pickle.load(f)      
    except FileNotFoundError:
        df_all = pd.DataFrame()
        for rel in reliabilities:
            for key in measures:
                print(f"Calculating evals for Result{[key]}[{rel}]", end="\n")
                df = ef.evaluate(Result[key][rel], dist_df=dMatDict[key], Threshold=.75, strictCycles =True )
                df["dMat"] = key
                df["Reliability"] = rel
                df_all = pd.concat([df_all, df], axis=0)
        
        df_reduced = df_all.groupby(["dMat" ,'Reliability']).mean()
        with open(f"{path_to_here}/Results Archive/{file[:-4]} Evaluated.pkl", "wb") as f: 
            pickle.dump(df_reduced, f)
    
    import matplotlib.pyplot as plt
    plt.style.use("default")
    markerDict = dict(zip(measures,["s","D","v"]))
    colorDict=dict(zip(measures, ["#C55D", "#55CD","#EC7D"]))
    figureSize = 6,8
    # First the ones where before and after are displayed:
    metrics = ["prox2sp","prox2spl", "clusters", ]
    for metric in metrics:
        plt.figure(figsize = figureSize)    
        for dMat in measures:
            if dMat[:2] == "KS": plt.plot(df_reduced.loc[dMat, metric+"_before"], label="before", color ="#000")
            plt.plot(df_reduced.loc[dMat, metric+"_after"], label=f"{dMat}",  marker= markerDict[dMat], color = colorDict[dMat], markeredgecolor=colorDict[dMat])
        plt.xlabel('self-reliability in percent', fontsize='x-large')
        plt.ylabel(metric, fontsize='x-large')
        plt.legend(loc='center left')
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')
        plt.legend(loc='center left', fontsize='x-large')
        plt.savefig(path_to_here+"/Figures/"+ f"{file[:file.find('.')]} {metric}", bbox_inches='tight')
        plt.show()
    
    
    # Second the difference-ones:
    for metric in ["dist2C"]:
        plt.figure(figsize = figureSize)    
        for dMat in measures:
            plt.plot(df_reduced.loc[dMat, metric+"_before"]-df_reduced.loc[dMat, metric+"_after"], label=f"{dMat}", color=colorDict[dMat], marker= markerDict[dMat])
        plt.xlabel('self-reliability in percent', fontsize='x-large')
        plt.ylabel(metric + "before - after", fontsize='x-large')
        plt.legend(loc='center left')
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')
        plt.legend(loc='lower left', fontsize='x-large')
        plt.savefig(path_to_here+"/Figures/"+ f"{file[:file.find('.')]} {metric}", bbox_inches='tight')
        plt.show()
    # Third the quotient-ones:
        
    for metric in ["intransitiveProfiles", "cycles"]:
        plt.figure(figsize = figureSize)    
        for dMat in measures:
            plt.plot(df_reduced.loc[dMat, metric+"_after"]/df_reduced.loc[dMat, metric+"_before"], label=f"{dMat}", color=colorDict[dMat], marker= markerDict[dMat])
        plt.xlabel('self-reliability in percent', fontsize='x-large')
        plt.ylabel(metric+"(after/before)", fontsize='x-large')
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')
        plt.legend(loc='center left', fontsize='x-large')
        plt.savefig(path_to_here+"/Figures/"+ f"{file[:file.find('.')]} {metric}", bbox_inches='tight')
        plt.show()