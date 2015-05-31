#****************************************************************
# Copyright (c) 2015, Georgia Tech Research Institute
# All rights reserved.
#
# This unpublished material is the property of the Georgia Tech
# Research Institute and is protected under copyright law.
# The methods and techniques described herein are considered
# trade secrets and/or confidential. Reproduction or distribution,
# in whole or in part, is forbidden except by the express written
# permission of the Georgia Tech Research Institute.
#****************************************************************/

from ..utils import Algorithm 
import numpy as np
from scipy.spatial.distance import pdist, squareform


class Dbscan(Algorithm):
    def __init__(self):
        super(Dbscan, self).__init__()
        self.parameters = ['minPts']
        self.inputs = ['matrix.csv']
        self.outputs = ['assignments.csv']
        self.name ='DBSCAN'
        self.type = 'Clustering'
        self.description = 'Performs DBSCAN clustering on the input dataset.'
        self.parameters_spec = [ { "name" : "Minimum points", "attrname" : "minPts", "value" : 5, "type" : "input" , "step": 1, "max": 15, "min": 1} ]

    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')      

        outputSeeds=True
        data = self.inputData
        numSamples = np.shape(data)[0]
        
        # compute distances between each samples
        distanceVector = pdist(data, 'euclidean')
        distanceData = squareform(distanceVector, force='tomatrix')
        
        # estimate eps based on standard deviation of the distances
        averageDistance = np.sum(distanceData, axis=0) / (numSamples-1)
        eps = np.std(averageDistance)
        
        # intialize class labels and isSeed variable
        nClass = np.zeros(numSamples)
        labels = np.zeros(numSamples)
        isSeed = labels > 1
            
        idx = 0
        for index in range(0, numSamples):
            
            unClassed = np.array(range(numSamples))[labels < 1]
            
            if labels[index] == 0:
                
                # find which samples are within eps of sample point index
                reachables = unClassed[distanceData[index, unClassed] <= eps]
                
                if len(reachables) + nClass[index] < self.minPts:
                    labels[index] = -1
                else:
                    idx = idx + 1
                    labels[index] = idx
                    isSeed[index] = True
                    reachables = np.setdiff1d(reachables, np.array(index))
                    unClassed = np.setdiff1d(unClassed, np.array(index))
                    nClass[reachables] = nClass[reachables] + 1
                    
                    # form a cluster till the cluster stablizes when all of the
                    # reachable samples for the sample index are properly clustered
                    while np.array([reachables]).any():
                        labels[reachables] = idx
                        reachableTemp = reachables
                        reachables = []
                        
                        # for each reachable samples of the sample index and of
                        # each reachable sample not yet clustered, try to cluster
                        # the sample
                        #bug fix...
                        if np.size(np.shape(reachableTemp)) == 0:
                            reachableTemp = np.array([reachableTemp])

                        for index2 in range(0, len(reachableTemp)):
                            idx2 = reachableTemp[index2]
                            idx2Reachables = unClassed[distanceData[idx2,unClassed] <= eps]
                            
                            if len(idx2Reachables) + nClass[idx2] >= self.minPts:
                                isSeed[idx2] = True
                                
                                # the line below seems to be always empty ....
                                labels[idx2Reachables[labels[idx2Reachables] < 0]] = idx
                                if np.size(np.shape(reachables)) == 0:
                                    reachables = np.union1d([reachables], idx2Reachables[labels[idx2Reachables] == 0]) 
                                else:
                                    reachables = np.union1d(reachables, idx2Reachables[labels[idx2Reachables] == 0])
                                #reachables = np.union1d(reachables, 
                                #                           idx2Reachables[labels[idx2Reachables] == 0])
                                reachables = np.int0(reachables)                      

                            nClass[idx2Reachables] = nClass[idx2Reachables] + 1
                            unClassed = np.setdiff1d(unClassed, idx2)
                    
            if len(unClassed) == 0:
                # break out of the for loop because all of the samples are clustered
                break
            
        if np.any(labels == -1):
            # set noise samples as 0
            labels[labels == -1] = 0
        
        # output estimated labels, eps, self.minPts, and isSeed if seeds = True
        if outputSeeds == True and idx > 0:
            dbscanStruct = {'clusterLabels':labels, 'eps':eps, 'minPts':self.minPts, 
            'isSeed':isSeed}
        else:
            dbscanStruct = {'clusterLabels':labels, 'eps':eps, 'minPts':self.minPts}          
                    
                    
        self.clusters = dbscanStruct["clusterLabels"].astype(int)
        self.results = {'assignments.csv':self.clusters}
