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

from bedrock.analytics.utils import * 
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

class Hierclust(Algorithm):
    def __init__(self):
        super(Hierclust, self).__init__()
        self.parameters=['numClusters','hierMethod','hierMetric']
        self.inputs = ['matrix.csv']
        self.outputs = ['assignments.csv']
        self.name ='Hierarchical'
        self.type = 'Clustering'
        self.description = 'Performs Hierarchical clustering on the input dataset.'
        self.parameters_spec = [ { "name" : "Clusters", "attrname" : "numClusters", "value" : 3, "type" : "input" , "step": 1, "max": 15, "min": 1},
            { "name" : "Method", "attrname" : "hierMethod", "value" : "ward", "type" : "select" , "options": ['ward','average','complete']},
            { "name" : "Metric", "attrname" : "hierMetric", "value" : "euclidean", "type" : "select" , "options": ['euclidean','manhattan','cityblock'] }]
    
    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')      
        if len(self.inputData.shape) == 1:
            self.inputData.shape=[self.inputData.shape[0],1]
        linkageOut = linkage(self.inputData, method=str(self.hierMethod), metric=str(self.hierMetric))
        self.clusters = fcluster(linkageOut, t=self.numClusters, criterion="maxclust").astype(int)
        self.results = {'assignments.csv':self.clusters}
