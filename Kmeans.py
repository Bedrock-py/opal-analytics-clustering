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

<<<<<<< HEAD
from analytics.utils import Algorithm 
=======
from analytics.utils import * 

import time, os
>>>>>>> a71c93a0611904a761387203a29cfb852794aae1
from sklearn.cluster import KMeans
import numpy as np


class Kmeans(Algorithm):
    def __init__(self):
        super(Kmeans, self).__init__()
        self.parameters = ['numClusters']
        self.inputs = ['matrix.csv']
        self.outputs = ['assignments.csv']
        self.name ='KMeans'
        self.type = 'Clustering'
        self.description = 'Performs K-means clustering on the input dataset.'
        self.parameters_spec = [ { "name" : "Clusters", "attrname" : "numClusters", "value" : 3, "type" : "input", "step": 1, "max": 15, "min": 1 }]

    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')
        kmeansResult = KMeans(init='k-means++', n_clusters=int(self.numClusters), n_init=30, max_iter=1000)
        kmeansResult.fit(self.inputData)
        self.clusters = kmeansResult.labels_.astype(int)
        self.results = {'assignments.csv': self.clusters}
