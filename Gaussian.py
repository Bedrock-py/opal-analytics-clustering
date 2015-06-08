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

from analytics.utils import Algorithm 
from sklearn.mixture import GMM

import time, os
import numpy as np


class Gaussian(Algorithm):
    def __init__(self):
        super(Gaussian, self).__init__()
        self.parameters = ['numClusters','covType']
        self.inputs = ['matrix.csv']
        self.outputs = ['assignments.csv']
        self.name ='Gaussian Mixture'
        self.type = 'Clustering'
        self.description = 'Performs Gaussian Mixture clustering on the input dataset.'
        self.parameters_spec = [ { "name" : "Clusters", "attrname" : "numClusters", "value" : 3, "type" : "input", "step": 1, "max": 15, "min": 1 }, 
            { "name" : "Covariance Type", "attrname" : "covType", "value" : "diag", "type" : "select", "options": ['spherical', 'tied', 'diag', 'full'] }] 

    
    def compute(self, filepath, **kwargs):
        self.inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',')
        print 'gaussian started...'
        gaussianMixtureResult = GMM(n_components=self.numClusters, covariance_type=self.covType)
        gaussianMixtureResult.fit(self.inputData)
        self.clusters = gaussianMixtureResult.predict(self.inputData)
        self.results = {'assignments.csv': self.clusters.astype(int)}
