# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 17:48:04 2019

@author: yangy
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components

class globalCoordinator:
    def getGlobalOrderedAssignment(self, particles, targets):
        """return
        orderIdx: assembly order, targets defined by shortest distance to the core
        levelOrder: list of lists
        assignement: an dictionary
        """
        self.N = targets.shape[0]
        distMat = euclidean_distances(particles, targets)
        row, col = linear_sum_assignment(distMat)
        assignment = dict(zip(row, col))
        orderIdx, levelOrder = self.getOrder(targets)
        return orderIdx, levelOrder, assignment
        
    def getOrder(self, targets):
        csr_graph, myGraph = self.generateTargetGraphMatrix(targets)
        centralTarget = self.findCentralTarget(csr_graph)
        levelOrder = self.levelOrderFromCentralTarget(myGraph, centralTarget)
        
        orderOut = {}
        for orderIdx, order in enumerate(levelOrder):
            for particleIdx in order:
                orderOut[particleIdx] = orderIdx
        
        return orderOut, levelOrder
    
    def generateTargetGraphMatrix(self, targets):
        N = targets.shape[0]
        distMat = euclidean_distances(targets, targets)
        for i in range(N):
            distMat[i, i] = 0.0
        distMat[distMat > 3] = 0.0
        distMat[distMat > 0.01] = 1
        csr_graph = csr_matrix(distMat)
        n_components, labels = connected_components(csgraph=csr_graph, directed=False, return_labels=True)
        print('graph components:', n_components)
        myGraph = {i: [] for i in range(N)}
        for i in range(N):
            dist = distMat[i]
            myGraph[i] = [j for j in range(N) if dist[j] > 0.0]   
        #print(csr_graph)
        #print(myGraph)
        return csr_graph, myGraph
    
    def findCentralTarget(self, graph):
        shortestDist = floyd_warshall(csgraph=graph, directed=False, return_predecessors=False)
        sumDist = np.sum(shortestDist, axis=0)
        centralTarget = np.argmin(sumDist)
        return centralTarget
        
        
    def levelOrderFromCentralTarget(self, graph, targetIdx):
        queue = [targetIdx]
        visited = set([targetIdx])
        levelOrder = []
        while queue:
            layer = []
            nextQueue = []
            for node in queue:
                for nb in graph[node]:
                    if nb not in visited:
                        visited.add(nb)                        
                        nextQueue.append(nb)
                layer.append(node)
            levelOrder.append(layer)
            queue = nextQueue
        return levelOrder    
    
    
    
    
    
    