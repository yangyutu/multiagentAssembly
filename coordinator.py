# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 17:48:04 2019

@author: yangy
"""

import numpy as np
from copy import deepcopy
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

        print("run level order for targets and particles")
        targetOrderIdx, targetLevelOrder = self.getTargetLevelOrder(targets)
        particleOrderIdx, particleLevelOrder = self.getParticleLevelOrder(particles)
        print("run Hungarian algorithm")

        assignment, inverseAssignment = self.getLayerSequentialAssignment(particles, particleLevelOrder,
                                                                          targets, targetLevelOrder)

        return assignment, inverseAssignment, \
               targetOrderIdx, targetLevelOrder, \
               particleOrderIdx, particleLevelOrder

    def getHungarianAssignment(self, particles, targets):
        # assign particles to targets
        # assume number of particles <= number of targets
        distMat = euclidean_distances(particles, targets)
        row, col = linear_sum_assignment(distMat)
        assignment = dict(zip(row, col))
        inverseAssignment = dict(zip(col, row))
        return assignment, inverseAssignment

    def getTargetLevelOrder(self, targets):
        csr_graph, myGraph = self.generateGraphMatrix(targets)
        centralTarget = self.findCentralTarget(csr_graph)
        levelOrder = self.levelOrderFromCentralTarget(myGraph, centralTarget)
        
        orderOut = {}
        for orderIdx, order in enumerate(levelOrder):
            for particleIdx in order:
                orderOut[particleIdx] = orderIdx
        
        return orderOut, levelOrder
    
    def generateGraphMatrix(self, targets):
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
        
        
    def levelOrderFromCentralTarget(self, graph, centerTargetIdx):
        queue = [centerTargetIdx]
        visited = set([centerTargetIdx ])
        levelOrder = []
        while queue:
            layer = []
            nextQueue = []
            for node in queue:
                layer.append(node)
                for nb in graph[node]:
                    if nb not in visited:
                        visited.add(nb)                        
                        nextQueue.append(nb)
            levelOrder.append(layer)
            queue = nextQueue
        return levelOrder

    def getParticleLevelOrder(self, particles):
        csr_graph, particleGraph = self.generateGraphMatrix(particles)

        particleNbCount = {}
        for k, v in particleGraph.items():
            particleNbCount[k] = len(particleGraph[k])

        queue = []
        visited = set()
        levelOrder = []
        for k, v in particleNbCount.items():
            if v < 6:
                queue.append(k)
                visited.add(k)
        while queue:
            nextLevel = []
            layer = []
            for node in queue:
                layer.append(node)
                for nb in particleGraph[node]:
                    particleNbCount[nb] -= 1
                    if particleNbCount[nb] < 6 and nb not in visited:
                        visited.add(nb)
                        nextLevel.append(nb)
            levelOrder.append(layer)
            queue = nextLevel

        orderOut = {}
        for orderIdx, order in enumerate(levelOrder):
            for particleIdx in order:
                orderOut[particleIdx] = orderIdx

        return orderOut, levelOrder



    def getLayerSequentialAssignment(self, particles, particleLevelOrder_origin, targets, targetLevelOrder_origin):
        targetLevelOrder = deepcopy(targetLevelOrder_origin)
        particleLevelOrder = deepcopy(particleLevelOrder_origin)
        finalAssignment = {}
        targetIdx, particleIdx = 0, 0

        while True:
            print(targetIdx)
            if len(targetLevelOrder[targetIdx]) == len(particleLevelOrder[particleIdx]):
                assignment, inverseAssignment = self.getHungarianAssignment(particles[particleLevelOrder[particleIdx]],
                                                                            targets[targetLevelOrder[targetIdx]])
                for k, v in inverseAssignment.items():
                    finalAssignment[particleLevelOrder[particleIdx][k]] = targetLevelOrder[targetIdx][v]
                targetIdx += 1
                particleIdx += 1
            elif len(targetLevelOrder[targetIdx]) < len(particleLevelOrder[particleIdx]):
                assignment, inverseAssignment = self.getHungarianAssignment(targets[targetLevelOrder[targetIdx]],
                                                                particles[particleLevelOrder[particleIdx]])
                # inverseAssignment maps from particles to targets
                for k, v in inverseAssignment.items():
                    finalAssignment[particleLevelOrder[particleIdx][k]] = targetLevelOrder[targetIdx][v]
                targetIdx += 1
                toRemove = [particleLevelOrder[particleIdx][i] for i in inverseAssignment.keys()]
                for i in toRemove:
                    particleLevelOrder[particleIdx].remove(i)
            elif len(targetLevelOrder[targetIdx]) > len(particleLevelOrder[particleIdx]):
                assignment, inverseAssignment = self.getHungarianAssignment(particles[particleLevelOrder[particleIdx]],
                                                                            targets[targetLevelOrder[targetIdx]])
                # inverseAssignment maps from particles to targets
                for k, v in assignment.items():
                    finalAssignment[particleLevelOrder[particleIdx][k]] = targetLevelOrder[targetIdx][v]
                particleIdx += 1

                toRemove = [targetLevelOrder[targetIdx][i] for i in assignment.values()]
                for i in toRemove:
                    targetLevelOrder[targetIdx].remove(i)

            if targetIdx == len(targetLevelOrder) or particleIdx == len(particleLevelOrder):
                break

        # sanity check, target number should be the same of particle order
        if len(finalAssignment.keys()) == len(particles):
            print("finish assignment!")

        finalInverseAssignment = dict(zip(finalAssignment.values(), finalAssignment.keys()))

        return finalAssignment, finalInverseAssignment
    