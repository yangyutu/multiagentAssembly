import numpy as np
import random
import json
import os
from coordinator import globalCoordinator
import GridWorldPython as gw
from sklearn.metrics.pairwise import euclidean_distances
import math
import sys

class HierchicalAssemblySimulator:
    def __init__(self, configName, randomSeed = 1, metaTargetPlanner = None, pathPlanner = None):
        """
        A model take in a particle configuration and actions and return updated a particle configuration
        """

        with open(configName) as f:
            self.config = json.load(f)

        self.randomSeed = randomSeed
        self.coordinator = globalCoordinator()
        self.model = gw.GridWorldPython(configName, randomSeed)
        self.read_config()
        self.initilize()

        # self.padding = self.config['']

    def read_config(self):
        self.N = self.config['N']
        self.totalSteps = self.config['totalSteps']
        self.arrivalDistThresh = self.config['arrivalDistThresh']
        self.controlMode = self.config['controlMode']
        self.controlModes = ['AllOn', 'SequentialVanilla', 'SequentialWithCoarseGoal', 'SequentialWithAll']
        self.trajOutputIntervalPython = self.config['trajOutputIntervalPython']
        if self.controlMode not in self.controlModes:
            print('controlMode Not allowed!')
            exit(1)

        self.assignmentFreq = self.config['assignmentFreq']

    def initilize(self):

        self.model.reset()

        if not os.path.exists('Traj'):
            os.makedirs('Traj')

        self.maxDisp = self.config['dt'] * self.config['maxSpeed'] * self.config['GridWorldNStep']

        # get particle and target positions
        iniConfigPos = np.genfromtxt(self.config['iniConfig'])
        iniConfig = []
        for i in range(self.N):
            pos = iniConfigPos[i]
            iniConfig.append(pos[1:3].tolist() + [0])
        self.particles = np.array(iniConfig)

        iniConfigPos = np.genfromtxt(self.config['targetConfig'])
        iniConfig = []
        for i in range(self.N):
            pos = iniConfigPos[i]
            iniConfig.append(pos[1:3].tolist())

        self.targets = np.array(iniConfig)
        self.targets = self.targets[np.lexsort(self.targets.T)]
        # get the whole coordination plan

        self.orderIdx, self.levelOrder, self.assignment \
            = self.coordinator.getGlobalOrderedAssignment(self.particles[:, 0:2], self.targets)

        self.invAssignment = dict(zip(self.assignment.values(), self.assignment.keys()))

        assignedTargets = []
        for i in range(self.N):
            assignedTargets.append(self.targets[self.assignment[i]])
        self.assignedTargets = np.array(assignedTargets)

        self.assignedProxyTargets = self.assignedTargets.copy()

        self.workingSet = list(range(self.N))
        self.toDoSet = []
        if self.controlMode in ['SequentialLevelVanilla', 'SequentialVanilla', 'SequentialWithCoarseGoal', 'SequentialWithAll']:
            self.workingSet = []
            self.toDoSet = list(range(self.N))

        # construct target dependence
        # target graph is constructed based on assignedTargets, not the original targets
        self.targetDeps = {}
        distMat = euclidean_distances(self.assignedTargets, self.assignedTargets)
        for i in range(self.N):
            distMat[i, i] = 0.0
        distMat[distMat > 3] = 0.0
        distMat[distMat > 0.01] = 1
        for i in range(self.N):
            self.targetDeps[i] = []
            temp = list(np.where(distMat[i] == 1)[0])
            for t in temp:
                if self.orderIdx[self.assignment[i]] > self.orderIdx[self.assignment[t]]:
                    self.targetDeps[i].append(t)
            if len(self.targetDeps[i]) < 2:
                for t in temp:
                    if self.orderIdx[self.assignment[i]] == self.orderIdx[self.assignment[t]]:
                        self.targetDeps[i].append(t)

        print("target dependence:", self.targetDeps)

        self.outputAssignment()
        self.getInitialSet()
        self.finishSet = set()
        self.trajOutputFile = None
        self.OPOutputFile = None
    def outputAssignment(self):
        output = []
        for i, part in enumerate(self.particles[:, 0:2]):
            res = part.tolist() + [self.assignment[i]]
            output.append(res)

        output = np.array(output)
        np.savetxt('assignment.txt', output, fmt='%.3f')

        output = []
        for i, part in enumerate(self.targets):
            res = part.tolist() + [self.orderIdx[i]]
            output.append(res)

        output = np.array(output)
        np.savetxt('targetOrder.txt', output, fmt='%.3f')

    def getInitialSet(self):

        self.initialSet = [self.invAssignment[i] for i in self.levelOrder[0]] \
                                       + [self.invAssignment[i] for i in self.levelOrder[1]] \
                                      # + [self.invAssignment[i] for i in self.levelOrder[2]]

        self.levelOrderIdx = 2

    def getAssignment(self):



        if self.controlMode in ['SequentialVanilla', 'SequentialLevelVanilla', 'SequentialWithCoarseGoal', 'SequentialWithAll']:
            if self.controlMode == 'SequentialLevelVanilla':
                # return the index set of particles and their assignment targets
                dist = np.sqrt(np.sum(np.square(self.assignedTargets[self.workingSet, :] \
                                                - self.particles[self.workingSet, 0:2]), axis=1))
                meanDist = np.mean(dist)
                if meanDist < self.arrivalDistThresh:
                    self.finishSet = [1 if i in self.workingSet else 0 for i in range(self.N)]
                if not self.workingSet:
                    self.workingSet = list(self.initialSet)
                    self.toDoSet = list(set(range(self.N)) - set(self.workingSet))

                else:
                    if np.all(dist < self.arrivalDistThresh):
                        for i in self.levelOrder[self.levelOrderIdx]:
                            self.workingSet.append(self.invAssignment[i])
                            self.toDoSet.remove(self.invAssignment[i])

                        self.levelOrderIdx += 1
            else:
                # return the index set of particles and their assignment targets
                dist = np.sqrt(np.sum(np.square(self.assignedTargets \
                                                - self.particles[:, 0:2]), axis=1))
                for i in self.workingSet:
                    if dist[i] < self.arrivalDistThresh:
                        self.finishSet.add(i)

                if not self.workingSet:
                    self.workingSet = list(self.initialSet)
                    self.toDoSet = list(set(range(self.N)) - set(self.workingSet))
                    print('initial working set: ', self.workingSet)
                else:
                    for i in self.toDoSet:
                        if np.all(dist[self.targetDeps[i]] < self.arrivalDistThresh):
                            print("add new target: ", i, "with level order: ", self.orderIdx[self.assignment[i]], 'dependence dist: ',self.targetDeps[i], dist[self.targetDeps[i]])
                            self.workingSet.append(i)
                            self.toDoSet.remove(i)

    def recordTraj(self, speeds = None):

        if not self.trajOutputFile:
            self.trajOutputFile = open(self.config['trajOutputNamePython'], 'w')

        if not self.OPOutputFile:
            self.OPOutputFile = open(self.config['OPOutputNamePython'], 'w')


        self.distToTarget = np.sqrt(np.sum(np.square(self.assignedTargets - self.particles[:, 0:2]), axis=1))
        self.distToProxyTargets = np.sqrt(np.sum(np.square(self.assignedProxyTargets - self.particles[:, 0:2]), axis=1))

        for i in range(self.N):
            temp = [self.stepCount, i] \
                   + self.particles[i].tolist() \
                   + self.assignedTargets[i].tolist() \
                   + self.assignedProxyTargets[i].tolist() \
                   + [0 if speeds is None else speeds[i]] \
                   + [self.distToTarget[i], self.distToProxyTargets[i]] \
                   + [1 if i in self.workingSet else 0] + [1 if i in self.toDoSet else 0]
            temp = np.array(temp)
            temp.tofile(self.trajOutputFile, sep =' ',format='%.3f')
            self.trajOutputFile.write('\n')

        temp = [self.stepCount, np.sum(self.distToTarget), np.sum(self.distToProxyTargets), np.sum(self.distToProxyTargets[self.workingSet])]
        temp = np.array(temp)
        temp.tofile(self.OPOutputFile, sep =' ', format='%.3f')
        self.OPOutputFile.write('\n')
        print("step:", self.stepCount, 'working set size', len(self.workingSet))
        print("total dist to target, total dist to proxy targets, total dist to proxy target working set")
        print(temp)

    def _calSpeedFromProjection(self, projection, projectionCos):
        if projectionCos > 0.5:
            if projection > self.maxDisp:
                return 1.0
            else:
                return projection / self.maxDisp
        else:
            return 0.0

    def _getProxyTarget(self, dist):
        pass

    def getControlledSpeeds(self):

        # get current particle positions
        pos = self.model.getPositions()
        pos.shape = (self.N, 3)
        self.particles = pos


        if self.stepCount % self.assignmentFreq == 0:
            self.getAssignment()

        speeds = np.zeros(self.N)
        if self.controlMode == 'AllOn':
            phi = self.particles[:, 2]
            projection = (self.assignedTargets[:, 0] - self.particles[:, 0]) \
                         * np.cos(phi) + (self.assignedTargets[:, 1] - self.particles[:, 1]) * np.sin(phi)
            dist = np.sqrt(np.sum(np.square(self.assignedTargets - self.particles[:,0:2]), axis=1))
            projectionCos = projection / dist
            for i in range(self.N):
                speeds[i] = self._calSpeedFromProjection(projection[i], projectionCos[i])
        elif self.controlMode == 'SequentialVanilla':
            phi = self.particles[self.workingSet, 2]
            projection = (self.assignedTargets[self.workingSet, 0] - self.particles[self.workingSet, 0]) \
                         * np.cos(phi) + (self.assignedTargets[self.workingSet, 1] - self.particles[self.workingSet, 1]) * np.sin(phi)
            dist = np.sqrt(np.sum(np.square(self.assignedTargets[self.workingSet,:] - self.particles[self.workingSet, 0:2]), axis=1))
            projectionCos = projection / dist
            for i1, i2 in enumerate(self.workingSet):
                speeds[i2] = self._calSpeedFromProjection(projection[i1], projectionCos[i1])
        elif self.controlModes == 'SequentialWithCoarseGoal':
            dist = np.sqrt(np.sum(np.square(self.assignedTargets[self.workingSet, :] - self.particles[self.workingSet, 0:2]),
                       axis=1))
            self.getProxyTargets(dist)
            phi = self.particles[self.workingSet, 2]
            projection = (self.assignedProxyTargets[self.workingSet, 0] - self.particles[self.workingSet, 0]) \
                         * np.cos(phi) + (self.assignedProxyTargets[self.workingSet, 1] - self.particles[self.workingSet, 1]) * np.sin(phi)
            proxyDist = np.sqrt(np.sum(np.square(self.assignedTargets[self.workingSet, :] - self.particles[self.workingSet, 0:2]),
                       axis=1))
            projectionCos = projection / proxyDist
            for i1, i2 in enumerate(self.workingSet):
                speeds[i2] = self._calSpeedFromProjection(projection[i1], projectionCos[i1])

        return speeds

    def start(self):

        for self.stepCount in range(self.totalSteps):
            speeds = self.getControlledSpeeds()
            if self.stepCount % self.trajOutputIntervalPython == 0:
                self.recordTraj(speeds)
            self.model.step(speeds)
        self.recordTraj(speeds)
        self.trajOutputFile.close()
        self.OPOutputFile.close()
        print(self.workingSet)
        print('idleSet:', self.toDoSet)
        for i in self.toDoSet:
            print(self.particles[i], self.assignedTargets[i], self.targetDeps[i])






