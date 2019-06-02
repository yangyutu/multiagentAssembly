import numpy as np
import random
import json
import os
from sklearn.metrics.pairwise import euclidean_distances
import math
import sys

class GoalSelectionEnv:
    def __init__(self, configName, randomSeed = 1):
        """
        A model take in a particle configuration and actions and return updated a particle configuration
        """
        
        with open(configName) as f:
            self.config = json.load(f)
        self.randomSeed = randomSeed
        self.read_config()
        self.initilize()

        #self.padding = self.config['']

    def initilize(self):
        if not os.path.exists('Traj'):
            os.makedirs('Traj')
        # import parameter for vector env
        self.viewer = None
        self.steps_beyond_done = None
        self.stepCount = 0
        self.nbActions = 2
        self.info = {}

        random.seed(self.randomSeed)
        np.random.seed(self.randomSeed)
        
        self.initObsMat()
        self.constructSensorArrayIndex()
        self.epiCount = -1

    def read_config(self):

        self.receptHalfWidth = self.config['receptHalfWidth']
        self.padding = self.config['obstacleMapPaddingWidth']
        self.receptWidth = 2 * self.receptHalfWidth + 1
        self.targetClipLength = 2 * self.receptHalfWidth
        self.stateDim = (self.receptWidth, self.receptWidth)
       
        self.sensorArrayWidth = (2*self.receptHalfWidth + 1)
        

        self.episodeEndStep = 500
        if 'episodeLength' in self.config:
            self.episodeEndStep = self.config['episodeLength']
        
        self.startThresh = 1
        self.endThresh = 1
        self.distanceThreshDecay = 10000

        self.targetThreshFlag = False

        if 'targetThreshFlag' in self.config:
            self.targetThreshFlag = self.config['targetThreshFlag']

        if 'target_start_thresh' in self.config:
            self.startThresh = self.config['target_start_thresh']
        if 'target_end_thresh' in self.config:
            self.endThresh = self.config['target_end_thresh']
        if 'distance_thresh_decay' in self.config:
            self.distanceThreshDecay = self.config['distance_thresh_decay']

        self.obstacleFlg = True
        if 'obstacleFlag' in self.config:
            self.obstacleFlg = self.config['obstacleFlag']
            
        self.nStep = self.config['modelNStep']

        self.actionPenalty = 0.0
        if 'actionPenalty' in self.config:
            self.actionPenalty = self.config['actionPenalty']

        self.stepSize = self.config['stepSize']

        self.pixelSize = 1
        if 'pixelSize' in self.config:
            self.pixelSize = self.config['pixelSize']

        self.distanceScale = 20
        if 'distanceScale' in self.config:
            self.distanceScale = self.config['distanceScale']

        self.obstaclePenalty = 1
        if 'obstaclePenalty' in self.config:
            self.obstaclePenalty = self.config['obstaclePenalty']

    def thresh_by_episode(self, step):
        return self.endThresh + (
                self.startThresh - self.endThresh) * math.exp(-1. * step / self.distanceThreshDecay)
    def constructSensorArrayIndex(self):
        x_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        y_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        [Y, X] = np.meshgrid(y_int, x_int)
        self.senorIndex = np.stack((X.reshape(-1), Y.reshape(-1)), axis=1)

    def getSensorInfo(self):
    # sensor information needs to consider orientation information
    # add integer resentation of location
    #    index = self.senorIndex + self.currentState + np.array([self.padding, self.padding])
        transIndex = self.senorIndex.copy()
        i = math.floor(self.currentState[0] + 0.5)
        j = math.floor(self.currentState[1] + 0.5)

        transIndex[:, 0] += self.padding + i
        transIndex[:, 1] += self.padding + j

        # use augumented obstacle matrix to check collision
        self.sensorInfoMat = self.obsMap[transIndex[:, 0], transIndex[:, 1]].reshape(self.receptWidth, -1)


    def getSensorInfoFromPos(self, position):

        transIndex = self.senorIndex.copy()
        i = math.floor(self.currentState[0] + 0.5)
        j = math.floor(self.currentState[1] + 0.5)

        transIndex[:, 0] += self.padding + i
        transIndex[:, 1] += self.padding + j

        # use augumented obstacle matrix to check collision
        sensorInfoMat = self.obsMap[transIndex[:, 0], transIndex[:, 1]].reshape(self.receptWidth, -1)

        # use augumented obstacle matrix to check collision
        return np.expand_dims(sensorInfoMat, axis = 0)

    def getHindSightExperience(self, state, action, nextState, info):

        #if not self.hindSightInfo['obstacle']:
        if True:
            targetNew = self.hindSightInfo['currentState'][0:2]

            distance = targetNew - self.hindSightInfo['previousState'][0:2]

            if self.obstacleFlg:

                sensorInfoMat = self.getSensorInfoFromPos(self.hindSightInfo['previousState'])
                stateNew = {'sensor': sensorInfoMat,
                            'target': distance / self.distanceScale}
            else:
                stateNew = distance / self.distanceScale
            actionNew = action
            rewardNew = 1.0 + self.actionPenaltyCal(action)
            return stateNew, actionNew, None, rewardNew
        else:
            return [None, action, None, 0]

    def constructSensorArrayIndex(self):
        x_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        y_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        [Y, X] = np.meshgrid(y_int, x_int)
        self.senorIndex = np.stack((X.reshape(-1), Y.reshape(-1)), axis=1)

    def actionPenaltyCal(self, action):
        actionNorm = np.linalg.norm(action, ord=2)
        return -self.actionPenalty * actionNorm ** 2

    def obstaclePenaltyCal(self):
        i = math.floor(self.currentState[0] + 0.5)
        j = math.floor(self.currentState[1] + 0.5)

        xIdx = self.padding + i
        yIdx = self.padding + j

        if self.obsMap[xIdx, yIdx] == 1:
            return -self.obstaclePenalty, True
        else:
            return 0, False

    def step(self, action):
        self.hindSightInfo['obstacle'] = False
        self.hindSightInfo['previousState'] = self.currentState.copy()
        reward = 0.0
        #if self.customExploreFlag and self.epiCount < self.customExploreEpisode:
        #    action = self.getCustomAction()
        action = action * self.stepSize
        self.currentState += action # instant movement
        self.hindSightInfo['currentState'] = self.currentState.copy()

        distance = self.targetState - self.currentState

        done = False

        if self.is_terminal(distance):
            reward = 1.0
            done = True

        # penalty for actions

        reward += self.actionPenaltyCal(action)
        #if np.linalg.norm(self.currentState, ord=np.inf) > 25:
        #    reward = -1

        # update sensor information
        if self.obstacleFlg:
            self.getSensorInfo()
            penalty, flag = self.obstaclePenaltyCal()
            reward += penalty
            if flag:
                self.hindSightInfo['obstacle'] = True
                self.currentState -= action

        # update step count
        self.stepCount += 1

        self.info['previousTarget'] = self.targetState.copy()
        self.info['currentState'] = self.currentState.copy()
        self.info['currentTarget'] = self.targetState.copy()

        if self.obstacleFlg:
            state = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                     'target': distance.copy() / self.distanceScale}
        else:
            state = distance / self.distanceScale
        return state, reward, done, self.info.copy()

    def is_terminal(self, distance):
        return np.linalg.norm(distance, ord=np.inf) < self.stepSize

    def reset_helper(self):
        # set target information
        if self.config['dynamicTargetFlag']:
            while True:
                col = random.randint(0, self.mapMat.shape[1] - 1) + self.padding
                row = random.randint(0, self.mapMat.shape[0] - 1) + self.padding
                if np.sum(self.obsMap[row-2:row+3, col-2:col+3]) == 0:
                    break
            self.targetState = np.array([row - self.padding, col - self.padding], dtype=np.int32)



        targetThresh = float('inf')
        if self.targetThreshFlag:
            targetThresh = self.thresh_by_episode(self.epiCount) * max(self.mapMat.shape)
            print('target Thresh', targetThresh)


        if self.config['dynamicInitialStateFlag']:
            while True:

                col = random.randint(0, self.mapMat.shape[1] - 1) + self.padding
                row = random.randint(0, self.mapMat.shape[0] - 1) + self.padding
                distanctVec = np.array([row - self.padding, col - self.padding], dtype=np.float32) - self.targetState
                distance = np.linalg.norm(distanctVec, ord=np.inf)
                if np.sum(self.obsMap[row-2:row+3, col-2:col+3]) == 0 and distance < targetThresh and not self.is_terminal(distanctVec):
                    break
            # set initial state
            print('target distance', distance)
            self.currentState = np.array([row - self.padding, col - self.padding], dtype=np.float32)


    def reset(self):
        self.stepCount = 0
        self.hindSightInfo = {}
        self.info = {}
        self.epiCount += 1

        self.currentState = np.array(self.config['currentState'], dtype=np.float32)
        self.targetState = np.array(self.config['targetState'], dtype=np.float32)

        self.reset_helper()
        # update sensor information
        if self.obstacleFlg:
            self.getSensorInfo()
            
        distance = self.targetState - self.currentState

        self.info['currentTarget'] = self.targetState.copy()

        #angleDistance = math.atan2(distance[1], distance[0]) - self.currentState[2]
        if self.obstacleFlg:
            combinedState = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                             'target': distance / self.distanceScale}
            return combinedState
        else:
            return distance / self.distanceScale

    # def constructObsMap(self, particles):
    #
    #     localPos = particles - self.currentState[0:2]
    #     x = localPos + self.sensorOffSet
    #     y = localPos + self.sensorOffSet
    #
    #     xIdx = np.floor(x / self.sensorPixelSize).astype(np.int)
    #     yIdx = np.floor(y / self.sensorPixelSize).astype(np.int)
    #
    #     xIdx[xIdx < 0] = 0
    #     yIdx[yIdx < 0] = 0
    #     xIdx[xIdx >= self.sensorMatSize] = self.sensorMatSize - 1
    #     yIdx[yIdx >= self.sensorMatSize] = self.sensorMatSize - 1
    #     sensorMat = np.zeros((self.sensorMatSize, self.sensorMatSize), dtype=np.int32)
    #     sensorMat[xIdx, yIdx] = 1

    def initObsMat(self):
        fileName = self.config['mapName']
        self.mapMat = np.genfromtxt(fileName + '.txt')
        self.mapShape = self.mapMat.shape
        padW = self.config['obstacleMapPaddingWidth']
        obsMapSizeOne = self.mapMat.shape[0] + 2*padW
        obsMapSizeTwo = self.mapMat.shape[1] + 2*padW
        self.obsMap = np.ones((obsMapSizeOne, obsMapSizeTwo))
        self.obsMap[padW:-padW, padW:-padW] = self.mapMat
        np.savetxt(self.config['mapName']+'obsMap.txt', self.obsMap, fmt='%d', delimiter='\t')
    

