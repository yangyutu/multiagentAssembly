import numpy as np
import random
import json
import os
from sklearn.metrics.pairwise import euclidean_distances
import math
import sys



def InstantMoveModel(particleState, goalAction):
    return goalAction


class GoalSelectionEnv:
    




class MultiActiveParticleBaseEnv:
    def __init__(self, configName, randomSeed = 1, goalSectionModel = None):
        """
        A model take in a particle configuration and actions and return updated a particle configuration
        """
        
        with open(configName) as f:
            self.config = json.load(f)
        self.randomSeed = randomSeed
        self.goalSelectionModel = goalSectionModel
        self.model = model
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

        #self.customExploreFlag = False
        #if 'customExploreFlag' in self.config:
        #    self.customExploreFlag = self.config['customExploreFlag']
        #    self.customExploreEpisode = self.config['customExploreEpisode']


    def thresh_by_episode(self, step):
        return self.endThresh + (
                self.startThresh - self.endThresh) * math.exp(-1. * step / self.distanceThreshDecay)
    def constructSensorArrayIndex(self):
        x_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        y_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        [Y, X] = np.meshgrid(y_int, x_int)
        self.senorIndex = np.stack((X.reshape(-1), Y.reshape(-1)), axis=1)
        

    def getSensorInfoFromPos(self, position):
        phi = position[2]

        rotMatrx = np.matrix([[math.cos(phi),  -math.sin(phi)],
                              [math.sin(phi), math.cos(phi)]])
        transIndex = np.matmul(self.senorIndex, rotMatrx.T).astype(np.int)

        i = math.floor(position[0] + 0.5)
        j = math.floor(position[1] + 0.5)

        transIndex[:, 0] += self.padding + i
        transIndex[:, 1] += self.padding + j

        # use augumented obstacle matrix to check collision
        sensorInfoMat = self.obsMap[transIndex[:, 0], transIndex[:, 1]].reshape(self.receptWidth, -1)

        # use augumented obstacle matrix to check collision
        return np.expand_dims(sensorInfoMat, axis = 0)

    def getHindSightExperience(self, state, action, nextState, info):

        targetNew = self.hindSightInfo['currentState'][0:2]

        distance = targetNew - self.hindSightInfo['previousState'][0:2]
        phi = self.hindSightInfo['previousState'][2]



        # distance will be changed from lab coordinate to local coordinate
        dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
        dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)

        angle = math.atan2(dy, dx)
        if math.sqrt(dx**2 + dy**2) > self.targetClipLength:
            dx = self.targetClipLength * math.cos(angle)
            dy = self.targetClipLength * math.sin(angle)

        # recover the global target position after target mapping
        globalTargetX = self.currentState[0] + dx * math.cos(phi) - dy * math.sin(phi)
        globalTargetY = self.currentState[1] + dx * math.sin(phi) + dy * math.cos(phi)


        if self.obstacleFlg:

            sensorInfoMat = self.getSensorInfoFromPos(self.hindSightInfo['previousState'])
            stateNew = {'sensor': sensorInfoMat,
                        'target': np.array([dx , dy])}
        else:
            stateNew = np.array([dx , dy])

        actionNew = action
        rewardNew = 1.0
        return stateNew, actionNew, None, rewardNew

    def constructSensorArrayIndex(self):
        x_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        y_int = np.arange(-self.receptHalfWidth, self.receptHalfWidth + 1)
        [Y, X] = np.meshgrid(y_int, x_int)
        self.senorIndex = np.stack((X.reshape(-1), Y.reshape(-1)), axis=1)
        # sensormap maps a location (x, y) to to an index. for example (-5, -5) to 0
        # self.sensorMap = {}
        # for i, x in enumerate(x_int):
        #     for j, y in enumerate(y_int):
        #         self.sensorMap[(x, y)] = i * self.receptWidth + j


    def getCustomAction(self):

        if self.config['particleType'] == 'FULLCONTROL':
            choice = np.random.randint(0, 3)
            if choice == 0:
                action = np.array([1, 0])
            elif choice == 1:
                action = np.array([1, -1])
            elif choice == 2:
                action = np.array([1, 1])
        elif self.config['particleType'] == 'VANILLASP':
            action = np.array([1])
        elif self.config['particleType'] == 'CIRCLER':
            action = np.array([1])
        elif self.config['particleType'] == 'SLIDER':
            choice = np.random.randint(0, 3)
            if choice == 0:
                action = np.array([1])
            elif choice == 1:
                action = np.array([0])
            elif choice == 2:
                action = np.array([-1])
        return action

    def step(self, action):
        self.hindSightInfo['previousState'] = self.currentState.copy()
        reward = 0.0
        #if self.customExploreFlag and self.epiCount < self.customExploreEpisode:
        #    action = self.getCustomAction()
        self.model.step(self.nStep, action)
        self.currentState = self.model.getPositions()

        self.hindSightInfo['currentState'] = self.currentState.copy()

        distance = self.targetState - self.currentState[0:2]

        done = False

        if self.is_terminal(distance):
            reward = 1.0
            done = True


        # penalty for actions
        actionNorm = np.linalg.norm(action, ord = 0)
        reward -= self.actionPenalty * actionNorm


        # update sensor information
        if self.obstacleFlg:
            self.getSensorInfo()
        # update step count
        self.stepCount += 1

        # distance will be changed from lab coordinate to local coordinate
        phi = self.currentState[2]
        dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
        dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)

        angle = math.atan2(dy, dx)
        if math.sqrt(dx**2 + dy**2) > self.targetClipLength:
            dx = self.targetClipLength * math.cos(angle)
            dy = self.targetClipLength * math.sin(angle)

        # recover the global target position after target mapping
        globalTargetX = self.currentState[0] + dx * math.cos(phi) - dy * math.sin(phi)
        globalTargetY = self.currentState[1] + dx * math.sin(phi) + dy * math.cos(phi)


        self.info['previousTarget'] = self.info['currentTarget'].copy()
        self.info['currentState'] = self.currentState.copy()
        self.info['targetState'] = self.targetState.copy()
        self.info['currentTarget'] = np.array([globalTargetX, globalTargetY])
        self.info['currentDistance'] = math.sqrt(dx**2 + dy**2)

        if self.obstacleFlg:

            state = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                     'target': np.array([dx , dy])}
        else:
            state = np.array([dx , dy])
        return state, reward, done, self.info.copy()

    def is_terminal(self, distance):
        return np.linalg.norm(distance, ord=np.inf) < self.endThresh

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
            self.currentState = np.array([row - self.padding, col - self.padding, random.random()*2*math.pi], dtype=np.float32)


    def reset(self):
        self.stepCount = 0
        self.hindSightInfo = {}
        self.info = {}
        self.epiCount += 1

        self.currentState = np.array(self.config['currentState'], dtype=np.float32)
        self.targetState = np.array(self.config['targetState'], dtype=np.int32)

        self.reset_helper()
        self.model.createInitialState(self.currentState[0], self.currentState[1], self.currentState[2])
        # update sensor information
        if self.obstacleFlg:
            self.getSensorInfo()
        distance = self.targetState - self.currentState[0:2]

        # distance will be change to local coordinate
        phi = self.currentState[2]
        dx = distance[0] * math.cos(phi) + distance[1] * math.sin(phi)
        dy = - distance[0] * math.sin(phi) + distance[1] * math.cos(phi)

        angle = math.atan2(dy, dx)
        if math.sqrt(dx**2 + dy**2) > self.targetClipLength:
            dx = self.targetClipLength * math.cos(angle)
            dy = self.targetClipLength * math.sin(angle)

        globalTargetX = self.currentState[0]+ dx * math.cos(phi) - dy * math.sin(phi)
        globalTargetY = self.currentState[1]+ dx * math.sin(phi) + dy * math.cos(phi)

        self.info['currentTarget'] = np.array([globalTargetX, globalTargetY])

        #angleDistance = math.atan2(distance[1], distance[0]) - self.currentState[2]
        if self.obstacleFlg:
            combinedState = {'sensor': np.expand_dims(self.sensorInfoMat, axis=0),
                             'target': np.array([dx , dy ])}
            return combinedState
        else:
            return np.array([dx , dy])

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
