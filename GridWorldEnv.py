import GridWorldPython as gw
import numpy as np
import random
import json
import os
from sklearn.metrics.pairwise import euclidean_distances
import math

class GridWorldEnv():
    def __init__(self, configName, randomSeed = 1):
        
        with open(configName) as f:
            self.config = json.load(f)

        self.model = gw.GridWorldPython(configName, randomSeed)

        self.receptHalfWidth = self.config['receptHalfWidth']
        self.receptWidth = 2 * self.receptHalfWidth + 1
        if not os.path.exists('Traj'):
            os.makedirs('Traj')

        self.stateDim = (self.receptWidth, self.receptWidth)
        self.numP = self.config['N']
        self.n_channel = self.config['n_channel']

        self.sensorArrayWidth = (2*self.receptHalfWidth + 1)
        self.episodeCount = 0

        self.episodeEndStep = 500
        if 'BDModelEpisodeEndStep' in self.config:
            self.episodeEndStep = self.config['BDModelEpisodeEndStep']

        self.customIniConfig = False
        if 'BDModelcustomIniConfig' in self.config:
            self.customIniConfig = self.config['BDModelcustomIniConfig']

        self.useChannel = 1
        if 'useChannels' in self.config:
            self.customIniConfig = self.config['useChannels']

        self.constantPropel = False
        self.nbActions = 4
        if 'constantPropel' in self.config:
            self.constantPropel = self.config['constantPropel']
            self.nbActions = 3
        # import parameter for vector env
        self.viewer = None
        self.steps_beyond_done = None
        self.stepCount = 0

        self.infoDict = {}
        self.randomSeed = randomSeed
        random.seed(self.randomSeed)
        np.random.seed(self.randomSeed)

        self.mapInfo_origin = np.genfromtxt(self.config['mapName']+'.txt')
        self.mapInfo = self.mapInfo_origin.copy()
        self.startThresh = 1
        self.endThresh = 1
        self.distanceThreshDecay = 10000

        if 'dist_start_thresh' in self.config:
            self.startThresh = self.config['dist_start_thresh']
        if 'dist_end_thresh' in self.config:
            self.endThresh = self.config['dist_end_thresh']
        if 'distance_thresh_decay' in self.config:
            self.distanceThreshDecay = self.config['distance_thresh_decay']

        self.iniConfigCenter = [0, 0]
        if 'iniConfigCenter' in self.config:
            self.iniConfigCenter = self.config['iniConfigCenter']

        self.rewardShareThresh = 1000
        if 'rewardShareThresh' in self.config:
            self.rewardShareThresh = self.config['rewardShareThresh']

        self.rewardShareCoeff =1
        if 'rewardShareCoeff' in self.config:
            self.rewardShareCoeff = self.config['rewardShareCoeff']

        self.rewardScale = 1
        if 'rewardScale' in self.config:
            self.rewardShareCoeff = self.config['rewardScale']

        self.attackFoodThresh = self.numP - 1
        if 'attackFoodThresh' in self.config:
            self.attackFoodThresh = self.config['attackFoodThresh']

        self.stayInGroupReward = 0.025/self.numP/self.numP
        if 'stayInGroupReward' in self.config:
            self.stayInGroupReward = self.config['stayInGroupReward']
        #self.padding = self.config['']

        self.foodCount = 0
        self.epiCount = -1
        self.posXSet = []
        self.posYSet = []


        self.rewardFromCooperation = 0

        self.foodPositionList = []
        self.partPositions = None

    def thresh_by_episode(self, step):
        return self.endThresh + (
                self.startThresh - self.endThresh) * math.exp(-1. * step / self.distanceThreshDecay)


    def generateIniconfig(self):
        # set target information

        distVec1 = np.array(self.mapInfo.shape) - np.array(self.iniConfigCenter)
        distVec2 = np.array(self.iniConfigCenter)
        distThresh = self.thresh_by_episode(self.epiCount) * np.min(np.vstack((distVec1, distVec2))) - 2
        print('distThresh,', distThresh)

        while True:
            pos_x = np.random.randint(self.iniConfigCenter[0] - distThresh, self.iniConfigCenter[0] + distThresh + 1, self.numP)
            pos_y = np.random.randint(self.iniConfigCenter[1] - distThresh, self.iniConfigCenter[1] + distThresh + 1, self.numP)

            pos = np.vstack((pos_x, pos_y)).T
            #pos = np.reshape(pos, (self.numP, -1))
            distMat = euclidean_distances(pos, pos)
            for i in range(distMat.shape[0]):
                distMat[i, i] = 10
            if np.all(distMat > 2.3):
                break
        iniConfig = np.vstack((pos_x, pos_y, np.random.rand(self.numP)*2*np.pi)).T
        iniConfig = iniConfig.flatten()
        self.posXSet.append(pos_x)
        self.posYSet.append(pos_y)

        return iniConfig


    def reset(self):
        self.epiCount += 1
        self.infoDict = {}
        self.rewardFromCooperation = 0
        self.mapInfo = self.mapInfo_origin.copy()

        self.foodPositionList = np.array(np.where(self.mapInfo == 1)).T
        self.foodCount = len(self.foodPositionList)

        self.stepCount = 0
        self.model.reset()

        if self.customIniConfig:
            iniConfig = self.generateIniconfig()
            self.model.setIniConfig(iniConfig)

        self.infoDict['reset config'] = self.model.getPositions()
        self.infoDict['food Count'] = self.foodCount
        obs = self.getObservation()


        return obs

    def getObservation(self):
        self.partPositions = self.model.getPositions()
        self.partPositions.shape = (self.numP, 3)

        obs = self.model.getObservation()
        # we only change the shape without create a new array
        obs.shape = (self.numP, self.n_channel, self.sensorArrayWidth, self.sensorArrayWidth)

        # only get the first two channels
        #idx = np.arange(0, self.numP * self.n_channel, self.n_channel).tolist()
        idx = [0, 1]

        obs = obs[:, idx, :, :]

        # now get food observation
        self.setFoodObservation(obs)
        return obs

    def setFoodObservation(self, obs):

        # for foodPos in self.foodPositionList:
        #     for i, partPos in enumerate(self.partPositions):
        #         localPos = foodPos - partPos[0:2]
        #         phi = partPos[2]
        #         theta = math.atan2(localPos[1], localPos[0])
        #         dist = math.sqrt(localPos[0]**2 + localPos[1]**2)
        #         if dist > (self.receptHalfWidth - 1):
        #             dist = self.receptHalfWidth - 1
        #         localPos[0] = dist*math.cos(theta)
        #         localPos[1] = dist*math.sin(theta)
        #         # now transform to rotated local frame
        #         rotMatrx = np.matrix([[math.cos(phi),  math.sin(phi)],
        #                               [-math.sin(phi), math.cos(phi)]])
        #         transIndex = np.matmul(localPos, rotMatrx.T)
        #         transIndex = np.floor(transIndex + 0.5).astype(np.int).tolist()
        #         # set food location info to first channel of particle i
        #         obs[i, 0, transIndex[0][0] + self.receptHalfWidth, transIndex[0][1] + self.receptHalfWidth] = 1


        for i, partPos in enumerate(self.partPositions):
            localPos = self.foodPositionList - partPos[0:2]
            phi = partPos[2]
            theta = np.arctan2(localPos[:,1], localPos[:,0])
            dist = np.sqrt(np.sum(np.square(localPos), axis=1))

            dist[dist > (self.receptHalfWidth - 1)] = self.receptHalfWidth - 1
            localPos[:, 0] = dist * np.cos(theta)
            localPos[:, 1] = dist * np.sin(theta)
            # now transform to rotated local frame
            rotMatrx = np.matrix([[math.cos(phi), math.sin(phi)],
                                  [-math.sin(phi), math.cos(phi)]])
            transIndex = np.matmul(localPos, rotMatrx.T)
            transIndex = np.floor(transIndex + 0.5).astype(np.int)
            # set food location info to first channel of particle i
            obs[i, 0, transIndex[:,0] + self.receptHalfWidth, transIndex[:,1] + self.receptHalfWidth] = 1

    def step(self, action):
        self.stepCount += 1
        self.model.step(action)
        rewards, done = self.calRewards(action)
        obs = self.getObservation()

        return obs, rewards, done, self.infoDict.copy()

    def calRewards(self, action):

        pos = self.partPositions[:, [0, 1]]
        distMat = euclidean_distances(pos, pos)

        inTargetFlag = np.zeros((self.numP,),dtype=np.int)
        done = False
        rewards = np.zeros((self.numP,),dtype=np.float)

        for i, pos in enumerate(self.partPositions):
            x_int = int(math.floor(pos[0] + 0.5))
            y_int = int(math.floor(pos[1] + 0.5))
            xIdx = [max(x_int-1, 0), x_int, min(x_int + 1, self.mapInfo.shape[0]-1)]
            yIdx = [max(y_int - 1, 0), y_int, min(y_int + 1, self.mapInfo.shape[1]-1)]

            nbIdx = np.where(distMat[i] < self.rewardShareThresh)[0]

            if np.any(self.mapInfo[xIdx,yIdx] == 1) and len(nbIdx) > self.attackFoodThresh:
                rewards[i] = 1
                self.mapInfo[xIdx,yIdx] = 0
                inTargetFlag[i] = 1
                self.foodCount -= 1
                print('### clear food at ', x_int, y_int, 'by ', i)
            if self.mapInfo[x_int,y_int] == -1:
                # hazard
                rewards[i] = -1
                #self.mapInfo[x_int, y_int] += 1
                inTargetFlag[i] = 1
            if action[i] > 0:
                # penalty for using energy
                rewards[i] -= 0.00

        # reward for staying close to each other
        rewards += self.stayInGroupReward*len(np.where(distMat < self.rewardShareThresh)[0])

        # reward sharing
        reward_share = rewards.copy()
        if np.any(inTargetFlag == 1):
            # if food is consumed by any particle, we need to update the food location
            self.foodPositionList = np.array(np.where(self.mapInfo == 1)).T
            self.foodCount = len(self.foodPositionList)


            #for i in range(distMat.shape[0]):
                #distMat[i, i] = 100

            for i, dist in enumerate(distMat):
                idx = np.where(dist < self.rewardShareThresh)[0]
                # if a particle gain reward and it has neighbors, we also give extra reward to the particle to encourage its behavior of waiting for neighbors when hunting for food
                if len(idx) > 1:
                    reward_share[i] += np.sum(rewards[idx]) * self.rewardShareCoeff * 0.5

                    print('reward sharing:', i, idx)

        reward_share /= self.rewardScale
        self.rewardFromCooperation += np.sum(reward_share - rewards/self.rewardScale)
        self.infoDict['rewardFromCooperation'] = self.rewardFromCooperation

        if self.foodCount == 0:
            done = True
            self.infoDict['finish config'] = self.model.getPositions()
#        if np.all(inTargetFlag == 1):
#            done = True
#            self.infoDict['finish config'] = self.model.getPositions()

        # handle reward sharing
        #xy = partPositions[:,0:2]
        #distMat = euclidean_distances(xy, xy)
        return reward_share, done

    def getSingleParticleSensorInfoFromPos(self, partConfig):

        obs = self.model.getObservation()
        obs.shape = (self.numP, self.n_channel, self.sensorArrayWidth, self.sensorArrayWidth)
        return obs


