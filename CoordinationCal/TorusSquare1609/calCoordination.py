# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:55:25 2019

@author: yangy
"""

import numpy as np
import sys
import pickle
sys.path.append('..')
from coordinator import globalCoordinator

coordinator = globalCoordinator()
config = {}
config['iniConfig'] = 'torus3276CPP.txt'
config['targetConfig'] = 'squareTargetN1609CPP.txt'
config['loadAssignmentFlag'] = False
config['assignmentFile'] = 'assignmentResults.pkl'
N = 1609
# get particle and target positions
iniConfigPos = np.genfromtxt(config['iniConfig'])
iniConfig = []
for i in range(N):
    pos = iniConfigPos[i]
    iniConfig.append(pos[1:3].tolist() + [0])
particles = np.array(iniConfig)

iniConfigPos = np.genfromtxt(config['targetConfig'])
iniConfig = []
for i in range(N):
    pos = iniConfigPos[i]
    iniConfig.append(pos[1:3].tolist())

targets = np.array(iniConfig)
targets = targets[np.lexsort(targets.T)]
# get the whole coordination plan
print("perform target assignment and coordination")


if not config['loadAssignmentFlag']:
    orderIdx, levelOrder, assignment \
        = coordinator.getGlobalOrderedAssignment(particles[:, 0:2], targets)
    with open(config['assignmentFile'], 'wb') as f:
        pickle.dump([orderIdx, levelOrder, assignment], f)
else:
    with open(config['assignmentFile'], 'rb') as f:
        orderIdx, levelOrder, assignment = pickle.load(f)


