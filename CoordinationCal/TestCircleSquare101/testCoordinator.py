# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:55:25 2019

@author: yangy
"""

import numpy as np
import sys
sys.path.append('..')
from coordinator import globalCoordinator

coordinator = globalCoordinator()
config = {}
config['iniConfig'] = 'circle451CPP.txt'
config['targetConfig'] = 'Square101CPP.txt'
N = 101

# get particle and target positions
iniConfigPos = np.genfromtxt(config['iniConfig'])
iniConfig = []
for i in range(N):
    pos = iniConfigPos[i]
    iniConfig.append(pos[1:3].tolist())
particles = np.array(iniConfig)

iniConfigPos = np.genfromtxt(config['targetConfig'])
iniConfig = []
for i in range(N):
    pos = iniConfigPos[i]
    iniConfig.append(pos[1:3].tolist())

targets = np.array(iniConfig)
orderIdx, levelOrder,  assignment, _ = coordinator.getGlobalOrderedAssignment(particles, targets)

output = []
for i, part in enumerate(particles):
    res = part.tolist() + [assignment[i]]
    output.append(res)
    
output = np.array(output)
np.savetxt('assignment.txt', output, fmt='%.3f')


output = []
for i, part in enumerate(targets):
    res = part.tolist() + [orderIdx[i]]
    output.append(res)
    
output = np.array(output)
np.savetxt('targetOrder.txt', output, fmt='%.3f')



