# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:55:25 2019

@author: yangy
"""

import numpy as np
import sys
sys.path.append('..')
from coordinator import globalCoordinator

coord = globalCoordinator() 
target = np.genfromtxt('squareTarget.txt')

particle = target + 15


orderIdx, levelOrder,  assignment = coord.getGlobalOrderedAssignment(particle, target)

output = []
for i, part in enumerate(particle):
    res = part.tolist() + [assignment[i]]
    output.append(res)
    
output = np.array(output)
np.savetxt('assignment.txt', output, fmt='%.3f')


output = []
for i, part in enumerate(target):
    res = part.tolist() + [orderIdx[i]]
    output.append(res)
    
output = np.array(output)
np.savetxt('targetOrder.txt', output, fmt='%.3f')



