import GridWorldPython as gw
import json
import numpy as np
import math

configName = 'config.json'

model = gw.GridWorldPython(configName, 1)

with open(configName,'r') as file:
    config = json.load(file)

N = config['N']
model.reset()
pos = model.getPositions()
pos.shape = (N, 3)
state = pos[0]
updateTime = config['GridWorldNStep'] * config['dt']
maxDisp = config['maxSpeed'] * updateTime
print(updateTime)
print(maxDisp)
for i in range(1000):
    x = state[0]
    y = state[1]
    phi = state[2]
    projection = ( - x) * math.cos(phi) + ( - y) * math.sin(phi)
    dist = math.sqrt((x) ** 2 + (y) ** 2)
    projectionCos = projection / dist
    # 0.382 is the angle of 67.5
    if projection > maxDisp and projectionCos > 0.5:
        action = 1
    else:
        action = projection / maxDisp


    model.step([action])
    pos = model.getPositions()
    pos.shape = (N, 3)
    state = pos[0]
    #print(pos)
