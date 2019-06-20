import GridWorldPython as gw
import json
import numpy as np
import math
from timeit import default_timer as timer
configName = 'config.json'

model = gw.GridWorldPython(configName, 1)

with open(configName,'r') as file:
    config = json.load(file)

N = config['N']
iniConfigPos = np.genfromtxt('squareTarget.txt')
iniConfigPos = iniConfigPos[:N,:]
targets = iniConfigPos.copy()
iniConfig = []
for pos in iniConfigPos:
    pos += 15
    iniConfig.append(pos.tolist() + [0])
model.reset()
pos = model.getPositions()
pos.shape = (N, 3)
state = pos[:, 0:2]
updateTime = config['GridWorldNStep'] * config['dt']
maxDisp = config['maxSpeed'] * updateTime

start = timer()

model.setIniConfig(np.array(iniConfig))
#print(iniConfig)
for i in range(10000):
    phi = pos[:,2]
    projection = (targets[:, 0] - state[:,0]) * np.cos(phi) + (targets[:, 1] - state[:,1]) * np.sin(phi)
    dist = np.sqrt(np.sum(np.square(targets - state), axis=1))
    if i % 1000 == 0:
        totalDist = np.sum(dist)
        print("total Dist, step : ", totalDist, i)
    projectionCos = projection / dist
    # 0.382 is the angle of 67.5
    speeds = []
    for i in range(N):
        if projectionCos[i] > 0.5:
            if projection[i] > maxDisp:
                speeds.append(1)
            else:
                speeds.append(projection[i] / maxDisp)
        else:
            speeds.append(0)
    model.step(speeds)
    pos = model.getPositions()
    pos.shape = (N, 3)
    state = pos[:, 0:2]

print("done")
end = timer()
print(end - start) # time in seconds