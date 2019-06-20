import GridWorldPython as gw
import json
import numpy as np


configName = 'config.json'

model = gw.GridWorldPython(configName, 1)

with open(configName,'r') as file:
    config = json.load(file)

N = config['N']
iniConfigPos = np.genfromtxt('squareTarget.txt')
iniConfig = []
for pos in iniConfigPos:
    iniConfig.append(pos.tolist() + [0])
model.reset()
model.setIniConfig(np.array(iniConfig))
#print(iniConfig)
for i in range(100000):
    speeds = [0 for _ in range(N)]
    model.step(speeds)
    pos = model.getPositions()
    pos.shape = (N, 3)
