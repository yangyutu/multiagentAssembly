import GridWorldPython as gw
import json
import numpy as np


configName = 'config.json'

model = gw.GridWorldPython(configName, 1)

with open(configName,'r') as file:
    config = json.load(file)

N = config['N']
model.reset()
for i in range(100):
    speeds = [1 for _ in range(N)]
    model.step(speeds)
    pos = model.getPositions()
    pos.shape = (N, 3)
    print(pos)
