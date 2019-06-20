import GridWorldPython as gw
import json
import numpy as np
from timeit import default_timer as timer

configName = 'config.json'

model = gw.GridWorldPython(configName, 1)

with open(configName,'r') as file:
    config = json.load(file)

N = config['N']
iniConfigPos = np.genfromtxt('squareTargetN1609CPP.txt')
iniConfig = []
for pos in iniConfigPos:
    iniConfig.append(pos[1:3].tolist() + [0])
model.reset()
model.setIniConfig(np.array(iniConfig))
#print(iniConfig)
start = timer()
for i in range(100):
    speeds = [0 for _ in range(N)]
    model.step(speeds)
    pos = model.getPositions()
    pos.shape = (N, 3)
end = timer()
print("timecost", end - start)