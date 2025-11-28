import sys, os
sys.path.insert(0, os.getcwd())
from env import SimpleEnv
import random
import numpy as np

env=SimpleEnv(seed_value=1)
obs=env.reset()
steps=200
for t in range(steps):
    actions=[]
    for i in range(env.n):
        angle=random.random()*2*3.14159
        dx=np.cos(angle)
        dy=np.sin(angle)
        dist=1.0
        actions.append([dx,dy,dist])
    obs, rewards, done, info = env.step(actions)
    if done:
        break
print('after', t+1, 'steps explored_ratio=', info['explored_ratio'])
print('newly_explored in last step:', info.get('newly_explored'))
