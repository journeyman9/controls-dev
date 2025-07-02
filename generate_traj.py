import gym
import gym_truck_backerupper
import numpy as np
import pandas as pd
import json

env = gym.make('TruckBackerUpper-v0').unwrapped
# Trajectory
q0 = [2.0, -3.0, 123.0] # x, y, theta
qg = [-8.0, -29.0, 215.0]

# Debug
#q0 = [-25.0, 0.0, 0.0] 
#qg = [25.0, 0.0, 0.0]

# Measurement
#q0 = [21.0, -16.0, 180.0]
#qg = [14.0, 11.0, 3.0]
env.manual_params(L2=10.192, h=-0.29)
env.manual_course(q0, qg)
env.dt = 0.008
print(env.dt)
env.num_steps = int((160 - 0)/env.dt) + 1

done = False
s = env.reset()

data = []

psi_1 = env.psi_1[env.sim_i-1]
psi_2 = env.psi_2[env.sim_i-1]
y2 = env.y2[env.sim_i-1]

psi_1d = 0
psi_2d = 0
y2d = 0
actions = 0

# Collect data at each step
data.append({
    "psi_1": psi_1,
    "psi_2": psi_2,
    "y2": y2,
    "psi_1d": psi_1d,
    "psi_2d": psi_2d,
    "y2d": y2d,
    "action": actions,
})

while not done:
    #env.render()

    # Example action from LQR, replace with RL policy
    #K = np.array([-24.7561, 94.6538, -7.8540])
    K = np.array([-27.60653245, 99.8307537, -7.85407596])
    a = np.clip(K.dot(s), env.action_space.low, env.action_space.high)

    s_, r, done, info = env.step(a)
    
    psi_1 = env.psi_1[env.sim_i-1]
    psi_2 = env.psi_2[env.sim_i-1]
    y2 = env.y2[env.sim_i-1]

    psi_1d = env.xd[0]
    psi_2d = env.xd[1]
    y2d = env.xd[5]

    # Collect data at each step
    data.append({
        "psi_1": psi_1,
        "psi_2": psi_2,
        "y2": y2,
        "psi_1d": psi_1d,
        "psi_2d": psi_2d,
        "y2d": y2d,
        "action": a[0],
    })


    s = s_
env.close()

# Convert to a DataFrame and save as CSV
df = pd.DataFrame(data)

df['psi_1'] -= np.pi
df['psi_2'] -= np.pi
df.to_csv("trajectory_data.csv", index=False)