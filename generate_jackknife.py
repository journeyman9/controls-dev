import gym
import gym_truck_backerupper
import numpy as np
import pandas as pd
import json
import time

env = gym.make('TruckBackerUpper-v0').unwrapped
# Trajectory
#q0 = [2.0, -3.0, 123.0] # x, y, theta
#qg = [-8.0, -29.0, 215.0]

# Debug
#q0 = [-25.0, 0.0, 0.0] 
#qg = [25.0, 0.0, 0.0]

# Measurement
#q0 = [21.0, -16.0, 180.0]
#qg = [14.0, 11.0, 3.0]

## Jackknife
#q0 = [-4.0, -2.0, 439.0-90.0]  # 7.661995, 
#qg = [-25.0, -5.0, 353.0-90.0]  # 6.1610122,
q0 = [10.0, -4.0, 186.0] # 366
qg = [9.0, 29.0, 61.0] # 240


"""
min_x = -40.0
min_y = -40.0
max_x = 40.0
max_y = 40.0

min_psi_1 = np.radians(0.0)
min_psi_2 = np.radians(0.0)
max_psi_1 = np.radians(360.0)
max_psi_2 = np.radians(360.0)

x_start = np.random.randint(min_x, max_x)
y_start = np.random.randint(min_y, max_y)
psi_start = np.random.randint(
                            np.degrees(min_psi_2), 
                            np.degrees(max_psi_2))

x_goal = np.random.randint(min_x, max_x)
y_goal = np.random.randint(min_y, max_y)
psi_goal = np.random.randint(
                           np.degrees(min_psi_2),
                           np.degrees(max_psi_2))

q0 = [x_start, y_start, psi_start - 180.0]
qg = [x_goal, y_goal, psi_goal - 180.0]
"""

env.manual_params(L2=10.192, h=-0.29)
env.manual_course(q0, qg)
#env.manual_velocity(v=-1.0)
env.lookahead(3)
#env.dt = 0.008
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

previous_action = 0.0
max_rate = np.radians(4)  # Maximum change in control per time step
hitch_limit = np.radians(18)

while not done:
    env.render()

    hitch = s[0] - s[1]
    
    K_aggr = np.array([-27.6065, 99.8308, -7.8541])   # original
    #K_safe = np.array([-2.56273982, 5.15301535, -0.15811388])  
    K_safe = np.array([-2.47340514, 4.77639661, -0.14142136])  

    if abs(hitch) < hitch_limit:
        K = K_aggr                  # track tightly
    else:
        K = K_safe                   # track safely
    #a = np.clip(K.dot(s), env.action_space.low, env.action_space.high)

    # Example action from LQR, replace with RL policy
    #K = np.array([-24.7561, 94.6538, -7.8540])
    #K = np.array([-27.60653245, 99.8307537, -7.85407596])
    #a = np.clip(K.dot(s), env.action_space.low, env.action_space.high)

    #K = np.array([-27.60653245, 99.8307537, -7.85407596])
    a_desired = np.clip(K.dot(s), env.action_space.low, env.action_space.high)

    # Rate limit the control
    a_change = a_desired[0] - previous_action
    a_change = np.clip(a_change, -max_rate, max_rate)
    a = np.array([previous_action + a_change])
    previous_action = a[0]
    
    
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
    
    if info['jackknife']:
        q0_j = env.q0.copy()
        qg_j = env.qg.copy()
        q0_j[2] = np.degrees(q0_j[2])
        qg_j[2] = np.degrees(qg_j[2])
        print("q0:", q0_j)
        print("qg:", qg_j)
        print("Jackknife detected!")
    
print(
    f"info: {json.dumps(info)}"
)

env.close()

# Convert to a DataFrame and save as CSV
df = pd.DataFrame(data)

#df['psi_1'] -= np.pi
#df['psi_2'] -= np.pi
df.to_csv("./MPC/jackknife.csv", index=False)