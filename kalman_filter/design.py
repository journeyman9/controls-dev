import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy.signal import cont2discrete


L1 = 5.74
L2 = 10.192
h = -0.29
v1x = -2.012

A = np.array(
    [
        [0, 0, 0], 
        [v1x/L2, -v1x/L2, 0],
        [0, v1x, 0]
    ]
)

B = np.array([v1x/L1, -h*v1x/(L1*L2), 0]).reshape(-1, 1)
C = np.eye(3)
D = np.zeros((3, 1))

dt = 0.008
sys = ct.ss(A, B, C, D)
dsys = ct.c2d(sys, dt, method='zoh')

# Print as floats
print("A matrix:")
print(np.array2string(dsys.A, precision=8, suppress_small=True))
print("B matrix:")
print(np.array2string(dsys.B, precision=8, suppress_small=True))

print("Controllable? ", np.linalg.matrix_rank(ct.ctrb(A, B)) == 3)
print("Observable? ", np.linalg.matrix_rank(ct.obsv(A, B)) == 3)

# Plot pole zero

# Print poles and zeros before pole placement  
"""
poles_before = ct.poles(sys)  
zeros_before = ct.zeros(sys)  
print("Poles before pole placement:", poles_before)  
print("Zeros before pole placement:", zeros_before)

# Plot Pole-Zero Map
plt.figure()
ct.pzmap(sys, title='Pole-Zero Map Before Pole Placement')
plt.show()

# Pole Placement
K = ct.place(A, B, [-1, -2, -3])
#K = ct.place(A, B, [-5-1j, -5+1j, -2])
print("K: ", K)

# New A matrix after pole placement
A_new = A - B @ K

# Create new state space system with new A matrix
sys_new = ct.ss(A_new, B, C, D)

# Plot Pole-Zero Map After Pole Placement
plt.figure()
ct.pzmap(sys_new, title='Pole-Zero Map After Pole Placement')
plt.show()
"""

# LQR
"""
Q = np.array(
    [
        [1.0, 0, 0],
        [0, 1.0, 0],
        [0, 0, 1.0]
    ]
)

R = 1.0
"""
Q = np.array(
    [
        [820.7016, 0, 0],
        [0, 820.7016, 0],
        [0, 0, 100]
    ]
)

R = 1.6211

#Q /= dt
#R /= dt

print(Q)
print(R)

K, S, E = ct.dlqr(dsys, Q, R)
#K, S, E = ct.lqr(sys, Q, R)
print("K: ", K)
print("Eigen:", E)