import numpy as np

# --- LATTICE CONSTANTS (D3Q19) ---
Q = 19
D = 3

# Velocity vectors
Cij = np.array([
    [0,0,0], # Rest particle
    [1,0,0], [0,1,0], [0,0,1], # Mag 1
    [-1,0,0], [0,-1,0], [0,0,-1], # Mag 1
    [1,1,0], [1,0,1], [0,1,1], # Mag sqrt(2)
    [-1,1,0], [-1,0,1], [0,-1,1], # Mag sqrt(2)
    [1,-1,0], [1,0,-1], [0,1,-1], # Mag sqrt(2)
    [-1,-1,0], [-1,0,-1], [0,-1,-1] # Mag sqrt(2)
]).reshape(Q, 1, 1, 1, D)

# Weights
Wi = np.array([
    1/3, # Rest particle
    *[1/18]*6, # Straight directions
    *[1/36]*12 # Diagonal directions
]).reshape(Q, 1, 1, 1, 1)
