import numpy as np

# --- LATTICE CONSTANTS (D2Q9) ---
Q = 9
D = 2

# Velocity vectors
Cij = np.array([
    [0,0], # Rest particle
    [1,0], [0,1], [-1,0], [0,-1], # Straight directions
    [1,1], [-1,1], [-1,-1], [1,-1] # Diagonal directions
]).reshape(Q, 1, 1, 1, D)

# Weights
Wi = np.array([
    4/9, # Rest particle
    *[1/9]*4, # Straight directions
    *[1/36]*4 # Diagonal directions
]).reshape(Q, 1, 1, 1, 1)
