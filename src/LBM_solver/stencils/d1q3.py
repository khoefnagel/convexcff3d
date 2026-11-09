import numpy as np

# --- LATTICE CONSTANTS (D1Q3) ---
Q = 3
D = 1

# Velocity vectors
Cij = np.array([
    [0], # Rest particle
    [1], [-1] # Straight directions
]).reshape(Q, 1, 1, 1, D)

# Weights
Wi = np.array([
    2/3, # Rest particle
    *[1/6]*2 # Straight directions
]).reshape(Q, 1, 1, 1, 1)
