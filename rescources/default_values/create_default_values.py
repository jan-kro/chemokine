import numpy as np
import h5py
import os
from tools import interaction_indices, get_interaction_probabilities, get_type_contains

#* Script that creates the default simulation parameters from experimental Kd Values and estimated Diffusivities *#

# filname where the default values should be saved
fname_h5 = "default_values_scaled_energies.h5"


particle_names = ["empty site",     # 0 
                  "CCL5",           # 1
                  "Net1",           # 2
                  "Hep",            # 3
                  "Coll",           # 4
                  "CCL5-Net1",      # 5
                  "CCL5-Hep",       # 6
                  "Net1-Hep",       # 7
                  "Net1-Coll",      # 8
                  "CCL5-Net1-Coll"] # 9
    
n_ptypes = len(particle_names)


# Experimental dissociation constants in nM
Kd_C_N_exp     = 100            # Chemokine-Netrin bond
Kd_C_H_exp     = 53             # Chemokine-Heparan Sulfate bond
Kd_N_H_exp     = 26             # Netrin-Heparan Sulfate bond
Kd_N_Coll_exp  = 17             # Netrin-Collagen bond
Kd_CN_Coll_exp = Kd_N_Coll_exp  # Chemokine-Netrin-Collagen bond

scaling_factor_Kd_exp = 1 / 3e5   #! SCALE KD VALUES

# dissociation constants in 1/a^3
Kd_C_N     = scaling_factor_Kd_exp * Kd_C_N_exp  
Kd_C_H     = scaling_factor_Kd_exp * Kd_C_H_exp  
Kd_N_H     = scaling_factor_Kd_exp * Kd_N_H_exp  
Kd_N_Coll  = scaling_factor_Kd_exp * Kd_N_Coll_exp  
Kd_CN_Coll = scaling_factor_Kd_exp * Kd_CN_Coll_exp 


# binding (affinity) constants K_a = 1 / K_d
K_a = np.ones((n_ptypes, n_ptypes))

K_a[1, 2] = 1/Kd_C_N
K_a[2, 1] = 1/Kd_C_N
K_a[1, 3] = 1/Kd_C_H
K_a[3, 1] = 1/Kd_C_H
K_a[2, 3] = 1/Kd_N_H
K_a[3, 2] = 1/Kd_N_H
K_a[2, 4] = 1/Kd_N_Coll
K_a[4, 2] = 1/Kd_N_Coll
K_a[5, 4] = 1/Kd_CN_Coll
K_a[4, 5] = 1/Kd_CN_Coll

# particle diffusivities (μm^2/s)                                                                  
diffusivities = np.array([0, 111.3, 88.9, 0, 0, 77.5, 0, 0, 0, 0])

print("Particle diffusivities:")
for D, name in zip(diffusivities, particle_names):
    print(f"{name:15s}: D = {D:6.1f} μm^2/s")

diffusivities *= 1e-12  # convert to m^2/s

print("\nBinding Constants:")
for i, name_i in enumerate(particle_names):
    for j, name_j in enumerate(particle_names):
        if K_a[i, j] != 1:
            print(f"{name_i:15s} + {name_j:15s}: K_a = {K_a[i, j]:.2f} a^3")

# create interaction matrix

# define which particle types result from bond
resulting_ptype = np.zeros((n_ptypes,n_ptypes), dtype=int)

# bonding with empty site is just a move
for k in range(n_ptypes):
    resulting_ptype[k,0] = k

resulting_ptype[1,2] = 5 # 1
resulting_ptype[1,3] = 6
resulting_ptype[2,1] = 5 # 2
resulting_ptype[2,3] = 7
resulting_ptype[2,4] = 8
resulting_ptype[3,1] = 6 # 3
resulting_ptype[3,2] = 7
resulting_ptype[4,2] = 8 # 4
resulting_ptype[4,5] = 9
resulting_ptype[5,4] = 9 # 5

print("\nPossible resulting particle types:")
for i, name_i in enumerate(particle_names):
    for j, name_j in enumerate(particle_names):
        if resulting_ptype[i, j] != 0:
            if j != 0:
                print(f"{name_i:15s} + {name_j:15s} -> {particle_names[resulting_ptype[i, j]]}")

# define interaction_probabilities and add every interaction
interaction_probabilities, interaction_prob_normalization = get_interaction_probabilities(
    diffusivities=diffusivities, 
    binding_constants=K_a, 
    resulting_ptype=resulting_ptype
    )

print("\nInteraction Probability Normalization:")
print(interaction_prob_normalization)

# print interaction matrix
print("\nNon-zero Interaction Probabilities:")
print("i               + j               --> n               + m              : probability")
print("------------------------------------------------------------------------------------")
# print possible moves
for i in range(n_ptypes):
    for j in range(n_ptypes):
        if i != 0 and j == 0:
            for k in range(n_ptypes**2):
                n, m = interaction_indices(k, n_ptypes=n_ptypes)
                if n!=i or m!=j:
                    if n==0:
                        if interaction_probabilities[i, j, k] > 0:
                            print(f"{particle_names[i]:15s} + {particle_names[j]:15s} --> {particle_names[n]:15s} + {particle_names[m]:15s}: {interaction_probabilities[i, j, k]:.2e}")

# print bonding interactions
for i in range(n_ptypes):
    for j in range(n_ptypes):
        if i != 0 and i != j:
            for k in range(n_ptypes**2):
                n, m = interaction_indices(k, n_ptypes=n_ptypes)
                if n!=i or m!=j:
                    if not (j==0 and n==0):
                        if interaction_probabilities[i, j, k] > 0:
                            print(f"{particle_names[i]:15s} + {particle_names[j]:15s} --> {particle_names[n]:15s} + {particle_names[m]:15s}: {interaction_probabilities[i, j, k]:.2e}")


# print what particle types contain what other particle types
type_contains = get_type_contains(interaction_probabilities)
                    
print("\nType contains:")
for i in range(n_ptypes):
    for j in range(n_ptypes):
        if type_contains[i, j]:
            print(f"{particle_names[i]:15s} contains {particle_names[j]:15s}")                            
                            
                            
# create h5py dataset containing the interaction matrix and all parameters

if not os.path.exists(fname_h5):
    with h5py.File(fname_h5, "w") as f:
        f.create_dataset("interaction_probabilities", data=interaction_probabilities)
        f.create_dataset("interaction_prob_normalization", data=interaction_prob_normalization)
        f.create_dataset("diffusivities", data=diffusivities)
        f.create_dataset("binding_constants", data=K_a)
        f.create_dataset("binding_energy", data=np.log(K_a).tolist())        
        f.create_dataset("particle_names", data=particle_names)
    print(f"\nDefault values saved to {fname_h5}")
else:
    print(f"\nWARNING: File {fname_h5} already exists. Not overwriting!")                            