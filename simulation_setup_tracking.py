import toml
import utils
import numpy as np
import os

#*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*#
#* Set up simulation to test particle tracking *#
#*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*#
 
c_chem = 0.05
c_net = 0.05
c_hep = 0.25

comment = f"Chemokine, netrin and heperan in the system. particle tracking is turned on. No reflective boundary."

# DEFINE FILENAMES
# ----------------

name_sys = f"test_tracking"
dir_sys = f"/net/data05/janmak98/results/september/chemokine/tests/"


name_sys = utils.validate_name_sys(
    dir_sys=dir_sys, 
    name_sys=name_sys, 
    overwrite_sys=False
)

fname_traj = utils.get_fname(dir_sys, name_sys, ftype="trajectory")
fname_cfg = utils.get_fname(dir_sys, name_sys, ftype="config")

# SIMULATION PARAMETERS
# ---------------------

num_steps = 1000
simulation_stride = 1

write_particle_traj = True

# SYSTEM PARAMETERS
# -----------------

# SET UP INITIAL POSITIONS

grid_size = [160, 120]
reflection = [False, False, False, False]

n_types = 9

particle_names =[
    "empty site",
    "CCL5",
    "Net1",
    "Heparansulfate",
    "CCL5-Net1",
    "CCL5-Heparansulfate",
    "Collagen Site",
    "CCL5-Net1-Collagen",
    "Net1-Collagen"
]

# defines the region in which the particles are distributed
regions_fraction = [
    [[0,1],[0,1]],       # empty grid site
    [[0,1],[0,1]],       # chemokine
    [[0,1],[0,1]],       # netrin
    [[0,1],[0,1]],       # heparansulfate
    [[0,1],[0,1]],       # chemokine-netrin
    [[0,1],[0,1]],       # chemokine-heparansulfate
    [[0,1],[0.48,0.52]], # collagen site
    [[0,1],[0,1]],       # chemokine-netrin-collagen
    [[0,1],[0,1]]        # netrin-collagen
]

# concentrations of the particles for the specified regions.
# therefore this is NOT NECESSARILY the concentration of the whole grid.
concentrations = [
    0.0,                      # 0 empty grid site
    c_chem,                   # 1 chemokine
    c_net,                    # 2 netrin
    c_hep,                    # 3 heparansulfate
    0.0,                      # 4 chemokine-netrin
    0.0,                      # 5 chemokine-heparansulfate
    0.0,                      # 6 collagen site
    0.0,                      # 7 chemokine-netrin-collagen
    0.0                       # 8 netrin-collagen
]

initial_positions = utils.get_initial_positions(
    grid_size = grid_size,
    concentrations = concentrations,
    regions_fraction = regions_fraction
)

# CALCULATE PROBABILITIES OUT OF PHYSICAL PROPERTIES

m_chemokine = 26.725          # kDa (CCL5) https://www.rcsb.org/structure/5COY
m_netrin = 52.437             # kDa (Netrin-1) https://www.rcsb.org/structure/4OVE
m_heperan = 5000              # kDa (heperansulfate) made up

molecule_mass = [
    1e15,                   # empty grid site (not accessed)
    m_chemokine,            # kDa chemokine
    m_netrin,               # kDa netrin
    m_heperan,              # kDa heperansulfate
    m_chemokine + m_netrin, # chemokine-netrin
    m_heperan + m_netrin,   # chemokine-heperansulfate
    1e15,                   # collagen site (not accessed)
    1e15,                   # chemokine-netrin-collagen (not accessed)
    1e15                    # netrin-collagen (not accessed)
]

# DEFINE PARTICLE RELATIONS FOR TRACKING

# define which particle types are contained in other particle types
type_contains = np.eye(n_types, dtype=bool) # each type is contained  in itself
type_contains[4, 1] = True # chemokine-netrin               contains chemokine
type_contains[4, 2] = True # chemokine-netrin               contains netrin 
type_contains[5, 1] = True # chemokine-heparansulfate       contains chemokine
type_contains[5, 3] = True # chemokine-heparansulfate       contains heparansulfate
type_contains[7, 1] = True # chemokine-netrin-collagen_site contains chemokine
type_contains[7, 2] = True # chemokine-netrin-collagen_site contains netrin
type_contains[7, 4] = True # chemokine-netrin-collagen_site contains chemokine-netrin
type_contains[7, 6] = True # chemokine-netrin-collagen_site contains collagen_site
type_contains[8, 2] = True # netrin-collagen_site           contains netrin
type_contains[8, 6] = True # netrin-collagen_site           contains collagen_site

# define which particle types should be tracked
track_types = np.zeros(n_types, dtype=bool)
track_types[1] = True # track chemokines
track_types[2] = True # track netrins


binding_energy_exponential = np.zeros((n_types, n_types))
# [K_N_C        K_N_Coll,       K_CN_Coll,    K_C_H]
# 247.20777497 1256.41214255 2082.27329478   24.5325302
binding_energy_exponential[1, 2] = 247.20777497  #10.0 #0.1405 #1/69.07e-9 # chemokine binds to netrin
binding_energy_exponential[2, 1] = 247.20777497  #10.0 #0.1405 #1/69.07e-9
binding_energy_exponential[2, 6] = 1256.41214255 #50.0 #0.7142 #1/13.59e-9 # netrin binds to collagen site 
binding_energy_exponential[6, 2] = 1256.41214255 #50.0 #0.7142 #1/13.59e-9  
binding_energy_exponential[1, 3] = 24.5325302    #1.0 #0.0139 #1/696e-9   # chemokine binds to heparansulfate
binding_energy_exponential[3, 1] = 24.5325302    #1.0 #0.0139 #1/696e-9
binding_energy_exponential[4, 6] = 2082.27329478 #70.0 #1.0    #1/9.706e-9 # chemokine-netrin binds to collagen_site
binding_energy_exponential[6, 4] = 2082.27329478 #70.0 #1.0    #1/9.706e-9 

binding_energy_exponential[:, 0] = 1 # all particles bind to empty grid site
binding_energy_exponential[0, :] = 1 

#! HALF ENERGY FOR BONDING !!!
binding_energy_exponential = np.sqrt(binding_energy_exponential)


diffusivities = utils.estimate_diffusivity(np.array(molecule_mass))
ratio = utils.calc_time_step_ratio(np.log(binding_energy_exponential), diffusivities)
move_probability, p_stay = utils.calc_probabilities_free(ratio, diffusivities)


# TRANSITION TYPES

bond_possible = np.zeros((n_types, n_types), dtype=bool)
unbond_possible = np.zeros((n_types, n_types), dtype=bool)

# diffusion of a particle (bonding to empty grid site)
bond_possible[1,0] = True # chemokine #! False for fixing ccl5
bond_possible[2,0] = True # netrin
#bond_possible[3,0] = True # heparansulfate #! False for fixing heparan
bond_possible[4,0] = True # chemokine-netrin #! False for fixing ccl5
#bond_possible[5,0] = True # chemokine-heparansulfate #! False for fixing heparan

# bonding of two particles
bond_possible[1,2] = True # chemokine + netrin -> chemokine-netrin complex
bond_possible[2,1] = True
bond_possible[1,3] = True # chemokine + heparansulfate -> chemokine-heparansulfate complex 
#bond_possible[3,1] = True
bond_possible[4,6] = True # chemokine-netrin + collagen_site -> chemokine-netrin-collagen_site 
bond_possible[2,6] = True # netrin + collagen_site -> chemokine-heparansulfate-collagen_site

# changing bond partners
unbond_possible[4,3] = True # chemokine-netrin + heparansulfate -> netrin + chemokine-heparansulfate
unbond_possible[5,2] = True # chemokine-heparansulfate + netrin -> heperansulfate + chemokine-netrin
unbond_possible[8,1] = True # netrin-collagen_site + chemokine -> collagen_site + chemokine-netrin

# unbonding of two particles
unbond_possible[4,0] = True # chemokine-netrin -> chemokine + netrin
unbond_possible[5,0] = True # chemokine-heparansulfate -> chemokine + heparansulfate
unbond_possible[7,0] = True # chemokine-netrin-collagen_site -> chemokine-netrin + collagen_site
unbond_possible[8,0] = True # netrin-collagen_site -> netrin + collagen_site

# update move probabilities so that probabilities for moves, that are not possible, are set to zero
move_probability *= bond_possible[:,0]

bond_idx = np.zeros((n_types, n_types, 2), dtype=int)

# diffusion (bonding to empty grid site)
bond_idx[1,0] = [0,1] # chemokine
bond_idx[2,0] = [0,2] # netrin
bond_idx[3,0] = [0,3] # heparin
bond_idx[4,0] = [0,4] # CN
bond_idx[5,0] = [0,5] # CH

# bonding indices
bond_idx[1,2] = [0,4] # chemokine + netrin -> chemokine-netrin complex
bond_idx[2,1] = [0,4]
bond_idx[1,3] = [0,5] # chemokine + heparansulfate -> chemokine-heparansulfate complex 
bond_idx[3,1] = [0,5]
bond_idx[4,6] = [0,7] # chemokine-netrin + collagen_site -> chemokine-netrin-collagen_site 
bond_idx[2,6] = [0,8] # netrin + collagen_site -> chemokine-heparansulfate-collagen_site

# related probabilities
bond_probability = np.zeros((n_types, n_types))

# diffusion (bonding to empty grid site)
bond_probability[1,0] = move_probability[1] # chemokine
bond_probability[2,0] = move_probability[2] # netrin
bond_probability[3,0] = move_probability[3] # heparin
bond_probability[4,0] = move_probability[4] # CN
bond_probability[5,0] = move_probability[5] # CH

# # bonding
bond_probability[1,2] = move_probability[1] * binding_energy_exponential[1,2] # chemokine + netrin -> chemokine-netrin complex
bond_probability[2,1] = move_probability[2] * binding_energy_exponential[2,1]
bond_probability[1,3] = move_probability[1] * binding_energy_exponential[1,3] # chemokine + heparansulfate -> chemokine-heparansulfate complex
bond_probability[3,1] = move_probability[3] * binding_energy_exponential[3,1]
bond_probability[4,6] = move_probability[4] * binding_energy_exponential[4,6] # chemokine-netrin + collagen_site -> chemokine-netrin-collagen_site
bond_probability[2,6] = move_probability[2] * binding_energy_exponential[2,6] # netrin + collagen_site -> chemokine-heparansulfate-collagen_site

# unbonding
unbond_idx = np.zeros((n_types, n_types, 2), dtype=int)

# unbond to another particle
unbond_idx[4,3] = [2,5] # chemokine-netrin + heparansulfate -> netrin + chemokine-heparansulfate
unbond_idx[5,2] = [3,4] # chemokine-heparansulfate + netrin -> heperansulfate + chemokine-netrin
unbond_idx[8,1] = [6,4] # netrin-collagen_site + chemokine -> collagen_site + chemokine-netrin

# unbond to empty grid site
unbond_idx[4,0] = [2,1] # chemokine-netrin -> netrin + chemokine
unbond_idx[5,0] = [3,1] # chemokine-heparansulfate -> heparansulfate + chemokine
unbond_idx[7,0] = [6,4] # chemokine-netrin-collagen_site -> collagen_site + chemokine-netrin
unbond_idx[8,0] = [6,2] # netrin-collagen_site -> collagen_site + netrin

#related probabilities
unbond_probability = np.zeros((n_types, n_types))

# unbond to another particle
unbond_probability[4,3] = move_probability[1] * binding_energy_exponential[1,3] / binding_energy_exponential[1,2] # chemokine-netrin + heparansulfate -> netrin + chemokine-heparansulfate
unbond_probability[5,2] = move_probability[1] * binding_energy_exponential[1,2] / binding_energy_exponential[1,3] # chemokine-heparansulfate + netrin -> heperansulfate + chemokine-netrin
unbond_probability[8,1] = move_probability[2] * binding_energy_exponential[1,2] / binding_energy_exponential[2,6] # netrin-collagen_site + chemokine -> collagen_site + chemokine-netrin

# ADDED PROB
# unbond to empty grid site
unbond_probability[4,0] = (move_probability[1] + move_probability[2]) / binding_energy_exponential[1,2] # chemokine-netrin -> chemokine + netrin
unbond_probability[5,0] = (move_probability[1] + move_probability[3]) / binding_energy_exponential[1,3] # chemokine-heparansulfate -> chemokine + heparansulfate
unbond_probability[7,0] = move_probability[4] / binding_energy_exponential[4,6] # chemokine-netrin-collagen_site -> chemokine-netrin + collagen_site
unbond_probability[8,0] = move_probability[2] / binding_energy_exponential[2,6] # netrin-collagen_site -> netrin + collagen_site

max_prob = np.max((np.max(bond_probability), np.max(unbond_probability)))

print('\nProbabilities that are different from zero and wether the transition is possible:\n')

print("------------------------------------------------------")
print("| Bond Probabilities                      | Possible |")
print("------------------------------------------------------")
for i in range(n_types):
    for j in range(n_types):
        if bond_probability[i, j] != 0:
            if bond_possible[i,j]:
                possible = 'yes'
            else:
                possible = 'no'
            print(f"| {particle_names[i]:13s} -> {particle_names[j]:16s}: {bond_probability[i,j]:7.5f} | {possible:^8s} |")
print()

print("------------------------------------------------------------")
print("| Unbond Probabilities                          | Possible |")
print("------------------------------------------------------------")

for i in range(n_types):
    for j in range(n_types):
        if unbond_probability[i, j] != 0:
            if unbond_possible[i,j]:
                possible = 'yes'
            else:
                possible = 'no'
            print(f"| {particle_names[i]:19s} -> {particle_names[j]:16s}: {unbond_probability[i,j]:7.5f} | {possible:^8s} |")        
print()

# because of some weird rounding errors
bond_probability /= np.max(bond_probability)
bond_probability *= 0.25

assert np.all(bond_probability <= 0.25), f"Bond probabilities are too big: {bond_probability}\n{bond_probability<=0.25}"
#assert np.all(unbond_probability <= 0.25), "Unbond probabilities are too big"


# plot initial positions for double checking of setup
#utils.plot_grid(initial_positions, particle_names, show=True, save=False)
#raise SystemExit

# save config

with open(fname_cfg, "w") as f:
    toml.dump({
        "comment": comment,
        "num_steps": num_steps,
        "stride": simulation_stride,
        "write_particle_traj": write_particle_traj,
        "reflection": reflection,
        "particle_names": particle_names,
        "name_sys": name_sys,
        "dir_sys": dir_sys,
        "type_contains": type_contains.tolist(),
        "track_types": track_types.tolist(),
        "bond_possible": bond_possible.tolist(),
        "unbond_possible": unbond_possible.tolist(),
        "bond_idx": bond_idx.tolist(),
        "unbond_idx": unbond_idx.tolist(),
        "bond_probability": bond_probability.tolist(),
        "unbond_probability": unbond_probability.tolist(),
        "initial_positions": initial_positions.tolist(),
        "binding_energy": np.log(binding_energy_exponential**2).tolist(),
        "diffusivities": diffusivities.tolist(),
    }, f)

print("\nConfig saved to:\n", fname_cfg, "\n")
print("Execution command:\n")
print(f"python run.py {name_sys} {dir_sys}")