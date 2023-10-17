import toml
import utils
import numpy as np

# comment for config file

comment = "First test run for the grid simulation, where unbonding is allowed and the ratio of binding probabilities is taken from the real binding energy but they are changed, so that the transition probabilities are in an order of magnitude that makes sense."


# DEFINE FILENAMES
# ----------------

# name_sys = "test_setup_no_netrin"
# dir_sys = "/local_scratch2/janmak98/chemokine/results/tests/"

name_sys = "gradient_double_netrin"
dir_sys = "/local_scratch2/janmak98/chemokine/results/scaled_energies/"


name_sys = utils.validate_name_sys(
    dir_sys=dir_sys, 
    name_sys=name_sys, 
    overwrite_sys=True
)

fname_traj = utils.get_fname(dir_sys, name_sys, ftype="trajectory")
fname_cfg = utils.get_fname(dir_sys, name_sys, ftype="config")

# SIMULATION PARAMETERS
# ---------------------

num_steps = 5000000
simulation_stride = 50

# num_steps = 5000
# simulation_stride = 10

# SYSTEM PARAMETERS
# -----------------

# SET UP INITIAL POSITIONS

grid_size = [180, 60]
#             left  right bottom top
reflection = [True, True, False, False]

n_types = 9

particle_names =[
    "empty site",
    "CCL5",
    "Net-1",
    "Heperin",
    "CCL5-Net-1",
    "CCL5-Heperin",
    "Collagen Site",
    "CCL5-Net-1-Collagen",
    "Net-1-Collagen"
]

# defines the region in which the particles are distributed
regions_fraction = [
    [[0,1],[0,1]],       # empty grid site
    [[0,0.2],[0,1]],       # chemokine
    [[0,0.2],[0,1]],       # netrin
    [[0,1],[0,1]],       # heparansulfate
    [[0,1],[0,1]],       # chemokine-netrin
    [[0,1],[0,1]],       # chemokine-heparansulfate
    [[0,1],[0.48,0.52]], # collagen site
    [[0,1],[0,1]],       # chemokine-netrin-collagen
    [[0,1],[0,1]]        # netrin-collagen
]

# concentrations of the particles for the specified regions.
# therfore this is NOT NECESSARILY the concentration of the whole grid.
concentrations = [
    0,                        # empty grid site
    0.1,                      # 0.1,  # chemokine
    0.2,#.1,                      # 0.1,  # netrin
    0.2,                     # 0.15, # heparansulfate
    0.0,                      # chemokine-netrin
    0.0,                      # chemokine-heparansulfate
    0.2,                      # 0.2 # collagen site
    0.0,                      # chemokine-netrin-collagen
    0.0                       # netrin-collagen
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
    1e15,                   # empty grid site (not accesed)
    m_chemokine,            # kDa chemokine
    m_netrin,               # kDa netrin
    m_heperan,              # kDa heperansulfate
    m_chemokine + m_netrin, # chemokine-netrin
    m_heperan + m_netrin,   # chemokine-heperansulfate
    1e15,                   # collagen site (not accesed)
    1e15,                   # chemokine-netrin-collagen (not accesed)
    1e15                    # netrin-collagen (not accesed)
]


binding_probability = np.zeros((n_types, n_types))
binding_probability[1, 2] = 10.0 #0.1405 #1/69.07e-9 # chemokine binds to netrin
binding_probability[2, 1] = 10.0 #0.1405 #1/69.07e-9
binding_probability[2, 6] = 51.21 #0.7142 #1/13.59e-9 # netrin binds to collagen site 
binding_probability[6, 2] = 51.21 #0.7142 #1/13.59e-9  
binding_probability[1, 3] = 1.0 #0.0139 #1/696e-9   # chemokine binds to heparansulfate
binding_probability[3, 1] = 1.0 #0.0139 #1/696e-9
binding_probability[4, 6] = 71.71 #1.0 #1/9.706e-9 # chemokine-netrin binds to collagen_site
binding_probability[6, 4] = 71.71 #1.0 #1/9.706e-9 


#binding_probability /= 4

binding_probability[:, 0] = 1 # all particles bind to empty grid site
binding_probability[0, :] = 1 
    
diffusivities = utils.estimate_diffusivity(np.array(molecule_mass))
ratio = utils.calc_time_step_ratio(np.log(binding_probability), diffusivities)
move_probability, p_stay = utils.calc_probabilities_free(ratio, diffusivities)


# TRANSITION TYPES
    
bond_possible = np.zeros((n_types, n_types), dtype=bool)
unbond_possible = np.zeros((n_types, n_types), dtype=bool)

# diffusion of a particle (bonding to empty grid site)
bond_possible[1,0] = True # chemokine
bond_possible[2,0] = True # netrin
#bond_possible[3,0] = True # heparansulfate
bond_possible[4,0] = True # chemokine-netrin
#bond_possible[5,0] = True # chemokine-heparansulfate

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

# bonding
bond_probability[1,2] = move_probability[1] * binding_probability[1,2] # chemokine + netrin -> chemokine-netrin complex
bond_probability[2,1] = move_probability[2] * binding_probability[2,1]
bond_probability[1,3] = move_probability[1] * binding_probability[1,3] # chemokine + heparansulfate -> chemokine-heparansulfate complex
bond_probability[3,1] = move_probability[3] * binding_probability[3,1]
bond_probability[4,6] = move_probability[4] * binding_probability[4,6] # chemokine-netrin + collagen_site -> chemokine-netrin-collagen_site
bond_probability[2,6] = move_probability[2] * binding_probability[2,6] # netrin + collagen_site -> chemokine-heparansulfate-collagen_site

# unbonding
unbond_idx = np.zeros((n_types, n_types, 2), dtype=int)

# unbond to another particle
unbond_idx[4,3] = [2,5] # chemokine-netrin + heparansulfate -> netrin + chemokine-heparansulfate
unbond_idx[5,2] = [3,4] # chemokine-heparansulfate + netrin -> heperansulfate + chemokine-netrin
unbond_idx[8,1] = [6,4] # netrin-collagen_site + chemokine -> collagen_site + chemokine-netrin

# unbond to empty grid site
unbond_idx[4,0] = [1,2] # chemokine-netrin -> chemokine + netrin
unbond_idx[5,0] = [3,1] # chemokine-heparansulfate -> chemokine + heparansulfate
unbond_idx[7,0] = [6,4] # chemokine-netrin-collagen_site -> chemokine-netrin + collagen_site
unbond_idx[8,0] = [6,2] # netrin-collagen_site -> netrin + collagen_site

#related probabilities
unbond_probability = np.zeros((n_types, n_types))

# unbond to another particle
unbond_probability[4,3] = move_probability[1] * binding_probability[1,3] / binding_probability[1,2] # chemokine-netrin + heparansulfate -> netrin + chemokine-heparansulfate
unbond_probability[5,2] = move_probability[1] * binding_probability[1,2] / binding_probability[1,3] # chemokine-heparansulfate + netrin -> heperansulfate + chemokine-netrin
unbond_probability[8,1] = move_probability[2] * binding_probability[1,2] / binding_probability[2,6] # netrin-collagen_site + chemokine -> collagen_site + chemokine-netrin

# unbond to empty grid site
unbond_probability[4,0] = (move_probability[1] + move_probability[2]) / 2 / binding_probability[1,2] # chemokine-netrin -> chemokine + netrin
unbond_probability[5,0] = (move_probability[1] + move_probability[3]) / 2 / binding_probability[1,3] # chemokine-heparansulfate -> chemokine + heparansulfate
unbond_probability[7,0] = move_probability[4] / binding_probability[4,6] # chemokine-netrin-collagen_site -> chemokine-netrin + collagen_site
unbond_probability[8,0] = move_probability[2] / binding_probability[2,6] # netrin-collagen_site -> netrin + collagen_site


max_prob = np.max((np.max(bond_probability), np.max(unbond_probability)))

# bond_probability /= max_prob*4
# unbond_probability /= max_prob*4

# print()
# print(bond_probability)
# print()
# print(unbond_probability)

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
            print(f"| {particle_names[i]:13s} -> {particle_names[j]:13s}: {bond_probability[i,j]:7.5f} | {possible:^8s} |")
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
            print(f"| {particle_names[i]:19s} -> {particle_names[j]:13s}: {unbond_probability[i,j]:7.5f} | {possible:^8s} |")        
print()

assert np.all(bond_probability <= 0.25), "Bond probabilities are too big"
assert np.all(unbond_probability <= 0.25), "Unbond probabilities are too big"

#raise SystemExit

utils.plot_grid(initial_positions, particle_names, show=True, save=False)

# save config

with open(fname_cfg, "w") as f:
    toml.dump({
        "comment": comment,
        "num_steps": num_steps,
        "stride": simulation_stride,
        "reflection": reflection,
        "particle_names": particle_names,
        "name_sys": name_sys,
        "dir_sys": dir_sys,
        "bond_possible": bond_possible.tolist(),
        "unbond_possible": unbond_possible.tolist(),
        "bond_idx": bond_idx.tolist(),
        "unbond_idx": unbond_idx.tolist(),
        "bond_probability": bond_probability.tolist(),
        "unbond_probability": unbond_probability.tolist(),
        "initial_positions": initial_positions.tolist()
    }, f)
    
print("\nConfig saved to:\n", fname_cfg, "\n")
print("Execution command:\n")
print(f"python run.py {name_sys} {dir_sys}")