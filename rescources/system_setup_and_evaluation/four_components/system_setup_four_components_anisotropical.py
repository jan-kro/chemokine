import chemokine as ck
import numpy as np 
import h5py
import toml
import os

from scipy.optimize import fsolve

#* CREATE SYSTEM FOUR COMPONENT SYSTEM WITH ANISTROPICAL INITIAL PARTICLE DISTRIBUTION, 
#* REFLECTIVE BOUNDARY AND PARTICLE SOURCE AT THE LEFT BOUNDARY

# load default values into config dict
fname_default_values = "/path/to/default_values_scaled_energies.h5"

# calculate the equilibrium concentrations of the three components
n_grid_x = 200
n_grid_y = 150
n_grid = n_grid_x*n_grid_y

# total initial concentration of all four particle species
c_chem_all = [0.0025]
c_net_all = [0, 0.01]
c_hep_all = [0.035] 
c_coll_all = [0.005]

n1 = len(c_chem_all)
n2 = len(c_net_all)
n3 = len(c_hep_all)
n4 = len(c_coll_all)

# get the default values for the binding constantes
fh5 = h5py.File(fname_default_values, "r")

particle_names = fh5['particle_names'][()].astype(str)

# define particle types
ptype_C   = 1
ptype_N   = 2
ptype_H   = 3
ptype_G   = 4 # Collagen
ptype_CN  = 5
ptype_CH  = 6
ptype_NH  = 7
# ptype_NG  = 8 # Netrin-Collagen
# ptype_CNG = 9 # Chemokine-Netrin-Collagen

bond_prob_CN  = fh5["interaction_probabilities"][ptype_C,ptype_N,ptype_CN]/4
bond_prob_CH  = fh5["interaction_probabilities"][ptype_C,ptype_H,ptype_CH]/4
bond_prob_NH  = fh5["interaction_probabilities"][ptype_N,ptype_H,ptype_NH]/4
# bond_prob_NG  = fh5["interaction_probabilities"][ptype_N,ptype_G,ptype_NG]/4
# bond_prob_CNG = fh5["interaction_probabilities"][ptype_C,ptype_NG,ptype_CNG]/4

D_C = fh5["interaction_probabilities"][ptype_C,0,ptype_C]/4
D_N = fh5["interaction_probabilities"][ptype_N,0,ptype_N]/4
D_H = fh5["interaction_probabilities"][ptype_H,0,ptype_H]/4


K_CN_setup = (bond_prob_CN / D_C)**2
K_CH_setup = (bond_prob_CH / D_C)**2
K_NH_setup = (bond_prob_NH / D_N)**2
fh5.close()

K_1 = K_CN_setup  
K_2 = K_CH_setup  
K_3 = K_NH_setup 

c_chem_th_all = []
c_net_th_all = []
c_hep_th_all = []
c_chem_net_th_all = []
c_chem_hep_th_all = []
c_net_hep_th_all = []

dir_cfg_all = []
for kk, A_0 in enumerate(c_chem_all):
    for jj, B_0 in enumerate(c_net_all):
        for ii, C_0 in enumerate(c_hep_all):
            for nn, c_coll in enumerate(c_coll_all):
                
                grid_size = [n_grid_x, n_grid_y]
                
                
                # defines the region in which the particles are distributed
                # the regions are defined by the fractions of the grid in x and y direction
                regions_fraction = [
                    [[0,1],[0,1]],       # empty grid site
                    [[0,0.05],[0,1]],    # chemokine
                    [[0,0.05],[0,1]],    # netrin
                    [[0,1],[0,1]],       # heparansulfate
                    [[0,1],[0.47,0.51]], # collagen site
                    [[0,1],[0,1]],       # chemokine-netrin
                    [[0,1],[0,1]],       # chemokine-heparansulfate
                    [[0,1],[0,1]],       # netrin-collagen
                    [[0,1],[0,1]],       # chemokine-netrin-collagen
                ]
                
                # calculate the concentration in the regions with the specified fractions
                # and the total concentrations
                
                n_chem_region = int(n_grid*A_0)
                n_net_region = int(n_grid*B_0)
                
                frac_region = (regions_fraction[ptype_N][0][1] - regions_fraction[ptype_N][0][0])*(regions_fraction[ptype_N][1][1] - regions_fraction[ptype_N][1][0])
                
                n_grid_net_region = int(n_grid*frac_region)
                
                c_chem_region = n_chem_region/n_grid_net_region
                c_net_region = n_net_region/n_grid_net_region


                n_coll_region = int(n_grid*c_coll)
                frac_coll_region = (regions_fraction[ptype_G][0][1] - regions_fraction[ptype_G][0][0])*(regions_fraction[ptype_G][1][1] - regions_fraction[ptype_G][1][0])
                n_grid_region = int(n_grid*frac_coll_region)
                
                c_coll_region = n_coll_region/n_grid_region
                
                # concentrations of the particles for the specified regions.
                # therefore this is NOT NECESSARILY the concentration of the whole grid.
                concentrations = [
                    0.0,                 # 0 empty grid site
                    c_chem_region,       # 1 chemokine
                    c_net_region,        # 2 netrin
                    C_0,                 # 3 heparansulfate
                    c_coll_region,       # 4 collagen site
                    0,                   # 5 chemokine-netrin
                    0,                   # 6 chemokine-heparansulfate
                    0,                   # 7 netrin-heparansulfate
                    0.0,                 # 8 netrin-collagen
                    0.0,                 # 9 chemokine-netrin-collagen
                ]
                
                n_ptypes = len(concentrations)
                
                initial_positions = ck.utils.get_initial_positions(
                    grid_size = grid_size,
                    concentrations = concentrations, # this is the concentrations in the specified regions
                    regions_fraction = regions_fraction
                )
                
                comment = f"Collagen System {A_0*100:.3f}% Chemokine, {B_0*100:.3f}% netrin and {C_0*100:.3f}% heparan sulafte {c_coll*100:.3f}% collagen in the system. new bond mechanism particle tracking is turned on. No reflective boundary."
                
                # DEFINE FILENAMES
                # ----------------
                    
                name_sys = f"four_component_system_c_chem_{kk:02d}_c_net_{jj:02d}_c_hep_{ii:02d}_c_coll_{nn:02d}"
                dir_sys = f"/path/to/four_component_system/"
                
                name_sys = ck.utils.validate_name_sys(
                    dir_sys=dir_sys,
                    name_sys=name_sys,
                    overwrite_sys=False,
                )
                
                fname_cfg = ck.utils.get_fname(dir_sys, name_sys, ftype="config")

                # SIMULATION PARAMETERS
                # ---------------------

                num_steps = 20000000
                simulation_stride = 5000
                
                # the particles are reflected at the left and right boundaries
                reflection = [True, True, False, False]

                # define which particle types should be tracked
                # particle tracking is necessarily turned off, since it is not supported
                # for a system with changing number of particles
                track_types = np.zeros(n_ptypes, dtype=bool) 

                
                # setup for adding particles during simulation
                n_1_init = np.sum(initial_positions==1)
                n_2_init = np.sum(initial_positions==2)
                
                # preserve the ratio of particles added
                # n1i... initial number of particles of component 1
                # n1a... number of particles of component 1 to be added
                # (n1i + n1a)/(n2i + n2a) = n1i/n2i
                # n2a = = (n1i + n1a)*n2i/n1i - n2i
                
                # Add two times the number of initial chemokines to the system
                n_1_add = 2*n_1_init
                
                # scale the number of addad netrins accordingly
                n_2_add = (n_1_init + n_1_add)*n_2_init/n_1_init - n_2_init
                
                # calculate how many steps have to pass until one particle gets added
                if n_1_add != 0:
                    n_steps_add_1 = num_steps//n_1_add
                else:
                    # no particles to add? -> make sure the modulo never equates zero 
                    n_steps_add_1 = num_steps + 2
                    
                if n_2_add != 0:     
                    n_steps_add_2 = num_steps//n_2_add
                else:
                    n_steps_add_2 = num_steps + 2
                
                l_x = n_grid_x
                l_y = n_grid_y
                    
                # add to the leftmost 0.5 percent of the box
                idx_add_right = int(0.005*l_x)
                
                if idx_add_right == 0:
                    print("WARNING: idx_add_right is zero")
                    idx_add_right = 3
                
                # set how many particles need to be added per step for each type
                # this number must be <= 1
                add_particles_per_step = np.zeros(n_ptypes, dtype=float)
                add_particles_per_step[1] = 1/n_steps_add_1
                add_particles_per_step[2] = 1/n_steps_add_2
                
                # define the region in which the particles are added 
                # (same notation as for the initial positions)
                region_fraction_add_particles = np.zeros([n_ptypes, 2, 2], dtype=float)
                region_fraction_add_particles[:, :, 1] = 1.0
                region_fraction_add_particles[1] = [[0, idx_add_right/l_x], [0, 1]]
                region_fraction_add_particles[2] = [[0, idx_add_right/l_x], [0, 1]]

                # CREATE CONFIG DICTIONARY
                # ------------------------

                config = {} 
                config["comment"]                       = comment
                config["num_steps"]                     = num_steps
                config["stride"]                        = simulation_stride
                config["reflection"]                    = reflection
                config["name_sys"]                      = name_sys
                config["dir_sys"]                       = dir_sys
                config["track_types"]                   = track_types.tolist()
                config["initial_positions"]             = initial_positions.tolist()
                config["add_particles_per_step"]        = add_particles_per_step.tolist()
                config["region_fraction_add_particles"] = region_fraction_add_particles.tolist()

                # LOAD DEFAULT VALUES
                with h5py.File(fname_default_values, "r") as fh5:
                    config["interaction_probabilities"] = fh5["interaction_probabilities"][:]
                    config["diffusivities"] = fh5["diffusivities"][:]
                    config["binding_constants"] = fh5["binding_constants"][:]
                    config["binding_energy"] = fh5["binding_energy"][:]
                    config["particle_names"] = [name.decode("utf-8") for name in fh5["particle_names"][:]]
                
                # PRINT CONFIGURATION
                # -------------------
                
                diffusivities             = config["diffusivities"]
                particle_names            = config["particle_names"]
                interaction_probabilities = config["interaction_probabilities"]
                K_a                       = config["binding_constants"]

                print("Particle diffusivities:")
                for D, name in zip(diffusivities, particle_names):
                    print(f"{name:15s}: D = {D:6.1f} μm^2/s")

                print("\nBinding Constants:")
                for i, name_i in enumerate(particle_names):
                    for j, name_j in enumerate(particle_names):
                        if K_a[i, j] != 1:
                            print(f"{name_i:15s} + {name_j:15s}: K_a = {K_a[i, j]:.2e} 1/Mol")
                                
                # print interaction matrix
                print("\nNon-zero Interaction Probabilities:")
                print("i               + j               --> n               + m              : probability")
                print("------------------------------------------------------------------------------------")
                for i in range(n_ptypes):
                    for j in range(n_ptypes):
                        if i != 0 and i != j:
                            # print(f"{particle_names[i]:15s} + {particle_names[j]:15s}:")
                            for k in range(n_ptypes**2):
                                n = k//n_ptypes
                                m = k%n_ptypes
                                if n!=i or m!=j:
                                    if not (j==0 and n==0):
                                        if interaction_probabilities[i, j, k] > 0:
                                            print(f"{particle_names[i]:15s} + {particle_names[j]:15s} --> {particle_names[n]:15s} + {particle_names[m]:15s}: {interaction_probabilities[i, j, k]:.2e}")

                # SAVE CONFIGURATION
                # ------------------
                
                with open(fname_cfg, "w") as f:
                    toml.dump(config, f)
                
                dir_cfg_all.append(os.path.dirname(fname_cfg))
    
run_str = "python run_parallel.py"
for ds in set(dir_cfg_all):
    run_str += " " + os.path.join(ds)
    
print("Execution command:")
print(run_str) 
    