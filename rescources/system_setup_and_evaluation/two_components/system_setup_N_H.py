import chemokine as ck
import numpy as np 
import h5py
import toml
import os
from scipy.optimize import fsolve


#* CREATE SYSTEM CONTAINING NETRIN AND HEPARAN SULFATE FOR DIFFERENT CHEMOKINE CONCENTRATIONS *#
 
# Since we want to create initial positions where the system is already equilibrated (i.e. the initial 
# concentrations are the equilibrium concentrations) we need to solve the system of equations that 
# describes the concentration balance as a function of Kd values and absolute concentrations.
 
# load default values into config dict
fname_default_values = "/home/janmak98/Projects/chemokine_skripts/default_values/default_values_scaled_energies_times_100.h5"

with h5py.File(fname_default_values, "r") as fh5:
    interaction_prob_normalization = fh5["interaction_prob_normalization"][()]
    interaction_probabilities = fh5["interaction_probabilities"][()]

ptype_C = 1     # Chemokine
ptype_N = 2     # Netrin
ptype_H = 3     # Heparan Sulfate
ptype_CN = 5    # Chemokine-Netrin
ptype_CH = 6    # Chemokine-Heparan Sulfate
ptype_NH = 7    # Netrin-Heparan Sulfate

# get the bond probabilities and diffusivities

# one needs to divide all values by four since this is also done in the simulation setup 
# in the ck.simulate() function. The reason is that the sum of the probabilities for all 
# four neighbouring moves must be 1.

# get bond probabilities
bond_prob_CN = interaction_probabilities[ptype_C,ptype_N,ptype_CN]/4
bond_prob_CH = interaction_probabilities[ptype_C,ptype_H,ptype_CH]/4
bond_prob_NH = interaction_probabilities[ptype_N,ptype_H,ptype_NH]/4

# get diffusivities
D_C      = interaction_probabilities[ptype_C,0,ptype_C]/4
D_N      = interaction_probabilities[ptype_N,0,ptype_N]/4

# calculate equilibrium constants
K_CN      = (bond_prob_CN / D_C)**2
K_CH      = (bond_prob_CH / D_C)**2
K_NH      = (bond_prob_NH / D_N)**2

# define all concentrations that should be tested in units of particles per unit volume (1/a^3)
c_C_all = [0.0]
c_N_all = np.logspace(-4, -1, 8)
c_H_all = [2e-3]

n_C = len(c_C_all)
n_N = len(c_N_all)
n_H = len(c_H_all)

# list af all config directories
dir_cfg_all = []

# loop through all possible concentration combinations
idx_sys = 0
for kk, C_0 in enumerate(c_C_all):
    for jj, N_0 in enumerate(c_N_all):
        for ii, H_0 in enumerate(c_H_all):
            
            C, N, H = fsolve(
                ck.analysis_tools.three_component_system, 
                x0=[C_0, N_0, H_0], 
                args=(C_0, N_0, H_0, K_CN, K_CH, K_NH),
                xtol=1e-10)
            
            # calculate equilibrium concentrations of the particle complexes 
            CN = K_CN*C*N
            CH = K_CH*C*H
            NH = K_NH*N*H

            # rename for readability
            c_chem     = C 
            c_net      = N
            c_hep      = H
            c_chem_net = CN
            c_chem_hep = CH
            c_net_hep  = NH

            # add comment entry to config dictionary
            comment = f"{c_chem*100:.3f}% Chemokine, {c_net*100:.3f}% netrin {c_hep*100:.3f}% heparan sulafte {c_chem_hep*100:.3f} CH {c_chem_net*100:.3f} CN {c_net_hep*100:.3f} in the system."
            
            # SIMULATION PARAMETERS-
            # ---------------------

            num_steps         = 150000000
            simulation_stride = 100
     
            # DEFINE FILENAMES
            # ----------------
                
            name_sys = f"two_component_sys_150M_steps_stride_100_{idx_sys:02d}"
            dir_sys = f"path/to/two_component_systems/N_H"
            name_sys = ck.utils.validate_name_sys(
                dir_sys=dir_sys,
                name_sys=name_sys,
                overwrite_sys=False,
            )
            
            # create config filename
            fname_cfg = ck.utils.get_fname(dir_sys, name_sys, ftype="config")

            # SYSTEM PARAMETERS
            # -----------------

            # SET UP INITIAL POSITIONS

            grid_size = [150, 150]
            reflection = [False, False, False, False]
            
            # concentrations of the particles for the specified regions.
            # therefore this is NOT NECESSARILY the concentration of the whole grid.
            concentrations = [
                0.0,                      # 0 empty grid site
                c_chem,                   # 1 chemokine
                c_net,                    # 2 netrin
                c_hep,                    # 3 heparansulfate
                0.0,                      # 4 collagen site
                c_chem_net,               # 5 chemokine-netrin
                c_chem_hep,               # 6 chemokine-heparansulfate
                c_net_hep,                # 7 netrin-heparansulfate
                0.0,                      # 8 netrin-collagen
                0.0,                      # 9 chemokine-netrin-collagen
            ]

            n_ptypes = len(concentrations)

            initial_positions = ck.utils.get_initial_positions_homogeneous(grid_size, concentrations)

            # define which particle types should be tracked
            track_types = np.zeros(n_ptypes, dtype=bool) # in this case no particles are tracked

            # CREATE CONFIG DICTIONARY
            # ------------------------

            config = {} 
            config["comment"]             = comment
            config["num_steps"]           = num_steps
            config["stride"]              = simulation_stride
            config["reflection"]          = reflection
            config["name_sys"]            = name_sys
            config["dir_sys"]             = dir_sys
            config["track_types"]         = track_types.tolist()
            config["initial_positions"]   = initial_positions.tolist()

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
            
            idx_sys += 1
            
run_str = "python run_parallel.py"
for ds in set(dir_cfg_all):
    run_str += " " + os.path.join(ds)
    
print("Execution command:")
print(run_str) 
    