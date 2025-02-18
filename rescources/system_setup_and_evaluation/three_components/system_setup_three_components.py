import chemokine as ck
import numpy as np 
import h5py
import toml
import os
from scipy.optimize import fsolve


#* CREATE SYSTEM CONTAINING CHEMOKINE NETRIN AND HEPARAN SULFATE *#


# define grid size for all systems
n_grid_x = 100*4
n_grid_y = 100*3
n_grid = n_grid_x*n_grid_y

# define the minimum (start) and maximum (end) concentration fractions of chemokine (c1) and netrin (c2)
# i.e. the systems concentrations will be in the range [c1_start, c1_end] for chemokine and [c2_start, c2_end]
# for netrin, where c2/c1 ranges from 0.1 to 10. (this is defined in this way in case we want to simulate systems 
# for multiple chemokine concentrations)
c2_c1_start = 0.1
c2_c1_end = 10

c_chem_all = [0.001]
c_hep_all = np.array([0, 1e-3, 2e-3, 4e-3])

# define the number of different netrin concentrations that should be tested
n1 = len(c_chem_all)
n2 = 10
n3 = len(c_hep_all)

# Since we want to create initial positions where the system is already equilibrated (i.e. the initial 
# concentrations are the equilibrium concentrations) we need to solve the system of equations that 
# describes the concentration balance as a function of Kd values and absolute concentrations.

# load default values into config dict
fname_default_values = "/path/to/defailt_values_scaled_energies.h5"

# get the default values for the binding constantes
fh5 = h5py.File(fname_default_values, "r")

# define particle types
ptype_C = 1     # Chemokine
ptype_N = 2     # Netrin
ptype_H = 3     # Heparan Sulfate
ptype_CN = 5    # Chemokine-Netrin
ptype_CH = 6    # Chemokine-Heparan Sulfate
ptype_NH = 7    # Netrin-Heparan Sulfate

# get bond probabilities
bond_prob_CN = fh5["interaction_probabilities"][ptype_C,ptype_N,ptype_CN]/4
bond_prob_CH = fh5["interaction_probabilities"][ptype_C,ptype_H,ptype_CH]/4
bond_prob_NH = fh5["interaction_probabilities"][ptype_N,ptype_H,ptype_NH]/4

# get particle diffusivities
D_C = fh5["interaction_probabilities"][ptype_C,0,ptype_C]/4
D_N = fh5["interaction_probabilities"][ptype_N,0,ptype_N]/4
D_H = fh5["interaction_probabilities"][ptype_H,0,ptype_H]/4

# calculate equilibrium constants
K_CN_setup = (bond_prob_CN / D_C)**2
K_CH_setup = (bond_prob_CH / D_C)**2
K_NH_setup = (bond_prob_NH / D_N)**2

fh5.close()

c_chem_th_all = []
c_net_th_all = []
c_hep_th_all = []
c_chem_net_th_all = []
c_chem_hep_th_all = []
c_net_hep_th_all = []

dir_cfg_all = []
for kk, C_0 in enumerate(c_chem_all):
    
    # define the initial netrin concentrations
    c2_start = c2_c1_start*C_0
    c2_end = c2_c1_end*C_0
    c_net_all = np.logspace(np.log10(c2_start), np.log10(c2_end), n2)
    
    for jj, N_0 in enumerate(c_net_all):
        for ii, H_0 in enumerate(c_hep_all):
            
            # solve system of equations to calculate equilibrium concentrations
            C, N, H = fsolve(
                ck.analysis_tools.three_component_system, 
                x0=[C_0, N_0, H_0], 
                args=(C_0, N_0, H_0, K_CN_setup, K_CH_setup, K_NH_setup),
                xtol=1e-10)
            
            # calculate equilibrium concentrations of the particle complexes 
            CN = K_CN_setup*C*N
            CH = K_CH_setup*C*H
            NH = K_NH_setup*N*H

            # rename for readability
            c_chem_th     = C 
            c_net_th      = N
            c_hep_th      = H
            c_chem_net_th = CN
            c_chem_hep_th = CH
            c_net_hep_th  = NH

            # add comment entry to config dictionary
            comment = f"Equilibrated System {c_chem_th*100:.3f}% Chemokine, {c_net_th*100:.3f}% netrin and {c_hep_th*100:.3f}% heparan sulafte {c_chem_net_th*100:.3f}% chem-net {c_chem_hep_th*100:.3f}% chem-hep {c_net_hep_th*100:.3f}% net_hep in the system. new bond mechanism particle tracking is turned on. No reflective boundary."

            # SIMULATION PARAMETERS
            # ---------------------

            num_steps = 5000000
            simulation_stride = 100
            
            # DEFINE FILENAMES
            # ----------------
                
            name_sys = f"three_component_system_c_chem_{kk:02d}_c_net_{jj:02d}_c_hep_{ii:02d}"
            dir_sys = f"/path/to/thee_component_system/"
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

            grid_size = [n_grid_x, n_grid_y]
            reflection = [False, False, False, False] # periodic boundaries at all sides
            
            # concentrations of the particles for the specified regions.
            # therefore this is NOT NECESSARILY the concentration of the whole grid.
            concentrations = [
                0.0,                      # 0 empty grid site
                c_chem_th,                # 1 chemokine
                c_net_th,                 # 2 netrin
                c_hep_th,                 # 3 heparansulfate
                0.0,                      # 4 collagen site
                c_chem_net_th,            # 5 chemokine-netrin
                c_chem_hep_th,            # 6 chemokine-heparansulfate
                c_net_hep_th,             # 7 netrin-heparansulfate
                0.0,                      # 8 netrin-collagen
                0.0,                      # 9 chemokine-netrin-collagen
            ]

            n_ptypes = len(concentrations)

            initial_positions = ck.utils.get_initial_positions_homogeneous(grid_size, concentrations)

            # define which particle types should be tracked
            track_types = np.zeros(n_ptypes, dtype=bool)
            
            # track chemokines! 
            track_types[1] = True 


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
    
run_str = "python run_parallel.py"
for ds in set(dir_cfg_all):
    run_str += " " + os.path.join(ds)
    
print("Execution command:")
print(run_str) 
    