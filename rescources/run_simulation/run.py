import os
import sys
import numpy as np
import chemokine as ck


#* RUN SIMULATIONS
#* ---------------

#* this script runs the simulation for all config files in the given directories via
#* python run.py /path/to/config_directory1 /path/to/config_directory2 ...

# handle input directories
dirs_sys = []
for kk in range(1, len(sys.argv)):
    dir_name = os.path.abspath(sys.argv[kk])
    if os.path.isdir(dir_name):
        # convert to system path in case configs path is given
        if os.path.basename(dir_name) == "configs":
            dir_name = os.path.dirname(dir_name)
        if not os.path.exists(dir_name):
            raise ValueError(f"Directory {dir_name} does not exist.")
        dirs_sys.append(dir_name)
    else:
        raise ValueError(f"Directory {dir_name} does not exist.")

# check if config directories are unique
if len(set(dirs_sys)) != len(dirs_sys):
    raise ValueError("Config directories must be unique!")
else:
    n_unique_dirs = len(dirs_sys)

# create list of all config filenames
fnames_config, dirs_sys, names_sys = ck.analysis_tools.list_fnames(dirs_sys) 

total_sims = len(fnames_config)
k_sim = np.arange(total_sims) + 1
    
print(f"Number of total simulations: {total_sims}")

# define function to report current staus, load config and run the simulation
def run_simulation(fname_config, k_sim, total_sims):
    config = ck.utils.load_config_dict(fname_config)
    print(f"[{k_sim}/{total_sims}] Simulating {config['name_sys']} ...")
    ck.grid_simulation.simulate(config)
    print(f"\nsimulation {k_sim}/{total_sims} finished")
    print(f"Name:      {config['name_sys']}")
    print(f"Directory: {config['dir_sys']}")

# loop over all config files and run the simulation
for fname, k in zip(fnames_config, k_sim):
    run_simulation(fname, k, total_sims)
    
