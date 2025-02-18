import toml
import os
import sys
import h5py
import numpy as np
import chemokine as ck
from joblib import Parallel, delayed

#* NOTES ON PARALLEL COMPUTING
#* ---------------------------

#* With parallel computing we mean that we run multiple simulations at the same time.
#! BUT: Each simulation is still running on a single CPU core!
#* This script is run with the following command:
#* python run_parallel.py /path/to/config_directory1 /path/to/config_directory2 ...
#* The script will then run all simulations in the given directories using the 
#* number of cores specified with the 'num_jobs' variable.

#* Additionally, if one wants to use multiple machines, we can split the
#* simulations into parts using the 'N_SPLIT' and 'IDX_SPLIT' variables.
#* i.e. one can split the simulations into 4 parts and run each part on a different
#* machine, using the indices 0,1,2,3 to specify which part to run.

N_SPLIT = 1 # 1 means no split
IDX_SPLIT = 0 # STARTS AT 0

# SET NUMBER OF JOBS
num_jobs = -1 # -1 use maximum available CPUs

#* RUN SIMULATIONS
#* ---------------

#* validate config paths

# handle path names
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

# validate config filenames
for fname in fnames_config:
    if not os.path.exists(fname): 
        raise ValueError(f"File {fname} does not exist!")
    if not fname[-5:] == ".toml":
        raise ValueError(f"File {fname} is not a config file")

if N_SPLIT != 1:
    print(f"Splitting {total_sims} simulations into {N_SPLIT} parts.")
    fnames_config = np.array_split(fnames_config, N_SPLIT)[IDX_SPLIT]
    k_sim = np.arange(len(fnames_config)) + 1
    total_sims = len(fnames_config)    

#* manually verify input parameters

print("\nSimulations will be done for the following parameters:\n")
print("Number of total simulations: ", total_sims)
print("Number of different directories: ", n_unique_dirs)
print("Number of jobs: ", num_jobs)
if N_SPLIT != 1:
    print(f"Split: True (Total Simulations in given directories is split into {N_SPLIT} parts.)")
    print(f"       Using part {IDX_SPLIT+1}/{N_SPLIT}.")

proceed = "y" == input("Do you want to proceed? (y/n): ")
if not proceed:
    sys.exit()

#* run simulations
        
print(f"Number of total simulations: {total_sims}")

def run_simulation(fname_config, k_sim, total_sims):
    config = ck.utils.load_config_dict(fname_config)
    fname_traj = ck.utils.get_fname(config["dir_sys"], config["name_sys"], "trajectory")
    
    # check if simulations hasnt been done yet
    if os.path.exists(fname_traj):
        # check if h5 file is corrupted:
        
        if ck.analysis_tools.is_file_in_use(fname_traj):
            print(f"Simulation {k_sim}/{total_sims} already started. Skipping ...")
            return
        
        try:
            with h5py.File(fname_traj, "r") as fh5:
                if 'completion_status' in fh5.attrs:
                    print("completion status:", fh5['completion_status'][()])
                    if fh5['completion_status'][()] == "completed":
                        print(f"Simulation {k_sim}/{total_sims} already exists. Skipping ...")
                        return
                    else:
                        print(f"Simulation {k_sim}/{total_sims} is corrupted. Restarting ...")
        except:
            print(f"Simulation {k_sim}/{total_sims} is corrupted. Restarting ...")

    print(f"[{k_sim}/{total_sims}] Simulating {config['name_sys']} ...")
    ck.grid_simulation.simulate(config)
    print(f"\nsimulation {k_sim}/{total_sims} finished")
    print(f"Name:      {config['name_sys']}")
    print(f"Directory: {config['dir_sys']}")


Parallel(n_jobs=num_jobs)(delayed(run_simulation)(fname, k, total_sims) for fname, k in zip(fnames_config, k_sim))

print("Simulations finished.")