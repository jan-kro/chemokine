import chemokine as ck
import numpy as np
import sys
import os

from joblib import Parallel, delayed

NUM_JOBS = 6

N_SPLIT = 1 # 1 means no split
IDX_SPLIT = 0 # STARTS AT 0

OVERWRITE = True

USE_CUSTOM_DIR = False
CUSTOM_DIR = ""

CALCULATION_METHOD = "fast"

# handle path names
dirs_sys = ck.analysis_tools.validate_dir_sys_input(sys.argv)

# get config filenames
fnames_config = ck.analysis_tools.list_fnames(dirs_sys, return_dirs=False, return_names=False) 

total_sims = len(fnames_config)

if N_SPLIT != 1:
    print(f"Splitting {total_sims} simulations into {N_SPLIT} parts.")
    fnames_config = np.array_split(fnames_config, N_SPLIT)[IDX_SPLIT]
    k_sim = np.arange(len(fnames_config)) + 1
    total_sims = len(fnames_config)   

if CALCULATION_METHOD == "fast":
    explanation_str = "Loads the whole trajectory into memory. Fast but memory intensive."
elif CALCULATION_METHOD == "efficient":
    explanation_str = "Loads the trajectory frame by frame into memory. Efficient but slow."
else:
    raise ValueError(f"Method {CALCULATION_METHOD} not implemented for concentration calculation.")

print("\nThe following parameters have been set for the concentration calculation:\n")
print(f"Calculation method: {CALCULATION_METHOD} ({explanation_str})")
print(f"Overwrite: {OVERWRITE}")
print(f"Number of jobs: {NUM_JOBS}")
if USE_CUSTOM_DIR:
    print(f"Using custom directory: {CUSTOM_DIR}")
if N_SPLIT != 1:
    print(f"Split: True (Total simulations in given directories are split into {N_SPLIT} parts.)")
    print(f"       Using part {IDX_SPLIT+1}/{N_SPLIT}.")
    
print("")
proceed = "y" == input("Do you want to proceed? (y/n): ")
if not proceed:
    sys.exit()
    
def calc_concentration(fname_config, k_sim, total_sims, overwrite):
    config = ck.utils.load_config_dict(fname_config)
    print(f"[{k_sim:03d}/{total_sims:03d}] System Name: {config['name_sys']}")
    
    if USE_CUSTOM_DIR:
        fname_concentration = os.path.join(CUSTOM_DIR, f"concentration_{config['name_sys']}.npy")
    else:
        fname_concentration = os.path.join(config["dir_sys"], "data", f"concentration_{config['name_sys']}.npy")
    
    if not overwrite:
        try:
            concentration = np.load(fname_concentration)
            print(f"Concentration already calculated for {config['name_sys']}. Skipping ...")
            return
        except:
            pass
    
    if overwrite:
        try:
            os.remove(fname_concentration)
        except:
            pass
        
    fname_traj = ck.utils.get_fname(config["dir_sys"], config["name_sys"], "trajectory")
    if os.path.exists(fname_traj):
        if ck.analysis_tools.is_file_in_use(fname_traj):
            print(f"Trajectory file {fname_traj} is currently in use. Skipping ...")
            return
    else:
        print(f"Trajectory file {fname_traj} does not exist. Skipping ...")
        return
    
    if not os.path.exists(os.path.dirname(fname_concentration)):
        # precaution for parallel processing
        try:
            os.makedirs(os.path.dirname(fname_concentration))
        except:
            pass

    n, concentration = ck.analysis_tools.get_particle_numbers_concentrations(config, total=False)
    np.save(fname_concentration, concentration)
    
    print(f'Finished concentration calculation {k_sim:03d}/{total_sims:03d} for {config["name_sys"]}')

print("\nStarting concentration calculation ...\n")
Parallel(n_jobs=NUM_JOBS)(delayed(calc_concentration)(fname_config, k_sim, total_sims, OVERWRITE) for k_sim, fname_config in enumerate(fnames_config))