import toml
import os
import sys
import h5py
import numpy as np
import chemokine as ck
from joblib import Parallel, delayed

#* In this script the given simulations are run in parallel and the 
#* mean squared displacements for both the x- and y- direction are 
#* calculated and saved for each simulation. This is done in the same 
#* way as in the script run_parallel.py (you can find further infos there) 

#* The MSDs are then saved in the system directory:
#* />dir_sy</data/msd_x_ptype_>ptype<_>name_sys<.npy 
#* />dir_sy</data/msd_y_ptype_>ptype<_>name_sys<.npy

N_SPLIT = 1
IDX_SPLIT = 1 # STARTS AT 0

#! CHOOSE MSD METHOD
MSD_METHOD = "fast"
#MSD_METHOD = "efficient"

#! SET PARAMETERS
MSD_STRIDE = 1    
MSD_PTYPE = 1

#! SET NUMBER OF JOBS
num_jobs = 10 # -1 use maximum available CPUs
num_jobs_msd = -1 # -1 use maximum available CPUs


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

#* verify input parameters

if MSD_STRIDE != 1:
    print(f"WARNING: Calculating MSDs with stride {MSD_STRIDE}.")
    user_agreed = "y" == input("Are you sure you want to continue? (y/n): ")
    if not user_agreed:
        sys.exit()
        
if MSD_METHOD == "fast":
    explanation_str = "Loads the whole trajectory into memory. Fast but memory intensive."
elif MSD_METHOD == "efficient":
    explanation_str = "Loads the trajectory frame by frame into memory. Efficient but slow."
else:
    raise ValueError(f"Method {MSD_METHOD} not implemented for MSD calculation.")

#* manually verify input parameters

print("\nSimulations will be done for the following parameters:\n")
print("Number of total simulations: ", total_sims)
print("Number of different directories: ", n_unique_dirs)
print("Number of jobs: ", num_jobs)
if N_SPLIT != 1:
    print(f"Split: True (Total Simulations in given directories is split into {N_SPLIT} parts.)")
    print(f"       Using part {IDX_SPLIT+1}/{N_SPLIT}.")

print("\nThe following parameters have been set for the MSD calculation:\n")
print(f"MSD method: {MSD_METHOD} ({explanation_str})")
print(f"Stride: {MSD_STRIDE}")
print(f"Particle type: {MSD_PTYPE}")
print(f"Number of jobs: {num_jobs_msd}\n")

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
print("\n Calculating MSDs ...")

msds_x = []
msds_y = []

def calc_msd(fname_config, ptype, k_sim, total_sims, stride, msd_method):
    config = ck.utils.load_config_dict(fname_config)
    print(f"[{k_sim:03d}/{total_sims:03d}] System Name: {config['name_sys']}")
    fname_msd_x = os.path.join(config["dir_sys"], "data", f"msd_x_ptype_{ptype:d}_{config['name_sys']}.npy")
    fname_msd_y = os.path.join(config["dir_sys"], "data", f"msd_y_ptype_{ptype:d}_{config['name_sys']}.npy")
    try:
        msd_x = np.load(fname_msd_x)
        msd_y = np.load(fname_msd_y)
        print(f"MSD already calculated for {config['name_sys']}. Skipping ...")
        return
    except:
        pass
    
    fname_traj = ck.utils.get_fname(config["dir_sys"], config["name_sys"], "trajectory")
    if os.path.exists(fname_traj):
        if ck.analysis_tools.is_file_in_use(fname_traj):
            print(f"Simulation of {config['name_sys']} still running. Skipping ...")
            return
        
    fname_particle_traj = ck.utils.get_fname(config["dir_sys"], config["name_sys"], "particle_traj")
    if os.path.exists(fname_particle_traj):
        if ck.analysis_tools.is_file_in_use(fname_particle_traj):
            print(f"MSD calculation of {config['name_sys']} already started. Skipping ...")
            return
    else:
        print(f"WARNING: No particle trajectory file found for {config['name_sys']}. Skipping ...")
        return
    
    if msd_method == "fast":
        msd_x, msd_y = ck.utils.get_msd_xy(config, stride=stride, ptype=ptype)
    elif msd_method == "efficient":
        msd_x, msd_y = ck.utils.get_msd_xy_2(config, stride=stride, ptype=ptype) 
    else:
        raise ValueError(f"Method {msd_method} not implemented for MSD calculation.")
    
    if not os.path.exists(os.path.dirname(fname_msd_x)):
        # precaution for parallel processing
        try:
            os.makedirs(os.path.dirname(fname_msd_x))
        except:
            pass
        
    np.save(fname_msd_x, msd_x)
    np.save(fname_msd_y, msd_y)
    print(f'Finished MSD calculation {k_sim:03d}/{total_sims:03d} for {config["name_sys"]}')


Parallel(n_jobs=num_jobs_msd)(delayed(calc_msd)(fname, MSD_PTYPE, k, total_sims, MSD_STRIDE, MSD_METHOD) for fname, k in zip(fnames_config, np.arange(total_sims) + 1))