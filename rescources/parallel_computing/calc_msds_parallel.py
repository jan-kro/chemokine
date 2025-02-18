import toml
import os
import sys
import h5py
import numpy as np
import chemokine as ck
from tools import list_fnames
from joblib import Parallel, delayed

# run with:
# python calc_msds_parallel.py /path/to/dir1 /path/to/dir2 ...

overwrite = True

use_custom_dir = False
custom_dir = "/home/janmak98/Projects/chemokine_skripts/december_24/three_components_varying_net/data_incomplete"

#! CHOOSE METHOD HERE
msd_method = "fast"
#msd_method = "efficient"

if overwrite:
    overwrite = "y" == input("Are you sure you want to overwrite the existing MSDs? (y/n): ")

N_SPLIT = 1
IDX_SPLIT = 1 # STARTS AT 0

STRIDE = 1
PTYPE = 1
num_jobs = -1 # -1 use maximum available CPUs


if STRIDE != 1:
    print(f"WARNING: Calculating MSDs with stride {STRIDE}.")
    user_agreed = "y" == input("Are you sure you want to continue? (y/n): ")
    if not user_agreed:
        sys.exit()

if msd_method == "fast":
    explanation_str = "Loads the whole trajectory into memory. Fast but memory intensive."
elif msd_method == "efficient":
    explanation_str = "Loads the trajectory frame by frame into memory. Efficient but slow."
else:
    raise ValueError(f"Method {msd_method} not implemented for MSD calculation.")

dirs_sys = []
for kk in range(1, len(sys.argv)):
    fname = sys.argv[kk]
    if os.path.isdir(fname):
        # convert to system path in case configs path is given
        fname = os.path.abspath(fname)
        if os.path.basename(os.path.abspath(fname)) == "configs":
            fname = os.path.dirname(fname)
        dirs_sys.append(fname)
    else:
        raise ValueError(f"Directory {sys.argv[kk]} does not exist.")

if len(set(dirs_sys)) != len(dirs_sys):
    raise ValueError("Config directories must be unique!")

# for every sys directory, sort the fnames ascendingly
fnames_config, dirs_sys, names_sys = list_fnames(dirs_sys) 

total_sims = len(fnames_config)

if N_SPLIT != 1:
    print(f"Splitting {total_sims} simulations into {N_SPLIT} parts.")
    fnames_config = np.array_split(fnames_config, N_SPLIT)[IDX_SPLIT]
    k_sim = np.arange(len(fnames_config)) + 1
    total_sims = len(fnames_config)   

print("\nThe following parameters have been set for the MSD calculation:\n")
print(f"MSD method: {msd_method} ({explanation_str})")
print(f"Stride: {STRIDE}")
print(f"Overwrite: {overwrite}")
print(f"Particle type: {PTYPE}")
print(f"Number of jobs: {num_jobs}")
if use_custom_dir:
    print(f"Using custom directory: {custom_dir}")
if N_SPLIT != 1:
    print(f"Split: True (Total simulations in given directories are split into {N_SPLIT} parts.)")
    print(f"       Using part {IDX_SPLIT+1}/{N_SPLIT}.")

print("")
proceed = "y" == input("Do you want to proceed? (y/n): ")
if not proceed:
    sys.exit()


def calc_msd(fname_config, ptype, k_sim, total_sims, stride, overwrite, msd_method):
    config = ck.utils.load_config_dict(fname_config)
    print(f"[{k_sim:03d}/{total_sims:03d}] System Name: {config['name_sys']}")
    
    if use_custom_dir:
        fname_msd_x = os.path.join(custom_dir, f"msd_x_ptype_{ptype:d}_{config['name_sys']}.npy")
        fname_msd_y = os.path.join(custom_dir, f"msd_y_ptype_{ptype:d}_{config['name_sys']}.npy")
    else:
        fname_msd_x = os.path.join(config["dir_sys"], "data", f"msd_x_ptype_{ptype:d}_{config['name_sys']}.npy")
        fname_msd_y = os.path.join(config["dir_sys"], "data", f"msd_y_ptype_{ptype:d}_{config['name_sys']}.npy")
    
    if not overwrite:
        try:
            msd_x = np.load(fname_msd_x)
            msd_y = np.load(fname_msd_y)
            print(f"MSD already calculated for {config['name_sys']}. Skipping ...")
            return
        except:
            pass
    
    if overwrite:
        try:
            os.remove(fname_msd_x)
            os.remove(fname_msd_y)
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
            try:
                with h5py.File(fname_particle_traj, "r") as fh5:
                    if True:
                        pass
            except:
                print(f"Simulation of {config['name_sys']} is not completed or corrupted. Skipping ...")
                return
    else:                   
        print(f"WARNING: No particle trajectory file found for {config['name_sys']}. Skipping ...")
        return
    
    if msd_method == "fast":
        msd_x, msd_y = ck.utils.get_msd_xy(config, stride=stride, ptype=ptype)
    elif msd_method == "efficient":
        msd_x, msd_y = ck.utils.get_msd_xy_2(config, stride=stride, ptype=ptype) #! DECIDE HERE
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


print(f"\n Calculating MSDs for particle type {PTYPE} ...")
Parallel(n_jobs=num_jobs)(delayed(calc_msd)(fname, PTYPE, k, total_sims, STRIDE, overwrite, msd_method) for fname, k in zip(fnames_config, np.arange(total_sims) + 1))