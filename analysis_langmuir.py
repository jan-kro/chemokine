import os
import sys
import h5py
import toml
import numpy as np
import matplotlib.pyplot as plt
from animation import animate

# take first input argument as directory to trajectories
sys_dir = sys.argv[1]
if len(sys_dir) == 0:
    sys_dir = "/local_scratch2/janmak98/chemokine/results/langmuir_ccl5_netrin_only_logscale_eq_3"

def get_trajectory_fnames(sys_dir):
    fnames = []
    for fname in os.listdir(os.path.join(sys_dir, "trajectories")):
        if fname.startswith("traj"):
            fnames.append(os.path.join(sys_dir, "trajectories", fname))
    return fnames

def get_parameters(sys_dir):
    fnames = []
    for fname in os.listdir(os.path.join(sys_dir, "configs")):
        if fname.startswith("cfg"):
            fnames.append(os.path.join(sys_dir, "configs", fname))
    with open(os.path.join(sys_dir, "configs", "parameters.toml"), "r") as f:
        parameters = toml.load(f)
    return parameters
