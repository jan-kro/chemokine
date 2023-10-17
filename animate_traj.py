import h5py
import numpy as np
import utils

name_sys = "gradient_double_netrin"
dir_sys = "/local_scratch2/janmak98/chemokine/results/scaled_energies/"


fname_traj = utils.get_fname(dir_sys, name_sys, ftype="trajectory")
#fname_traj = "/local_scratch2/janmak98/chemokine/results/langmuir_complete_new/trajectories/traj_cn0.031pct.hdf5"
traj = h5py.File(fname_traj, "r")["trajectory"]

print(traj.shape)

fname_anim = utils.get_fname(dir_sys, name_sys, ftype="animation")

utils.animate(traj, show=False, save=True, fname=fname_anim)

