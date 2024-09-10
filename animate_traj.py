import h5py
import numpy as np
import utils

# gradient simulation with non-zero ccl-hep binding energy
names_all = ["langmuir_1pctccl_0pctnet_10pcthep",
             "langmuir_1pctccl_1pctnet_10pcthep",
             "langmuir_1pctccl_2pctnet_10pcthep",
             "langmuir_1pctccl_5e-1pctnet_10pcthep",
             "langmuir_2pctccl_0pctnet_10pcthep",
             "langmuir_2pctccl_1pctnet_10pcthep",
             "langmuir_2pctccl_2pctnet_10pcthep",
             "langmuir_2pctccl_5e-1pctnet_10pcthep",
             "langmuir_5e-1pctccl_0pctnet_10pcthep",
             "langmuir_5e-1pctccl_1pctnet_10pcthep",
             "langmuir_5e-1pctccl_2pctnet_10pcthep",
             "langmuir_5e-1pctccl_5e-1pctnet_10pcthep"]

names = [names_all[4], names_all[5], names_all[6], names_all[7]]
dir_sys = '/local_scratch2/janmak98/chemokine/results/gradients/gradient_15pct_rightscaling/'

name_sys = names[1]

dir_sys = "/net/data05/janmak98/results/september/chemokine/tests/"
name_sys = "langmuir_i0_1e-03pctccl5_j0_1e-03pctnet1_k0_1e-02pcthep"

fname_traj = utils.get_fname(dir_sys, name_sys, ftype="trajectory")
traj = h5py.File(fname_traj, "r")["trajectory"]


fname_anim = utils.get_fname(dir_sys, name_sys, ftype="animation")
#fname_anim = "/local_scratch2/janmak98/movies_mid_term/chemokine_diffusion/anim_2pctccl_1pctnet_10pcthep.mp4"

#utils.animate(traj, show=True, save=False, fname=fname_anim)
utils.animate(traj, show=False, save=True, fname=fname_anim)