import os
import h5py
import mpltex
import numpy as np
import chemokine as ck
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

#* Plotting script for the fraction bound of chemokine to Heparan Sulfate 
#* and Netrin to Heparan Sulfate 

# (1) LOAD DEFAULT VALUES
# -----------------------
 
fname_default_values = "/path/to/default_values_scaled_energies.h5"

with h5py.File(fname_default_values, "r") as fh5:
    interaction_prob_normalization = fh5["interaction_prob_normalization"][()]
    interaction_probabilities = fh5["interaction_probabilities"][()]

# get the diffusivities and equilibrium constants used in the simulation
    
ptype_C = 1
ptype_N = 2
ptype_H = 3
ptype_CN = 5
ptype_CH = 6
ptype_NH = 7

bond_prob_CN = interaction_probabilities[ptype_C,ptype_N,ptype_CN]/4
bond_prob_CH = interaction_probabilities[ptype_C,ptype_H,ptype_CH]/4
bond_prob_NH = interaction_probabilities[ptype_N,ptype_H,ptype_NH]/4

# free diffusivities of the particles
D_C = interaction_probabilities[ptype_C,0,ptype_C]/4 # Chemokine
D_N = interaction_probabilities[ptype_N,0,ptype_N]/4 # netrin

# equilibrium constants    
K_CN = (bond_prob_CN / D_C)**2 # Chemokine-Netrin
K_CH = (bond_prob_CH / D_C)**2 # Chemokine-Heparan Sulfate
K_NH = (bond_prob_NH / D_N)**2 # Netrin-Heparan Sulfate

# (2) CALCULATE THEORETICAL CURVES
# --------------------------------

# (2.a) NETRIN - HEPARAN SULFATE

# initial lists where the particles concentrations of all systems are saved
c_chem_all = []
c_net_all = []
c_hep_all = []
c_chem_net_all = []
c_chem_hep_all = []
c_net_hep_all = []

# define the initial absolute concentrations (must be same as in the simulations)
c_C_all = [0.0]
c_N_all = np.logspace(-4, -1, 100)
c_H_all = [2e-3]

n_C = len(c_C_all)
n_N = len(c_N_all)
n_H = len(c_H_all)

# loop through all possible concentration combinations
idx_sys = 0
for kk, C_0 in enumerate(c_C_all):
    for jj, N_0 in enumerate(c_N_all):
        for ii, H_0 in enumerate(c_H_all):
            
            # numerically solve system of equations to calculate equilibrium concentrations
            C, N, H = fsolve(
                ck.analysis_tools.three_component_system, 
                x0=[0, N_0, H_0], 
                args=(0, N_0, H_0, K_CN, K_CH, K_NH),
                xtol=1e-10)
            
            CN = K_CN*C*N
            CH = K_CH*C*H
            NH = K_NH*N*H

            c_chem_all.append(C)
            c_net_all.append(N)
            c_hep_all.append(H)
            c_chem_net_all.append(CN)
            c_chem_hep_all.append(CH)
            c_net_hep_all.append(NH)
            
c_chem_all = np.array(c_chem_all)
c_net_all = np.array(c_net_all)
c_hep_all = np.array(c_hep_all)
c_chem_net_all = np.array(c_chem_net_all)
c_chem_hep_all = np.array(c_chem_hep_all)
c_net_hep_all = np.array(c_net_hep_all)

# (2.b) CHEMOKINE - HEPARAN SULFATE

c_C_2_all = np.logspace(-5, -1, 100)
c_N_2_all = [0.0]
c_H_2_all = [2e-3]

n_C_2 = len(c_C_2_all)
n_N_2 = len(c_N_2_all)
n_H_2 = len(c_H_2_all)

c_chem_2_all = []
c_net_2_all = []
c_hep_2_all = []
c_chem_net_2_all = []
c_chem_hep_2_all = []
c_net_hep_2_all = []

idx_sys = 0
for kk, C_0 in enumerate(c_C_2_all):
    for jj, N_0 in enumerate(c_N_2_all):
        for ii, H_0 in enumerate(c_H_2_all):
            
            C, N, H = fsolve(
                ck.analysis_tools.three_component_system, 
                x0=[C_0, 0, H_0], 
                args=(C_0, 0, H_0, K_CN, K_CH, K_NH),
                xtol=1e-10)
            
            CN = K_CN*C*N
            CH = K_CH*C*H
            NH = K_NH*N*H

            c_chem_2_all.append(C)
            c_net_2_all.append(N)
            c_hep_2_all.append(H)
            c_chem_net_2_all.append(CN)
            c_chem_hep_2_all.append(CH)
            c_net_hep_2_all.append(NH)
            
c_chem_2_all = np.array(c_chem_2_all)
c_net_2_all = np.array(c_net_2_all)
c_hep_2_all = np.array(c_hep_2_all)
c_chem_net_2_all = np.array(c_chem_net_2_all)
c_chem_hep_2_all = np.array(c_chem_hep_2_all)
c_net_hep_2_all = np.array(c_net_hep_2_all)

# (3) CALCULATE FRACTION BOUND FROM SIMULATIONS
# ---------------------------------------------

dir_sys = "path/to/two_component_systems/N_H"
dir_sys_2 = "path/to/two_component_systems/C_H"

# get all config file names in the directories
fnames_config, dirs_sys, names_sys = ck.analysis_tools.list_fnames(dir_sys)
fnames_config_2, dirs_sys_2, names_sys_2 = ck.analysis_tools.list_fnames(dir_sys_2)

data_dir_N_H = "path/to/two_component_systems/data/N_H"
data_dir_C_H = "path/to/two_component_systems/data/C_H"

# (3.a) NETRIN - HEPARAN SULFATE

# try to load the fraction bound from the simulations
# if it does not exist, calculate it from the simulations
try:
    n_C_all = np.load(os.path.join(data_dir_N_H, "n_C_all.npy"))
    n_N_all = np.load(os.path.join(data_dir_N_H, "n_N_all.npy"))
    n_H_all = np.load(os.path.join(data_dir_N_H, "n_H_all.npy"))
    n_CN_all = np.load(os.path.join(data_dir_N_H, "n_CN_all.npy"))
    n_CH_all = np.load(os.path.join(data_dir_N_H, "n_CH_all.npy"))
    n_NH_all = np.load(os.path.join(data_dir_N_H, "n_NH_all.npy"))
except:    
    print("DATA DOES NOT EXIST. Calculating fraction bound for N-H system...")
    
    if not os.path.exists(data_dir_N_H):
        os.makedirs(data_dir_N_H)
    
    n_C_all = []
    n_N_all = []
    n_H_all = []
    n_CN_all = []
    n_CH_all = []
    n_NH_all = []

    n_N_traj_all = []
    idx_start = 0

    for i, (dir_sys, name_sys) in enumerate(zip(dirs_sys, names_sys)):
        cfg = ck.utils.load_config_dict(fnames_config[i])
        fname_traj = ck.utils.get_fname(dir_sys, name_sys, "trajectory")
        with h5py.File(fname_traj, "r") as fh5:
            traj = fh5["trajectory"]
            n_frames = traj.shape[0]
            n_C = 0
            n_N = 0
            n_H = 0
            n_CN = 0
            n_CH = 0
            n_NH = 0
            
            n_N_traj = np.zeros(n_frames-idx_start)
            
            for idx_frame in range(idx_start, n_frames):
                frame = traj[idx_frame]
                n_C += np.sum(frame == ptype_C)
                
                n_N_traj[idx_frame-idx_start] = np.sum(frame == ptype_N)
                n_N += n_N_traj[idx_frame-idx_start]
                
                n_H += np.sum(frame == ptype_H)
                n_CN += np.sum(frame == ptype_CN)
                n_CH += np.sum(frame == ptype_CH)
                n_NH += np.sum(frame == ptype_NH) 
            
            n_C  /= n_frames - idx_start
            n_N  /= n_frames - idx_start
            n_H  /= n_frames - idx_start
            n_CN /= n_frames - idx_start
            n_CH /= n_frames - idx_start
            n_NH /= n_frames - idx_start
            
            n_C_all.append(n_C)
            n_N_all.append(n_N)
            n_H_all.append(n_H)
            n_CN_all.append(n_CN)
            n_CH_all.append(n_CH)
            n_NH_all.append(n_NH)
            
            n_N_traj_all.append(n_N_traj)
            
    n_C_all = np.array(n_C_all)
    n_N_all = np.array(n_N_all)
    n_H_all = np.array(n_H_all)
    n_CN_all = np.array(n_CN_all)
    n_CH_all = np.array(n_CH_all)
    n_NH_all = np.array(n_NH_all)
    

    np.save(os.path.join(data_dir_N_H, "n_C_all.npy"), n_C_all)
    np.save(os.path.join(data_dir_N_H, "n_N_all.npy"), n_N_all)
    np.save(os.path.join(data_dir_N_H, "n_H_all.npy"), n_H_all)
    np.save(os.path.join(data_dir_N_H, "n_CN_all.npy"), n_CN_all)
    np.save(os.path.join(data_dir_N_H, "n_CH_all.npy"), n_CH_all)
    np.save(os.path.join(data_dir_N_H, "n_NH_all.npy"), n_NH_all)

# (3.b) CHEMOKINE - HEPARAN SULFATE

try:
    n_C_2_all = np.load(os.path.join(data_dir_C_H, "n_C_all.npy"))
    n_N_2_all = np.load(os.path.join(data_dir_C_H, "n_N_all.npy"))
    n_H_2_all = np.load(os.path.join(data_dir_C_H, "n_H_all.npy"))
    n_CN_2_all = np.load(os.path.join(data_dir_C_H, "n_CN_all.npy"))
    n_CH_2_all = np.load(os.path.join(data_dir_C_H, "n_CH_all.npy"))
    n_NH_2_all = np.load(os.path.join(data_dir_C_H, "n_NH_all.npy"))
    
except:        
    print("DATA DOES NOT EXIST. Calculating fraction bound for N-H system...")
    
    if not os.path.exists(data_dir_C_H):
        os.makedirs(data_dir_C_H)
        
    n_C_2_all = []
    n_N_2_all = []
    n_H_2_all = []
    n_CN_2_all = []
    n_CH_2_all = []
    n_NH_2_all = []

    idx_start_2 = 0

    for i, (dir_sys, name_sys) in enumerate(zip(dirs_sys_2, names_sys_2)):
        cfg = ck.utils.load_config_dict(fnames_config_2[i])
        fname_traj = ck.utils.get_fname(dir_sys, name_sys, "trajectory")
        with h5py.File(fname_traj, "r") as fh5:
            traj = fh5["trajectory"]
            n_frames = traj.shape[0]
            n_C = 0
            n_N = 0
            n_H = 0
            n_CN = 0
            n_CH = 0
            n_NH = 0
            
            for idx_frame in range(idx_start_2, n_frames):
                frame = traj[idx_frame]
                n_C += np.sum(frame == ptype_C)
                n_N += np.sum(frame == ptype_N)
                n_H += np.sum(frame == ptype_H)
                n_CN += np.sum(frame == ptype_CN)
                n_CH += np.sum(frame == ptype_CH)
                n_NH += np.sum(frame == ptype_NH) 
            
            
            n_C  /= n_frames - idx_start_2
            n_N  /= n_frames - idx_start_2
            n_H  /= n_frames - idx_start_2
            n_CN /= n_frames - idx_start_2
            n_CH /= n_frames - idx_start_2
            n_NH /= n_frames - idx_start_2
            
            n_C_2_all.append(n_C)
            n_N_2_all.append(n_N)
            n_H_2_all.append(n_H)
            n_CN_2_all.append(n_CN)
            n_CH_2_all.append(n_CH)
            n_NH_2_all.append(n_NH)
            
            
    n_C_2_all = np.array(n_C_2_all)
    n_N_2_all = np.array(n_N_2_all)
    n_H_2_all = np.array(n_H_2_all)
    n_CN_2_all = np.array(n_CN_2_all)
    n_CH_2_all = np.array(n_CH_2_all)
    n_NH_2_all = np.array(n_NH_2_all)
    
    np.save(os.path.join(data_dir_C_H, "n_C_all.npy"), n_C_all)
    np.save(os.path.join(data_dir_C_H, "n_N_all.npy"), n_N_all)
    np.save(os.path.join(data_dir_C_H, "n_H_all.npy"), n_H_all)
    np.save(os.path.join(data_dir_C_H, "n_CN_all.npy"), n_CN_all)
    np.save(os.path.join(data_dir_C_H, "n_CH_all.npy"), n_CH_all)
    np.save(os.path.join(data_dir_C_H, "n_NH_all.npy"), n_NH_all)


@mpltex.acs_decorator
def plot_fraction_bound():
    
    fig, ax = plt.subplots(1)
    # fig.set_size_inches(3*1.61803399,3*1)
    fig.set_dpi(300)
    
    ax.set_xscale("log")
    
    ls = mpltex.linestyles()
    
    col_ccl5 = "#00982C" # chemokine
    col_net1 = "#F80808" # netrin
    
    n_grid = 150*150 #! INSERT NUMBER OF GRID POINTS HERE
    
    x_chem_2_sim = n_C_2_all/n_grid
    y_chem_2_sim = n_CH_2_all/(n_H_2_all + n_CH_2_all)
    
    scale_x = 3e5 #! INSERT THE SCALE OF KD VALUES HERE
    
    x_chem_2_th = c_chem_2_all
    y_chem_2_th = c_chem_hep_2_all/(c_hep_2_all + c_chem_hep_2_all)
    c = next(ls)['color']    
    ax.plot(scale_x*x_chem_2_sim, y_chem_2_sim, "o", color=col_ccl5, markerfacecolor='none',label="C (sim)")
    ax.plot(scale_x*x_chem_2_th, y_chem_2_th,        color=col_ccl5, label="C (th)")
    
    x_net_sim = n_N_all/n_grid
    y_net_sim = n_NH_all/(n_H_all + n_NH_all)
    
    x_net_th = c_net_all
    y_net_th = c_net_hep_all/(c_hep_all + c_net_hep_all)
    c = next(ls)['color']
    ax.plot(scale_x*x_net_sim, y_net_sim, "o", color=col_net1, markerfacecolor='none',label="N (sim)")
    ax.plot(scale_x*x_net_th, y_net_th,        color=col_net1, label="N (th)")
    
    ax.set_xlim((1e-1, 1e5))
    ax.set_ylim((0,1.1))
    
    leg_handles = []
    leg_labels = []
    
    
    ls = mpltex.linestyles()
    
    c = next(ls)['color']
    leg_handles.append(plt.Line2D([0], [0], linestyle='none', marker='s', color=col_ccl5, markerfacecolor=col_ccl5, label='simulation'))
    leg_labels.append("CCL5 to Hep")
    
    c = next(ls)['color']
    leg_handles.append(plt.Line2D([0], [0], linestyle='none', marker='s', color=col_net1, markerfacecolor=col_net1, label='simulation'))
    leg_labels.append("Net-1 to Hep")
    
    leg_handles.append(plt.Line2D([0],[0], linestyle='none', marker="o", color="black", markerfacecolor="none"))
    leg_labels.append("simulation")
    
    leg_handles.append(plt.Line2D([0],[0], color="black"))
    leg_labels.append("theory")
    
    ax.legend(leg_handles, leg_labels, loc="upper left", bbox_to_anchor=(1.05, 0.95))
    
    ax.set_xlabel("analyte concentration")
    ax.set_ylabel("fraction bound")
    
    plt.show()

plot_fraction_bound()