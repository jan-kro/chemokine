import matplotlib.pyplot as plt
import chemokine as ck 
import numpy as np
import mpltex
import h5py
import os

from scipy.ndimage import gaussian_filter1d

#* To create the extended data figure, we average over 11 simulations 
#* with the exact same parameters to get better statistics.

# path where the calculated data of this skript will be stored
#! This will be around 10 GB of data
data_path_CNCI_all = "/path/to/analysis_data/occupation_distribution_x_CNCI_all_systems.npy"

dir_fig = "/path/to/figure_directory"

# list config file names
dir_sys = "/path/to/simulation_directory"
fnames_config, dirs_sys, names_sys = ck.analysis_tools.list_fnames(dir_sys)

# EVALUATE SIMULATION DATA
# ------------------------

ptype_CNCI = 9 # type of Chemokine-Netrin-Collagen-I complex

c_chem_all = [0.0025]
c_net_all = [0.01] * len(fnames_config) # assuming all simulations have the same netrin concentration
c_hep_all = [0.035]
c_coll_all = [0.005]

# loop over all systems and bit the amount of CNCI complexes over the x-axis for every frame 

dist_CNCI_all = []
for k in range(len(c_net_all)):
    # get trajectory filename
    fname_cfg = fnames_config[k]
    cfg = ck.utils.load_config_dict(fname_cfg)
    fname_traj = ck.utils.get_fname(cfg["dir_sys"], cfg["name_sys"], "trajectory")
    
    # load trajectory 
    with h5py.File(fname_traj, "r") as f:
        traj = f["trajectory"]
        n_frames = traj.shape[0]
        n_x = traj.shape[1] # number of bins in x
        n_y = traj.shape[2]
        print(f"Calculating occupation distribution for sys {k} ...")
        
        # loop over all frames
        dist_CNCI = np.zeros((n_frames, n_x)) # initialize distribution array
        for tt in range(n_frames):
            if tt % int(n_frames/10) == 0:
                print(f"frame {tt}/{n_frames}")
            dist_CNCI[tt] = np.sum(traj[tt] == ptype_CNCI, axis=1)
            
        dist_CNCI_all.append(dist_CNCI)
dist_CNCI_all = np.array(dist_CNCI_all)

# save distributions of all systems #! (around 10 GB of data)
np.save(data_path_CNCI_all, dist_CNCI_all)

# get the inital distribution of free and/or bound collagen-I sites
ptype_CI = 4 # type of collagen-I
dist_CI_0_all = []
for k in range(len(c_net_all)):
    fname_cfg = fnames_config[k]
    cfg = ck.utils.load_config_dict(fname_cfg)
    fname_traj = ck.utils.get_fname(cfg["dir_sys"], cfg["name_sys"], "trajectory")
    with h5py.File(fname_traj, "r") as f:
        traj = f["trajectory"]
        dist_CI = np.sum(traj[0] == ptype_CI, axis=1)            
        dist_CI_0_all.append(dist_CI)
dist_CI_0_all = np.array(dist_CI_0_all)
dist_CI_0 = np.mean(dist_CI_0_all, axis=0)

# PLOT EVALUATED DATA
# -------------------

# average the distributions over all systems
dist_CNCI = np.sum(dist_CNCI_all, axis=0)/dist_CNCI_all.shape[0]


# define plotting function
@mpltex.acs_decorator
def plot_collagen_occupation_dist():
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
    
    fig.set_dpi(300)
    
    #! THIS NEED TO BE ADJUSTED TO THE ACTUAL SIMULATION PARAMETERS
    simulation_stride = 50
    dt = simulation_stride*6.519 # ns
    a =  0.017689 # um
    
    print(f"total time: {dt*n_frames*1e-6:.2f} ms" )
    
    # 
    stride = 100000 # number of frames that should pass between two plotted distributions
    window = 2 # number of frames a curve should be averaged over
    smoothing_window = 20 # smoothing window for the gaussian filter in units of the grid constant (i.e. the width of the gaussian)
    print(f"smoothing window: {smoothing_window*a:.2f} um")
    n_curves = n_frames//stride
    
    averaging_time = window*dt*1e-6 # ms
    
    # filtered distributions of free and/or bound collagen-I sites
    dist_CI_filtered = gaussian_filter1d(dist_CI_0, sigma=smoothing_window)
    
    # plot unfiltered data
    ls = mpltex.linestyles()    
    for tt in range(0,n_curves*stride, stride):
        dist_s = dist_CNCI[0]
        for tt_s in range(tt+1, tt+window):
                dist_s += dist_CNCI[tt_s]
        dist_s /= window
    
        c = next(ls)["color"]
        
        # the total Collagen-I distibution is in some sense the local density of 
        # collagen sites. To unbias the data, we divide the CNCI distribution by the
        # initial distribution of collagen sites.
        ax.plot(np.arange(len(dist_s))*a, dist_s/dist_CI_filtered, color=c, alpha=0.33)
    
    # filter data and plot
    ls = mpltex.linestyles()
    for tt in range(0,n_curves*stride, stride):
        dist_s = dist_CNCI[0]
        for tt_s in range(tt+1, tt+window):
                dist_s += dist_CNCI[tt_s]
        dist_s /= window
        dist_filtered = gaussian_filter1d(dist_s, sigma=smoothing_window)
        
        c = next(ls)["color"]
        ax.plot(np.arange(len(dist_filtered))*a, dist_filtered/dist_CI_filtered, color=c, label=fr"$t={tt*dt*1e-6+averaging_time/2:.1f}~\mathrm{{ms}}$")

    
    ax.set_xlabel(r"$x / \mathrm{\mu m}$") 
    ax.set_ylabel(r"$c_{\mathrm{CNCI}} / c_{\mathrm{CI}}^0$")
    
    ax.legend()
    
    # safe figure
    plt.savefig(f"{dir_fig}/collagen_occupation_distribution_gaussian_filter.pdf", bbox_inches='tight')
    
    plt.show()
    
plot_collagen_occupation_dist()