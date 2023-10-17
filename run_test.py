from grid_simulation_test import simulate
from utils import animate, get_fname
import toml
import h5py

name_system = "test_setup_no_netrin"     
dir_system = "/local_scratch2/janmak98/chemokine/results/tests"

fname_config = get_fname(dir_system, name_system, ftype="config")

with open(fname_config, "r") as f:
    config = toml.load(f)

simulate(config)

fname_traj = get_fname(dir_system, name_system, ftype="trajectory")

traj = h5py.File(fname_traj, 'r')['trajectory']

animate(traj,
        show = True,
        save = False,
        fps = 15,
        marker_scale = 1.0,
        plot_edge = 5)
