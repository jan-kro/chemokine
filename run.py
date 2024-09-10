from grid_simulation import simulate
import utils
import sys

name_system = sys.argv[1] 
dir_system = sys.argv[2]

fname_config = utils.get_fname(dir_system, name_system, ftype="config")


config = utils.load_config_dict(fname_config)

simulate(config)

print("\nsimulation finished")
print(f"Name:      {config['name_sys']}")
print(f"Directory: {config['dir_sys']}")

# fname_traj = get_fname(dir_system, name_system, ftype="trajectory")

# traj = h5py.File(fname_traj, 'r')['trajectory']

# animate(traj,
#         show = True,
#         save = False,
#         fps = 15,
#         marker_scale = 1.0,
#         plot_edge = 5)
