from monte_carlo_chemokine import monte_carlo_simulation as mcc
from animation import animate
import numpy as np
import os

grid_size = [300, 80]
n_steps = 15*10
n_particles = 300 # 0.7
particle_types = np.random.choice([0, 1, 2], size=(n_particles), p=[0.66, 0.34, 0])
particle_diffusivity = [3, 2, 1, 0.05, 0.03]
c_heperansulfate = 0.03


traj_chemokine, traj_netrin, traj_heparansulfate, traj_chemokine_netrin,  traj_chemokine_heparansulfate = mcc(num_steps = n_steps,
                                                                                                              grid_size = grid_size,
                                                                                                              particle_diffusivity=particle_diffusivity,
                                                                                                              n_particles = n_particles,
                                                                                                              fraction_x=0.3,
                                                                                                              c_heparansulfate=c_heperansulfate,
                                                                                                              reflection=True,
                                                                                                              particle_types=particle_types)

animate(traj_chemokine,
        traj_netrin,
        traj_heparansulfate,
        traj_chemokine_netrin,
        traj_chemokine_heparansulfate,
        grid_size,
        show = False,
        save = True,
        fname = None,
        fps = 15,
        collagen_y = None,
        marker_scale = 1.0,
        plot_edge = 5)
