from monte_carlo_chemokine import monte_carlo_simulation as mcc
from animation import animate
from plot_grid import plot_grid
import numpy as np
import physical_units_diffusion as pud
import os 

num_steps = 100
grid_size = [300, 100]

# CALCULATE PHYSICAL UNITS

molecule_mass = [
        15.77, # kDa (CCL5) https://www.rcsb.org/structure/5COY
        50.63, # kDa (Netrin-1) https://www.rcsb.org/structure/4OVE
        500, # kDa (heperansulfate) made up
        15.77 + 50.63,
        500 + 50.63,
        99999999,
        99999999,
        99999999
]

diffusivities = []

for m in molecule_mass:
    diffusivities.append(pud.estimate_diffusivity(m))
    


diffusion_probability = np.array([0, 3, 2, 0.05, 1, 0.03, 0, 0, 0])
diffusion_probability = diffusion_probability/max(diffusion_probability)/5

binding_probability = np.zeros((9, 9))
binding_probability[1, 2] = 6 # chemokine binds to netrin
binding_probability[2, 1] = 6
binding_probability[2, 6] = 9 # netrin binds to collagen site 
binding_probability[6, 2] = 9 # netrin binds to collagen site 
binding_probability[1, 3] = 3 # chemokine binds to heparansulfate
binding_probability[3, 1] = 3
binding_probability[4, 6] = 9 # chemokine-netrin binds to collagen_site
binding_probability[6, 4] = 9 # chemokine-netrin binds to collagen_site
binding_probability[:, 0] = 1 # all particles bind to empty grid site
binding_probability[0, :] = 1 # all particles bind to empty grid site

binding_probability = binding_probability/np.max(binding_probability*diffusion_probability)/4
print(binding_probability[binding_probability != 0])
print(np.sum(binding_probability))

p =[3, 0.1, 0.08, 0.3, 0.01, 0.01]
initial_positions = np.random.choice([0, 1, 2, 3, 4, 5], size=(grid_size[0], grid_size[1]), p=p/np.sum(p))
initial_positions[int(grid_size[0]/2), ::3] = 6 # collagen fiber

traj = mcc(num_steps = num_steps,
           initial_positions = initial_positions,
           diffusion_probability = diffusion_probability, 
           binding_probability = binding_probability,
           reflection_x = True)

animate(traj,
        show = False,
        save = True,
        fname = None,
        fps = 15,
        marker_scale = 1.0,
        plot_edge = 5)
