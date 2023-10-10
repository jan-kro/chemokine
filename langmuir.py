from monte_carlo_chemokine import monte_carlo_simulation as mcc
from initial_position import get_initial_positions
from plot_grid import plot_grid
import physical_units_diffusion as pud
import numpy as np
import os

outdir_traj = "/local_scratch2/janmak98/chemokine/results/langmuir_ccl5_netrin_homogeneous/trajectories"

if not os.path.exists(outdir_traj):
    os.makedirs(outdir_traj)
    
netrin_concentrations = np.linspace(0, 0.5, 10)

simulation_stride = 1
animation_stride = 30
num_steps = 5000
grid_size = [180, 60]


for kk, netrin_concentration in enumerate(netrin_concentrations):
    
    print(f'Simulation {kk+1}/{len(netrin_concentrations)} started\nnetrin_concentration = {netrin_concentration:.3f} %')
    
    fname_traj = os.path.join(outdir_traj, f'traj_cn{netrin_concentration:.3f}pct.hdf5')

    # CALCULATE PHYSICAL UNITS

    molecule_mass = [
            99999999,       # empty grid site
            15.77,          # kDa (CCL5) https://www.rcsb.org/structure/5COY
            50.63,          # kDa (Netrin-1) https://www.rcsb.org/structure/4OVE
            500,            # kDa (heperansulfate) made up
            15.77 + 50.63,
            500 + 50.63,
            99999999,
            99999999,
            99999999
    ]

    diffusivities = pud.estimate_diffusivity(np.array(molecule_mass))
    #print(diffusivities)



    # diffusion_probability = np.array([0, 3, 2, 0.05, 1, 0.03, 0, 0, 0])
    # diffusion_probability = diffusion_probability/max(diffusion_probability)/5

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

    ratio = pud.calc_time_step_ratio(np.log(binding_probability), diffusivities)
    diffusion_probability, p_stay = pud.calc_probabilities_free(ratio, diffusivities)


    for i in range(9):
            for j in range(9):
                    if np.any(diffusivities/binding_probability[i, j] > 0.25) and binding_probability[i, j] != 0:
                            print('too big')
                            print(diffusivities/binding_probability[i, j])
                            L = diffusivities/binding_probability[i, j] > 0.25
                            print(diffusivities[L])
                            print(binding_probability[i, j])


    # set up initial configuration
    initial_positions = get_initial_positions(grid_size = grid_size,
                                              concentrations = [0.5,   # empty grid site
                                                                0.07,  # chemokine
                                                                netrin_concentration,  # netrin
                                                                0.15,  # heparansulfate
                                                                0.0,   # chemokine-netrin
                                                                0.0,   # chemokine-heparansulfate
                                                                0.5,   # collagen site
                                                                0.0,   # chemokine-netrin-collagen
                                                                0.0],  # netrin-collagen
                                                                              #   x       y
                                              regions_fraction = [[[0, 1],   [0, 1]],   # empty grid site
                                                                  [[0, 1],   [0, 1]],   # chemokine
                                                                  [[0, 1],   [0, 1]],   # netrin
                                                                  [[0, 1],   [0, 1]],   # heparansulfate
                                                                  [[0, 1],   [0, 1]],   # chemokine-netrin
                                                                  [[0, 1],   [0, 1]],   # chemokine-heparansulfate
                                                                  [[0, 1],   [0.49, 0.51]],   # collagen site
                                                                  [[0, 1],   [0, 1]],   # chemokine-netrin-collagen
                                                                  [[0, 1],   [0, 1]]]   # netrin-collagen
                                              )
    
    mcc(num_steps = num_steps, 
        initial_positions = initial_positions,
        diffusion_probability = diffusion_probability, 
        binding_probability = binding_probability,
        reflection_x = True,
        stride=simulation_stride,
        fname_traj=fname_traj)
                            

# plot_grid(initial_positions,
#           show = False,
#           save = True,
#           fname = fname_figures)