# from monte_carlo_chemokine import monte_carlo_simulation as mcc
from grid_simulation import simulate as mcc
from initial_position import get_initial_positions
from plot_grid import plot_grid
import physical_units_diffusion as pud
import numpy as np
import os
import h5py

import matplotlib.pyplot as plt

"""
LANGMUIR ISOTHERM SIMULATION

This script simulates the system for different concentrations of netrin-1, while kkeping all other concentrations constant.
The particles are distributed uniformly in the grid, and the simulation is run for 10000 steps.
"""

outdir_traj = "/local_scratch2/janmak98/chemokine/results/langmuir_complete_new/trajectories"
indir_traj = "/local_scratch2/janmak98/chemokine/results/langmuir_ccl5_netrin_only_logscale/trajectories"

fnames_indir = []
for fname_indir in os.listdir(indir_traj):
    fnames_indir.append(os.path.join(indir_traj, fname_indir))

pct = np.zeros_like(fnames_indir, dtype=float)
for idx, fname_indir in enumerate(fnames_indir):
    pct[idx] = float(fname_indir.split('_cn')[-1].split('pct')[0])

idx_sort = np.argsort(pct)

fnames_indir = np.array(fnames_indir)[idx_sort]
pct = np.array(pct)[idx_sort]

for fname_indir in fnames_indir:
    print(h5py.File(fname_indir, 'r')['trajectory'][-1].shape)

if not os.path.exists(outdir_traj):
    os.makedirs(outdir_traj)
    
netrin_concentrations = np.logspace(-3, np.log10(0.15), 20)
#netrin_concentrations = np.array([0, 0.05, 0.1, 0.2])


simulation_stride = 10
animation_stride = 30
num_steps = 100000
grid_size = [180, 60]

# pNC = 69.07 # bonding probability Chemokine-Netrin
chemokine_concentration = 0.1

# def eq_complex_conc(cN, cC, pNC):
#     return 0.5 * (np.sqrt(cC) * np.sqrt(4*cN*pNC + cC) - cC)

# def eq_complex_conc_2(cN, cC, pNC):
#     return np.sqrt(cC**2/4+cC*cN*pNC)-cC/2


# plt.figure()

# for kk, netrin_concentration in enumerate(netrin_concentrations):
    
#     cn_complex_concentration = eq_complex_conc(netrin_concentration, chemokine_concentration, pNC)
#     cn_complex_concentration_2 = eq_complex_conc_2(netrin_concentration, chemokine_concentration, pNC)

#     #print(f"{netrin_concentration:6.3f} {cn_complex_concentration:6.3f} {cn_complex_concentration/(cn_complex_concentration+chemokine_concentration):6.3f}")
#     print(f"{netrin_concentration:6.3f} {cn_complex_concentration_2:6.3f} {cn_complex_concentration_2/(cn_complex_concentration_2+chemokine_concentration):6.3f}")
    
#     #plt.plot(netrin_concentration, cn_complex_concentration/(cn_complex_concentration+chemokine_concentration), 'o')
#     plt.plot(netrin_concentration, cn_complex_concentration_2/(cn_complex_concentration_2+chemokine_concentration), 'x')
# plt.xscale('log')
# plt.legend()
# plt.show()
#raise SystemExit  

cn_complex_concentration = 0

for kk, netrin_concentration in enumerate(netrin_concentrations):
    
    # cn_complex_concentration = eq_complex_conc(netrin_concentration, chemokine_concentration, pNC)
    
    concentrations = [
        0,                        # empty grid site
        chemokine_concentration,  # 0.07,  # chemokine
        netrin_concentration,     # netrin
        0.0,                     # 0.15 # heparansulfate
        cn_complex_concentration, # chemokine-netrin
        0.0,                      # chemokine-heparansulfate
        0.0,                      # 0.2 # collagen site
        0.0,                      # chemokine-netrin-collagen
        0.0                       # netrin-collagen
    ]

    
    regions_fraction = [
        [[0,1],[0,1]],       # empty grid site
        [[0,1],[0,1]],       # chemokine
        [[0,1],[0,1]],       # netrin
        [[0,1],[0,1]],       # heparansulfate
        [[0,1],[0,1]],       # chemokine-netrin
        [[0,1],[0,1]],       # chemokine-heparansulfate
        [[0,1],[0.48,0.52]], # collagen site
        [[0,1],[0,1]],       # chemokine-netrin-collagen
        [[0,1],[0,1]]        # netrin-collagen
    ]
    

    
    print(f'Simulation {kk+1}/{len(netrin_concentrations)} started\nnetrin_concentration = {netrin_concentration:.3f} %')
    
    fname_traj = os.path.join(outdir_traj, f'traj_cn{netrin_concentration:.3f}pct.hdf5')

    # CALCULATE PHYSICAL UNITS

    molecule_mass = [
        1e15,        # empty grid site
        26.725,          # kDa (CCL5) https://www.rcsb.org/structure/5COY
        52.437,          # kDa (Netrin-1) https://www.rcsb.org/structure/4OVE
        500,             # kDa (heperansulfate) made up
        26.725 + 52.437,
        500 + 52.437,
        1e15,
        1e15,
        1e15
    ]

    diffusivities = pud.estimate_diffusivity(np.array(molecule_mass))

    binding_probability = np.zeros((9, 9))
    binding_probability = np.ones((9, 9))
    binding_probability[1, 2] = 69.07 # 6 # chemokine binds to netrin
    binding_probability[2, 1] = 69.07
    binding_probability[2, 6] = 13.59 # 9 # netrin binds to collagen site 
    binding_probability[6, 2] = 13.59 # 9 # netrin binds to collagen site 
    binding_probability[1, 3] = 696 # 3 # chemokine binds to heparansulfate
    binding_probability[3, 1] = 696
    binding_probability[4, 6] = 9.706 # 9 # chemokine-netrin binds to collagen_site
    binding_probability[6, 4] = 9.706 # 9 # chemokine-netrin binds to collagen_site
    
    binding_probability *= 1e-9 # nanomolar to molar
    binding_probability = 1/binding_probability
    
    binding_probability[:, 0] = 1 # all particles bind to empty grid site
    binding_probability[0, :] = 1 # all particles bind to empty grid site
    
    ratio = pud.calc_time_step_ratio(np.log(binding_probability), diffusivities)
    diffusion_probability, p_stay = pud.calc_probabilities_free(ratio, diffusivities)

    diffusion_probability = diffusion_probability/np.max(diffusion_probability)/4

    # check if probabilities fit
    # for i in range(9):
    #     for j in range(9):                
    #         for dp in diffusion_probability:
    #             if not binding_probability[i, j] == 0:
    #                 if dp/binding_probability[i, j] > 0.25:
    #                     print('too big')
    #                     print(dp/binding_probability[i, j])
    #                     print(dp)
    #                     print(binding_probability[i, j])
    #                 if dp*binding_probability[i, j] > 0.25:
    #                     print('too big')
    #                     print(dp*binding_probability[i, j])
    #                     print(dp)
    #                     print(binding_probability[i, j])
                        
    
    # set up initial configuration
    initial_positions = get_initial_positions(
        grid_size = grid_size,
        concentrations = concentrations,
        regions_fraction = regions_fraction
    )
    
    # initial_positions = h5py.File(fnames_indir[kk], 'r')['trajectory'][9000]
    # fname_traj = os.path.join(outdir_traj, f'traj_cn{pct[kk]:.3f}pct.hdf5')
    
    mcc(
        num_steps = num_steps, 
        initial_positions = initial_positions,
        move_probability = diffusion_probability, 
        binding_probability = binding_probability,
        reflection_x = True,
        stride=simulation_stride,
        fname_traj=fname_traj
    )
                            

# plot_grid(initial_positions,
#           show = False,
#           save = True,
#           fname = fname_figures)