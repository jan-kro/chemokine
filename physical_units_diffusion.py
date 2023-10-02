import numpy as np


# the indexing of the concentrations, diffusivities and energies has to be always in the same order, i.e. concentrations=[c1, c2, ...] then diffusivitiy_array=[D1, D2, ...]
# function that takesin the matrix containing all binding energies in kbT and diffusivities in [m^2/s] and outputs the ratio of time step to grid constant squared Delta/a^2
def calc_time_step_ratio(energy_matrix, diffusivity_array):
    # compute ratio of time step to grid constant squared Delta/a^2 by setting the maximum movement probability to be 1/4
    if len(diffusivity_array) != len(energy_matrix):
        print("Error: length of diffusivity array must be the same as the number of rows in the energy matrix")
        return
    max_dexp = np.max(np.exp(energy_matrix) * diffusivity_array) # since energy matrix is symmetric, it is sufficient to multiply the diffusivity array, such that D1 exp(E12) and D2 exp(E12) = D2 exp(E21) are both considered
    ratio = 1 / (5 * max_dexp)
    return ratio

# function that takes the ratio of time step to grid constant squared Delta/a^2 in [s/m^2] and a diffusivity in [m^2/s] and returns the probability of a particle to move in one direction and the probability to stay in the same position
def calc_probabilities(ratio, diffusivity_array):
    p_move = diffusivity_array * ratio * 1.25
    p_stay = 1 - 4 * p_move
    return p_move, p_stay

# function taking concentrations of particle species in [mol/l], wanted number of rarest species and ratio Delta/a^2 in [s/m^2] and outputs the grid constant a in [m], the time step in [s] and the number of all species
def calc_nspecies_gridconst(ratio, concentrations, n_min, grid_dim):
    a_grid = (n_min / ((grid_dim[0] - 1) * (grid_dim[1] - 1) * np.min(concentrations) * 6.022e26)) ** (1/3)
    n_species = int(n_min * concentrations / np.min(concentrations))
    time_step = ratio * a_grid**2
    return n_species, a_grid, time_step


# estimate diffusivity from mass of molecule in [kDa] in case no experimental diffusivity in water is given
# input can be a numpy array or float
def estimate_diffusivity(mass):
    # approximate hydrodynamic radius of protein by average protein density of 1.38 g/cm^3 = 1.38 * 1e3 kg/m^3
    # for 10kDa protein this gives a radius of 1.42 nm and at 300K a diffusion coefficient of 1.5e-10 m^2/s
    mass_kg = mass * 1.66054e-24 #for given mas in [kDa] calculate mass in [kg]
    r = (3 * mass_kg / (1.38 * 1e3 * 4 * np.pi)) ** (1/3) # calculate radius in [m]
    # use stokes law D=kBT/gamma with gamma=6*pi*eta*r and water viscosity ~ 1e-3 Pa*s to calculate diffusivity in [m^2/s]
    D = 1.38e-23 * 300 / (6 * np.pi * 1e-3 * r) 
    return D

