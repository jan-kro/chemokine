import numpy as np


# the indexing of the concentrations, diffusivities and energies has to be always in the same order, i.e. concentrations=[c1, c2, ...] then diffusivitiy_array=[D1, D2, ...]
# function that takesin the matrix containing all binding energies in kbT and diffusivities in [m^2/s] and outputs the ratio of time step to grid constant squared Delta/a^2
def calc_time_step_ratio(energy_matrix, diffusivity_array):
    # compute ratio of time step to grid constant squared Delta/a^2 by setting the maximum movement probability to be 1/4
    if len(diffusivity_array) != len(energy_matrix):
        print("Error: length of diffusivity array must be the same as the number of rows in the energy matrix")
        return
    max_dexp2 = 0 # initialize maximum product of diffusivity with exp(energy)
    for i in range(len(energy_matrix)):
        for j in range(i + 1, len(energy_matrix)):
            if np.exp(energy_matrix[i, j]) * diffusivity_array[j] > max_dexp2:
                max_dexp2 = np.exp(energy_matrix[i, j]) * diffusivity_array[j] # maximum of occuring product

    max_dexp = np.max(diffusivity_array * np.exp(energy_matrix))
    max_tot = np.max([max_dexp, max_dexp2]) # if all binding energies are negative, moving probability is higher for pure diffusion
    ratio = 1 / (5 * max_tot)
    return ratio


# function that takes  thye ratio of time step to grid constant squared Delta/a^2 in [s/m^2] and a diffusivity in [m^2/s] and returns the probability of a particle to move in one direction and the probability to stay in the same position
def calc_probabilities_free(ratio, diffusivity_array):
    p_move = diffusivity_array * ratio * 1.25
    p_stay = 1 - 4 * p_move
    return p_move, p_stay


# function computes moving probability in one direction with a neighbor present in this direction
def calc_probabilities_with_neighbor(energy_matrix, diffusivity_array):
    ratio = calc_time_step_ratio(energy_matrix, diffusivity_array)
    p_move_free = calc_probabilities_free(ratio, diffusivity_array)[0]
    p_move_neighbor = np.zeros(int(len(energy_matrix) * (len(energy_matrix) - 1) / 2))
    # one could make a matrix in the shape of energy_matrix, but i+j-1 is also unique
    for i in range(len(energy_matrix)):
        for j in range(i + 1, len(energy_matrix)):
            p_move_neighbor[i + j - 1] = p_move_free[i] * np.exp(energy_matrix[i, j])
            # print(i + j - 1)
    return p_move_neighbor


# function taking concentrations of particle species in [mol/l], wanted number of rarest species and ratio Delta/a^2 in [s/m^2] and outputs the grid constant a in [m], the time step in [s] and the number of all species
def calc_nspecies_gridconst(ratio, concentrations, n_min, grid_dim):
    a_grid = (n_min / ((grid_dim[0] - 1) * (grid_dim[1] - 1) * np.min(concentrations) * 6.022e26)) ** (1/3)
    n_species = np.int64(n_min * concentrations / np.min(concentrations))
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


# return concentration in [mol/l] for given number of n_species particles in given grid
# assumes volume of (n-1)a x (m-1)a x a in simulation slab
def physical_concentration(n_species, grid_dim, a_grid):
    concentration = n_species / ((grid_dim[0] - 1) * (grid_dim[1] - 1) * a_grid**3 * 6.022e26)
    return concentration