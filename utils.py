from matplotlib.animation import FuncAnimation
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import toml
import os


def calc_time_step_ratio(energy_matrix: np.ndarray, diffusivity_array: np.ndarray, only_diff: bool=False):
    """
    Computes the ratio of time step to grid constant squared Delta/a^2 by setting 
    the maximum movement probability to be 1/4.
    
    Parameters
    ----------
    energy_matrix : np.ndarray
        2d array containing all binding energies in kbT for each particle pair.
        The order in the energy matrix and the diffusivity array must be the same.
    diffusivity_array : np.ndarray
        1d array containing the diffusivity of every particle in [m^2/s].
    only_diff : bool, optional
        If True, only the diffusivity is considered for calculating the step probability.
    
    Returns
    -------
    ratio : float
        Ratio of time step to grid constant squared Delta/a^2 in [s/m^2].
    """
    
    assert len(diffusivity_array) == energy_matrix.shape[0], "length of diffusivity array must be the same as the number of rows and colums in the energy matrix"
    assert len(diffusivity_array) == energy_matrix.shape[1], "length of diffusivity array must be the same as the number of rows and colums in the energy matrix"
    
    if only_diff:
        ratio = 1 / (5 * np.max(diffusivity_array))
    else:
        max_dexp2 = 0 # initialize maximum product of diffusivity with exp(energy)
        for i in range(len(energy_matrix)):
            for j in range(i + 1, len(energy_matrix)):
                if np.exp(energy_matrix[i, j]) * diffusivity_array[j] > max_dexp2:
                    max_dexp2 = np.exp(energy_matrix[i, j]) * diffusivity_array[j] # maximum of occuring product

        max_dexp = np.max(diffusivity_array * np.exp(energy_matrix))
        max_tot = np.max([max_dexp, max_dexp2]) # if all binding energies are negative, moving probability is higher for pure diffusion
        ratio = 1 / (5 * max_tot)
    return float(ratio)


def calc_probabilities_free(ratio: float, diffusivity_array: np.ndarray):
    """
    Takes the ratio of time step to grid constant squared Delta/a^2 in [s/m^2] and 
    a diffusivity in [m^2/s] to calculate the probability of a particle to move in one 
    direction and the probability to stay in the same position.
    
    Parameters
    ----------
    ratio : float
        Ratio of time step to grid constant squared Delta/a^2 in [s/m^2].
    diffusivity_array : np.ndarray
        Diffusivity of every particle in [m^2/s].
        
    Returns
    -------
    p_move : np.ndarray
        Probability of every particle to move in one direction.
    p_stay : np.ndarray
        Probability of every particle to stay in the same position.
    """
    
    p_move = diffusivity_array * ratio * 1.25
    p_stay = 1 - 4 * p_move
    return p_move, p_stay

def calc_probabilities_with_neighbor(energy_matrix: np.ndarray, diffusivity_array: np.ndarray, only_diff: bool=False):
    """
    Calculates the moving probability in one direction with a neighbor present in this direction.
    
    Parameters
    ----------
    energy_matrix : np.ndarray
        2d array containing all binding energies in kbT for each particle pair.
        The order in the energy matrix and the diffusivity array must be the same.
    diffusivity_array : np.ndarray
        1d array containing the diffusivity of every particle in [m^2/s].
    only_diff : bool, optional
        If True, only the diffusivity is considered for calculating the step probability.
        
    Returns
    -------
    p_move_neighbor : np.ndarray
        Moving probability in one direction with a neighbor present in this direction.
    """
    
    ratio = calc_time_step_ratio(energy_matrix, diffusivity_array, only_diff=only_diff)
    p_move_free = calc_probabilities_free(ratio, diffusivity_array)[0]
    p_move_neighbor = np.zeros(int(len(energy_matrix) * (len(energy_matrix) - 1) / 2))
    # one could make a matrix in the shape of energy_matrix, but i+j-1 is also unique
    for i in range(len(energy_matrix)):
        for j in range(i + 1, len(energy_matrix)):
            p_move_neighbor[i + j - 1] = p_move_free[i] * np.exp(energy_matrix[i, j])
            # print(i + j - 1)
    return p_move_neighbor


def calc_nspecies_gridconst(ratio: float, concentrations: np.ndarray, n_min : int, grid_dim: tuple):
    """
    Takes the concentrations of particle species in [mol/l] and a desired number of 
    particles for the rarest particle type to calcualte the number of particles for
    every particle type, the grid constant and the time step.
    
    Parameters
    ----------
    ratio : float
        Ratio of time step to grid constant squared Delta/a^2 in [s/m^2].
    concentrations : np.ndarray
        Concentration of every particle type in [mol/l].
    n_min : int
        Desired number of particles for the rarest particle type.
    grid_dim : tuple
        Tuple of integers containing the dimensions of the grid in an ndarray.shape like fashion.
        
    Returns
    -------
    n_species : np.ndarray
        Number of particles for every particle type.
    a_grid : float
        Grid constant in [m].
    time_step : float
        Time step in [s].
    """
    a_grid = (n_min / ((grid_dim[0] - 1) * (grid_dim[1] - 1) * np.min(concentrations) * 6.022e26)) ** (1/3)
    n_species = np.int64(n_min * concentrations / np.min(concentrations))
    time_step = ratio * a_grid**2
    return n_species, a_grid, time_step


def estimate_diffusivity(mass):
    """
    Calculate the diffusivity of a particle by approximating the hydrodynamic radius of a protein by average protein 
    density of 1.38 g/cm^3 = 1.38 * 1e3 kg/m^3 for 10kDa protein. This would give a radius of 1.42 nm and at 300K a 
    diffusion coefficient of 1.5e-10 m^2/s for a 10kDa protein.
    
    Parameters
    ----------
    mass : float or np.ndarray
        Mass of the particle in [kDa].
    
    Returns
    -------
    D : float or np.ndarray
        Diffusivity of the particle in [m^2/s]. 
    """
    mass_kg = mass * 1.66054e-24 #for given mas in [kDa] calculate mass in [kg]
    r = (3 * mass_kg / (1.38 * 1e3 * 4 * np.pi)) ** (1/3) # calculate radius in [m]
    # use stokes law D=kBT/gamma with gamma=6*pi*eta*r and water viscosity ~ 1e-3 Pa*s to calculate diffusivity in [m^2/s]
    D = 1.38e-23 * 300 / (6 * np.pi * 1e-3 * r) 
    return D


def physical_concentration(n_species: np.ndarray, grid_dim: tuple, a_grid: float):
    """
    Calculate physical concentration in [mol/l] for given number of n_species particles in given grid.
    
    Parameters
    ----------
    n_species : np.ndarray
        Number of particles for every particle type.
    grid_dim : tuple
        Tuple of integers containing the dimensions of the grid in an ndarray.shape like fashion.
    a_grid : float
        Grid constant in [m].
        
    Returns
    -------
    concentration : np.ndarray
        Physical concentration in [mol/l] for every particle type.
    """
    concentration = n_species / ((grid_dim[0] - 1) * (grid_dim[1] - 1) * a_grid**3 * 6.022e26)
    return concentration


def animate(trajectory:                    np.ndarray,
            particle_names:                list        = None,
            show:                          bool        = True,
            save:                          bool        = False,
            fname:                         str         = None,
            fps:                           int         = 15,
            marker_scale:                  float       = 1.0,
            plot_edge:                     int         = 5):
    """
    Animates the 2D trajectories of the particles. Saves the animation as mp4 file if specified.
    
    Parameters
    ----------
    traj : np.ndarray
        Trajectory of the particles of shape (n_steps, L_x, L_y).
    show : bool, optional 
        Show the animation, by default True
    save : bool, optional
        Save the animation, by default False
    fname : str, optional
        Filename of the animation, by default file gets saved in ./figures with timestamp
        Filename must end with .mp4
    fps : int, optional
        Frames per second, by default 15
    marker_scale : float, optional
        scales the markers in case they are to big or to small
    plot_edge : int, optional
        edge around the grid, that is included in the plot, by default 5 grid steps
    """
    
    if particle_names==None:
        particle_names =[
            "empty site",
            "CCL5",
            "Net-1",
            "Heperin",
            "CCL5-Net-1",
            "CCL5-Heperin",
            "Collagen Site",
            "CCL5-Net-1-Collagen",
            "Net-1-Collagen"
        ]
    
    n_steps = trajectory.shape[0]
    grid_size = trajectory[0].shape

    fig, ax = plt.subplots()
    fig.set_figwidth(grid_size[0]/max(grid_size)*20)
    fig.set_figheight(grid_size[1]/max(grid_size)*20)

    box = ax.get_position()

    def update(i):
        
        #clear the frame
        ax.clear()
        
        ax.scatter(*np.where(trajectory[i] == 1), label=particle_names[1], color="blue",                        s=80*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 2), label=particle_names[2], color="red",    marker=r"--$\cdot$", s=250*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 3), label=particle_names[3], color="grey",   marker=r"$\sim$",    s=150*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 4), label=particle_names[4], color="green",                       s=90*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 5), label=particle_names[5], color="purple", marker="+",          s=80*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 6), label=particle_names[6], color="k",      marker="x",          s=90*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 7), label=particle_names[7], color="k",      marker="o",          s=90*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 8), label=particle_names[8], color="k",      marker="*",          s=90*marker_scale)
        

        ax.set_xlim(-plot_edge, grid_size[0]+plot_edge)    
        ax.set_ylim(-plot_edge, grid_size[1]+plot_edge)
        
        ax.plot([0, grid_size[0]], [0, 0],                       "--", color="black")
        ax.plot([grid_size[0], 0], [grid_size[1], grid_size[1]], "--", color="black")
        ax.plot([grid_size[0],  grid_size[0]], [0, grid_size[1]],      color="black")
        ax.plot([0, 0], [grid_size[1], 0],                             color="black")
        
        
        # draw legend outside of the plot
        
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #draw frame
        plt.draw()
        #pause to show the frame
        #plt.pause(0.05)
        
    anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)

    if show:
        plt.show()

    if save:
        if fname is None:
            now = datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
            fname = os.path.join(os.getcwd(), 'movies', 'sim_'+now+'.mp4')
        print(f'saving animation (fps = {fps:d}) ...')
        anim.save(fname, fps=fps)
        print(f'animation saved at {fname}.')
        
def plot_grid(positions,
              particle_names: list = None,
              show: bool = True,
              save: bool = True,
              fname: str = None):
    """
    Plots the positions of the particles in the grid. Saves plot as png file if specified.
    
    Parameters
    ----------
    positions : np.ndarray
        Positions of the particles of shape (L_x, L_y).
    show : bool, optional
        Show the plot, by default True
    save : bool, optional
        Save the plot, by default True
    fname : str, optional
        Filename of the plot, by default file gets saved in ./figures with timestamp
        Filename must end with .png
    """
    
    # todo
    # import background image of collagen fiber
    # img = plt.imread("collagen_fiber.png")
    
    if particle_names==None:
        particle_names =[
            "empty site",
            "CCL5",
            "Net-1",
            "Heperin",
            "CCL5-Net-1",
            "CCL5-Heperin",
            "Collagen Site",
            "CCL5-Net-1-Collagen",
            "Net-1-Collagen"
        ]
    
    grid_size = positions.shape
        
    fig, ax = plt.subplots()
    fig.set_figwidth(grid_size[0]/max(grid_size)*20)
    fig.set_figheight(grid_size[1]/max(grid_size)*20)

    box = ax.get_position()

    marker_scale = 1
 
    plot_edge = 4

    ax.clear()

    #ax.imshow(img, extent=[0, grid_size[0], 0, grid_size[1])

    ax.scatter(*np.where(positions == 1), label=particle_names[1], color="blue",                        s=80*marker_scale)
    ax.scatter(*np.where(positions == 2), label=particle_names[2], color="red",    marker=r"--$\cdot$", s=250*marker_scale)
    ax.scatter(*np.where(positions == 3), label=particle_names[3], color="grey",   marker=r"$\sim$",    s=150*marker_scale)
    ax.scatter(*np.where(positions == 4), label=particle_names[4], color="green",                       s=90*marker_scale)
    ax.scatter(*np.where(positions == 5), label=particle_names[5], color="purple", marker="+",          s=80*marker_scale)
    ax.scatter(*np.where(positions == 6), label=particle_names[6], color="k",      marker="x",          s=90*marker_scale)
    ax.scatter(*np.where(positions == 7), label=particle_names[7], color="k",      marker="o",          s=90*marker_scale)
    ax.scatter(*np.where(positions == 8), label=particle_names[8], color="k",      marker="*",          s=90*marker_scale)

    ax.set_xlim(-plot_edge, grid_size[0]+plot_edge)    
    ax.set_ylim(-plot_edge, grid_size[1]+plot_edge)

    ax.plot([0, grid_size[0]], [0, 0],                       "--", color="black")
    ax.plot([grid_size[0], 0], [grid_size[1], grid_size[1]], "--", color="black")
    ax.plot([grid_size[0],  grid_size[0]], [0, grid_size[1]],      color="black")
    ax.plot([0, 0], [grid_size[1], 0],                             color="black")


    # draw legend outside of the plot

    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #draw frame
    if show:
        plt.show()
    if save:
        if fname is None:
            now = datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
            fname = os.path.join(os.getcwd(), 'plots', 'plot_'+now+'.png')
        plt.savefig(fname)
        print(f'plot saved at {fname}.')
        
        
def get_parameters_dict(fname: str):
    """
    Retruns the simulation parameters as dictionary from a toml file.
    The parameters include:
    
    - comment: str
        Comment string for the simulation.
    - num_steps: int
        Number of simulation steps.
    - reflection_x: bool
        If True, particles are reflected at the x boundaries.
    - fname_traj: str
        Filename of the trajectory file.
    - move_probability : np.ndarray
        Probability of every particle to move one step in one direction.
    - binding_probability : np.ndarray
        Probability of every particle to bind to another particle.
    - initial_positions : np.ndarray
        Initial positions of the particles.
    
    
    Parameters
    ----------
    fname : str
        Filename of the toml file.
        
    Returns
    -------
    parameters_dict : dict
        Dictionary containing the simulation parameters.
    """
    
    
    with open(fname, 'r') as f:
        parameters_dict = toml.load(f)
    
    # backwards compatibility
    if "comment" not in parameters_dict:
        parameters_dict["comment"] = ""
        
    if "diffusion_probability" in parameters_dict:
        parameters_dict["move_probability"] = np.array(parameters_dict["diffusion_probability"])
        parameters_dict.dump("diffusion_probability")
    
    parameters_dict["move_probability"] = np.array(parameters_dict["move_probability"])
    parameters_dict["binding_probability"] = np.array(parameters_dict["binding_probability"])
    parameters_dict["initial_positions"] = np.array(parameters_dict["initial_positions"])
    
    return parameters_dict


def get_initial_positions(grid_size:        np.ndarray = [300, 100],
                          concentrations:   np.ndarray = [0.0,   # empty grid site (unused because of normalization)
                                                          0.05,  # chemokine
                                                          0.05,  # netrin
                                                          0.2,   # heparansulfate
                                                          0.0,   # chemokine-netrin
                                                          0.0,   # chemokine-heparansulfate
                                                          0.1, # collagen site
                                                          0.0,   # chemokine-netrin-collagen
                                                          0.0],  # netrin-collagen
                                                        #   x       y
                          regions_fraction: np.ndarray = [[[0, 1],   [0, 1]],   # empty grid site
                                                          [[0, 0.2], [0, 1]],   # chemokine
                                                          [[0, 1],   [0, 1]],   # netrin
                                                          [[0, 1],   [0, 1]],   # heparansulfate
                                                          [[0, 1],   [0, 1]],   # chemokine-netrin
                                                          [[0, 1],   [0, 1]],   # chemokine-heparansulfate
                                                          [[0, 1],   [0.49, 0.51]],   # collagen site
                                                          [[0, 1],   [0, 1]],   # chemokine-netrin-collagen
                                                          [[0, 1],   [0, 1]]]   # netrin-collagen
                        ):
    '''
    Creates initial configuration for a certain grid size and particle concentrations
    
    Parameters
    ----------
    grid_size : np.ndarray, optional
        size of the grid, by default [300, 100]
    concentrations : np.ndarray, optional
        concentrations of the particles for the specified regions.
        The first entry is the concentration of empty grid sites.
        the concentration of the empty grid size is used to normalize the other concentrations.
    regions_fraction : np.ndarray, optional
        the regions in which the particles are distributed.
        the concentrations refer to these regions, NOT the whole grid.
        
    Returns
    -------
    np.ndarray
        initial configuration of the particles 
    '''
    
    initial_positions = np.zeros(grid_size, dtype=int)
    
    particle_types = np.arange(len(concentrations))
    
    # normlize concentrations
    assert np.sum(concentrations[1:]) <= 1, 'concentrations are not valid'
    concentrations[0] = 1 - np.sum(concentrations[1:])
    concentrations = np.array(concentrations)
    
    # part the grid into different regions
    x_regions = []
    y_regions = []
    
    # regions
    idx_regions = np.zeros_like(regions_fraction, dtype=int)
    
    for i in range(len(regions_fraction)):
        assert regions_fraction[i][0][0] >= 0 and regions_fraction[i][0][1] <= 1, 'x region is not valid'
        assert regions_fraction[i][1][0] >= 0 and regions_fraction[i][1][1] <= 1, 'y region is not valid'
        assert regions_fraction[i][0][0] <= regions_fraction[i][0][1], 'x region is not valid'
        assert regions_fraction[i][1][0] <= regions_fraction[i][1][1], 'y region is not valid'
        
        idx_regions[i] = [[np.round(regions_fraction[i][0][0]*grid_size[0]),
                           np.round(regions_fraction[i][0][1]*grid_size[0])],
                          [np.round(regions_fraction[i][1][0]*grid_size[1]),
                           np.round(regions_fraction[i][1][1]*grid_size[1])]]
        
        x_regions.append(idx_regions[i, 0, 0])
        x_regions.append(idx_regions[i, 0, 1])
        y_regions.append(idx_regions[i, 1, 0])
        y_regions.append(idx_regions[i, 1, 1])
    
    x_regions = np.sort(list(set(x_regions)))
    y_regions = np.sort(list(set(y_regions)))    
    
    use_particle = np.zeros(((len(x_regions)-1)*((len(y_regions)-1)), len(concentrations)), dtype=bool)
    
    # set init pos
    ii = 0
    for idx_x in range(len(x_regions)-1):
        for idx_y in range(len(y_regions)-1):
            ixl = x_regions[idx_x]
            ixr = x_regions[idx_x+1]-1
            iyl = y_regions[idx_y]
            iyr = y_regions[idx_y+1]-1
            
            for k in range(len(idx_regions)):
                if (ixl >= idx_regions[k, 0, 0] and ixr < idx_regions[k, 0, 1]) and (iyl >= idx_regions[k, 1, 0] and iyr < idx_regions[k, 1, 1]):
                    use_particle[ii, k] = True
            #print(use_particle) 
            ii += 1 
    
    ii = 0
    for idx_x in range(len(x_regions)-1):
        for idx_y in range(len(y_regions)-1):
            ixl = x_regions[idx_x]
            ixr = x_regions[idx_x+1]
            iyl = y_regions[idx_y]
            iyr = y_regions[idx_y+1]

            current_particle_types = particle_types[use_particle[ii]]
            current_concentrations = concentrations[use_particle[ii]]
            current_concentrations[0] = 1-np.sum(current_concentrations[1:])
            
            initial_positions[ixl:ixr, iyl:iyr] = np.random.choice(current_particle_types, 
                                                                   size=[ixr-ixl, iyr-iyl], 
                                                                   p=current_concentrations)
              
            ii += 1    
    
    return initial_positions
    
def validate_name_sys(dir_sys: str, name_sys: str, overwrite_sys: bool = False):
    # create dir if it doesnt exist
    if not os.path.exists(dir_sys):
        os.makedirs(dir_sys)

    # if sys exists append version number
    if os.path.exists(os.path.join(dir_sys, "traj_"+name_sys+".toml")) and not overwrite_sys:
        print(f"System '{name_sys}' already exists!")
        name_sys_0 = name_sys.split("_version")[0]
        version = 0
        while os.path.exists(os.path.join(dir_sys, "traj_"+name_sys_0+"_ver"+str(version)+".toml")):
            version += 1
        name_sys = name_sys_0+"_ver"+str(version)
        print(f"System name has been chaged to '{name_sys}'.")
    
    dir_traj = os.path.dirname(get_fname(dir_sys, name_sys, ftype="trajectory"))
    dir_cfg = os.path.dirname(get_fname(dir_sys, name_sys, ftype="config"))
    
    if not os.path.exists(dir_traj):
        os.makedirs(dir_traj)
    if not os.path.exists(dir_cfg):
        os.makedirs(dir_cfg)
    
    return name_sys

def get_fname(dir_sys, name_sys, ftype="trajectory"):
    """
    filetypes:
        trajectory
        config
        animation
    """
    
    if ftype == "trajectory":
        fname = os.path.join(dir_sys, "trajectories", "traj_"+name_sys+".toml")
    if ftype == "config":
        fname = os.path.join(dir_sys, "configs", "cfg_"+name_sys+".toml")
    if ftype == "animation":
        fname = os.path.join(dir_sys, "animations", "anim_"+name_sys+".mp4")
        
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
        
    return fname