from numba import njit
from datetime import datetime
from matplotlib.animation import FuncAnimation
import numpy as np
import tidynamics as tid
import matplotlib.pyplot as plt
import os
import toml
import h5py
from tqdm import tqdm
import ffmpeg

#! TODO create FileType class to avoid plain text in file creation etc...


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
        If True, only the diffusivity is considered for calculating the step probability. cd /home/janmak98/Projects/chemokine_skripts; conda activate chemokine
    
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
    a_grid = (n_min / ((grid_dim[0]) * (grid_dim[1]) * np.min(concentrations) * 6.022e26)) ** (1/3)
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


def get_physical_concentration(n_species: np.ndarray, grid_dim: tuple, a_grid: float):
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
    concentration = n_species / (grid_dim[0] * grid_dim[1] * a_grid**3 * 6.022e26)
    return concentration

def get_grid_constant(n_particles: np.ndarray, concentration: np.ndarray, grid_dim: tuple):
    """
    calcualte the grid constant of a system of given number of particles for a grid of grid_dim[0]*grid_dim[1] sites,
    with a desired physical particle concentration.
    
    Parameters
    ----------
    n_particles : int
        Number of particles of desired particle type.
    concentration : float
        Physical concentration of desired particle type in [mol/l].
    grid_dim : tuple
        Tuple of integers containing the dimensions of the grid in an ndarray.shape like fashion.
        
    Returns
    -------
    a_grid : float
        Grid constant in [m].
    """

    a_grid = (n_particles / ((grid_dim[0]) * (grid_dim[1]) * concentration * 6.022e26)) ** (1/3)
    return a_grid
    
    
    


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
    
    # Chemokine: green_Hex color # 00A200
    # Netrin-1: red_Hex color # FF0000
    # Chemokine + netrin-1_Hex color # FF00FF
    
    if particle_names==None:
        particle_names =[
            "empty site",
            "CCL5",
            "Net-1",
            "Heperansulfate",
            "CCL5-Net-1",
            "CCL5-Heperansulfate",
            "Collagen Site",
            "CCL5-Net-1-Collagen",
            "Net-1-Collagen"
        ]
    
    n_steps = trajectory.shape[0]
    grid_size = trajectory[0].shape

    # import background image of collagen fiber
    img = plt.imread("images/collagen_fiber_long.png")
    
    fig, ax = plt.subplots(dpi=300)
    fig.set_figwidth(grid_size[0]/max(grid_size)*20)
    fig.set_figheight(grid_size[1]/max(grid_size)*20)

    box = ax.get_position()

    def update(i):
        
        #clear the frame
        ax.clear()
        
        ax.imshow(img, extent=[0, grid_size[0], 0, grid_size[1]])
        
        ax.scatter(*np.where(trajectory[i] == 1), label=particle_names[1], color="#00A200",                      s=80*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 2), label=particle_names[2], color="#FF0000", marker=r"--$\cdot$", s=250*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 3), label=particle_names[3], color="grey",    marker=r"$\sim$",    s=150*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 4), label=particle_names[4], color="#FF00FF",                      s=90*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 5), label=particle_names[5], color="purple",  marker="+",          s=80*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 6), label=particle_names[6], color="k",       marker="x",          s=90*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 7), label=particle_names[7], color="k",       marker="o",          s=90*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 8), label=particle_names[8], color="k",       marker="*",          s=90*marker_scale)
        

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

def animate_ffmpeg_from_config(
        config:               dict,
        stride:               int   = 1,
        frame_range:          tuple = None,
        particle_names:       list  = None,
        fname:                str   = None,
        fps_in:               int   = 15,
        fps_out:              int   = 30,
        marker_scale:         float = 1.0,
        plot_edge:            int   = 5,
        show_collagen:        bool  = True,
        colors:               list  = None,
        markers:              list  = None,
        fname_collagen_image: str   = None,
        delete_frames:        bool  = True,
        a_scale:              float = 1.0,
        axis_labels:          list  = None,
    ):


    fname_traj =  get_fname(config["dir_sys"], config["name_sys"], ftype="trajectory")
    
    if particle_names is None:
        if "particle_names" in config.keys():
            particle_names = config["particle_names"]
    
    if fname is None:
        fname = get_fname(config["dir_sys"], config["name_sys"], ftype="animation")
    
    with h5py.File(fname_traj, "r") as f:
        # useful because if f["trajectory"] is not indexed, it is not loaded into memory
        animate_ffmpeg(
            f["trajectory"],
            stride=stride,
            frame_range=frame_range, 
            particle_names=particle_names, 
            fname=fname, 
            fps_in=fps_in, 
            fps_out=fps_out, 
            marker_scale=marker_scale, 
            plot_edge=plot_edge,
            show_collagen=show_collagen,
            colors=colors,
            markers=markers,
            fname_collagen_image=fname_collagen_image, 
            delete_frames=delete_frames,
            a_scale=a_scale,
            axis_labels=axis_labels,
        )


def animate_ffmpeg(
        trajectory,
        stride:               int   = 1,
        frame_range:          tuple = None,
        particle_names:       list  = None,
        fname:                str   = None,
        fps_in:               int   = 15,
        fps_out:              int   = 30,
        marker_scale:         float = 1.0,
        plot_edge:            int   = 5,
        colors:               list  = None,
        markers:              list  = None,
        show_collagen:        bool  = True,
        fname_collagen_image: str   = None,
        delete_frames:        bool  = True,
        a_scale:              float = 1.0,
        axis_labels:          list  = None
    ):
    
    if not fname.endswith(".mp4"):
        raise ValueError("Filename must end with .mp4")
    
    if frame_range is None:
        frame_start = 0
        frame_end = trajectory.shape[0]
    else:
        frame_start = frame_range[0]
        frame_end = frame_range[1]
    
    
    print(f"Creating animation at {fname} ...")
    
    print("Saving individual frames ...")
    # save individual frames
    #n_digits = int(np.ceil(np.log10(n_frames)))
    for k, i in enumerate(range(frame_start, frame_end, stride)):
        fname_frame = fname.replace(".mp4", "_frame") + str(k) + ".png" #f"{i:0{n_digits}d}.png"
        plot_grid(
            trajectory[i], 
            show=False, 
            save=True, 
            show_collagen=show_collagen, 
            fname=fname_frame, 
            particle_names=particle_names, 
            plot_edge=plot_edge, 
            marker_scale=marker_scale,
            colors=colors,
            markers=markers,
            fname_collagen_image=fname_collagen_image,
            a_scale=a_scale,
            axis_labels=axis_labels,
        )
    
    print("Converting frames to video ...")
    # convert to video
    # ffmpeg -framerate 1 -i frame%d.png -c:v libx264 -r 30 output.mp4
    # ffmpeg.input(fname.replace(".mp4", "_frame") + "%d.png", framerate=fps_in).output(fname, framerate=fps_out, loglevel="warning").run()
    # ffmpeg.input(fname.replace(".mp4", "_frame") + "%d.png", framerate=fps_in).output(fname, {"codec:v": "libx264"}, framerate=fps_out, loglevel="warning").run()
    ffmpeg.input(
        fname.replace(".mp4", "_frame") + "%d.png", 
        framerate=fps_in
    ).output(
        fname, 
        vcodec="libx264", 
        pix_fmt="yuv420p",
        framerate=fps_out, 
        loglevel="warning"
    ).run()

    if delete_frames:
        print("Deleting individual frames ...")
        # delete frames
        for k, i in enumerate(range(frame_start, frame_end, stride)):
            os.remove(fname.replace(".mp4", "_frame") + str(k) + ".png")
        
    print("Done.")
    
def plot_grid(
    positions,
    particle_names:       list  = None,
    show:                 bool  = True,
    save:                 bool  = True,
    fname:                str   = None,
    show_collagen:        bool  = True,
    fname_collagen_image: str   = None,
    marker_scale:         float = 1.0,
    plot_edge:            float = 4.0,
    colors:               list  = None,
    markers:              list  = None,
    marker_sizes:         list  = None,
    a_scale:              float = 1.0,
    axis_labels:          list  = None):
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
    
    # import background image of collagen fiber
    
    if show_collagen:
        
        if fname_collagen_image is None:
            fname_collagen_image = "/home/janmak98/chemokine/chemokine/images/collagen_fiber_long.png"
        
        img = plt.imread(fname_collagen_image)
    
    if particle_names is None:
        particle_names =[
            "empty site",
            "CCL5",
            "Net-1",
            "Heperansulfate",
            "Collagen Site",
            "CCL5-Net-1",
            "CCL5-Heperansulfate",
            "Net-1-Heparansulfate",
            "Net-1-Collagen"
            "CCL5-Net-1-Collagen",
        ]
    
    n_ptypes = len(particle_names) - 1 # exclude empty site
    
    if colors is None:
        colors = [
            "#00982C", # chemokine
            "#F80808", # netrin
            "#8CD9F4", # heparansulfate
            "none",    # collagen site
            "#FF7532", # chemokine-netrin
            "#358A9F", # chemokine-heparansulfate
            "#358A9F", # netrin-heparansulfate
            "purple",  # netrin-collagen
            "purple",  # chemokine-netrin-collagen
        ]
    
        if len(colors) < n_ptypes:
            colors += ["k"]*(n_ptypes-5)
    
    if markers is None:
        markers = [
            "o",          # chemokine
            r"--$\cdot$", # netrin
            r"$\sim$",    # heparansulfate
            ".",          # collagen site
            "o",          # chemokine-netrin
            "o",          # chemokine-heparansulfate
            r"--$\cdot$", # netrin-heparansulfate
            r"--$\cdot$", # netrin-collagen
            "o",          # chemokine-netrin-collagen
        ]
        if len(markers) < n_ptypes:
            markers += ["o"]*(n_ptypes-8)
            
    if marker_sizes is None:
        marker_sizes = [
            80, 
            250, 
            150, 
            90, 
            80,
            90,
            250,
            250,
            80]
        if len(marker_sizes) < n_ptypes:
            marker_sizes += [90]*(n_ptypes-5)
    
    grid_size = np.array(positions.shape)*a_scale
     
    fig, ax = plt.subplots()
    fig.set_figwidth(grid_size[0]/max(grid_size)*20)
    fig.set_figheight(grid_size[1]/max(grid_size)*20)

    
    
    box = ax.get_position()

    ax.clear()

    if show_collagen:
        ax.imshow(img, extent=[0, grid_size[0], 0, grid_size[1]])
    
    for i in range(1, n_ptypes+1):
        x, y = np.where(positions == i)
        if False:#i == 4:
            ax.scatter(x*a_scale, y*a_scale, color=colors[i-1], marker=markers[i-1], s=marker_sizes[i-1]*marker_scale)
        else:
            ax.scatter(x*a_scale, y*a_scale, label=particle_names[i], color=colors[i-1], marker=markers[i-1], s=marker_sizes[i-1]*marker_scale)
    
    ax.set_xlim(-plot_edge, grid_size[0]+plot_edge)    
    ax.set_ylim(-plot_edge, grid_size[1]+plot_edge)

    ax.plot([0, grid_size[0]], [0, 0],                       "--", color="black")
    ax.plot([grid_size[0], 0], [grid_size[1], grid_size[1]], "--", color="black")
    ax.plot([grid_size[0],  grid_size[0]], [0, grid_size[1]],      color="black")
    ax.plot([0, 0], [grid_size[1], 0],                             color="black")


    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
    
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
        fig.savefig(fname, bbox_inches='tight')
        #print(f'plot saved at {fname}.')
        
    fig.clf()
    plt.close()
        

def load_config(fname: str):
    """
    loads config from filetype, depending on file handle
    """

    file_suffix = fname.split(".")[-1]
    if file_suffix == "toml":
        return load_config_dict(fname)
    
def load_config_dict(fname: str):
    """
    Returns the simulation config as dictionary from a toml file.
    The lists in the config are converted to numpy arrays with the respective dtype.
    The config keys include:
    comment: str
        Comment string for the simulation.
        Can contain any information about the simulation.
    num_steps : int
        number of steps the simulation should run
    stride : int
        the trajectory is saved every stride steps
    reflection : tuple
        tuple of booleans with len=4 determining at which walls the particles are reflected
        the order is [x0 x1 y0 y1] i.e. left, right, up, down
    particle_names : list
        list of strings containing the names of the particles in the same order as the particle types
    initial_positions : np.ndarray
        2d array of the initial positions of the particles with shape (N_sites_x, N_sites_y)
        The array consists of N different integer values, where N is the number of particle types.
        Empty grid sites are also particle types.
        Example: init_pos[3, 7] = 1 means that there is a particle of type 1 at position x=3, y=7
    interaction_probabilities : np.ndarray
        3d array with shape (N, N, N^2)
        It contains all probabilities (divided by 4) for every possible outcome state between two particles.
        The first index i_in is the particle type of the first particle, the second index j_in is the particle type of the second particle.
        The third index ij_out contains the probability that the state [i_out, j_out] = [ij_out//N, ij_out%N] occurs.
        The sum over the third index should be 0.25 for every i_in, j_in.         
    name_sys : str
        system name which is used for the traj file
    dir_sys: str
        directory which is used for the traj file
        the file will be saved at 'dir_sys/trajectories/traj_name_sys.hdf5'
    track_types : np.ndarray
        1d array of booleans with length N
        If track_types[i] = True, the trajectory of particle type i is saved
    diffusivities : np.ndarray
        1d array of floats with length N
        The diffusivity of particle type i in units of m^2/s
    binding_energy : np.ndarray
        2d array of floats with length (N,N)
        The binding energy of particle type i to particle of type j in units of kBT
    binding_constants : np.ndarray
        exponential of the binding energy
        
    
    Parameters
    ----------
    fname : str
        Filename of the toml file.
        
    Returns
    -------
    config_dict : dict
        Dictionary containing the simulation parameters.
    """
    
    
    with open(fname, 'r') as f:
        config_dict = toml.load(f)
    
    # convert dict entries to appropriate data types
    config_dict['initial_positions']         = np.array(config_dict['initial_positions'], dtype=np.int8)
    config_dict["binding_energy"]            = np.array(config_dict["binding_energy"], dtype=np.float64)
    config_dict["diffusivities"]             = np.array(config_dict["diffusivities"], dtype=np.float64)
    config_dict["track_types"]               = np.array(config_dict["track_types"], dtype=bool)
    config_dict["interaction_probabilities"] = np.array(config_dict["interaction_probabilities"], dtype=np.float64)
    config_dict["reflection"]                = np.array(config_dict["reflection"], dtype=bool)
    
    # set default values and take care of backwards compatability
    if not "add_particles_per_step" in config_dict.keys():
        n = len(config_dict["diffusivities"])
        config_dict["add_particles_per_step"] = np.zeros(n, dtype=float)
    else:
        config_dict["add_particles_per_step"] = np.array(config_dict["add_particles_per_step"], dtype=float)
    
    if not "region_fraction_add_particles" in config_dict.keys():
        n = len(config_dict["diffusivities"])
        add = np.zeros((n, 2, 2), dtype=float)
        add[:, :, 1] = 1
        config_dict["region_fraction_add_particles"] = add 
    else:
        config_dict["region_fraction_add_particles"] = np.array(config_dict["region_fraction_add_particles"], dtype=float)
    
    return config_dict


def get_initial_positions(
    grid_size:        np.ndarray = [300, 100],
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

def get_initial_positions_homogeneous(grid_size: np.ndarray, concentrations: np.ndarray):
    '''
    Creates Initial configuration for a certain grid size and a set of particle concentrations
    
    Parameters
    ----------
    grid_size : tuple
        size of the grid in x and y direction: [l_x, l_y]
    concentrations : tuple
        concentrations of the particles in the grid. The sum of the concentrations must be smaller than 1.
        The number of entries in the tuple must be equal to the number of particle types, where the first 
        entry corresponds to the empty grid sites, the second entry is the concentration of particle type 1, 
        the third entry the concentration of particle type 2, and so on.

    Returns
    -------
    np.ndarray
        2D array of shape [l_x, l_y], containing the initial configuration of the particles
    '''
    
    initial_positions = np.zeros(grid_size)

    if np.sum(concentrations) > 1:
        raise ValueError("Concentrations are too high and sum over 1")


    maxit = 1e7
    for ptype, con in enumerate(concentrations):
        
        if ptype == 0:
            continue
        
        n_tot = np.round(con*grid_size[0]*grid_size[1])
        n = 0
        it  = 0
        while n<n_tot:
            i = np.random.randint(0,grid_size[0])
            j = np.random.randint(0,grid_size[1])
        
            if initial_positions[i,j] == 0:
                initial_positions[i,j] = ptype
                n += 1
            it+=1
            
            if it >= maxit:
                raise ValueError("maxit reached")
            
    return initial_positions

def _get_state_index(i, j, n_ptypes):
    return i*n_ptypes + j

def _get_interaction_indices(state_index, n_ptypes):
    return [state_index//n_ptypes, state_index%n_ptypes]

def _get_type_contains(interaction_probabilities):
    """
    Create a lookup table for which particle type contains which other particle type.
    If particle type i contains particle type j, then type_contains[i, j] is True.
    
    Parameters
    ----------
    interaction_probabilities : np.ndarray
        3D array with shape (n_ptypes, n_ptypes, n_ptypes**2) containing the probabilities of interactions.
        The first index is the particle that wants to bind, the second index is the particle that it will bind to
        and the third index is the state index of the resulting particles.
        
    Returns
    -------
    type_contains : np.ndarray
        2D array with shape (n_ptypes, n_ptypes) containing boolean values indicating which particle types are contained in other particle types.
        Example: if type_contains[1, 2] is True, this means that particle type 2 is contained in particle type 1.
    """
    n_ptypes = interaction_probabilities.shape[0]
    type_contains = np.zeros((n_ptypes, n_ptypes), dtype=bool)

    for i in range(n_ptypes):
        for j in range(n_ptypes):
            for k in range(n_ptypes**2):
                if interaction_probabilities[i, j, k] > 0:
                    n, m = _get_interaction_indices(k, n_ptypes=n_ptypes)
                    # if initial particle and resulting particle are both not empty sites
                    if i != 0 and m!=0:
                        # if whole particle moves, resulting particle contains initial particle
                        if n == 0:        
                            type_contains[m, i] = True
                        # if the bond partner is not an empty site, resulting particle contains the bond partner 
                        if j != 0:
                            type_contains[m, j] = True
                            
    # for any particle that is contained in another particle, the particle that contains it also contains all 
    # particles that the contained particle contains                    
    for i in range(n_ptypes):
        for j in range(n_ptypes):
            if type_contains[i,j]:
                for k in range(n_ptypes):
                    if type_contains[j,k]:
                        type_contains[i,k] = True
                        
    return type_contains


def get_tracked_particle_initial_positions(config):
    """
    For each particle type that should be tracked, an array is returned with the same shape as init_pos, where each 
    particle is assigned an index from 1 to N, where N is the number of particles of the type that should be tracked.
    This includes also particle complexes that contain the particle type to be tracked.
    
    Parameters
    ----------
    init_pos : np.ndarray
        2D array with shape (l_x, l_y) containing the initial positions of the particles.
    track_types : np.ndarray
        1D array with shape (n_types,) containing boolean values indicating which particle types should be tracked.
    
    Returns
    -------
    frame_trackers : np.ndarray
        3D array with shape (l_x, l_y, n_trackers) containing the indices of the particles that should be tracked.
        The third dimension corresponds to the different particle types that should be tracked. The entries that 
        are zero correspond to empty grid sites.
    """
    
    init_pos = config["initial_positions"]
    track_types = config["track_types"] 
    interaction_probabilities = config["interaction_probabilities"]
    
    type_contains = _get_type_contains(interaction_probabilities)
    types = np.arange(len(type_contains)) # array containing all particle types in ascending order
    n_trackers = np.sum(track_types) # number of particle types that should be tracked
    
    # initialize array to store the indices of the particles that should be tracked
    frame_trackers = np.zeros((init_pos.shape[0], init_pos.shape[1], n_trackers), dtype=np.uint16)
    
    # loop over particle types that should be tracked
    for idx_type, ptype in enumerate(types[track_types]):
        # get particle types in which ptype is contained
        contained_in = types[type_contains[:, ptype]]
        
        # mark which particles are contained in the tracker for the current type
        frame_trackers[:, :, idx_type] = np.isin(init_pos, contained_in)
        
        # give each particle an individual index from 1 to N
        frame_trackers[:, :, idx_type][frame_trackers[:, :, idx_type]==1] = np.arange(1, np.sum(frame_trackers[:, :, idx_type])+1)
    
    return frame_trackers

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

def get_fname_from_config(config, ftype="trajectory"):
    return get_fname(config["dir_sys"], config["name_sys"], ftype)

def get_fname(dir_sys, name_sys, ftype="trajectory", name=None):
    """
    filetypes:
        trajectory
        particle_traj
        config
        animation
    """
    if name is None:
        name = name_sys
    
    if ftype == "trajectory":
        fname = os.path.join(dir_sys, "trajectories", "traj_"+name+".hdf5")
    if ftype == "particle_traj":
        fname = os.path.join(dir_sys, "trajectories_tracked", "traj_particles"+name+".hdf5")
    if ftype == "config":
        fname = os.path.join(dir_sys, "configs", "cfg_"+name+".toml")
    if ftype == "animation":
        fname = os.path.join(dir_sys, "animations", "anim_"+name+".mp4")
    if ftype == "results":
        fname = os.path.join(dir_sys, "results", name+".hdf5")
        
    if not os.path.exists(os.path.dirname(fname)):
        #! This is necessary for parallel processing, because the directory might be created by another process
        try:
            os.makedirs(os.path.dirname(fname))
        except:
            print(f"Could not create directory {os.path.dirname(fname)}")
            
    return fname

@njit
def get_traj_xy(traj, n_particles: int = None):
    """
    Calculates a trajectory of shape (n_particles, n_frames, 2) from a trajectory of shape (n_frames, n_x, n_y), 
    where the last dimension is the x and y index (=position) of the particle in the frame.
    
    Parameters
    ----------
    traj : np.ndarray
        Trajectory of the particles of shape (n_frames, n_x, n_y), where each particle is represented by an individual integer value raneging from 1 to N+1, where N is the number of particles in the trajectory.
    n_particles : int
        Number of particles in the trajectory.
    """
    #! TODO there is no reason that one needs to pass the number of particles, this should be infered from the trajectory
    n_frames, n_x, n_y = traj.shape
    l_xy = np.array([n_x, n_y])
    
    # Infer number of particles
    if n_particles is None:
        n_particles = int(np.max(traj[0]))
    
    # Preallocate output
    traj_xy = np.zeros((n_particles, n_frames, 2))
    
    # Precompute positions for all frames
    particle_positions = np.zeros((n_frames, n_particles, 2))
    for frame_idx in range(n_frames):
        for particle_index in range(1, n_particles + 1):
            position = np.argwhere(traj[frame_idx] == particle_index)
            if position.size > 0:  # Particle is present
                particle_positions[frame_idx, particle_index - 1] = position[0]

    # Add initial positions
    traj_xy[:, 0, :] = particle_positions[0]
    
    # Unwrap trajectory using the minimum image convention
    for frame_idx in range(1, n_frames):   
        delta = particle_positions[frame_idx] - particle_positions[frame_idx - 1]
        traj_xy[:, frame_idx, :] = traj_xy[:, frame_idx - 1, :] + delta - l_xy * np.round(delta / l_xy)

    return traj_xy

def get_particle_numbers_concentrations(config, total=True, skip_frames=0):
    init_pos = config["initial_positions"]
    n_grid = init_pos.shape[0]*init_pos.shape[1]
    type_contains = _get_type_contains(config["interaction_probabilities"])
    ntypes = len(config["diffusivities"])
    ptypes = np.arange(ntypes)
    n_part = np.zeros(ntypes)
    if total:
        for k in range(ntypes):
            for j in range(ntypes):
                if type_contains[j, k] == True:
                    n_part[k] += np.count_nonzero(init_pos == ptypes[j])
    else:
        fname_particle_traj = get_fname(config["dir_sys"], config["name_sys"], "trajectory")
        with h5py.File(fname_particle_traj, "r") as fh5:
            traj = fh5["trajectory"]

            n_frames = traj.shape[0]

            for i in range(skip_frames, n_frames):
                for k in range(ntypes):
                    n_part[k] += np.sum(traj[i] == ptypes[k])
            n_part /= (n_frames-skip_frames)
    return n_part, n_part/n_grid 

def get_tracked_type_idx(config, ptype):
    track_types = config["track_types"]
    idx = 0
    for i in range(ptype):
        if track_types[i]:
            idx += 1
    return idx

def get_number_of_frames(config):
    fname_particle_traj = get_fname(config["dir_sys"], config["name_sys"], "trajectory")
    with h5py.File(fname_particle_traj, "r") as fh5:
        traj = fh5["trajectory"]
        n_frames = traj.shape[0]
    return n_frames

def _validate_frame_range(frame_range, n_frames):
    
    if frame_range is None:
        frame_range = (0, n_frames)
    else:
        if frame_range[0] < 0 or frame_range[1] > n_frames:
            raise ValueError("Frame range is not valid.")
    return frame_range

def get_msd_xy(config, ptype=0, frame_range=None, stride=1):

    type_idx = get_tracked_type_idx(config, ptype)
    fname_particle_traj = get_fname(config["dir_sys"], config["name_sys"], "particle_traj")
    
    print(f"Get MSD for system {config['name_sys']}")
    print(f"Loading .hdf5 trajectory file ...")
    with h5py.File(fname_particle_traj, "r") as fh5:
        
        frame_range = _validate_frame_range(frame_range, get_number_of_frames(config))
        
        traj = fh5["trajectory"][frame_range[0]:frame_range[1]:stride, :, :, type_idx]
        
        n_frames = traj.shape[0]
        
        l_xy = np.array([traj.shape[1], traj.shape[2]])
        
        n_particles = int(np.max(traj[0]))
        
        pname = config["particle_names"][ptype]
        
        print(f"Number of {pname} particles of in system {config['name_sys']}: {n_particles}")
        
        print("Unwrapping trajectory...")        
        traj_xy = get_traj_xy(traj, n_particles)
        
        msd_all_x = np.zeros((n_particles, n_frames))
        msd_all_y = np.zeros((n_particles, n_frames))
            
        print("Calculating MSD...")
        
        for k in range(n_particles):
            msd_all_x[k] = tid.msd(traj_xy[k,:,0])
            msd_all_y[k] = tid.msd(traj_xy[k,:,1])

        msd_x = np.mean(msd_all_x, axis=0)
        msd_y = np.mean(msd_all_y, axis=0)
        
        return msd_x, msd_y
    
def get_msd_xy_2(config, ptype=0, frame_range=None, stride=1):
    # this method calculates the unwrapped trajectory without loading the total particle_traj into memory
    # slower than get_msd_xy but no memory complications
    
    type_idx = get_tracked_type_idx(config, ptype)
    fname_particle_traj = get_fname(config["dir_sys"], config["name_sys"], "particle_traj")
    
    print(f"Get MSD for system {config['name_sys']}")
    print(f"Loading .hdf5 trajectory file ...")
    with h5py.File(fname_particle_traj, "r") as fh5:
        
        frame_range = _validate_frame_range(frame_range, get_number_of_frames(config))
        
        frame_indices = np.arange(frame_range[0], frame_range[1], stride)
        
        n_frames = len(frame_indices)
        
        traj = fh5["trajectory"]
        
        n_particles = int(np.max(traj[0, :, :, type_idx]))
        
        pname = config["particle_names"][ptype]
        
        print(f"Number of {pname} particles of in system {config['name_sys']}: {n_particles}")
        
        print("Unwrapping trajectory...")        
        # traj_xy = get_traj_xy(traj, n_particles)
        
        l_xy = np.array([traj.shape[1], traj.shape[2]])
                
        # Preallocate output
        traj_xy = np.zeros((n_particles, n_frames, 2), dtype=np.int64)
        
        # Precompute positions for all frames
        particle_positions = np.zeros((n_frames, n_particles, 2), dtype=np.int64)
        for idx_pos, frame_idx in enumerate(frame_indices):
            for particle_index in range(1, n_particles + 1):
                position = np.argwhere(traj[frame_idx, :, :, type_idx] == particle_index)
                if position.size > 0:  # Particle is present
                    particle_positions[idx_pos, particle_index - 1] = position[0]
                else:
                    raise ValueError(f"Particle {particle_index} is not present in frame {frame_idx}")
        
        # Add initial positions
        traj_xy[:, 0, :] = particle_positions[0]
        
        # Unwrap trajectory using the minimum image convention
        for pos_idx in range(1, n_frames):#frame_indices[1:]:
            delta = particle_positions[pos_idx] - particle_positions[pos_idx - 1]
            traj_xy[:, pos_idx, :] = traj_xy[:, pos_idx - 1, :] + delta - l_xy * np.round(delta / l_xy)
            
        msd_all_x = np.zeros((n_particles, n_frames))
        msd_all_y = np.zeros((n_particles, n_frames))
            
        print("Calculating MSD...")
        
        for k in range(n_particles):
            msd_all_x[k] = tid.msd(traj_xy[k,:,0])
            msd_all_y[k] = tid.msd(traj_xy[k,:,1])

        msd_x = np.mean(msd_all_x, axis=0)
        msd_y = np.mean(msd_all_y, axis=0)
        
        return msd_x, msd_y