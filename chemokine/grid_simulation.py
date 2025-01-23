import os
import h5py
import random
import bisect
from . import utils
from rust_chemokine import calculate_step 
import numpy as np
from tqdm import tqdm

def simulate(config):
    """
    monte carlo simulation of binding particles on a grid
    the first index is the x coordinate and the second index is the y coordinate
    i.e. grid[i, j] is the grid point at position x=i, y=j
    
    There are empty sites (0), 5 particle types (1, 2, 3, 4, 5) and 3 types of collagen interaction sites (6, 7, 8):
    0: empty grid point
    1: chemokine
    2: netrin
    3: heparansulfate
    4: collagen interaction site
    5: chemokine-netrin complex
    6: chemokine-heperansulfate complex
    7: netrin-heparansulfate complex 
    8: netrin-collagen_site complex
    9: chemokine-netrin-collagen_site complex
    
    Parameters:
    -----------
    config : dict or str
        dictionary containing the simulation parameters
        If config is a string, it gets interpreted as a path to a .toml file containing the simulation parameters
    
    Simulation parameters, that must be contained in config:
    --------------------------------------------------------
    num_steps : int
        number of steps the simulation should run
    stride : int
        the trajectory is saved every stride steps
    reflection : tuple
        tuple of booleans with len=4 determining at which walls the particles are reflected
        the order is [x0 x1 y0 y1] i.e. left, right, up, down
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
    name_sys : str
        system name which is used for the traj file
    dir_sys: str
        directory which is used for the traj file
        the file will be saved at 'dir_sys/trajectories/traj_name_sys.hdf5'
    track_types : list
        list of bools for each particle type, determining which particle types should be tracked during the simulation
        
    Returns
    -------
    trajectory: h5py.Dataset
        3d array of shape (num_steps//stride+1, init_pos.shape[0], init_pos.shape[1]) containing the trajectory of the particles
        Each entry corrsponds to the particle type at the respective grid site
    particle_trajectory: h5py.Dataset
        4d array of shape (num_steps//stride+1, init_pos.shape[0], init_pos.shape[1], n_tracked_particles) containing the trajectory of the tracked particles
        The last dimension indexes the tracked particle type, the entries for each type are the grid sites where the particles of each type are located. The
        particles for each type are numbered from 1 to n_type. Entries where the trajectory is zero correspond to an empty grid site.
    """
    

    
    if isinstance(config, str):
        # load config from file
        config = utils.load_config(config)
    
    # get simulation parameters
    num_steps                     = config['num_steps']
    stride                        = config['stride']
    reflection                    = config['reflection']
    initial_positions             = config['initial_positions']
    interaction_probabilities     = config['interaction_probabilities']
    name_sys                      = config['name_sys']
    dir_sys                       = config['dir_sys']
    track_types                   = config['track_types']
    add_particles_per_step        = config['add_particles_per_step']
    region_fraction_add_particles = config["region_fraction_add_particles"]
            
    fname_traj = utils.get_fname(dir_sys, name_sys, 'trajectory')
    
    # number of grid point types
    n_ptypes = interaction_probabilities.shape[0]
    n_ptypes_sq = n_ptypes**2
    
    # list all ptypes
    ptypes_all = np.arange(n_ptypes)
    
    # list of particle types that are tracked
    ptypes_tracked = np.array(ptypes_all[track_types],dtype=np.uint64)
    
    # get [n_ptypes x n_ptypes] lookup table of which particle types are contained in other particle types
    type_contains = utils._get_type_contains(interaction_probabilities)
    
    # create list of particle types that contain the tracked particle types to check if a particle moved
    tracked_particle_possible_moves = []
    for ptype in ptypes_tracked:
        tracked_particle_possible_moves.append(ptypes_all[type_contains[:,ptype]])
    
    # cumulative sum of the interaction probabilities
    interaction_probabilities_cumulative = np.cumsum(interaction_probabilities, axis=2)
    
    # check if the cumulative sum is correct
    if not np.allclose(interaction_probabilities_cumulative[:, :, -1], 1, atol=1e-14):
        raise ValueError("interaction probabilities do not sum to one")
    
    # divide by four because there is four neighbours
    interaction_probabilities_cumulative /= 4
    
    # grid lengths 
    l_x = initial_positions.shape[0]
    l_y = initial_positions.shape[1]
        
    # NOTE could change to something like a trajectory chunk that is written to file every x steps
    # current frame of the trajectory
    pos = np.array(initial_positions, dtype=np.uint64)

    # keep track of which particles already moved
    pos_moved = np.zeros_like(initial_positions, dtype=bool) 
    
    # move probability array
    p_choice = np.zeros(4*n_ptypes_sq)
    
    # for quick indexing in move_prob_array
    move_direction_base_idx   = np.array(np.arange(4)   * n_ptypes_sq, dtype=np.uint64)
    move_direction_base_idx_2 = np.array(np.arange(1,5) * n_ptypes_sq, dtype=np.uint64)
    
    # keep track of which moves are allowed
    allowed_moves = np.zeros(4, dtype=bool)
    
    # create trajectory file
    fh5 = h5py.File(fname_traj, 'w')
    fh5.attrs['completion_status'] = 'incomplete'
    traj_h5 = fh5.create_dataset('trajectory', shape=(num_steps//stride + 1, l_x, l_y), dtype=np.int8)
    
    # do any particles have to be tracked?
    write_particle_traj = bool(np.any(track_types))
    
    if write_particle_traj:
        fname_traj_particles = utils.get_fname(dir_sys, name_sys, 'particle_traj')
        fh5_particles = h5py.File(fname_traj_particles, 'w')
        fh5_particles.attrs['completion_status'] = 'incomplete'
        
        n_trackers = np.sum(track_types)
        
        traj_h5_particles = fh5_particles.create_dataset('trajectory', shape=(num_steps//stride+1, l_x, l_y, n_trackers), dtype=np.int16)
        # attach tracked typed to the dataset
        traj_h5_particles.attrs['track_types'] = track_types
        # save initial positions of the tracked particles
        traj_h5_particles[0] = utils.get_tracked_particle_initial_positions(config)      
        # save new particle positions
        pos_particles = np.array(traj_h5_particles[0], dtype=np.uint64)
    else:
        pos_particles = np.array([], dtype=np.uint64, ndmin=3)        
    
    
    # copy initial positions to trajectory
    traj_h5[0] = np.copy(pos)
    
    # check if particles should be added during simulation
    add_particles = np.any(add_particles_per_step != 0.0)
    
    if add_particles:
        # make sure that particles are not tracked when adding particles during simulation
        if write_particle_traj:
            raise ValueError("adding particles during simulation is not supported when tracking particles")
        
        # boolean array that is true for particle types that need to be added
        add_particle_type = add_particles_per_step != 0
        
        # calculate the number of steps that have to pass until one particle gets added of a certain type
        n_steps_add_particle = 2 * num_steps * np.ones_like(add_particles_per_step, dtype=np.uint64)
        n_steps_add_particle[add_particle_type] = np.floor(1/add_particles_per_step[add_particle_type]).astype(np.uint64)
        
        # validate reagions fraction input
        if region_fraction_add_particles[add_particle_type, :, 0].min() < 0.0:
            raise ValueError("region_fraction_add_particles must be larger than or equal to 0.0")
        if region_fraction_add_particles[add_particle_type, :, 1].max() > 1.0:
            raise ValueError("region_fraction_add_particles must be less than or equal to 1.0")
        if np.any((region_fraction_add_particles[add_particle_type, 0, 1] - region_fraction_add_particles[add_particle_type, 0, 0])*l_x < 1.0):
            raise ValueError("region_fraction_add_particles must result in a width of at least 1.0 in x")
        if np.any((region_fraction_add_particles[add_particle_type, 1, 1] - region_fraction_add_particles[add_particle_type, 1, 0])*l_y < 1.0):
            raise ValueError("region_fraction_add_particles must result in a width of at least 1.0 in y")
        
        # caclulate grid indices for adding particles
        region_idx_add_particles = np.zeros_like(region_fraction_add_particles, dtype=np.uint64)
        region_idx_add_particles[:, 0, :] = region_fraction_add_particles[:, 0, :] * (l_x - 1)
        region_idx_add_particles[:, 1, :] = region_fraction_add_particles[:, 1, :] * (l_y - 1)
    
    
    # start simulation
    for step in tqdm(range(1, num_steps+1)):                
    #for step in range(1, num_steps+1):                
        # no particle moved yet
        pos_moved.fill(False)    
        
        calculate_step(
            l_x=l_x,#:                                                                      usize,
            l_y=l_y,#:                                                                      usize,
            pos=pos,#:                                                                      &PyArray2<usize>, // Particle positions
            pos_moved=pos_moved,#:                                                          &PyArray2<bool>, // Whether the position has been moved
            pos_particles=pos_particles,#:                                                  &PyArray3<usize>, // Particle trajectories
            allowed_moves=allowed_moves,#:                                                  &PyArray1<bool>, // Movement flags
            p_choice=p_choice,#:                                                            &PyArray1<f64>, // Probabilities for movement
            move_direction_base_idx=move_direction_base_idx,#:                              &PyArray1<usize>, // Starting indices for move directions
            move_direction_base_idx_2=move_direction_base_idx_2,#:                          &PyArray1<usize>, // Ending indices for move directions
            reflection=reflection,#:                                                        &PyArray1<bool>, // Reflection conditions
            interaction_probabilities=interaction_probabilities,#:                          &PyArray3<f64>, // Interaction probabilities,
            interaction_probabilities_cumulative=interaction_probabilities_cumulative,#:    &PyArray3<f64>, // Interaction probabilities
            n_ptypes=n_ptypes,#:                                                            usize,
            n_ptypes_sq=n_ptypes_sq,#:                                                      usize,
            write_particle_traj=write_particle_traj,#:                                      bool,
            tracked_particle_possible_moves=tracked_particle_possible_moves,#:              &PyList, // Possible moves for tracked particles
        )
        
        if add_particles:
            for idx_type_add, add_type in enumerate(add_particle_type):
                if step%n_steps_add_particle[idx_type_add] == 0 and add_type:
                    for _ in range(10000):
                        idx_x_add = random.randint(region_idx_add_particles[idx_type_add, 0, 0], region_idx_add_particles[idx_type_add, 0, 1])
                        idx_y_add = random.randint(region_idx_add_particles[idx_type_add, 1, 0], region_idx_add_particles[idx_type_add, 1, 1])
                        if pos[idx_x_add, idx_y_add] == 0:
                            pos[idx_x_add, idx_y_add] = idx_type_add
                            break
                    else:
                        print("WARNING: Could not add particle of type", idx_type_add, "in step", step) 
        
        # update trajectory
        if step%stride == 0:
            traj_h5[step//stride] = np.copy(pos)
            
            if write_particle_traj:
                traj_h5_particles[step//stride] = np.copy(pos_particles)

    fh5.attrs['completion_status'] = 'completed'
    fh5.close()
    
    if write_particle_traj:
        fh5_particles.attrs['completion_status'] = 'completed'
        fh5_particles.close() 
