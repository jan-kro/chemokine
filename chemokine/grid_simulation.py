import os
import h5py
import random
import bisect
from . import utils
from rust_chemokine import calculate_step 
import numpy as np
from tqdm import tqdm

# def _get_move_probabilities(
#     pos, 
#     pos_moved,
#     tag, 
#     i, 
#     j, 
#     neighbours, 
#     allowed_moves,
#     p_choice,
#     n_ptypes,
#     move_direction_base_idx,
#     move_direction_base_idx_2,
#     interaction_probabilities_cumulative, 
#     reflection, 
#     l_x, 
#     l_y,
#     ):
#     """
#     Obtains the probability array for all possible moves and interactions of a particle at position i, j with its neighbouring particles.
    
#     Parameters
#     ----------
#     pos : np.ndarray
#         2d array of the current positions of the particles with shape (N_sites_x, N_sites_y)
#     pos_moved : np.ndarray
#         2d array of booleans with shape (N_sites_x, N_sites_y) determining if a particle at the respective grid site has already moved in the current time step
#     tag : int
#         tag of the particle that should move
#     i : int
#         x coordinate of the particle
#     j : int
#         y coordinate of the particle
#     neighbours : np.ndarray
#         1d array of length 4 containing the tags of the neighbouring particles (filled with zeros at input)
#     allowed_moves : np.ndarray
#         1d array of length 4 containing booleans determining if the particle can move in the respective direction (filled with zeros at input)
#     p_choice : np.ndarray
#         1d array of length 4*N containing the probabilities of the particle to move in the respective direction (filled with zeros at input)
#     n_ptypes : int
#         number of particle types
#     move_direction_base_idx : np.ndarray
#         1d array of length 4 containing the base indices for the four move directions (+x, -x, +y, -y) in the move probability array
#     move_direction_base_idx_2 : np.ndarray
#         1d array of length 4 containing the base indices + n_ptypes^2 for the four move directions (+x, -x, +y, -y) in the move probability array
#     interaction_probabilities_cumulative : np.ndarray
#         3d array with shape (N, N, N^2)
#         It contains a cumulative sum of all probabilities for every possible outcome state between two particles.
#         The first index i_in is the particle type of the first particle, the second index j_in is the particle type of the second particle.
#         The third index ij_out contains the cumulative probability that states [i_out, j_out] = [k//N, k%N] occur for every k = 0, 1, 2, ..., ij_out-1, ij_out.
#         The last entry in the third index must be 1.
#     reflection : tuple
#         tuple of booleans with len=4 determining at which walls the particles are reflected
#         the order is [x0 x1 y0 y1] i.e. left, right, up, down
#     l_x : int
#         number of grid sites in x direction
#     l_y : int
#         number of grid sites in y direction
        
#     Returns
#     -------
#     p_choice : np.ndarray
#         1d array of length 4*n_ptypes^2 containing the probabilities of all possible outcome states that can occur with the interaction of the current particle with its neighbours.
#     neighbours : np.ndarray
#         1d array of length 4 containing the tags of the neighbouring particles
#     neighbour_idx : np.ndarray
#         1d array of length 4 containing the indices of the neighbouring particles
#     """
    
#     allowed_moves.fill(True)
#     p_choice[move_direction_base_idx[0]:move_direction_base_idx_2[0]].fill(0)
#     p_choice[move_direction_base_idx[1]:move_direction_base_idx_2[1]].fill(0.25)
#     p_choice[move_direction_base_idx[2]:move_direction_base_idx_2[2]].fill(0.5)
#     p_choice[move_direction_base_idx[3]:move_direction_base_idx_2[3]].fill(0.75)
    
#     # get neighbouring indices and wrap around
#     neighbour_idx = np.array((((i+1)%l_x, j), ((i-1)%l_x, j), (i, (j+1)%l_y), (i, (j-1)%l_y)))
    
#     # check for neighbouring particles that already moved    
#     allowed_moves = ~pos_moved[neighbour_idx[:, 0], neighbour_idx[:, 1]]
    
#     # get neighbours
#     neighbours = pos[neighbour_idx[:, 0], neighbour_idx[:, 1]]
    
#     # check for walls
    
#     # reflection array: [x0 x1 y0 y1]
#     # allowed moves array: [+x -x +y -y]
    
#     if reflection[0] & (i == 0):
#         allowed_moves[1] = False # wall   
#     if reflection[1] & (i == l_x-1):
#         allowed_moves[0] = False # wall  
#     if reflection[2] & (j == 0):
#         allowed_moves[3] = False # wall   
#     if reflection[3] & (j == l_y-1):
#         allowed_moves[2] = False # wall
    
#     # loop thour the four neighbouring particles
#     for p_base_idx, p_base_idx_end, allowed, neighbour in zip(move_direction_base_idx, move_direction_base_idx_2, allowed_moves, neighbours):
#         # if move is allowed, the probability array for this direction if filled with interaction probabilities
#         if allowed:
#             p_choice[p_base_idx:p_base_idx_end] += interaction_probabilities_cumulative[tag, neighbour]
#         # if move is not allowed the particle will stay at its position
#         else:
#             p_choice[p_base_idx + tag*n_ptypes + neighbour:p_base_idx_end] += 0.25
#     return p_choice, neighbours, neighbour_idx

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
        config = utils.load_config_dict(config)
    
    # get simulation parameters
    num_steps                   = config['num_steps']
    stride                      = config['stride']
    reflection                  = np.array(config['reflection'], dtype=bool)
    initial_positions           = config['initial_positions']
    interaction_probabilities   = config['interaction_probabilities']
    name_sys                    = config['name_sys']
    dir_sys                     = config['dir_sys']
    track_types                 = config['track_types']
    
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
    # p_choice_2 = np.zeros(4*n_ptypes_sq) #! for testing
    
    # for quick indexing in move_prob_array
    move_direction_base_idx   = np.array(np.arange(4)   * n_ptypes_sq, dtype=np.uint64)
    move_direction_base_idx_2 = np.array(np.arange(1,5) * n_ptypes_sq, dtype=np.uint64)
    
    # neighbour array for certain particle
    neighbours = np.zeros(4, dtype=int) # +x, -x, +y, -y
    # neighbours_2 = np.zeros(4, dtype=int) # +x, -x, +y, -y #! For testing
    
    # keep track of which moves are allowed
    allowed_moves = np.zeros(4, dtype=bool)
    
    # create trajectory file
    fh5 = h5py.File(fname_traj, 'w')
    fh5.attrs['completion_status'] = 'incomplete'
    traj_h5 = fh5.create_dataset('trajectory', shape=(num_steps//stride+1, l_x, l_y), dtype=np.int8)
    
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
            
    
    # copy initial positions to trajectory
    traj_h5[0] = np.copy(pos)
    
    # #! FOR TESTING ONLY
    # n_actions = 0
    # n_stay = 0
    
    
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
        
        # update trajectory
        if step%stride == 0:
            traj_h5[step//stride] = np.copy(pos)
            
            if write_particle_traj:
                traj_h5_particles[step//stride] = np.copy(pos_particles)

    fh5.attrs['completion_status'] = 'completed'
    fh5.close()
    
    # print(f"Average number of actions per step: {n_actions/num_steps}")#! FOR TESTING ONLY
    # print(f"Average number of stays per step: {n_stay/num_steps}")#! FOR TESTING ONLY
    # print(f"Total number of actions: {n_actions}")#! FOR TESTING ONLY
    
    if write_particle_traj:
        fh5_particles.attrs['completion_status'] = 'completed'
        fh5_particles.close() 

# def simulate(config):
#     """
#     monte carlo simulation of binding particles on a grid
#     the first index is the x coordinate and the second index is the y coordinate
#     i.e. grid[i, j] is the grid point at position x=i, y=j
    
#     There are empty sites (0), 5 particle types (1, 2, 3, 4, 5) and 3 types of collagen interaction sites (6, 7, 8):
#     0: empty grid point
#     1: chemokine
#     2: netrin
#     3: heparansulfate
#     4: collagen interaction site
#     5: chemokine-netrin complex
#     6: chemokine-heperansulfate complex
#     7: netrin-heparansulfate complex
#     8: netrin-collagen_site complex
#     9: chemokine-netrin-collagen_site complex
    
#     Parameters:
#     -----------
#     config : dict or str
#         dictionary containing the simulation parameters
#         If config is a string, it gets interpreted as a path to a .toml file containing the simulation parameters
    
#     Simulation parameters, that must be contained in config:
#     --------------------------------------------------------
#     num_steps : int
#         number of steps the simulation should run
#     stride : int
#         the trajectory is saved every stride steps
#     reflection : tuple
#         tuple of booleans with len=4 determining at which walls the particles are reflected
#         the order is [x0 x1 y0 y1] i.e. left, right, up, down
#     initial_positions : np.ndarray
#         2d array of the initial positions of the particles with shape (N_sites_x, N_sites_y)
#         The array consists of N different integer values, where N is the number of particle types.
#         Empty grid sites are also particle types.
#         Example: init_pos[3, 7] = 1 means that there is a particle of type 1 at position x=3, y=7
#     interaction_probabilities : np.ndarray
#         3d array with shape (N, N, N^2)
#         It contains all probabilities (divided by 4) for every possible outcome state between two particles.
#         The first index i_in is the particle type of the first particle, the second index j_in is the particle type of the second particle.
#         The third index ij_out contains the probability that the state [i_out, j_out] = [ij_out//N, ij_out%N] occurs.
#     name_sys : str
#         system name which is used for the traj file
#     dir_sys: str
#         directory which is used for the traj file
#         the file will be saved at 'dir_sys/trajectories/traj_name_sys.hdf5'
#     track_types : list
#         list of bools for each particle type, determining which particle types should be tracked during the simulation
        
#     Returns
#     -------
#     trajectory: h5py.Dataset
#         3d array of shape (num_steps//stride+1, init_pos.shape[0], init_pos.shape[1]) containing the trajectory of the particles
#         Each entry corrsponds to the particle type at the respective grid site
#     particle_trajectory: h5py.Dataset
#         4d array of shape (num_steps//stride+1, init_pos.shape[0], init_pos.shape[1], n_tracked_particles) containing the trajectory of the tracked particles
#         The last dimension indexes the tracked particle type, the entries for each type are the grid sites where the particles of each type are located. The
#         particles for each type are numbered from 1 to n_type. Entries where the trajectory is zero correspond to an empty grid site.
#     """
    

    
#     if isinstance(config, str):
#         # load config from file
#         config = utils.load_config_dict(config)
    
#     # get simulation parameters
#     num_steps                   = config['num_steps']
#     stride                      = config['stride']
#     reflection                  = config['reflection']
#     initial_positions           = config['initial_positions']
#     interaction_probabilities   = config['interaction_probabilities']
#     name_sys                    = config['name_sys']
#     dir_sys                     = config['dir_sys']
#     track_types                 = config['track_types']
    
#     fname_traj = utils.get_fname(dir_sys, name_sys, 'trajectory')
    
#     # number of grid point types
#     n_ptypes = interaction_probabilities.shape[0]
#     n_ptypes_sq = n_ptypes**2
    
#     # list all ptypes
#     ptypes_all = np.arange(n_ptypes)
    
#     # list of particle types that are tracked
#     ptypes_tracked = ptypes_all[track_types]
    
#     # get [n_ptypes x n_ptypes] lookup table of which particle types are contained in other particle types
#     type_contains = utils._get_type_contains(interaction_probabilities)
    
#     # create list of particle types that contain the tracked particle types to check if a particle moved
#     tracked_particle_possible_moves = []
#     for ptype in ptypes_tracked:
#         tracked_particle_possible_moves.append(ptypes_all[type_contains[:,ptype]])
    
#     # cumulative sum of the interaction probabilities
#     interaction_probabilities_cumulative = np.cumsum(interaction_probabilities, axis=2)
    
#     # check if the cumulative sum is correct
#     if not np.allclose(interaction_probabilities_cumulative[:, :, -1], 1, atol=1e-14):
#         raise ValueError("interaction probabilities do not sum to one")
    
#     # divide by four because there is four neighbours
#     interaction_probabilities_cumulative /= 4
    
#     # grid lengths 
#     l_x = initial_positions.shape[0]
#     l_y = initial_positions.shape[1]
        
#     # NOTE could change to something like a trajectory chunk that is written to file every x steps
#     # current frame of the trajectory
#     pos = np.array(initial_positions, dtype=np.int8)

#     # keep track of which particles already moved
#     pos_moved = np.zeros_like(initial_positions, dtype=bool) 
    
#     # move probability array
#     p_choice = np.zeros(4*n_ptypes_sq)
#     p_choice_2 = np.zeros(4*n_ptypes_sq) #! for testing
    
#     # for quick indexing in move_prob_array
#     move_direction_base_idx   = np.arange(4)   * n_ptypes_sq
#     move_direction_base_idx_2 = np.arange(1,5) * n_ptypes_sq
    
#     # neighbour array for certain particle
#     neighbours = np.zeros(4, dtype=int) # +x, -x, +y, -y
#     neighbours_2 = np.zeros(4, dtype=int) # +x, -x, +y, -y #! For testing
    
#     # keep track of which moves are allowed
#     allowed_moves = np.zeros(4, dtype=bool)
    
#     # create trajectory file
#     fh5 = h5py.File(fname_traj, 'w')
#     traj_h5 = fh5.create_dataset('trajectory', shape=(num_steps//stride+1, l_x, l_y), dtype=np.int8)
    
#     # do any particles have to be tracked?
#     write_particle_traj = np.any(track_types)
    
#     if write_particle_traj:
#         fname_traj_particles = utils.get_fname(dir_sys, name_sys, 'particle_traj')
#         fh5_particles = h5py.File(fname_traj_particles, 'w')
        
#         n_trackers = np.sum(track_types)
        
#         traj_h5_particles = fh5_particles.create_dataset('trajectory', shape=(num_steps//stride+1, l_x, l_y, n_trackers), dtype=np.int16)
#         # attach tracked typed to the dataset
#         traj_h5_particles.attrs['track_types'] = track_types
#         # save initial positions of the tracked particles
#         traj_h5_particles[0] = utils.get_tracked_particle_initial_positions(config)      
#         # save new particle positions
#         pos_particles = np.copy(traj_h5_particles[0])
            
    
#     # copy initial positions to trajectory
#     traj_h5[0] = np.copy(pos)
    
#     #! FOR TESTING ONLY
#     n_actions = 0
#     n_stay = 0
    
    
#     # start simulation
#     for step in tqdm(range(1, num_steps+1)):                
#     #for step in range(1, num_steps+1):                
#         # no particle moved yet
#         pos_moved.fill(False)    
        
#         # loop through all particles and sites
#         for idx_x in range(l_x):
#             for idx_y in range(l_y):
                
#                 # get tag of the current particle 
#                 tag = pos[idx_x, idx_y]

#                 # check if site is not empty and if it has not moved yet
#                 if tag != 0 and not pos_moved[idx_x, idx_y]:
                    
#                     p_choice, neighbours, neighbour_idx = _get_move_probabilities(
#                         pos, 
#                         pos_moved,
#                         tag, 
#                         idx_x, 
#                         idx_y, 
#                         neighbours, 
#                         allowed_moves,
#                         p_choice,
#                         n_ptypes,
#                         move_direction_base_idx,
#                         move_direction_base_idx_2,
#                         interaction_probabilities_cumulative,
#                         reflection, 
#                         l_x, 
#                         l_y)
                    
#                     # first decide if the particle will break its bond
                    
#                     # choose the type of interaction that takes place                 
#                     step_idx = bisect.bisect_left(p_choice, random.uniform(0, 1))
                    
#                     # if p_choice[-1] != 1:
#                     #     raise ValueError("last entry in p_choice should be 1")

#                     # get the tag of the neighbour, that the particle interacts with, and its coordinates
#                     tag_neighbour = neighbours[step_idx//n_ptypes_sq]
                    
                    
#                     # get the new tags after the interaction takes place
#                     new_tags = [step_idx%n_ptypes_sq//n_ptypes, step_idx%n_ptypes]
                    
#                     # check if particles moved (i.e. if an interaction took place)
#                     if not (tag==new_tags[0] and tag_neighbour==new_tags[1]):
                        
#                         idx_new_x, idx_new_y = neighbour_idx[step_idx//n_ptypes_sq]
                        
#                         # update tags
#                         pos[idx_x, idx_y] = new_tags[0]
#                         pos[idx_new_x, idx_new_y] = new_tags[1]
                         
#                         # update pos moved
                        
#                         pos_moved[idx_x, idx_y] = new_tags[0] != 0  #? Should it also be flagged as moved if it is an empty grid site?
#                         pos_moved[idx_new_x, idx_new_y] = True      #* No, because the empty grid site is not a particle
                        
#                          # update particle trajectory
#                         if write_particle_traj:
#                             # check if current particle has to be tracked
#                             for idx_type, ptype in enumerate(ptypes_tracked):
#                                 #! after testing remove ptype and loop: for idx_type in range(n_ptypes_tracked): ...
#                                 # if the shifted particle contains particle type to be tracked, update position
#                                 # if type_contains[new_tags[1], ptype] and type_contains[tag, ptype]:
#                                 # if the particle that moved is a particle that has to be tracked
#                                 for ptype_moved in tracked_particle_possible_moves[idx_type]:
#                                     if (interaction_probabilities[ptype_moved, new_tags[0], tag] > 0) and (interaction_probabilities[ptype_moved, tag_neighbour, new_tags[1]] > 0):    
                                        
#                                         #! TODO: remove after testing
#                                         if pos_particles[idx_x, idx_y, idx_type] == 0:
#                                             print("p choice", p_choice)
#                                             print("step idx", step_idx)
#                                             print("p_choice[step_idx]", p_choice[step_idx])
#                                             print("interaction probability", interaction_probabilities[tag, tag_neighbour, new_tags[0]*n_ptypes + new_tags[1]])
#                                             print("interaction_probabilities[5, 5, 1*n_ptypes+5]")
#                                             print(interaction_probabilities[5, 5, 1*n_ptypes+5])
#                                             print("interaction_probabilities[5, 5, 2*n_ptypes+5]")
#                                             print(interaction_probabilities[5, 5, 2*n_ptypes+5])
#                                             print("n_ptypes", n_ptypes)
#                                             print("interaction_probabilies.shape[0]", interaction_probabilities.shape[0])
#                                             print("tag", tag, "tag_neighbour", tag_neighbour)
#                                             print("new tags", new_tags, "new_tags[0]", new_tags[0], "new_tags[1]", new_tags[1])
#                                             print("ptype", ptype)
#                                             print("particle pos", pos_particles[idx_x, idx_y, idx_type], pos_particles[idx_new_x, idx_new_y, idx_type])
#                                             print("new particle pos", 0, pos_particles[idx_x, idx_y, idx_type])
#                                             print("idx_x, idx_y", idx_x, idx_y)
#                                             print("pos[idx_x, idx_y]", pos[idx_x, idx_y] )
#                                             print("idx_new_x, idx_new_y", idx_new_x, idx_new_y)
#                                             print("pos[idx_new_x, idx_new_y]" ,pos[idx_new_x, idx_new_y] )

#                                             raise ValueError(f"particle type {ptype} not at position {idx_x, idx_y}")    
#                                         if pos_particles[idx_new_x, idx_new_y, idx_type] != 0:
#                                             print("p choice", p_choice)
#                                             print("step idx", step_idx)
#                                             print("p_choice[step_idx]", p_choice[step_idx])
#                                             print("interaction probability", interaction_probabilities[tag, tag_neighbour, new_tags[0]*n_ptypes + new_tags[1]])
#                                             print("interaction_probabilities[5, 5, 1*n_ptypes+5]")
#                                             print(interaction_probabilities[5, 5, 1*n_ptypes+5])
#                                             print("interaction_probabilities[5, 5, 2*n_ptypes+5]")
#                                             print(interaction_probabilities[5, 5, 2*n_ptypes+5])
#                                             print("n_ptypes", n_ptypes)
#                                             print("interaction_probabilies.shape[0]", interaction_probabilities.shape[0])
#                                             print("tag", tag, "tag_neighbour", tag_neighbour)
#                                             print("new tags", new_tags, "new_tags[0]", new_tags[0], "new_tags[1]", new_tags[1])
#                                             print("ptype", ptype)
#                                             print("particle pos", pos_particles[idx_x, idx_y, idx_type], pos_particles[idx_new_x, idx_new_y, idx_type])
#                                             print("new particle pos", 0, pos_particles[idx_x, idx_y, idx_type])
#                                             print("idx_x, idx_y", idx_x, idx_y)
#                                             print("pos[idx_x, idx_y]", pos[idx_x, idx_y] )
#                                             print("idx_new_x, idx_new_y", idx_new_x, idx_new_y)
#                                             print("pos[idx_new_x, idx_new_y]" ,pos[idx_new_x, idx_new_y] )
#                                             raise ValueError(f"particle type {ptype} already at position {idx_new_x, idx_new_y}")
                                        
#                                         pos_particles[idx_new_x, idx_new_y, idx_type] = pos_particles[idx_x, idx_y, idx_type]
#                                         pos_particles[idx_x, idx_y, idx_type] = 0
                
#                         n_actions += 1 #! FOR TESTING ONLY
                        
#                     else:
#                         n_stay += 1
                     
                            
#                     #? should the current particle also be tagged as "moved" if it did not move, but could have moved?
#                     #* Yes, because in a bon situation, the bond partner still needs to have the possibility to bond with it    

#         # update trajectory
#         if step%stride == 0:
#             traj_h5[step//stride] = np.copy(pos)
            
#             if write_particle_traj:
#                 traj_h5_particles[step//stride] = np.copy(pos_particles)

#     fh5.close()
    
#     print(f"Average number of actions per step: {n_actions/num_steps}")#! FOR TESTING ONLY
#     print(f"Average number of stays per step: {n_stay/num_steps}")#! FOR TESTING ONLY
#     print(f"Total number of actions: {n_actions}")#! FOR TESTING ONLY
    
#     if write_particle_traj:
#         fh5_particles.close() 


# def simulate(config):
#     """
#     monte carlo simulation of binding particles on a grid
#     the first index is the x coordinate and the second index is the y coordinate
#     i.e. grid[i, j] is the grid point at position x=i, y=j
    
#     There are empty sites (0), 5 particle types (1, 2, 3, 4, 5) and 3 types of collagen interaction sites (6, 7, 8):
#     0: empty grid point
#     1: chemokine
#     2: netrin
#     3: heparansulfate
#     4: chemokine-netrin complex
#     5: chemokine-heperansulfate complex
#     6: collagen interaction site
#     7: chemokine-netrin-collagen_site complex
#     8: netrin-collagen_site complex
    
#     Parameters:
#     -----------
#     config : dict or str
#         dictionary containing the simulation parameters
#         If config is a string, it gets interpreted as a path to a .toml file containing the simulation parameters
    
#     Simulation parameters, that must be contained in config:
#     --------------------------------------------------------
#     num_steps : int
#         number of steps the simulation should run
#     stride : int
#         the trajectory is saved every stride steps
#     reflection : tuple
#         tuple of booleans with len=4 determining at which walls the particles are reflected
#         the order is [x0 x1 y0 y1] i.e. left, right, up, down
#     initial_positions : np.ndarray
#         2d array of the initial positions of the particles with shape (N_sites_x, N_sites_y)
#         The array consists of N different integer values, where N is the number of particle types.
#         Empty grid sites are also particle types.
#         Example: init_pos[3, 7] = 1 means that there is a particle of type 1 at position x=3, y=7
#     bond_possible : np.ndarray
#         2d array of booleans with shape (N_types, N_types) determining if two particles can bind
#         The first index is the particle that wants to bind and the second index is the particle that it will bind to
#         a simple diffusion is also treated as a bond, where the particle binds to an empty grid site
#     unbond_possible : np.ndarray
#         2d array of booleans with shape (N_types, N_types) determining if two particles can unbind
#         The first index is the particle that wants to break its bond and the second index is the particle that it will unbind to.
#         A particle can unbind towards an empty grid site (=simple diffusion) or towards another particle, which can bond with one of the non-bonded particles.
#     bond_idx : np.ndarray
#         3d array of shape (N_types, N_types, 2) containing the tags of the new particles after a bond
#         The first index is the particle that wants to bind and the second index is the particle that it will bind to
#         The third index is the tag of the two particles after the bond
#         If the particle bonds to an empty grid site, the first index of the resulting types is zero 
#     unbond_idx : np.ndarray
#         3d array of shape (N_types, N_types, 2) containing the tags of the new particles after an unbond
#         The first index is the particle that wants to unbind and the second index is the particle that it will unbind to
#         The third index is the tag of the two particles after the process.
#     bond_probability : np.ndarray
#         2d array of shape (N_types, N_types) containing the probabilities that a bond between two particle types happens
#         If the particle bonds to an empty grid site, the bond_probability is the same as the move ("diffusion") probability
#         If a particle bonds to another particle, the bond_probability is the product of the move probability and the binding probability
#     unbond_probability : np.ndarray
#         2d array of shape (N_types, N_types) containing the probabilities that a bond between two particle types breaks
#         If the particle unbonds to another particle, the unbond_probability is the product of the move probability of the 
#         particle, that changes bond partner and the unbonding probability times the new bond probability
#         If a particle unbonds to an empty grid site, the unbond_probability is the product of the average move probability of the two 
#         particles (p1 + p2)/2 move probability and the unbinding probability
#     name_sys : str
#         system name which is used for the traj file
#     dir_sys: str
#         directory which is used for the traj file
#         the file will be saved at 'dir_sys/trajectories/traj_name_sys.hdf5'
#     write_particle_traj : bool
#         if True, a trajectory for every particle type is saved, where each particle is numbered from 1 to N
#     track_types : list
#         list of bools for each particle type, determining which particle types should be tracked during the simulation
        
#     Returns
#     -------
#     trajectory: h5py.Dataset
#         3d array of shape (num_steps//stride+1, init_pos.shape[0], init_pos.shape[1]) containing the trajectory of the particles
#     trajectory
#     """
    
    
#     if isinstance(config, str):
#         # load config from file
#         config = utils.load_config_dict(config)
    
#     # get simulation parameters
#     num_steps           = config['num_steps']
#     stride              = config['stride']
#     reflection          = config['reflection']
#     name_sys            = config['name_sys']
#     dir_sys             = config['dir_sys']
#     initial_positions   = config['initial_positions']
#     bond_possible       = config["bond_possible"]
#     unbond_possible     = config["unbond_possible"]
#     bond_idx            = config["bond_idx"]
#     unbond_idx          = config["unbond_idx"]
#     bond_probability    = config["bond_probability"]
#     unbond_probability  = config["unbond_probability"]
#     write_particle_traj = config["write_particle_traj"]
#     track_types         = config["track_types"]
#     type_contains       = config["type_contains"]
    
#     fname_traj = utils.get_fname(dir_sys, name_sys, 'trajectory')
    
#     # number of grid point types
#     n_types = len(bond_possible)
    
#     ptypes = np.arange(n_types)
    
#     # grid lengths 
#     l_x = initial_positions.shape[0]
#     l_y = initial_positions.shape[1]
    
#     # keep track of particles that can/cannot move to decrease runtime
#     move_possible = np.zeros(n_types, dtype=bool)
#     for k in range(n_types):
#         move_possible[k] = np.any(bond_possible[k]) or np.any(unbond_possible[k])
    
    
#     # NOTE could change to something like a trajectory chunk that is written to file every x steps
#     # current frame of the trajectory
#     pos = np.array(initial_positions, dtype=np.int8)

#     # keep track of which particles already moved
#     pos_moved = np.zeros_like(initial_positions, dtype=bool) 
    
#     # move probability array
#     move_prob = np.zeros(5)
    
#     # neighbour array for certain particle
#     neighbours = np.zeros(4, dtype=int) # +x, -x, +y, -y
    
#     # keep track of which moves are allowed
#     allowed_moves = np.zeros(4, dtype=bool)
#     allowed_bonds = np.zeros(4, dtype=bool)
#     allowed_unbonds = np.zeros(4, dtype=bool)
    
#     # create trajectory file
#     fh5 = h5py.File(fname_traj, 'w')
#     traj_h5 = fh5.create_dataset('trajectory', shape=(num_steps//stride+1, l_x, l_y), dtype=np.int8)
    
#     if write_particle_traj:
#         fname_traj_particles = utils.get_fname(dir_sys, name_sys, 'particle_traj')
#         fh5_particles = h5py.File(fname_traj_particles, 'w')
        
#         n_trackers = np.sum(track_types)
        
#         traj_h5_particles = fh5_particles.create_dataset('trajectory', shape=(num_steps//stride+1, l_x, l_y, n_trackers), dtype=np.int16)
        
#         # attach tracked typed to the dataset
#         traj_h5_particles.attrs['track_types'] = track_types
        
#         # save initial positions of the tracked particles
#         traj_h5_particles[0] = utils.get_tracked_particle_initial_positions(pos, track_types, type_contains) 
        
#         # save new particle positions
#         pos_particles = np.copy(traj_h5_particles[0])
            
    
#     # copy initial positions to trajectory
#     traj_h5[0] = np.copy(pos)
    
#     # start simulation
#     for step in tqdm(range(1, num_steps+1)):                
#     #for step in range(1, num_steps+1):                
#         # no particle moved yet
#         pos_moved.fill(False)    
        
#         # loop through all particles and sites
#         for idx_x in range(l_x):
#             for idx_y in range(l_y):
                
#                 # get tag of the current particle 
#                 tag = pos[idx_x, idx_y]

#                 # check if site is not empty and if it has not moved yet
#                 if move_possible[tag] and not pos_moved[idx_x, idx_y]:
#                     # get neighbours and allowed moves (allowed_moves is set to true in the function at the beginning)                                    
#                     neighbours, neighbour_idx, allowed_moves, allowed_bonds, allowed_unbonds = _get_neighbours(
#                         pos, 
#                         pos_moved, 
#                         idx_x, 
#                         idx_y, 
#                         neighbours, 
#                         allowed_moves, 
#                         allowed_bonds, 
#                         allowed_unbonds,
#                         bond_possible,
#                         unbond_possible,
#                         reflection=reflection, 
#                         l_x=l_x, 
#                         l_y=l_y
#                     )

                    
                    
#                     # first decide if the particle will break its bond
                    
#                     moved_flag = False
                    
#                     if np.any(allowed_unbonds):
#                         # loop through all unbonds
#                         for idx_move, neighbour_tag in enumerate(neighbours):
#                             # calcualate step probability for every direction
#                             move_prob[idx_move] = unbond_probability[tag, neighbour_tag]*allowed_unbonds[idx_move]                        
                        
#                         # normalize probabilities (calculate probability to stay)
#                         move_prob[4] = 1 - np.sum(move_prob[:4])

#                         # choose new position
#                         step_idx = np.random.choice(5, p=move_prob)
                        
#                         if step_idx != 4:
#                             moved_flag = True

#                             # tag of the bond partner
#                             tag_partner = neighbours[step_idx]
                            
#                             # get tags of the new particle types of the two grid sites
#                             #! TODO: maybe we need to randomize which particle stays and which leaves to avoid bias in MSD
#                             new_tags = unbond_idx[tag, tag_partner]
                    
#                     # if the particle did not unbond, it will try to bond/move
                    
#                     if not moved_flag and np.any(allowed_bonds):
#                         # loop through all possible bonds
#                         for idx_move, neighbour_tag in enumerate(neighbours):
#                             move_prob[idx_move] = bond_probability[tag, neighbour_tag]*allowed_bonds[idx_move] 

#                         # normalize probabilities
#                         move_prob[4] = 1 - np.sum(move_prob[:4])

#                         # choose new position
#                         step_idx = np.random.choice(5, p=move_prob)
                        
#                         if step_idx != 4:
#                             moved_flag = True
                            
#                             # tag of the bond partner
#                             tag_partner = neighbours[step_idx]
                            
#                             new_tags = bond_idx[tag, tag_partner]
                                                
                    
                        
#                     if moved_flag:
#                         # idx of the new position
#                         new_idx_x = neighbour_idx[step_idx][0]
#                         new_idx_y = neighbour_idx[step_idx][1]

#                         # update tags
#                         pos[idx_x, idx_y] = new_tags[0]
#                         pos[new_idx_x, new_idx_y] = new_tags[1]
                         
#                         # update pos moved
#                         pos_moved[idx_x, idx_y] = True          #! NOTE: should it also be flagged as moved if it is an empty grid site??
#                         pos_moved[new_idx_x, new_idx_y] = True
                        
#                         # update particle trajectory
#                         if write_particle_traj:
#                             # check if current particle has to be tracked
#                             for idx_type, ptype in enumerate(ptypes[track_types]):
#                                 # if the shifted particle contains particle type to be tracked, update position
#                                 if type_contains[new_tags[1], ptype] and type_contains[tag, ptype]:
                                    
#                                     #TODO: remove after testing
#                                     if pos_particles[idx_x, idx_y, idx_type] == 0:
#                                         print("old tags", tag, tag_partner)
#                                         print("new tags", new_tags)
#                                         print("ptype", ptype)
#                                         print("particle pos", pos_particles[idx_x, idx_y, idx_type], pos_particles[new_idx_x, new_idx_y, idx_type])
#                                         print("new particle pos", 0, pos_particles[idx_x, idx_y, idx_type])
#                                         raise ValueError(f"particle type {ptype} not at position {idx_x, idx_y}")    
#                                     if pos_particles[new_idx_x, new_idx_y, idx_type] != 0:
#                                         print("old tags", tag, pos[new_idx_x, new_idx_y])
#                                         print("new tags", new_tags)
#                                         print("ptype", ptype)
#                                         print("particle pos", pos_particles[idx_x, idx_y, idx_type], pos_particles[new_idx_x, new_idx_y, idx_type])
#                                         print("new particle pos", 0, pos_particles[idx_x, idx_y, idx_type])
#                                         raise ValueError(f"particle type {ptype} already at position {new_idx_x, new_idx_y}")
                                    
#                                     pos_particles[new_idx_x, new_idx_y, idx_type] = pos_particles[idx_x, idx_y, idx_type]
#                                     pos_particles[idx_x, idx_y, idx_type] = 0
                        
#                     #! TODO: should the current particle also be tagged as "moved" if it did not move, but could have moved?    

#         # update trajectory
#         if step%stride == 0:
#             traj_h5[step//stride] = pos
            
#             if write_particle_traj:
#                 traj_h5_particles[step//stride] = np.copy(pos_particles)
            
#     fh5.close()
    
#     if write_particle_traj:
#         fh5_particles.close() 



