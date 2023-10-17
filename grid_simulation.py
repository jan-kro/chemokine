import os
import h5py
import utils
import numpy as np
from datetime import datetime
from numba import njit
from tqdm import tqdm
import toml


def _get_neighbours(pos, 
                    pos_moved, 
                    i, 
                    j, 
                    neighbours, 
                    allowed_moves,
                    allowed_bonds,
                    allowed_unbonds, 
                    bond_possible,
                    unbond_possible, 
                    reflection: tuple = (False, False, False, False), 
                    l_x: int = None, 
                    l_y: int = None):
    """
    gets the four tags of the neighbours of the particle at pos[i, j]while considering boundery conditions
    also returns a boolean array that determines if the particle is allowed to move to the corresponding neighbouring site
    
    Parameters
    ----------
    pos : np.ndarray
        2d array of the positions of the particles with shape (grid_size[0], grid_size[1])
    pos_moved : np.ndarray
        2d array of booleans with shape = pos.shape determining wether a particle already moved
    i, j : int
        indices of the particle in pos. pos[i, j] = particle tag
    neighbours : np.ndarray
        1d array of len=4 with the neighbour tags of the particle in the following order: [+x -x +y -y]
        content of array does not matter because it gets overwritten
    allowed_moves : np.ndarray
        1d array of len=4 with booleans determining if the particle is allowed to move to the corresponding neighbour
        content of array does not matter because it gets overwritten
    reflection_x : bool, optional
        if True, the particles are reflected at the left and right wall, if False, they are wrapped around
    l_x : int, optional
        length of the grid in x direction, by default None
        only used if reflection_x is True
        
    Returns
    -------
    neighbours : np.ndarray
        1d array of len=4 with the neighbour tags of the particle in the following order: [+x -x +y -y]
        if the particle hits a wall (i. e. it is not allowed to move to that position), the corresponding entry is -1
    allowed_moves : np.ndarray
        1d array of len=4 with booleans determining if the particle is allowed to move to the corresponding neighbour
    """
    
    tag = pos[i, j]
    
    allowed_moves.fill(True)
    allowed_bonds.fill(True)
    allowed_unbonds.fill(True)
    
    # get neighbouring indices and wrap around
    neighbour_idx = np.array((((i+1)%l_x, j), ((i-1)%l_x, j), (i, (j+1)%l_y), (i, (j-1)%l_y)))
    
    # check for neighbouring particles that already moved    
    allowed_moves = ~pos_moved[neighbour_idx[:, 0], neighbour_idx[:, 1]]
    
    # get neighbours
    neighbours = pos[neighbour_idx[:, 0], neighbour_idx[:, 1]]
    
    # check for walls
    if reflection[0] & (i == 0):
        allowed_moves[1] = False # wall   
    if reflection[1] & (i == l_x-1):
        allowed_moves[0] = False # wall  
    if reflection[2] & (j == 0):
        allowed_moves[2] = False # wall   
    if reflection[3] & (j == l_y-1):
        allowed_moves[3] = False # wall
    
    # check for neighbours it cannot bond to
    for k, nb in enumerate(neighbours):
        allowed_bonds[k] = allowed_moves[k] & bond_possible[tag, nb]
    
    # check for neighbours it cannot unbond with    
    for k, nb in enumerate(neighbours):
        allowed_unbonds[k] = allowed_moves[k] & unbond_possible[tag, nb]
        
    allowed_moves = allowed_bonds | allowed_unbonds
                    
    return neighbours, neighbour_idx, allowed_moves, allowed_bonds, allowed_unbonds

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
    4: chemokine-netrin complex
    5: chemokine-heperansulfate complex
    6: collagen interaction site
    7: chemokine-netrin-collagen_site complex
    8: netrin-collagen_site complex
    
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
        2d array of the initial postitions of the particles with shape (N_sites_x, N_sites_y)
        The array consists of N different integer values, where N is the number of particle types.
        Empty grid sites are also particle types.
        Example: init_pos[3, 7] = 1 means that there is a particle of type 1 at position x=3, y=7
    bond_possible : np.ndarray
        2d array of booleans with shape (N_types, N_types) determining if two particles can bind
        The first index is the particle that wants to bind and the second index is the particle that it will bind to
        a simple diffusion is also treated as a bond, where the particle binds to an empty grid site
    unbond_possible : np.ndarray
        2d array of booleans with shape (N_types, N_types) determining if two particles can unbind
        The first index is the particle that wants to break its bond and the second index is the particle that it will unbind to.
        A particle can unbind twards an empty grid site or towards another particle, which can bond with one of the unbonded particles.
    bond_idx : np.ndarray
        3d array of shape (N_types, N_types, 2) containing the tags of the new particles after a bond
        The first index is the particle that wants to bind and the second index is the particle that it will bind to
        The third index is the tag of the two particles after the bond
        If the particle bonds to an empty grid site, the first index of the resulting types is zero 
    unbond_idx : np.ndarray
        3d array of shape (N_types, N_types, 2) containing the tags of the new particles after an unbond
        The first index is the particle that wants to unbind and the second index is the particle that it will unbind to
        The third index is the tag of the two particles after the process.
    bond_probability : np.ndarray
        2d array of shape (N_types, N_types) containing the probabilities that a bond between two particle types happens
        If the particle bonds to an empty grid site, the bond_probability is the same as the move ("diffusion") probability
        If a particle bonds to another particle, the bond_probability is the product of the move probability and the binding probability
    unbond_probability : np.ndarray
        2d array of shape (N_types, N_types) containing the probabilities that a bond between two particle types breaks
        If the particle unbonds to another particle, the unbond_probability is the product of the move probability of the 
        particle, that changes bond partner and the unbonding probability times the new bond probability
        If a particle unbonds to an empty grid site, the unbond_probability is the product of the average move probability of the two 
        particles (p1 + p2)/2 move probability and the unbinding probability
    name_sys : str
        system name which is used for the traj file
    dir_sys: str
        directory which is used for the traj file
        the file will be saved at 'dir_sys/trajectories/traj_name_sys.hdf5'
        
    Returns
    -------
    trajectory: h5py.Dataset
        3d array of shape (num_steps//stride+1, init_pos.shape[0], init_pos.shape[1]) containing the trajectory of the particles
    """
    
    
    if isinstance(config, str):
        # load config from file
        config = toml.load(config)
    
    # get simulation parameters
    num_steps          = config['num_steps']
    stride             = config['stride']
    reflection         = config['reflection']
    name_sys           = config['name_sys']
    dir_sys            = config['dir_sys']
    initial_positions  = np.array(config['initial_positions'], dtype=np.int8)
    bond_possible      = np.array(config["bond_possible"], dtype=bool)
    unbond_possible    = np.array(config["unbond_possible"], dtype=bool)
    bond_idx           = np.array(config["bond_idx"], dtype=np.int8)
    unbond_idx         = np.array(config["unbond_idx"], dtype=np.int8)
    bond_probability   = np.array(config["bond_probability"])
    unbond_probability = np.array(config["unbond_probability"])
    
    fname_traj = utils.get_fname(dir_sys, name_sys, 'trajectory')
    
    # number of grid point types
    n_types = len(bond_possible)
    
    # grid lengths 
    l_x = initial_positions.shape[0]
    l_y = initial_positions.shape[1]
    
    # keep track of particles that can/cannot move to decrease runtime
    move_possible = np.zeros(n_types, dtype=bool)
    for k in range(n_types):
        move_possible[k] = np.any(bond_possible[k]) or np.any(unbond_possible[k])
    
    
    # NOTE could change to something like a trajectory chunk that is written to file every x steps
    # current frame of the trajectory
    pos = np.array(initial_positions, dtype=np.int8)

    # keep track of which particles already moved
    pos_moved = np.zeros_like(initial_positions, dtype=bool) 
    
    # move probability array
    move_prob = np.zeros(5)
    
    # neighbour array for certain particle
    neighbours = np.zeros(4, dtype=int) # +x, -x, +y, -y
    
    # keep track of which moves are allowed
    allowed_moves = np.zeros(4, dtype=bool)
    allowed_bonds = np.zeros(4, dtype=bool)
    allowed_unbonds = np.zeros(4, dtype=bool)
    
    # create trajectory file
    fh5 = h5py.File(fname_traj, 'w')
    traj_h5 = fh5.create_dataset('trajectory', shape=(num_steps//stride+1, l_x, l_y), dtype=np.int8)
    
    # copy initial positions to trajectory
    traj_h5[0] = np.copy(pos)
    
    # start simulation
    for step in tqdm(range(1, num_steps+1)):                
        # no particle moved yet
        pos_moved.fill(False)    
        
        # loop through all particles and sites
        for idx_x in range(l_x):
            for idx_y in range(l_y):
                tag = pos[idx_x, idx_y]

                # check if site is not empty and if it has not moved yet
                if move_possible[tag] and not pos_moved[idx_x, idx_y]:
                    # get neighbours and allowed moves (allowed_moves is set to true in the function at the beginning)                                    
                    neighbours, neighbour_idx, allowed_moves, allowed_bonds, allowed_unbonds = _get_neighbours(
                        pos, 
                        pos_moved, 
                        idx_x, 
                        idx_y, 
                        neighbours, 
                        allowed_moves, 
                        allowed_bonds, 
                        allowed_unbonds,
                        bond_possible,
                        unbond_possible,
                        reflection=reflection, 
                        l_x=l_x, 
                        l_y=l_y
                    )

                    # update transition probability
                    
                    # get current neighbour sites, where the particle is allowed to move to
                    #current_neighbours = neighbours[allowed_moves]
                    
                    moved_flag = False
                    
                    if np.any(allowed_bonds):
                        # loop through all possible bonds
                        for idx_move, neighbour_tag in enumerate(neighbours):
                            move_prob[idx_move] = bond_probability[tag, neighbour_tag]*allowed_bonds[idx_move] 

                        # normalize probabilities
                        move_prob[4] = 1 - np.sum(move_prob[:4])

                        # choose new position
                        step_idx = np.random.choice(5, p=move_prob)
                        
                        if step_idx != 4:
                            moved_flag = True
                            
                            # tag of the bond partner
                            tag_partner = neighbours[step_idx]
                            new_tags = bond_idx[tag, tag_partner]
                                                
                    if not moved_flag and np.any(allowed_unbonds):
                        # loop through all unbonds
                        for idx_move, neighbour_tag in enumerate(neighbours):
                            move_prob[idx_move] = unbond_probability[tag, neighbour_tag]*allowed_unbonds[idx_move]                        
                        
                        # normalize probabilities
                        move_prob[4] = 1 - np.sum(move_prob[:4])

                        # choose new position
                        step_idx = np.random.choice(5, p=move_prob)
                        
                        if step_idx != 4:
                            moved_flag = True

                            # tag of the bond partner
                            tag_partner = neighbours[step_idx]
                            new_tags = unbond_idx[tag, tag_partner]
                        
                        
                    if moved_flag:
                        # idx of the new position
                        new_idx_x = neighbour_idx[step_idx][0]
                        new_idx_y = neighbour_idx[step_idx][1]

                        # update tags
                        pos[idx_x, idx_y] = new_tags[0]
                        pos[new_idx_x, new_idx_y] = new_tags[1]
                        
                        # update pos moved
                        pos_moved[idx_x, idx_y] = True
                        pos_moved[new_idx_x, new_idx_y] = True        

        # update trrajectory
        if step%stride == 0:
            traj_h5[step//stride] = pos
            
    fh5.close() 