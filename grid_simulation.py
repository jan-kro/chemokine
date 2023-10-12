import os
import h5py
import numpy as np
from datetime import datetime
from numba import njit
from tqdm import tqdm
import toml

def _get_neighbours(pos, pos_moved, i, j, neighbours, allowed_moves, transition_type, reflection_x: bool = True, l_x: int = None):
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

    right, left, up, down = i+1, i-1, j+1, j-1
    
    # check for neighbouring particles that already moved
    allowed_moves[[0,1]] = ~pos_moved[:, j].take([right, left], mode="wrap")
    allowed_moves[[2,3]] = ~pos_moved[i, :].take([up, down], mode="wrap")
    
    # get neighbours
    neighbours[[0,1]] = pos[:, j].take([right, left], mode="wrap") # +x, -x
    neighbours[[2,3]] = pos[i, :].take([up, down], mode="wrap") # +y, -y
    
    # check for neighbours it cannot bond to
    for i, nb in enumerate(neighbours):
        allowed_moves[i] = transition_type[tag, nb] != 0
    
    # check for walls
    if reflection_x:
        if i == 0:
            allowed_moves[1] = False # wall   
        elif i == l_x-1:
            allowed_moves[0] = False # wall  
            
    
    
    return neighbours, allowed_moves

def simulate(num_steps:               int  = None,
             initial_positions:       np.ndarray = None,
             move_probability:        np.ndarray = None, 
             binding_probability:     np.ndarray = None,
             reflection_x:            bool = True, 
             stride:                  int = 1,
             fname_traj:              str = None):
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
    
    
    Parameters
    ----------
    num_steps : int
        number of steps the simulation should run
    initial_positions : np.ndarray
        2d array of the initial postitions of the particles with shape (grid_size[0], grid_size[1])
        if entry is 0, the grid position is not occupied
    move_probability : list
        list of floats with len=9 determining the probability of a particle to make a step in x and y direction
        the first entry and the last 3 entries are 0, because the empty grid points and the collagen interaction 
        sites are not moving
    binding_probability : ndarray
        2d array of floats with shape (9,9) determining the probability of binding beween the different particle types
        example: bp[1, 2] is the probability, that particle type 1 binds to particle type 2
        the first row and column are 1, because every particle binds to an empty grid point
        the final tranision probability is the product of the diffusion probability and the binding probability
        the maximal possible value of this product has to be 1/4, otherwise there could arise cases where the sum of 
        transition probabilities is larger than 1
    reflection_x : bool, optional
        if True, the particles are reflected at the left and right wall, if False, they are wrapped around
    stride : int, optional
        the trajectory is saved every stride steps, by default 1
    fname_traj : str, optional
        path to the trajectory file, by default None
        if None, the trajectory is saved in a file with the current date and time in the filename
        the filename must have the ending .hdf5
    
    Returns
    -------
    trajectory: h5py.Dataset
        3d array of shape (num_steps//stride+1, grid_size[0], grid_size[1]) containing the trajectory of the particles
    """
    
    if fname_traj is None:
        now = datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
        fname_traj = os.path.join('trajectories', 'traj_' + now + '.hdf5')
    
    # if path to trajectory doesnt exist, create it
    if not os.path.exists(os.path.dirname(fname_traj)):
        os.makedirs(os.path.dirname(fname_traj))
    
    # get parent directory of trajectory file
    parent_dir = os.path.join(os.path.dirname(fname_traj), os.pardir)
    print(parent_dir)
    
    # get config directory, create if necessary
    config_dir = os.path.join(parent_dir, 'configs')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    # create config filename
    config_fname = os.path.join(config_dir, "cfg_"+os.path.basename(fname_traj)[:-5] + '.toml')
    print(config_fname)
    
    # save simulation parameters
    with open(config_fname, 'w') as f:
        toml.dump({
            'num_steps': num_steps,
            'stride': stride,
            'reflection_x': reflection_x,
            'fname_traj': fname_traj,
            'diffusion_probability': move_probability.tolist(),
            'binding_probability': binding_probability.tolist(),
            'initial_positions': initial_positions.tolist()
        }, f)
    
    # number of grid point types
    n_types = 9 
    
    # grid lengths 
    l_x = initial_positions.shape[0]
    l_y = initial_positions.shape[1]
    
    # possible steps for a particle
    step_choice = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]]) # +x, -x, +y, -y, stay
    
    # TRANSITION TYPES
    # ~~~~~~~~~~
    
    # 0: no transition allowed
    # 1: diffusions
    # 2: bonding of two particles
    # 3: switching of bond partners
    # 4: unbonding of two particles (unused)
    
    transition_type = np.zeros((n_types, n_types), dtype=int)
    
    # diffusion of two particles
    transition_type[1,0] = 1 # chemokine
    transition_type[2,0] = 1 # netrin
    transition_type[3,0] = 1 # heparansulfate
    transition_type[4,0] = 1 # chemokine-netrin
    transition_type[5,0] = 1 # chemokine-heparansulfate
    
    # bonding of two particles
    transition_type[1,2] = 2 # chemokine + netrin -> chemokine-netrin complex
    transition_type[2,1] = 2
    transition_type[1,3] = 2 # chemokine + heparansulfate -> chemokine-heparansulfate complex 
    transition_type[3,1] = 2
    transition_type[4,6] = 2 # chemokine-netrin + collagen_site -> chemokine-netrin-collagen_site 
    transition_type[2,6] = 2 # netrin + collagen_site -> chemokine-heparansulfate-collagen_site
    
    # changing bond partners
    transition_type[4,3] = 3
    transition_type[5,2] = 3
    transition_type[8,1] = 3
    
    # transition type (2)
    
    # bonding indices
    bond_idx = np.zeros((n_types, n_types), dtype=int)
    bond_idx[1,2] = 4 # chemokine + netrin -> chemokine-netrin complex
    bond_idx[2,1] = 4
    bond_idx[1,3] = 5 # chemokine + heparansulfate -> chemokine-heparansulfate complex 
    bond_idx[3,1] = 5
    bond_idx[4,6] = 7 # chemokine-netrin + collagen_site -> chemokine-netrin-collagen_site 
    bond_idx[2,6] = 8 # netrin + collagen_site -> chemokine-heparansulfate-collagen_site
    
    # transition type (3)
    
    # changing bond partners
    bond_switch_idx = np.zeros((n_types, n_types, 2), dtype=int)
    bond_switch_idx[4,3] = [2,5] # chemokine-netrin + heparansulfate -> netrin + chemokine-heparansulfate
    bond_switch_idx[5,2] = [3,4] # chemokine-heparansulfate + netrin -> heperansulfate + chemokine-netrin
    bond_switch_idx[8,1] = [6,4] # netrin-collagen_site + chemokine -> collagen_site + chemokine-netrin
    
    bond_switch_probability = np.zeros((n_types, n_types))
    bond_switch_probability[4,3] = binding_probability[1,3]/binding_probability[1,2] # chemokine-netrin + heparansulfate -> netrin + chemokine-heparansulfate
    bond_switch_probability[5,2] = binding_probability[1,2]/binding_probability[1,3] # chemokine-heparansulfate + netrin -> heperansulfate + chemokine-netrin
    bond_switch_probability[8,1] = binding_probability[1,2]/binding_probability[2,6] # netrin-collagen_site + chemokine -> collagen_site + chemokine-netrin
    
    # transition type (4)
    
    # unbonding  
    # unbond_idx = np.zeros((n_types, 2), dtype=int)
    # unbond_idx[4] = [1, 2] # chemokine-netrin -> chemokine + netrin
    # unbond_idx[5] = [1, 3] # chemokine-heparansulfate -> chemokine + heparansulfate
    # unbond_idx[7] = [4, 6] # chemokine-netrin-collagen_site -> chemokine-netrin + collagen_site
    # unbond_idx[8] = [2, 6] # netrin-collagen_site -> netrin + collagen_site
    
    
    # create move probabilities array and validate probability input
    move_probabilities = np.zeros((n_types, 5))
    for i in range(n_types):
        assert 4*move_probability[i] <= 1, 'Move probabilities too big'
        move_probabilities[i, :4] = move_probability[i]
        move_probabilities[i, 4] = 1 - 4*move_probability[i]
    
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
    
    # index array for moves
    idx_moves = np.arange(4)
    
    
    fh5 = h5py.File(fname_traj, 'w')
    traj_h5 = fh5.create_dataset('trajectory', shape=(num_steps//stride+1, l_x, l_y), dtype=np.int8)
    
    # copy initial positions to trajectory
    traj_h5[0] = np.copy(pos)
    
    for step in tqdm(range(1, num_steps+1)):                
        # no particle moved yet
        pos_moved.fill(False)    
        
        # loop through all particles and sites
        for idx_x in range(l_x):
            for idx_y in range(l_y):
                tag = pos[idx_x, idx_y]

                # check if site is not empty and if it has not moved yet
                if (tag != 0) and (pos_moved[idx_x, idx_y] == False):
                    # get neighbours and allowed moves (allowed_moves is set to true in the function at the beginning)                                    
                    neighbours, allowed_moves = _get_neighbours(pos, pos_moved, idx_x, idx_y, neighbours, allowed_moves, transition_type, reflection_x=reflection_x, l_x=l_x)
                    
                    # update transition probability
                    
                    # get current neighbour sites, where the particle is allowed to move to
                    current_neighbours = neighbours[allowed_moves]
                    
                    if np.all(current_neighbours == 0):
                        # if particle has no neighbours its probability to move is the probability arising from the diffusion
                        move_prob[:4] = move_probabilities[tag][:4]*allowed_moves
                        move_prob[4] = 1 - np.sum(move_prob[:4])
                        
                    elif np.any(allowed_moves):
                        # if the particle has neighbours which it can bond to it will bond
                        # the probability to do so is 1 for the case of a single neighbour
                        # for multiple neighbours it is calculated by the ratio of the different bond probabilities
                        
                        # fill all probabilities to 0 where there are no allowed moves, aswell as the stay probability
                        move_prob.fill(0)
                        
                        # loop through all possible moves
                        for idx_move, neighbour_tag in zip(idx_moves[allowed_moves], current_neighbours):
                            # NOTE the if statements are ordered in most probable frequency of occurence
                            if transition_type[tag, neighbour_tag] == 1:
                                # diffusion is negligable -> prabability is 0
                                pass
                            
                            elif transition_type[tag, neighbour_tag] == 3:
                                # switching bond partners case
                                move_prob[idx_move] = bond_switch_probability[tag, neighbour_tag]
                            
                            elif transition_type[tag, neighbour_tag] == 2:
                                # bonding case
                                move_prob[idx_move] = binding_probability[tag, neighbour_tag]
                            
                            # else:
                            #     # unbonding case
                            #     pass
                                
                        # normalize probabilities
                        move_prob /= np.sum(move_prob)
                    
                    else:
                        # particle is trapped and cannot move
                        move_prob.fill(0)
                        move_prob[4] = 1
                        # update pos moved
                        pos_moved[idx_x, idx_y] = True
                        
                    # choose new position
                    step_idx = np.random.choice(5, p=move_prob)
                                        
                    # if particle is shifted:
                    #  - wrap new indices
                    #  - check type of transition
                    #  - update positions 
                    #  - mark particle as moved
                    if step_idx != 4:
                        # new positions
                        new_idx_x = idx_x + step_choice[step_idx, 0]
                        new_idx_y = idx_y + step_choice[step_idx, 1]
                        
                        # wrap
                        new_idx_x = new_idx_x%l_x
                        new_idx_y = new_idx_y%l_y
                        
                        # check what type of grid point particle moved to
                        tag_2 = pos[new_idx_x, new_idx_y]
                        
                        # update tags
                        if transition_type[tag, tag_2] == 1:
                            # particle diffused
                            pos[idx_x, idx_y] = 0
                            pos[new_idx_x, new_idx_y] = tag
                                
                        elif transition_type[tag, tag_2] == 2:
                            # bonding case
                            pos[idx_x, idx_y] = 0
                            pos[new_idx_x, new_idx_y] = bond_idx[tag, tag_2]
                            
                        elif transition_type[tag, tag_2] == 3:
                            # switching bond partners case
                            pos[idx_x, idx_y] = bond_switch_idx[tag, tag_2, 0]
                            pos[new_idx_x, new_idx_y] = bond_switch_idx[tag, tag_2, 1]
                            # update pos moved
                            pos_moved[idx_x, idx_y] = True
                        else:
                            # unbonding case
                            pass          
                
                        # mark second particle as moved
                        pos_moved[new_idx_x, new_idx_y] = True

        # update trrajectory
        if step%stride == 0:
            traj_h5[step//stride] = pos
            
    fh5.close() 