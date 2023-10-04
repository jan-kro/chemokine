import os
import h5py
import numpy as np
from datetime import datetime
from numba import njit
from tqdm import tqdm

"""
GENERAL IDEA

Hab mir das jetzt so gedacht: Wir haben 3 verschiedene typen von particles und deren Anzahl
ändert sich ja im laufe der simulation. Auch die gesamtanzahl der particles ändert sich.
Wenn ein particle vom typ 0 und und einer von typ 1 zusammenstoßen wergibt sich nur ein 
particle von typ 2.

Habe jetzt mal allen fünf typen von particles ihre eigene trajectory gegeben. Da ja alles auf 
einem Grid passiert, kann man die coords in integers speichern. hab jetzt bewusst alles in positiven 
integers gespeichert, und den teil der trajectories der nicht belegt ist mit -10 gefüllt.

Damit kann man vorher schon die arrays definieren und muss nicht irgendwie reshapen, was wahrscheinlich 
sowieso nicht wirklich funktionieren würde. 

Den monte calro step an sich hab ich noch nicht geschriben, hab aber das wichtigste dazu unten kommentiert
"""

# TODO (maybe): write a _get_neighbours function that returns the neighbour tags 

def _get_transition_prob(pos, i, j, transition_prob, diffusion_probability, binding_probability, reflection_x: bool = True, l_x: int = None):
    """
    Calculates the 5 tranistion probabilities [+x -x +y -y stay] for the particle pos_idx in pos.
    Periodic boundaries are taken into account and an optional reflective boundary in x direction.
    The final transition probability is the product of the diffusion probability and the binding probability.
    
    Parameters
    ----------
    pos : np.ndarray
        2d array of the positions of the particles with shape (grid_size[0], grid_size[1])
    i, j : int
        indices of the particle in pos. pos[i, j] = particle tag
    transition_prob : np.ndarray
        1d array of len=5 to store the transition probabilities in the following order: [+x -x +y -y stay]
    diffusion_probability : list
        list of floats with len=9 determining the diffusivity of the particle in x and y direction
        the first and last entry are 0, because the empty grid points and the collagen interaction sites are not moving
        the sum of the entries must be 1
    binding_probability : ndarray
        2d array of floats with shape (9,9) determining the probability of binding beween the different particle types
        example: bp[1, 2] is the probability, that particle type 1 binds to particle type 2
        the first row and column are 1, because every particle binds to an empty grid point
        the final tranision probability is the product of the diffusion probability and the binding probability
        the maximal possible value of this product has to be 1/4, otherwise there could arise cases where the sum of 
        transition probabilities is larger than 1
    reflection_x : bool, optional
        if True, the particles are reflected at the left and right wall, if False, they are wrapped around
    l_x : int, optional
        length of the grid in x direction, by default None
        only used if reflection_x is True
        
    Returns
    -------
    transition_prob : np.ndarray
        1d array of len=5 with the transition probabilities in the following order: [+x -x +y -y stay]
    """
    
    tag = pos[i, j]
    
    if reflection_x:
        if i == 0:
            transition_prob[0] = binding_probability[tag, pos[:, j].take(i+1, mode="wrap")] # +x
            transition_prob[1] = 0 # wall                                                   # -x    
        elif i == l_x-1:
            transition_prob[0] = 0 # wall                                                   # +x
            transition_prob[1] = binding_probability[tag, pos[:, j].take(i-1, mode="wrap")] # -x
        else:
            transition_prob[[0,1]] = binding_probability[[tag, tag], pos[:, j].take([i+1, i-1], mode="wrap")] # +x, -x
        
        transition_prob[[2,3]] = binding_probability[[tag, tag], pos[i, :].take([j+1, j-1], mode="wrap")] # +y, -y
    else:  
        transition_prob[[0,1]] = binding_probability[[tag, tag], pos[:, j].take([i+1, i-1], mode="wrap")] # +x, -x
        transition_prob[[2,3]] = binding_probability[[tag, tag], pos[i, :].take([j+1, j-1], mode="wrap")] # +y, -y
    
    transition_prob *= diffusion_probability    
    transition_prob[4] = 1 - np.sum(transition_prob[:4]) 
    
    return transition_prob

def _get_unbonding_prob(pos, idx_x, idx_y, tag_move, tag_stay, neighbours, transition_prob, diffusion_probability_move, binding_probability, reflection_x: bool = True, l_x: int = None):
    '''
    Calculates the 5 tranistion probabilities [+x -x +y -y stay] for an unbonding complex.
    Periodic boundaries are taken into account and an optional reflective boundary in x direction.
    
    Parameters
    ----------
    pos : np.ndarray
        2d array of the positions of the particles with shape (grid_size[0], grid_size[1])
    idx_x, idx_y : int
        indices of the particle-complex in pos. pos[ix, iy] = tag of the complex
    tag_move : int
        tag of the particle that is moving
    tag_stay : int
        tag of the particle that stays
    neighbours : np.ndarray
        1d array of len=4 to store the neighbours of the particle in the following order: [+x -x +y -y]
    transition_prob : np.ndarray
        1d array of len=5 to store the transition probabilities in the following order: [+x -x +y -y stay]
    diffusion_probability_move : float
        diffusion probability of the moving particle
    binding_probability : ndarray
        2d array of floats with shape (9,9) determining the probability of binding beween the different particle types
        example: bp[1, 2] is the probability, that particle type 1 binds to particle type 2
        the first row and column are 1, because every particle binds to an empty grid point
        the final tranision probability is the product of the diffusion probability and the inverse binding probability
    reflection_x : bool, optional
        if True, the particles are reflected at the left and right wall, if False, they are wrapped around
    l_x : int, optional
        length of the grid in x direction, by default None
        
    Returns
    -------
    transition_prob : np.ndarray
        1d array of len=5 with the transition probabilities for the moving particle in the following order: [+x -x +y -y stay]
    '''
    
    binding_probability = binding_probability[tag_move, tag_stay]
    # check for (un)allowed moves
    if reflection_x:
        if idx_x == 0:
            neighbours[0] = pos[:, idx_y].take(idx_x+1, mode="wrap") # +x
            neighbours[1] = 1 # wall                                                   # -x    
        elif idx_x == l_x-1:
            neighbours[0] = 1 # wall                                                   # +x
            neighbours[1] = pos[:, idx_y].take(idx_x-1, mode="wrap") # -x
        else:
            neighbours[[0,1]] = pos[:, idx_y].take([idx_x+1, idx_x-1], mode="wrap") # +x, -x
        
        neighbours[[2,3]] =  pos[idx_x, :].take([idx_y+1, idx_y-1], mode="wrap") # +y, -y
    else:  
        neighbours[[0,1]] = pos[:, idx_y].take([idx_x+1, idx_x-1], mode="wrap") # +x, -x
        neighbours[[2,3]] =  pos[idx_x, :].take([idx_y+1, idx_y-1], mode="wrap") # +y, -y
    
    for k, nb in enumerate(neighbours):
        if nb==0:
            transition_prob[k] = diffusion_probability_move/binding_probability
        else:
            transition_prob[k] = 0
    
    #! CHOOSE BINDING AND DIFFUSION PROBABILITIES SUCH THAT THE MAXIMUM OF THE PRODUCT IS 1/4
    # TODO remove this, once fixed
    if np.sum(transition_prob[:4])>1:
        transition_prob[:4] = transition_prob[:4]/np.max(transition_prob[:4])*0.25
    
    # normalize transition probability
    transition_prob[4] = 1 - np.sum(transition_prob[:4])
    
    return transition_prob
    
                
def monte_carlo_simulation(num_steps:               int  = None,
                           initial_positions:       np.ndarray = None,
                           diffusion_probability:   np.ndarray = None, 
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
    diffusion_probability : list
        list of floats with len=9 determining the diffusivity of the particle in x and y direction
        the first and last entry are 0, because the empty grid points and the collagen interaction sites are not moving
        the sum of the entries must be 1
    binding_probability : ndarray
        2d array of floats with shape (9,9) determining the probability of binding beween the different particle types
        example: bp[1, 2] is the probability, that particle type 1 binds to particle type 2
        the first row and column are 1, because every particle binds to an empty grid point
        the final tranision probability is the product of the diffusion probability and the binding probability
        the maximal possible value of this product has to be 1/4, otherwise there could arise cases where the sum of 
        transition probabilities is larger than 1
    reflection : bool, optional
        if True, the particles are reflected at the left and right wall, if False, they are wrapped around
    
    Returns
    -------
    trajectory: ndarray
        3d array of shape (num_steps//stride+1, grid_size[0], grid_size[1]) containing the trajectory of the particles
    """
    
    if fname_traj is None:
        now = datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
        fname_traj = os.path.join('trajectories', 'traj_' + now + '.hdf5')
    
    # TODO write temporary functions for simulation testing:
    # check if number of particles stays the same
    # check if number of particles of each type stays the same
    
    n_types = 9 # number of grid point types

    # NOTE unused
    particle_tags = np.array([1, 2, 3, 4, 5]) # 1 = chemokine, 2 = netrin, ...
    
    complex_tags = np.array([4, 5, 7, 8]) # 4 = chemokine-netrin, 5 = chemokine-heparansulfate, ...
    
    # TODO remove after testing
    # save total number of particles
    n_initial = 0
    for i in [1, 2, 3, 6]:
        n_initial += np.sum(initial_positions == i)
    for i in [4, 5, 8]:
        n_initial += 2*np.sum(initial_positions == i)
    n_initial += 3*np.sum(initial_positions == 7)
    
    
    l_x = initial_positions.shape[0]
    l_y = initial_positions.shape[1]
    
    # possible steps for a particle
    step_choice = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]]) # +x, -x, +y, -y, stay
    
    # bond types
    bond_type_idx = np.zeros((n_types, n_types), dtype=int)
    bond_type_idx[1, 2] = 4 # chemokine + netrin -> chemokine-netrin complex
    bond_type_idx[2, 1] = 4
    bond_type_idx[1, 3] = 5 # chemokine + heparansulfate -> chemokine-heparansulfate complex 
    bond_type_idx[3, 1] = 5
    bond_type_idx[4, 6] = 7 # chemokine-netrin + collagen_site -> chemokine-netrin-collagen_site
    bond_type_idx[6, 4] = 7 # 
    bond_type_idx[2, 6] = 8 # netrin + collagen_site -> chemokine-heparansulfate-collagen_site
    bond_type_idx[6, 2] = 8 # 
    
    unbond_type_idx = np.zeros((n_types, 2), dtype=int)
    unbond_type_idx[4] = [1, 2] # chemokine-netrin -> chemokine + netrin
    unbond_type_idx[5] = [1, 3] # chemokine-heparansulfate -> chemokine + heparansulfate
    #! in the below case, is it possible for only chemokine to hop off?
    unbond_type_idx[7] = [4, 6] # chemokine-netrin-collagen_site -> chemokine-netrin + collagen_site
    unbond_type_idx[8] = [2, 6] # netrin-collagen_site -> netrin + collagen_site
    
    # create trajectory array
    #trajectory = np.zeros((num_steps, l_x, l_y), dtype=int)
    
    # first frame are the initial positions
    #trajectory[0] = np.copy(initial_positions)
    
    # trajectory
    pos = np.copy(initial_positions)

    # keep track of which particles already moved
    pos_moved = np.zeros_like(initial_positions, dtype=bool) 
    
    # transition probability array
    transition_prob = np.zeros(5)
    
    # neighbour array for certain particle
    neighbours = np.zeros(4, dtype=int) # +x, -x, +y, -y
    
    
    fh5 = h5py.File(fname_traj, 'w')
    traj_h5 = fh5.create_dataset('trajectory', shape=(num_steps//stride+1, l_x, l_y), dtype=int)
    traj_h5[0] = np.copy(pos)
    
    for step in tqdm(range(1, num_steps)):
        
        # update trajectory
        # trajectory[step] = np.copy(trajectory[step-1])
                
        # no particle moved yet
        pos_moved.fill(False)    
        
        # loop through all particles and sites
        for idx_x in range(l_x):
            for idx_y in range(l_y):
                # tag = trajectory[step, idx_x, idx_y]
                tag = pos[idx_x, idx_y]

                # check if site is not empty and if it has not moved yet
                if (tag != 0) and (pos_moved[idx_x, idx_y] == False):
                    # get transition probabilities
                    # transition_prob = _get_transition_prob(trajectory[step], 
                    #                                        idx_x, 
                    #                                        idx_y, 
                    #                                        transition_prob, 
                    #                                        diffusion_probability[tag], 
                    #                                        binding_probability, 
                    #                                        reflection_x=reflection_x, 
                    #                                        l_x=l_x)
                    transition_prob = _get_transition_prob(pos, 
                                                           idx_x, 
                                                           idx_y, 
                                                           transition_prob, 
                                                           diffusion_probability[tag], 
                                                           binding_probability, 
                                                           reflection_x=reflection_x, 
                                                           l_x=l_x)
                    
                    # choose new position
                    step_idx = np.random.choice(5, p=transition_prob)
                    new_idx_x = idx_x + step_choice[step_idx, 0]
                    new_idx_y = idx_y + step_choice[step_idx, 1]
                                        
                    # if particle is shifted:
                    #  - wrap new indices
                    #  - check if bonded
                    #  - update positions 
                    #  - mark particle as moved
                    if step_idx != 4:
                        # wrap
                        new_idx_x = new_idx_x%l_x
                        new_idx_y = new_idx_y%l_y
                        
                        # check what type of grid point particle moved to
                        tag_2 = pos[new_idx_x, new_idx_y]
                        
                        # if it is not an empty grid point a bond happend and particle types need to be updated
                        if tag_2 != 0:
                            # particle bonded
                            # get tag of new particle
                            tag_new = bond_type_idx[tag, tag_2]
                        else:
                            # particle moved to an empty grid point
                            tag_new = tag
                            
                        # update trajectory
                        pos[idx_x, idx_y] = 0
                        pos[new_idx_x, new_idx_y] = tag_new                
                
                        # mark particle as moved
                        pos_moved[new_idx_x, new_idx_y] = True
                        
                    # if not shifted, check for possibility of unbonding
                    else:
                        if tag in complex_tags:
                            # choose wich particle gets moved first
                            tag_1st, tag_2nd = unbond_type_idx[tag][np.random.choice(2, size=2, replace=False)]
                            
                            for tag_move, tag_stay in zip([tag_1st, tag_2nd], [tag_2nd, tag_1st]):
                                if not pos_moved[idx_x, idx_y]:   
                                    # get transition probabilities
                                    # unbonding_prob = _get_unbonding_prob(trajectory[step], 
                                    #                                      idx_x, 
                                    #                                      idx_y, 
                                    #                                      tag_move, 
                                    #                                      tag_stay, 
                                    #                                      neighbours, 
                                    #                                      transition_prob, 
                                    #                                      diffusion_probability[tag_move], 
                                    #                                      binding_probability, 
                                    #                                      reflection_x = reflection_x, 
                                    #                                      l_x = l_x)
                                    unbonding_prob = _get_unbonding_prob(pos, 
                                                                         idx_x, 
                                                                         idx_y, 
                                                                         tag_move, 
                                                                         tag_stay, 
                                                                         neighbours, 
                                                                         transition_prob, 
                                                                         diffusion_probability[tag_move], 
                                                                         binding_probability, 
                                                                         reflection_x = reflection_x, 
                                                                         l_x = l_x)
                            
                                    # choose new position
                                    step_idx = np.random.choice(5, p=unbonding_prob)
                                    
                                    # if particle is shifted:
                                    #  - wrap new indices
                                    #  - update positions
                                    #  - mark particle as moved
                                    if step_idx != 4:
                                        # update position and wrap around
                                        new_idx_x = (idx_x + step_choice[step_idx, 0])%l_x
                                        new_idx_y = (idx_y + step_choice[step_idx, 1])%l_y
                                        
                                        # update trajectory
                                        pos[idx_x, idx_y] = tag_stay # leave 2nd particle on same spot
                                        pos[new_idx_x, new_idx_y] = tag_move # move 1st particle to new spot
                                        
                                        # mark particles as moved
                                        pos_moved[idx_x, idx_y] = True
                                        pos_moved[new_idx_x, new_idx_y] = True

        # update trrajectory
        if step%stride == 0:
            traj_h5[step//stride] = np.copy(pos)
            
            # check if number of particles stays the same
            n_current = 0
            for i in [1, 2, 3, 6]:
                n_current += np.sum(pos == i)
            for i in [4, 5, 8]:
                n_current += 2*np.sum(pos == i)
            n_current += 3*np.sum(pos == 7)
            
            assert n_current == n_initial, f'Assertion error at step {step}\nNumber of particles changed: current = {n_current} vs initial = {n_initial}'
            
        
    fh5.close()
    #return trajectory
    #return                