import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit


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

@njit
def _monte_carlo_step():
    # TODO
    return

#@njit still not working
def _check_all(pos1, pos2, n1, n2):
    """
    Checks if particles are bonded (= on the same positions) and if particles are neighbours (= distance 1 to each other)
    
    Parameters
    ----------
    pos1 : ndarray
        2d array of the positions of the first particle type
    pos2 : ndarray
        2d array of the positions of the second particle type
    n1 : int
        number of particles of the first type
    n2 : int
        number of particles of the second type
        
    Returns
    -------
    bonded_idx : ndarray
        2d array of the indices of the bonded particles. Entries are [[bond1_type0, bond1_type1],
                                                                     [bond2_type0, bond2_type1],
                                                                               . . .          ]] 
    neighbour_idx : ndarray
        2d array of the indices of the neighbouring particles    
    """
    
    
    # NOTE: idk ob es effizienter ist die arrays in einer gewissen größe vorzudefinieren und dann den ungefüllten
    #       part wegzuslicen, oder ob append ok ist
    
    
    # initialize arrays
    
    # either the smaller amount of particles has 4 neighbours for every particle, or the bigger amount is the limit 
    # BUT its probably cheaper to always create the bigger array if the order of magnitude is similar (idk though)
    n_min = min(n1, n2)
    
    bonded_idx =  np.ones((n_min, 2), dtype=int)
    neighbour_idx =  np.ones((n_min*4, 2), dtype=int) 
    
    n_bonded = 0
    n_neighbour = 0
    
    for i in range(n1):
        for j in range(n2):
            dvec_x = pos2[j, 0] - pos1[i, 0]
            dvec_y = pos2[j, 1] - pos1[i, 1]
            
            # if the distance of both particles is 0, they are bonded
            if dvec_y == 0 and dvec_x == 0:
                bonded_idx[n_bonded] = [i,j]
                n_bonded += 1
            
            # if distance is 1, they are neighbours
            elif abs(dvec_x) + abs(dvec_y) == 1: 
                neighbour_idx[n_neighbour] = [i,j]
                n_neighbour += 1
    
    # if an atom is bonded it cannot be a neighbour, so remove it from the neighbour list 
       
    # Remove entries in idx_neighbours where the first element is in bond_elements
    # if neither particles of type 0 nor particles of type 1 are involven in a bond, they can be neighbours           
    neighbour_idx = [x for x in neighbour_idx if (x[0] not in bonded_idx[:, 0]) and (x[1] not in bonded_idx[:, 1]) and x[0] != -1]        
    
    return np.array(bonded_idx[:n_bonded], ndmin=2), np.array(neighbour_idx, ndmin=2)
                

def monte_carlo_simulation(num_steps:               int  = 1000,
                           grid_size:               list = [100, 50], 
                           particle_diffusivity:    list = [1., 1., 0.5, 0.2, 0.1], 
                           binding_probability:     list = [[0, 0.5], [0.5, 0]],
                           c_heparansulfate:        float = 0.005,
                           initial_positions:       np.ndarray = None,
                           particle_types:          list = None,
                           n_particles:             int  = 100, 
                           fraction_x:              float = 0.2,
                           collagen_y:              int   = None,
                           reflection:              bool = True):
    """
    monte carlo simulation of binding particles on a grid
    
    there are 5 particles types (0, 1, 2, 3, 4):
    0: chemokine
    1: netrin
    2: heparansulfate
    3: chemokine-heparansulfate complex
    4: chemokine-netrin complex
    
    Parameters
    ----------
    grid_size : list
        list of integers determining the length of the grid in x and y direction
    num_steps : int
        number of steps the simulation should run
    particle_diffusivity : list
        list of floats determining the diffusivity of the particle in x and y direction
        the first two entries are the diffusivity of particle types
    binding_probability : list
        2d list of floats determining the probability of binding beween the different particle types
        example: bp[0,1] is the probability, that particle type 1 binds to particle type 2
    c_heparansulfate: float
        concentration (fraction of total gridpoints over number of heparansulfates) 
        positions are drawn uniformly
    initial_positions : np.ndarray, optional
        2d list of the initial postitions of the chemokine (type 0) and netrin (type 1) particles shape (n0+n1, 2)
        if None, the particles are placed randomly on the grid
    particle_types : list, optional
        list of the particle types on length n0+n1
        if None the particle types are randomly assigned with a 50% probability to be of type 0 or 1
    n_particles : int
        if initial_positions is None, this determines the number of particles of each type.
        if initial positions are given, this is ignored
    fraction_x : float
        fraction of the grid in x direction, where the particles are placed, to create a gradient
        only used if initial_positions is None
    collagen_y : int, optional
        y-value of the collagen fiber, by default center of the grid
    reflection : bool, optional
        if True, the particles are reflected at the left and right wall, if False, they are wrapped around
    
    Returns
    -------
    trajectory_chemokine: ndarray
        3d array of the trajectory of the chemokine particles
    trajectory_netrin: ndarray
        3d array of the trajectory of the netrin particles
    trajectory_heparansulfate: ndarray
        3d array of the trajectory of the heparansulfate particles
    trajectory_chemokine_netrin: ndarray
        3d array of the trajectory of the chemokine-netrin complexes
    trajectory_chemokine_heparansulfate: ndarray
        3d array of the trajectory of the chemokine-heparansulfate complexes
    """
    
    
    
    # create initial positions of particles, with the condition that the do not overlap
    if initial_positions is None:
        # Initialize the particle positions randomly within the grid
        initial_positions_x = np.random.randint(0, int(grid_size[0]*fraction_x))
        initial_positions_y = np.random.randint(0, grid_size[1])
        initial_positions = np.array([initial_positions_x, initial_positions_y], ndmin=2)
        for i in range(n_particles-1):
            initial_positions_x = np.random.randint(0, int(grid_size[0]*fraction_x))
            initial_positions_y = np.random.randint(0, grid_size[1])
            new_init_pos = np.array([initial_positions_x, initial_positions_y], ndmin=2)
            maxit_init = 10000
            while np.any(np.all(new_init_pos == initial_positions, axis=1)):
                initial_positions_x = np.random.randint(0, int(grid_size[0]*fraction_x))
                initial_positions_y = np.random.randint(0, grid_size[1])
                new_init_pos = np.array([initial_positions_x, initial_positions_y], ndmin=2)
                if maxit_init == 0:
                    raise ValueError("Initial particle positions. the grid may be too small")
                maxit_init -= 1
            initial_positions = np.append(initial_positions, new_init_pos, axis=0)
    
    # if the particle types are not specified, randomly assign them (no bonded particles at the beginning)
    if particle_types is None:
        particle_types = np.random.randint(0, 2, size=(len(initial_positions)))
    
    # validate input
    if len(initial_positions) != len(particle_types):
        raise ValueError("Number of initial positions and particle types do not match")
    
    # if y value of the collagen fiber is not given, choose the middle of the grid
    if collagen_y is None:
        collagen_y = np.round(grid_size[1]/2)
    
    # TODO implement proper probability calculations
    # probabilities is an array of shape (n_particle_types, 3)
    # with dimension 0 being the particle type
    # and dimension 1 being the step choice from which to draw for every direction (0, 1, -1)
    # example: 
    #   probabilities[1, 2] is the probability of the netrin particle move -1 step along the grid
    #   the direction (x/y) is not specified 
    probabilities = np.array([np.array([1,1,1,1,1]), particle_diffusivity, particle_diffusivity]).T
    probabilities = probabilities/np.sum(probabilities, axis=1, keepdims=True)
    step_choice = np.array([0, 1, -1])

    
    # number of particles of each type
    n_particles_chemokine                = np.sum(particle_types == 0, dtype=int)
    n_particles_netrin                   = np.sum(particle_types == 1, dtype=int)
    n_particles_heparansulfate           = int(c_heparansulfate * grid_size[0] * grid_size[1])
    n_particles_chemokine_netrin         = 0
    n_particles_chemokine_heparansulfate = 0
    
    # value to fill the trajectory with, if a particle is not present
    deleted_value = -10 
    
    # initialize trajectories 
    trajectory_chemokine                = deleted_value * np.ones((num_steps,   n_particles_chemokine, 2),      dtype=int) 
    trajectory_netrin                   = deleted_value * np.ones((num_steps,   n_particles_netrin, 2),         dtype=int) 
    trajectory_heparansulfate           = deleted_value * np.ones((num_steps+1, n_particles_heparansulfate, 2), dtype=int)
    trajectory_chemokine_netrin         = deleted_value * np.ones((num_steps+1, n_particles_chemokine_netrin + int(np.min([n_particles_chemokine, n_particles_netrin])),  2), dtype=int)
    trajectory_chemokine_heparansulfate = deleted_value * np.ones((num_steps+1, n_particles_chemokine_heparansulfate + int(np.min([n_particles_chemokine, n_particles_heparansulfate])), 2), dtype=int)
    
    # define initial positions 
    pos_chemokine = initial_positions[particle_types == 0]
    pos_netrin    = initial_positions[particle_types == 1]
    
    # create heparan sulfates
    pos_hep_x = np.random.randint(0, grid_size[0], size=n_particles_heparansulfate)
    pos_hep_y = np.random.randint(0, grid_size[1], size=n_particles_heparansulfate)
    pos_heparansulfate = np.array([pos_hep_x, pos_hep_y]).T
    
    # no complexes present at the beginning 
    # TODO change at some point
    pos_chemokine_netrin         = trajectory_chemokine_netrin[0] 
    pos_chemokine_heparansulfate = trajectory_chemokine_heparansulfate[0]
    
    # add initial positions to trajectories
    trajectory_chemokine[0, :n_particles_chemokine, :]                               = np.copy(pos_chemokine)
    trajectory_netrin[0, :n_particles_netrin, :]                                     = np.copy(pos_netrin)
    trajectory_heparansulfate[0, :n_particles_heparansulfate, :]                     = np.copy(pos_heparansulfate)
    trajectory_chemokine_netrin[0, :n_particles_chemokine_netrin, :]                 = pos_chemokine_netrin[:n_particles_chemokine_netrin]
    trajectory_chemokine_heparansulfate[0, :n_particles_chemokine_heparansulfate, :] = pos_chemokine_heparansulfate[:n_particles_chemokine_heparansulfate]
    
    # check if any chemokine-netrin complexes are bonded to collagen
    idx_fixed = np.where(pos_chemokine_netrin[:, 1] == collagen_y)[0]
    pos_fixed = pos_chemokine_netrin[idx_fixed]

    for step in range(1, num_steps):
        
        # TODO take care of possibility that the particle complex can split again 
        # maybe like:
        #
        # idx_unbonded_1 = [1, 4, 33, 86, ...] # only 1D
        #
        # if idx_unbonded_1.shape[1] > 0:
        #   n_unbonded_1 = len(idx_unbonded_1)
        #   trajectory_chemokine[step, n_particles_chemokine:n_particles_chemokine+n_unbonded_1, :] = pos_chemokine_netrin[idx_unbonded_1, :]
        #   trajectory_netrin[step, n_particles_netrin:n_particles_netrin+n_unbonded_1, :] = pos_chemokine_netrin[idx_unbonded_1, :]
        #   
        #  n_particles_chemokine_netrin -= n_unbonded_1
        #  n_particles_chemokine += n_unbonded_1
        #  n_particles_netrin += n_unbonded_1
        #
        #  trajectory_chemokine_netrin[step, :n_particles_chemokine_netrin, :] = np.delete(pos_chemokine_netrin, idx_unbonded_1, axis=0)[:n_particles_chemokine_netrin]
        #
        # ! IMPORTANT: if this is implemented the way of fixing particles to the fiber does not work anymore 
        
        # check if chemokine is bonded to netrin
        idx_bonded_1, idx_neighbour_1 = _check_all(pos_chemokine, pos_netrin, n_particles_chemokine, n_particles_netrin)
        n_bonded_1 = idx_bonded_1.shape[0]
        
        # add to bonded and remove from unbonded
        if n_bonded_1 > 0: # if particles are bonded
            
            # add to complex
            
            # add old positions to new step
            trajectory_chemokine_netrin[step, :n_particles_chemokine_netrin, :] = pos_chemokine_netrin[:n_particles_chemokine_netrin]
            
            # add new positions to new step
            trajectory_chemokine_netrin[step, n_particles_chemokine_netrin:n_particles_chemokine_netrin+n_bonded_1, :] = pos_chemokine[idx_bonded_1[:,0], :]

            # update number of particles
            n_particles_netrin           -= n_bonded_1
            n_particles_chemokine_netrin += n_bonded_1
            
            # remove from unbonded
            trajectory_netrin[step, :n_particles_netrin]       = np.delete(pos_netrin, idx_bonded_1[:,1],    axis=0)[:n_particles_netrin]
            
        else:
            # nothing happens to complex and netrin
            trajectory_netrin[step]           = np.copy(pos_netrin)
            trajectory_chemokine_netrin[step] = np.copy(pos_chemokine_netrin)
            
        # check if chemokine bonded to heparan sulfate
        idx_bonded_2, idx_neighbour_2 = _check_all(pos_chemokine, pos_heparansulfate, n_particles_chemokine, n_particles_heparansulfate)
        n_bonded_2 = idx_bonded_2.shape[0]
        
        if  n_bonded_2> 0: # if particles are bonded
            
            # add to bonded
            trajectory_chemokine_heparansulfate[step, :n_particles_chemokine_heparansulfate, :] = pos_chemokine_heparansulfate[:n_particles_chemokine_heparansulfate]
            trajectory_chemokine_heparansulfate[step, n_particles_chemokine_heparansulfate:n_particles_chemokine_heparansulfate+n_bonded_2, :] = pos_chemokine[idx_bonded_2[:,0], :]
            
            # update number of particles
            n_particles_heparansulfate -= n_bonded_2
            n_particles_chemokine_heparansulfate += n_bonded_2
            
            # remove from unbonded
            trajectory_heparansulfate[step, :n_particles_heparansulfate] = np.delete(pos_heparansulfate, idx_bonded_2[:,1], axis=0)[:n_particles_heparansulfate]
        else:
            # noting happens to complex and heparansulfate
            trajectory_heparansulfate[step] = np.copy(pos_heparansulfate)
            trajectory_chemokine_heparansulfate[step] = np.copy(pos_chemokine_heparansulfate)
        
        # check if chemokine bonds to both netrin and heparansulfate
        if n_bonded_1 > 0 and n_bonded_2 > 0:
            # to not overwrite particles this must be in one step 
            n_particles_chemokine -= n_bonded_1 + n_bonded_2
            trajectory_chemokine[step, :n_particles_chemokine] = np.delete(pos_chemokine, np.append(idx_bonded_1[:,0], idx_bonded_2[:,0]), axis=0)[:n_particles_chemokine]
            
        elif n_bonded_1 > 0:
            n_particles_chemokine -= n_bonded_1
            trajectory_chemokine[step, :n_particles_chemokine] = np.delete(pos_chemokine, idx_bonded_1[:,0], axis=0)[:n_particles_chemokine]
            
        elif n_bonded_2 > 0:
            n_particles_chemokine -= n_bonded_2
            trajectory_chemokine[step, :n_particles_chemokine] = np.delete(pos_chemokine, idx_bonded_2[:,0], axis=0)[:n_particles_chemokine]
        
        else:
            # nothing happens to chemokine
            trajectory_chemokine[step] = np.copy(pos_chemokine)
            
        # ~~~~~~~~~~~~~~~~~~~~~~~
        # PERFORM MONTE CARLO STEP
        # ~~~~~~~~~~~~~~~~~~~~~~~
        # - accept/reject step
        #   - accaptance probability differs for all particle types (= different mobility)
        #   - additionally, the probability changes, when two particles are next to each other (-> neigbours)
        #   - direction is weighted if there are neighbours
        # - check if move is allowed
        #   - are two particles on the same position? 
        #   - would they move outside the box?
        # - update positions
        
        # simulate mc step
        # TODO remove once mc is implemented
        trajectory_chemokine[step, :n_particles_chemokine]                               += np.random.choice(step_choice, size=(n_particles_chemokine, 2), p=probabilities[0])
        trajectory_netrin[step, :n_particles_netrin]                                     += np.random.choice(step_choice, size=(n_particles_netrin, 2), p=probabilities[1])
        trajectory_chemokine_netrin[step, :n_particles_chemokine_netrin]                 += np.random.choice(step_choice, size=(n_particles_chemokine_netrin, 2), p=probabilities[2])
        trajectory_heparansulfate[step, :n_particles_heparansulfate]                     += np.random.choice(step_choice, size=(n_particles_heparansulfate, 2), p=probabilities[3])
        trajectory_chemokine_heparansulfate[step, :n_particles_chemokine_heparansulfate] += np.random.choice(step_choice, size=(n_particles_chemokine_heparansulfate, 2), p=probabilities[4])  
        
        
        if reflection:
            # boundary condition with left and right reflection
            # if particle goes outside the box, it is reflected at the wall
            
            # left wall
            idx_left = np.where(trajectory_chemokine[step, :n_particles_chemokine, 0] < 0)[0]
            trajectory_chemokine[step, idx_left, 0] = -trajectory_chemokine[step, idx_left, 0]
            
            idx_left = np.where(trajectory_netrin[step, :n_particles_netrin, 0] < 0)[0]
            trajectory_netrin[step, idx_left, 0] = -trajectory_netrin[step, idx_left, 0]
            
            idx_left = np.where(trajectory_chemokine_netrin[step, :n_particles_chemokine_netrin, 0] < 0)[0]
            trajectory_chemokine_netrin[step, idx_left, 0] = -trajectory_chemokine_netrin[step, idx_left, 0]
            
            idx_left = np.where(trajectory_heparansulfate[step, :n_particles_heparansulfate, 0] < 0)[0]
            trajectory_heparansulfate[step, idx_left, 0] = -trajectory_heparansulfate[step, idx_left, 0]
            
            idx_left = np.where(trajectory_chemokine_heparansulfate[step, :n_particles_chemokine_heparansulfate, 0] < 0)[0]
            trajectory_chemokine_heparansulfate[step, idx_left, 0] = -trajectory_chemokine_heparansulfate[step, idx_left, 0]
            
            # right wall
            idx_right = np.where(trajectory_chemokine[step, :n_particles_chemokine, 0] > grid_size[0])[0]
            trajectory_chemokine[step, idx_right, 0] = 2*grid_size[0] - trajectory_chemokine[step, idx_right, 0]
            
            idx_right = np.where(trajectory_netrin[step, :n_particles_netrin, 0] > grid_size[0])[0]
            trajectory_netrin[step, idx_right, 0] = 2*grid_size[0] - trajectory_netrin[step, idx_right, 0]
            
            idx_right = np.where(trajectory_chemokine_netrin[step, :n_particles_chemokine_netrin, 0] > grid_size[0])[0]
            trajectory_chemokine_netrin[step, idx_right, 0] = 2*grid_size[0] - trajectory_chemokine_netrin[step, idx_right, 0]
            
            idx_right = np.where(trajectory_heparansulfate[step, :n_particles_heparansulfate, 0] > grid_size[0])[0]
            trajectory_heparansulfate[step, idx_right, 0] = 2*grid_size[0] - trajectory_heparansulfate[step, idx_right, 0]
            
            idx_right = np.where(trajectory_chemokine_heparansulfate[step, :n_particles_chemokine_heparansulfate, 0] > grid_size[0])[0]
            trajectory_chemokine_heparansulfate[step, idx_right, 0] = 2*grid_size[0] - trajectory_chemokine_heparansulfate[step, idx_right, 0]
        
        
        # pbc
        for i in range(2):
            trajectory_chemokine[step, :n_particles_chemokine, i] = np.mod(trajectory_chemokine[step, :n_particles_chemokine, i], grid_size[i])
            trajectory_netrin[step, :n_particles_netrin, i] = np.mod(trajectory_netrin[step, :n_particles_netrin, i], grid_size[i])
            trajectory_chemokine_netrin[step, :n_particles_chemokine_netrin, i] = np.mod(trajectory_chemokine_netrin[step, :n_particles_chemokine_netrin, i], grid_size[i])
            trajectory_heparansulfate[step, :n_particles_heparansulfate, i] = np.mod(trajectory_heparansulfate[step, :n_particles_heparansulfate, i], grid_size[i])
            trajectory_chemokine_heparansulfate[step, :n_particles_chemokine_heparansulfate, i] = np.mod(trajectory_chemokine_heparansulfate[step, :n_particles_chemokine_heparansulfate, i], grid_size[i])

        
        
        # do not move fixed particles
        # TODO this falls apart once the particles are allowed to split again
        if idx_fixed.shape[0] > 0:
            trajectory_chemokine_netrin[step, idx_fixed] = np.copy(pos_fixed)
        
        # save new frame to pos
        pos_chemokine = np.copy(trajectory_chemokine[step])
        pos_netrin = np.copy(trajectory_netrin[step])
        pos_chemokine_netrin = np.copy(trajectory_chemokine_netrin[step])
        pos_heparansulfate = np.copy(trajectory_heparansulfate[step])
        pos_chemokine_heparansulfate = np.copy(trajectory_chemokine_heparansulfate[step])
        
        # check if any chemokine-netrin complexes are bonded to collagen
        idx_fixed = np.where(pos_chemokine_netrin[:, 1] == collagen_y)[0]
        pos_fixed = pos_chemokine_netrin[idx_fixed]
        
    return trajectory_chemokine, trajectory_netrin, trajectory_heparansulfate, trajectory_chemokine_netrin,  trajectory_chemokine_heparansulfate



# test 


# # initial_positions = np.array([[0, 0],[1, 5],[4, 2],[0, 0],[6, 4],[6, 6],[9, 9]])
# # particle_types = np.array([0, 0, 0, 1, 1, 1, 1])
# grid_size = [300, 80]
# n_steps = 15*10
# n_particles = 300 # 0.7
# particle_types = np.random.choice([0, 1, 2], size=(n_particles), p=[0.66, 0.34, 0])
# particle_diffusivity = [3, 2, 1, 0.05, 0.03]
# c_heparansulfate = 0.03


# pos_0, pos_1, pos_2, pos_3, pos_4 = monte_carlo_simulation(num_steps = n_steps,
#                                                            grid_size = grid_size,
#                                                            particle_diffusivity=particle_diffusivity,
#                                                            n_particles = n_particles,
#                                                            fraction_x=0.3,
#                                                            c_heparansulfate=c_heparansulfate,
#                                                            reflection=True,
#                                                            particle_types=particle_types)

# print("results:")
# for i in range(n_steps):
#     print(i)
#     print(pos_0[i])
#     print(pos_1[i])
#     print(pos_2[i])
#     print("\n")


# ---------------------------
#         ANIMATION
# ---------------------------



# collagen_y = np.round(grid_size[1]/2)

# fig, ax = plt.subplots()
# fig.set_figwidth(grid_size[0]/max(grid_size)*20)
# fig.set_figheight(grid_size[1]/max(grid_size)*20)

# box = ax.get_position()

# marker_scale = 1

# plot_edge = 4
# def update(i):
    
#     #clear the frame
#     ax.clear()
    
#     ax.axhspan(collagen_y-1, collagen_y+1, alpha=0.3, color="red", label="Collagen fiber")
    
#     ax.scatter(*pos_2[i].T, label="Heparansulfate",           color="grey",   marker=r"$\sim$",    s=150*marker_scale)
#     ax.scatter(*pos_4[i].T, label="Chemokine-Heparansulfate", color="purple", marker="+",          s=80*marker_scale)
#     ax.scatter(*pos_0[i].T, label="Chemokine",                color="blue",                        s=80*marker_scale)
#     ax.scatter(*pos_1[i].T, label="Netrin",                   color="red",    marker=r"--$\cdot$", s=250*marker_scale)
#     ax.scatter(*pos_3[i].T, label="Chemokine-Netrin",         color="green",                       s=90*marker_scale)
    

#     ax.set_xlim(-plot_edge, grid_size[0]+plot_edge)    
#     ax.set_ylim(-plot_edge, grid_size[1]+plot_edge)
    
#     ax.plot([0, grid_size[0]], [0, 0],                       "--", color="black")
#     ax.plot([grid_size[0], 0], [grid_size[1], grid_size[1]], "--", color="black")
#     ax.plot([grid_size[0],  grid_size[0]], [0, grid_size[1]],      color="black")
#     ax.plot([0, 0], [grid_size[1], 0],                             color="black")
    
    
#     # draw legend outside of the plot
    
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     #draw frame
#     plt.draw()
#     #pause to show the frame
#     #plt.pause(0.05)
    
# anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)

# #plt.show()

# print('saving animation...')
# anim.save('movies/sim_with_netrin_test.mp4', fps=15)#, writer='pillow')
# print('done')