
# Chemokine
This package models reaction-diffusion systems on a two dimensional grid. It is used to simulate chemokine, netrin and heparansulfates with possibility to build chemokine-netrin and chemokine-heparan sulfate complexes. The netrin and chemokine-netrin complexes can additionally bind to a collagen fiber, that spans simulation box. Additionally the package allows to add particles of different types to the system throughout the simulation. It also comes with various analysis and visualization tools such as a mean squared deviation calculation and an animation tool.

The package is built on top of the ```numpy``` and ```h5py``` libraries and uses the custom ```rust_chemokine``` library, such that the simulation itself is performed in Rust in order to optimize simulation speed and making simulations of long systems feasible, reaching up to 200 ms simulation time span in less than 24 hours on a single core for systems containing 1500 particles.

The package was originally developed to simulate chemokine speedup through netrin complexation, but it can be used for any reaction-diffusion system on a two-dimensional grid. The package is designed to be easily extendable to other systems by changing the interaction probabilities and the initial state of the system. It also contains tools to calculate interaction probabilities from particle diffusivities and binding constants for simple system setups.

# Installation

The package can be installed by cloning the repository and running the following command in the root directory of the repository:

```console
pip install .
```


# Quick Start

The simulation is run by specifying system and simulation parameters in a config dictionary saved in a .toml file and then executing

```console
python run.py /path/to/config.toml
```

# About

The reaction-diffusion simulation is performed on a two-dimesional grid, which is represented by an array of integers. Each integer represents a different particle type, where ```0``` always represents the empty grid site, which will also be treated as a particle type in the following. If two particles react and form a particle complex, the particle complex is represented by an integer different from the two species. The dynamics of the system is governed by the interaction probabilities between the different particle species. Assume the system contains the particle types $0,1,2,...,N-1$ then there are $N^2$ possible combinations of neighboring particle types, where each of the four adjacent grid sites to a particle is considered a neighbor. For each of the $N^2$ possible states there are $N^2$ possible states that can theoretically result from the interaction of the neighboring particles after one time step. Most of these states are physically not possible, thus their probability is zero. The interaction probabilities are stored in a matrix of size $N \times N \times N^2$, where the first two dimensions represent the particle types of the two neighboring particles and the third dimension represents the possible resulting states. 

For every time step the system is updated in the following way:

Loop over all grid sites:

1. Check if the grid site is an empty site. If yes, continue with the next grid site.
2. Check if the particle performed an action in the current time step. If yes, continue with the next grid site.
3. Get the neighboring sites of the current particle.
   - Check which neighboring particles the current particle is allowed to interact with. An interaction is allowed if the neighboring particle did not yet perform an action in the current simulation step.
   - Get the interaction probabilities of the current particle with the neighboring particles from the interaction probability matrix. This results in $4N$ probabilities which are normalized.
5. Draw which interaction will take place with the normalized probabilities and update the current position array with the new state.
6. If the new state is different from the previous state, mark that the resulting particles performed an action in the current time step.

In this way the system is evolved in time for the desired amount of steps.

For more information on the implementation of the simulation see ```resources/model_description/model_description.pdf```.

# Tutorial

The simulation parameters are specified in a config dictionary which is saved as a .toml file. The dictionary must contain the following keys:

- ```num_steps```: Number of simulation steps
- ```stride```: Save the state of the system every ```stride``` steps
- ```reflection```: tuple of four booleans determining at which walls the particles are reflected
        the order is ```[x0, x1, y0, y1]``` i.e. left, right, up, down. If the entry is ```False``` the boundary at that wall is periodic.
- ```track_types```: tuple of $N$ booleans, where $N$ is the number of particle types. If the $i$-th entry is ```True``` the particles of type $i$ are tracked in an individual trajectory where each particle is distinguishable.
- ```name_sys```: Name of the system. Determines all output filenames
- ```dir_sys```: Directory where the output files are saved
- ```initial_positions```: 2d array of integers of shape ```[lx, ly]```containing the initial state of the grid. The array consists of $N$ different integer values, where $N$ is the number of particle types. The integer values represent the particle types, with ```0``` being the empty grid site. The grid is of size ```[lx, ly]```. This means that the indices of a particle are at the same time the $x$ and $y$ coordinates of the particle.
- ```interaction_probabilities```: 3d array of floats of shape ```[N, N, N^2]``` containing the interaction probabilities between the different particle types. The first two dimensions represent the particle types of the two neighboring particles and the third dimension represents the possible resulting states. Suppose an initial state of two neighboring particles ```[i_in, j_in]```: The first index ```i_in``` is the particle type of the particle that can perform an action, the second index ```j_in``` is the particle type of the neighboring particle. The third index ```ij_out``` then indexes the probability ```interaction_probabilities[i_in, j_in, ij_out]``` that the state ```[i_out, j_out] = [ij_out//N, ij_out%N]``` occurs.
- ```add_particles_per_step```: tuple of floats of length $N$ containing the number of particles of each type that are added to the system in each time step. Each float must be between 0 and 1, since not more than one particle can be added per step.

The simulation can then be performed by running the following python script:
 
```
import chemokine as ck

config = ck.load_config('/path/to/config.toml')
ck.simulate(config)
```

This produces a trajectory file at ```>dir_sys</trajectories/traj_>name_sys<.hdf5```. The trajectory is a ```h5py``` file containing the following datasets:

- ```completion_status```: ```True``` if the simulation was completed, ```False``` if the simulation was interrupted before completion.
- ```trajectory```: 3d array of integers of shape ```[1+num_steps//stride, lx, ly]``` containing the state of the system at each time step for the stride set in the config. The first entry is the initial position of the system. The values contained in the array follow the same convention as the initial_positions array in the config.

To open the trajectory file, the following code can be used:

```
import h5py
import chemokine as ck

fname_cfg = '/path/to/config.toml'
config = ck.load_config(fname_cfg)
fname_traj = ck.utils.get_fname_from_config(config, ftype='trajectory')
with h5py.File(fname_traj, 'r') as fh5:
    completion_status = fh5['completion_status'][()]
    trajectory = fh5['trajectory'][:]
```

If the individual particle trajectories are tracked, then an additional trajectory file is saved at ```>dir_sys</trajectories/traj_particles>name_sys<.hdf5``` with the shape ```[num_steps//stride, lx, ly, n_tracked]```, where ```n_tracked``` is the number of particle types that are tracked. For example if ```config['track_types'] = [Flase, True, True, False]```, then ```n_tracked = 2```.

If ```config['track_types'][ptype] = True```, then the mean squared deviation for the particle type ```ptype``` can be calculated by running the following script:

```
import chemokine as ck

fname_cfg = '/path/to/config.toml'
config = ck.load_config(fname_cfg)

msd_x, msd_y = ck.utils.get_msd_xy(config, ptype=1) # ptype is the particle type
```

Both ```msd_x``` and ```msd_y``` are 1d arrays of length ```num_steps//stride``` containing the mean squared deviation of the x and y coordinates of the particles of type ```ptype``` averaged over all particles in the box. It is important to note that it is not differentiated between bound and free particles, since the particle trajectory contains all particles of type ```ptype```, regardless if they are bound or not; i.e. the number of particles is constant over time.

Often the trajectory is very large and cannot be loaded into memory. In this case the function ```ck.utils.get_msd_xy_2()``` can be used, which is slower but does not use a lot of memory.


# Default Systems



# Conda Environment
The file ```conda-spec-file-chemokine.txt``` contains all information to create a new conda environment, containing all the packages needed to run the code.

Make sure you are in the directory of the spec file
```console
> (base) user:/path/to/chemokine$
```

The command 

```console
conda create -n chemokine --file conda-spec-file-chemokine.txt
```

creates a new conda environment in your ``` /path/to/conda/envs``` directory called ```chemokine```.

If you want to save the environment at another location , use 

```console
mkdir /path/to/env/env-name
conda create --prefix /path/to/env/env-name --file conda-spec-file-chemokine.txt
```

For this you need to change your conda config to contain the path to the new environment

```console
conda config --append envs_dirs /path/to/env
```

If you activate the enviroment it will look like this:

```console
> (base) user:/path/to/cwd$ conda activate env-name
> (/path/to/env/env-name) user:/path/to/cwd$ 
```

If you want to change this to 

```console
> (env-name) user:/path/to/cwd$
```

paste the following command in the command line:

```console
conda config --set env_prompt '({name})'
```












