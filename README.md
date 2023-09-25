
# Chemokine
Monte Carlo simulation of diffusing chemokine, netrin and heparansulfates with possibility to build chemokine-netrin and chempkine-heparansulfate complexes. The chemokine-netrin complex additionally binds to a collagene fiber, that spans the whole simulation box.

The simulation is run by specifying the simulation parameters in the ```run.py``` file and the running the file

```console
$ python run.py
```

# Conda Environment
The file ```conda-spec-file-chemokine.txt``` contains all information to create a new conda environment, containing all the packages needed to run the code.

Make sure you are in the curren folder
```console
(base) user:/path/to/chemokine$
```

The command 

```console
$ conda create -n chemokine --file conda-spec-file-chemokine.txt
```

creates a new environment in your ``` /path/to/conda/envs``` directory

If you want to save the environment at another location , use 

```console
$ conda create --prefix /path/to/env/env-name --file conda-spec-file-chemokine.txt
```

For this you need to change your conda config to contain the path to the environments

```console
$ conda config --append envs_dirs /path/to/env
```

If you activate the enviroment it will look like this:

```console
(/path/to/env/env_name) user:/path/to/cwd$ 
```

If you want to change this to 

```console
(env_name) user:path/to/cwd$
```

paste the following command in the command line:

```console
conda config --set env_prompt '({name})'
```












