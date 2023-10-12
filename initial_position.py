import numpy as np

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

# initpos = get_initial_positions()

# plot_grid(initpos)
