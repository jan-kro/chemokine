import numpy as np

def state_index(i, j, n_ptypes):
    return i*n_ptypes + j

def interaction_indices(state_index, n_ptypes):
    return [state_index//n_ptypes, state_index%n_ptypes]

def add_state_probability(interaction_probabilities, i, j, n, m, k, diffusivities, binding_constants, n_ptypes):
    # to divide binding energy by 2, take the square root of the binding constant
    interaction_probabilities[i, j, state_index(n, m, n_ptypes)] = diffusivities[k]*np.sqrt(binding_constants[k,j]/binding_constants[k,n])
    return interaction_probabilities

def get_interaction_probabilities(diffusivities, binding_constants, resulting_ptype):

    n_ptypes = len(diffusivities) 
    
    interaction_probabilities = np.zeros((n_ptypes, n_ptypes, n_ptypes**2))
    
    for idx_in_1 in range(1,n_ptypes):
        for idx_in_2 in range(n_ptypes):
            for idx_out_1 in range(n_ptypes):
                for idx_out_2 in range(1,n_ptypes):
                    for idx_move in range(n_ptypes):
                        if resulting_ptype[idx_move, idx_out_1] == idx_in_1 and resulting_ptype[idx_move, idx_in_2] == idx_out_2:
                            interaction_probabilities = add_state_probability(
                                interaction_probabilities, 
                                idx_in_1, 
                                idx_in_2,
                                idx_out_1,
                                idx_out_2,
                                idx_move, 
                                diffusivities=diffusivities, 
                                binding_constants=binding_constants,
                                n_ptypes=n_ptypes
                                )
                            
    # normalize probabilities
    interaction_sum = np.sum(interaction_probabilities, axis=2)
    normalization = np.max(interaction_sum)
    interaction_probabilities = interaction_probabilities/normalization

    # calculate counter probability for not moving
    for i in range(n_ptypes):
        for j in range(n_ptypes):
            interaction_probabilities[i, j, state_index(i, j, n_ptypes)] = 1 - np.sum(interaction_probabilities[i,j])
       
    return interaction_probabilities, normalization

def get_interaction_probabilities_no_bp_switch(diffusivities, binding_constants, resulting_ptype):

    n_ptypes = len(diffusivities) 
    
    interaction_probabilities = np.zeros((n_ptypes, n_ptypes, n_ptypes**2))
    
    for idx_in_1 in range(1,n_ptypes):
        for idx_in_2 in range(n_ptypes):
            for idx_out_1 in range(n_ptypes):
                for idx_out_2 in range(1,n_ptypes):
                    for idx_move in range(n_ptypes):
                        if resulting_ptype[idx_move, idx_out_1] == idx_in_1 and resulting_ptype[idx_move, idx_in_2] == idx_out_2:
                            if not (idx_in_1 != 0 and idx_in_2 != 0 and idx_out_1 != 0 and idx_out_2 != 0):                            
                                interaction_probabilities = add_state_probability(
                                    interaction_probabilities, 
                                    idx_in_1, 
                                    idx_in_2,
                                    idx_out_1,
                                    idx_out_2,
                                    idx_move, 
                                    diffusivities=diffusivities, 
                                    binding_constants=binding_constants,
                                    n_ptypes=n_ptypes
                                    )
                            
    # normalize probabilities
    interaction_sum = np.sum(interaction_probabilities, axis=2)
    normalization = np.max(interaction_sum)
    interaction_probabilities = interaction_probabilities/normalization

    # calculate counter probability for not moving
    for i in range(n_ptypes):
        for j in range(n_ptypes):
            interaction_probabilities[i, j, state_index(i, j, n_ptypes)] = 1 - np.sum(interaction_probabilities[i,j])
       
    return interaction_probabilities, normalization

def get_resulting_ptype(interaction_probabilities):
    n_ptypes = interaction_probabilities.shape[0]
    resulting_ptype = np.zeros((n_ptypes,n_ptypes), dtype=int)
    for i in range(n_ptypes):
        for j in range(n_ptypes):
            for k in range(n_ptypes**2):
                n, m = interaction_indices(k, n_ptypes=n_ptypes)
                if n==0:
                    if interaction_probabilities[i, j, k] > 0:
                        resulting_ptype[i, j] = m
    
    return resulting_ptype

def get_type_contains(interaction_probabilities):
    
    n_ptypes = interaction_probabilities.shape[0]
    
    type_contains = np.zeros((n_ptypes, n_ptypes), dtype=bool)

    for i in range(n_ptypes):
        for j in range(n_ptypes):
            for k in range(n_ptypes**2):
                if interaction_probabilities[i, j, k] > 0:
                    n, m = interaction_indices(k, n_ptypes=n_ptypes)
                    # if initial particle and resulting particle are both not empty sites
                    if i != 0 and m!=0:
                        # if whole particle moves, resulting particle contains initial particle
                        if n == 0:        
                            type_contains[m, i] = True
                        # if the bond partner is not an empty site, resulting particle contains the bond partner 
                        if j != 0:
                            type_contains[m, j] = True
                            
    # for any particle that is contained in another particle, the particle that contains it also contains all particles that the contained particle contains                    
    for i in range(n_ptypes):
        for j in range(n_ptypes):
            if type_contains[i,j]:
                for k in range(n_ptypes):
                    if type_contains[j,k]:
                        type_contains[i,k] = True
                        
    return type_contains