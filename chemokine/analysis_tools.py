import os
import h5py
import psutil 
import numpy as np
import chemokine as ck

def is_file_not_in_use(file_path, i):
    if not os.path.exists(file_path):
        print(i, "File does not exist: ", file_path)
        return False
    for proc in psutil.process_iter(['pid', 'open_files']):
        for open_file in proc.info['open_files'] or []:
            if open_file.path == file_path:
                print(i, "File is in use: ", file_path)                
                return False    
    return True

def is_file_in_use(file_path):
    for proc in psutil.process_iter(['pid', 'open_files']):
        for open_file in proc.info['open_files'] or []:
            if open_file.path == file_path:
                print("File is in use: ", file_path)                
                return False
    return False 

def three_component_system_old(q, A_0, B_0, C_0, K_1, K_2, K_3, scale=1):
    A, B, C = q    
        
    # eq = [A_0 - A - K_1*A*B - K_2*A*C - K_1*K_4*A*B*C,
    #       B_0 - B - K_1*A*B - K_3*B*C - K_1*K_4*A*B*C,
    #       C_0 - C - K_2*A*C - K_3*B*C - K_1*K_4*A*B*C]
    
    eq = [A_0*scale - A*scale - K_1*A*B*scale - K_2*A*C*scale,
          B_0*scale - B*scale - K_1*A*B*scale - K_3*B*C*scale,
          C_0*scale - C*scale - K_2*A*C*scale - K_3*B*C*scale]
    
    return eq

def three_component_system(q, A_0, B_0, C_0, K_1, K_2, K_3):
    A, B, C = q    
    
    eq = [A_0 - A - K_1*A*B - K_2*A*C,
          B_0 - B - K_1*A*B - K_3*B*C,
          C_0 - C - K_2*A*C - K_3*B*C]
    
    return eq

def two_component_system(q, A_0, B_0, K_1):
    
    A, B = q
    
    eq = [A_0 - A - K_1*A*B,
          B_0 - B - K_1*A*B]
    
    return eq

def effective_diffusivity(A, B, C, D_A, D_AB, D_AC, K_1, K_2):
    
    AB  = K_1*A*B
    AC  = K_2*A*C
    
    return (A*D_A + AB*D_AB + AC*D_AC) / (A + AB + AC)

def effective_diffusivity_2(A, B, C, D_A, D_AB, D_AC, K_1, K_2):
    
    AB  = K_1*A*B*2
    AC  = K_2*A*C*2
    
    return (A*D_A + AB*D_AB + AC*D_AC) / (A + AB + AC)

def effective_diffusivity_two_comp(A, B, D_A, D_AB, K_1):
    
    AB  = K_1*A*B
    
    return (A*D_A + AB*D_AB) / (A + AB)

def linear_fit_function(x, k, d):
    return x*k + d

def get_particle_numbers_concentrations(config, total=True, skip_frames=0, method="efficient"):
    init_pos = config["initial_positions"]
    n_grid = init_pos.shape[0]*init_pos.shape[1]
    type_contains = ck.utils._get_type_contains(config["interaction_probabilities"])
    ntypes = len(config["diffusivities"])
    ptypes = np.arange(ntypes)
    n_part = np.zeros(ntypes)
    if total:
        for k in range(ntypes):
            for j in range(ntypes):
                if type_contains[k, j] == True:
                    n_part[k] += np.count_nonzero(init_pos == ptypes[j])
    else:
        fname_particle_traj = ck.utils.get_fname(config["dir_sys"], config["name_sys"], "trajectory")
        with h5py.File(fname_particle_traj, "r") as fh5:
            if method == "efficient":
                traj = fh5["trajectory"]
            elif method == "fast":
                traj = fh5["trajectory"][:]
            else:
                raise ValueError("Method not implemented.")


            n_frames = traj.shape[0]

            for i in range(skip_frames, n_frames):
                for k in range(ntypes):
                    n_part[k] += np.sum(traj[i] == ptypes[k])
            n_part /= (n_frames-skip_frames)
    return n_part, n_part/n_grid 

def validate_dir_sys_input(sys_argv):
    dirs_sys = []
    for kk in range(1, len(sys_argv)):
        dir_name = os.path.abspath(sys_argv[kk])
        if os.path.isdir(dir_name):
            # convert to system path in case configs path is given
            if os.path.basename(dir_name) == "configs":
                dir_name = os.path.dirname(dir_name)
            if not os.path.exists(dir_name):
                raise ValueError(f"Directory {dir_name} does not exist.")
            dirs_sys.append(dir_name)
        else:
            raise ValueError(f"Directory {dir_name} does not exist.")
    
    # check if config directories are unique
    if len(set(dirs_sys)) != len(dirs_sys):
        raise ValueError("Config directories must be unique!")
    
    return dirs_sys
    
def list_fnames(dirs_sys, return_dirs=True, return_names=True, sort=True):
    fnames_config_out = []
    names_sys_out = []
    dirs_sys_out = []
    if isinstance(dirs_sys, str):
        dirs_sys = [dirs_sys]
        
    for dir_sys in dirs_sys:
        names_sys = []
        fnames_config = []
        for rel_fname_cfg in os.listdir(os.path.join(dir_sys, "configs")):
            fname_config = os.path.join(dir_sys, "configs", rel_fname_cfg)
            config = ck.utils.load_config_dict(fname_config)
            name_sys = config["name_sys"]
            names_sys.append(name_sys)
            fnames_config.append(fname_config)
            dirs_sys_out.append(dir_sys)
        # sort ascendingly
        if sort:
            fnames_config_out += [fname for fname in sorted(fnames_config)]
            names_sys_out += [name for name in sorted(names_sys)]
    
    if return_dirs:
        if return_names:
            return fnames_config_out, dirs_sys_out, names_sys_out
        else:
            return fnames_config_out, dirs_sys_out
    elif return_names:
        return fnames_config_out, names_sys_out
    else:
        return fnames_config_out
    
def get_theta_WRONG(config, ptype_1, ptype_2, ptype_cplx, return_n=True):
    fname_particle_traj = ck.utils.get_fname(config["dir_sys"], config["name_sys"], "trajectory")

    # get particle types that contain ptype 1 excluding complex!
    type_contains = ck.utils._get_type_contains(config["interaction_probabilities"])
    n_types = len(type_contains)
    ptypes_contain_1 = []    
    for k in range(n_types):
        if k == ptype_cplx:
            continue
        if type_contains[k, ptype_1]:
            ptypes_contain_1.append(k)
            
    print(f"Get Fraction Bound of {config['particle_names'][ptype_1]} to {config['particle_names'][ptype_2]} for system {config['name_sys']}")
    print(f"Loading .hdf5 trajectory file ...")
    with h5py.File(fname_particle_traj, "r") as fh5:
        traj = fh5["trajectory"]
        
        n_frames = traj.shape[0]
        
        n_1 = 0
        n_2 = 0
        n_cplx = 0
        for i in range(n_frames):
            frame = traj[i]
            for ptype in ptypes_contain_1:
                n_1 += np.sum(frame == ptype)
            n_2 += np.sum(frame==ptype_2)
            n_cplx += np.sum(frame==ptype_cplx)
        
        n_1/=n_frames
        n_2/=n_frames
        n_cplx/=n_frames
        theta = n_2/(n_2+n_cplx)
        
        if return_n: 
            return theta, n_1, n_2, n_cplx
        else:
            return theta
        
def get_theta(config, ptype_1, ptype_2, ptype_cplx, return_n=True, c_part_rel=None, n_part_rel=None):
    """
    ptype1 binds to ptype2 to form ptype_cplx
    """
    if c_part_rel is None and n_part_rel is None:
        n_part_rel, c_part_rel = get_particle_numbers_concentrations(config, total=False)
    else: 
        init_pos = np.array(config["initial_positions"])
        n_grid = init_pos.shape[0] * init_pos.shape[1]
        if n_part_rel is None:
            n_part_rel = c_part_rel * n_grid
        if c_part_rel is None:
            c_part_rel = n_part_rel / n_grid
        
    theta = (c_part_rel[ptype_cplx])/(c_part_rel[ptype_1]+c_part_rel[ptype_cplx])
    
    n_1 = n_part_rel[ptype_1]
    n_2 = n_part_rel[ptype_2]
    n_cplx = n_part_rel[ptype_cplx]
        
    if return_n: 
        return theta, n_1, n_2, n_cplx
    else:
        return theta
        

def get_c_D_scale(move_prob, a_SI, D_SI):
    
    # Physical Units
    # mass = 15.77   # kDa (CCL5) https://www.rcsb.org/structure/5COY
    mass = 26.7   # kDa (CCL5 + tracking molecule)

    # approximating the hydrodynamic radius of a protein by average protein 
    # density of 1.38 g/cm^3 = 1.38 * 1e3 kg/m^3 for 10kDa protein.

    mass_kg = mass * 1.66054e-24 #for given mass in [kDa] calculate mass in [kg]
    r = (3 * mass_kg / (1.38 * 1e3 * 4 * np.pi)) ** (1/3) # calculate radius in [m]
    # use stokes law D=kBT/gamma with gamma=6*pi*eta*r and water viscosity ~ 1e-3 Pa*s to calculate diffusivity in [m^2/s]
    D_SI = 1.38e-23 * 300 / (6 * np.pi * 1e-3 * r) # [m^2/s]


    delta_t = move_prob * a_SI**2 / D_SI

    a_nm = a_SI * 1e9
    delta_t_ns = delta_t * 1e9

    a_mum = a_SI * 1e6

    # SCALE DIFFUSION CONSTANT
    D_scale = a_nm**2 / delta_t_ns

    D_scale_mums = a_mum**2 / delta_t

    # physical concentration

    unit_volume = a_SI**3 # m^3
    Avogardo_C = 6.022e23 # 1/mol
    m3_to_liters = 1e3
    # c_mol_per_liter_all = c_part_all / (unit_volume * Avogardo_C * m3_to_liters)

    c_scale_mol_per_liter = 1/(unit_volume * Avogardo_C * m3_to_liters) # scale concentration to mol/liters
    
    return c_scale_mol_per_liter, D_scale_mums

def binomial(n, k):
    if 0 <= k <= n:
        a= 1
        b=1
        for t in range(1, min(k, n - k) + 1):
            a *= n
            b *= t
            n -= 1
        return a // b
    else:
        return 0

def free_volume_correction_exponential(c, D_theory, gamma=3.3124219330860156, d=2.1411403308904657): 
    # c... absolute concentration of excluded volume particles
    return D_theory*np.exp(-gamma*c/(d-c))


def bond_diffusivity_correction(c_A, c_B, c_AB, c_A_0, c_free, D_A_free, K_AB):
    K_sqrt = np.sqrt(K_AB)
    p_bond = 0
    p_unbond = 0

    for k in range(1, 5):
        p_bond   += k * D_A_free * K_sqrt * binomial(4, k) * c_B**k  * (1 -  c_B)**(4-k)
        p_unbond += k * D_A_free / K_sqrt * binomial(4, k) * c_free**k * (1 - c_free)**(4-k)

    # D_corr =  c_Chem/(c_Chem+c_Cplx)/30 * p_bond * D_free * K + c_Cplx/(c_Chem+c_Cplx)/30 * p_unbond * D_free / K
    D_corr =  c_A/c_A_0 * p_bond * D_A_free * K_sqrt + c_AB/c_A_0 * p_unbond * D_A_free / K_sqrt
    
    return D_corr 