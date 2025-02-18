import os
import h5py
import mpltex
import numpy as np
import chemokine as ck
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit

# default values
fname_default_values = "path/to/default_values_scaled_energies.h5"

# define where figure should be saved
dir_fig = "path/to/figure_folder"

# create list of all config file names
dir_sys = f"/path/to/thee_component_system/"

# list_fnames() sorts all filenames ascendingly by name
fnames_config, dirs_sys, names_sys = ck.analysis_tools.list_fnames(dir_sys)

# create index array for easier data access
c_chem_all = [0.001]
c_hep_all = np.array([0, 1e-3, 2e-3, 4e-3])

c2_c1_start = 0.1
c2_c1_end = 10

n1 = len(c_chem_all)
n2 = 10
n3 = len(c_hep_all)

idx_sys = np.zeros((n1, n2, n3), dtype=int) - 1
labels_sys = np.empty((n1, n2, n3), dtype=object)

c_C_0_sim_all = []
c_N_0_sim_all = []
c_H_0_sim_all = []

n = 0
for kk, A_0 in enumerate(c_chem_all):
    for jj in range(n2):
        for ii, C_0 in enumerate(c_hep_all):
            
            # save initial concentrations into lists
            c2_start = c2_c1_start*A_0
            c2_end = c2_c1_end*A_0
            B_0 = np.logspace(np.log10(c2_start), np.log10(c2_end), n2)[jj]
            
            c_C_0_sim_all.append(A_0)
            c_N_0_sim_all.append(B_0)
            c_H_0_sim_all.append(C_0)
            
            idx_sys[kk, jj, ii] = n
            labels_sys[kk, jj, ii] = fr"c_\mathrm{{C}}^0={A_0:.2f} c_\mathrm{{N}}^0={B_0:.2f} c_\mathrm{{H}}^0={C_0:.2f}"
            n += 1

c_C_0_sim_all = np.array(c_C_0_sim_all)
c_N_0_sim_all = np.array(c_N_0_sim_all)
c_H_0_sim_all = np.array(c_H_0_sim_all)

# Calculate the Mean Squared Displacement (MSD) for all systems
# -------------------------------------------------------------

ptype = 1    # particle type for which the MSD should be calculated (1 = chemokine)
n_part_all = []
c_part_all = []
c_part_all_rel = []

msds_x = []
msds_y = []
strides = []

stride_analysis = 1

n_systems = len(fnames_config)
for idx, fname_config in enumerate(fnames_config):
    print(f"[{idx+1:03d}/{n_systems:03d}] System Name: {names_sys[idx]}")
    config = ck.utils.load_config_dict(fname_config)
    
    # calculate MSD
    msd_x, msd_y = ck.utils.get_msd_xy(config, ptype)
    
    # append to list
    msds_x.append(msd_x)
    msds_y.append(msd_y)
    strides.append(config["stride"])
    
    # calculate total particle numbers and concentrations for every particle type
    n_part, c_part = ck.analysis_tools.get_particle_numbers_concentrations(config)
    
    n_part_all.append(n_part)
    c_part_all.append(c_part)  
    
    n_steps = config["num_steps"]
    stride = config["stride"]
    n_frames = n_steps//stride
    n, c = ck.analysis_tools.get_particle_numbers_concentrations(config, total=False, skip_frames=int(n_frames*0.1))
    c_part_all_rel.append(c)
    
n_part_all = np.array(n_part_all)
c_part_all = np.array(c_part_all)
c_part_all_rel = np.array(c_part_all_rel, dtype=float)

msds_x = np.array(msds_x)
msds_y = np.array(msds_y)
strides = np.array(strides)

# Fit the MSDs to calculate the diffusivities
# -------------------------------------------

x_fits = []
y_fits = []
y_ideals = []
move_probs = []
linfuncs = []
msds = []

idx_min_fit = 5
idx_max_fit = 50

for msd_x, msd_y, stride in zip(msds_x, msds_y, strides):
    msd = 0.5*(msd_x + msd_y)
    msds.append(msd)
    
    x_fit = np.arange(idx_min_fit, idx_max_fit)*stride_analysis
    linfunc_x = curve_fit(ck.analysis_tools.linear_fit_function, x_fit, msd[idx_min_fit:idx_max_fit])[0]
    y_fit = x_fit*linfunc_x[0] + linfunc_x[1]
    linfuncs.append(linfunc_x)
    move_prob = config['interaction_probabilities'][ptype, 0, ptype]/4
    print("Diffusion coefficient fit: D =", linfunc_x[0]/2/stride)
    print("Move probability config: p =", move_prob)
    move_probs.append(move_prob)
    
    y_ideal = 2*move_prob*x_fit
    
    x_fits.append(x_fit)
    y_fits.append(y_fit)
    y_ideals.append(y_ideal)

# calcualte diffusivities from fit    
D_all = np.array([linfunc_tmp[0]/2/stride for linfunc_tmp, stride in zip(linfuncs, strides)])

x_fits = np.array(x_fits)
y_fits = np.array(y_fits)
y_ideals = np.array(y_ideals)
move_probs = np.array(move_probs)
linfuncs = np.array(linfuncs)
msds = np.array(msds)

# Calculate the diffusivities theoretically 
# -----------------------------------------

fh5 = h5py.File(fname_default_values, "r")
interaction_prob_normalization = fh5["interaction_prob_normalization"][()]

ptype_C = 1
ptype_N = 2
ptype_H = 3
ptype_CN = 5
ptype_CH = 6
ptype_NH = 7

bond_prob_CN = fh5["interaction_probabilities"][ptype_C,ptype_N,ptype_CN]/4
bond_prob_CH = fh5["interaction_probabilities"][ptype_C,ptype_H,ptype_CH]/4
bond_prob_NH = fh5["interaction_probabilities"][ptype_N,ptype_H,ptype_NH]/4

D_C = fh5["interaction_probabilities"][ptype_C,0,ptype_C]/4
D_N = fh5["interaction_probabilities"][ptype_N,0,ptype_N]/4
D_H = fh5["interaction_probabilities"][ptype_H,0,ptype_H]/4
D_CN = fh5["interaction_probabilities"][ptype_CN, 0, ptype_CN]/4
D_CH = fh5["interaction_probabilities"][ptype_CH, 0, ptype_CH]/4
D_NH = fh5["interaction_probabilities"][ptype_NH, 0, ptype_NH]/4

K_CN_setup = (bond_prob_CN / D_C)**2
K_CH_setup = (bond_prob_CH / D_C)**2
K_NH_setup = (bond_prob_NH / D_N)**2

fh5.close()

c2_c1_start = 0.1
c2_c1_end = 10

c_C_all = [0.001]
c_H_all = np.array([0, 1e-3, 2e-3, 4e-3])

n_C = len(c_C_all)
n_N = 50
n_H = len(c_H_all)

# c2_start = c2_c1_start*c_C_all[0]
# c2_end = c2_c1_end*c_C_all[-1]
# c_N_all = np.logspace(np.log10(c2_start), np.log10(c2_end), n_N)
            
K_CN = K_CN_setup
K_CH = K_CH_setup
K_NH = K_NH_setup

c_C_0_th_all = []
c_N_0_th_all = []
c_H_0_th_all = []

c_C_th_all = []
c_N_th_all = []
c_H_th_all = []
c_CN_th_all = []
c_CH_th_all = []
c_NH_th_all = []

D_th_all = []
D_th_corr_all = []
D_th_no_bond_corr_all = []

idx_sys_th = np.zeros((n_C, n_N, n_H), dtype=int)

dir_cfg_all = []

eq_all = []

nn = 0
for kk, C_0 in enumerate(c_C_all):
    for jj in range(n_N):
        for ii, H_0 in enumerate(c_H_all):
            
            c_N_all = np.logspace(np.log10(c2_c1_start*C_0), np.log10(c2_c1_end*C_0), n_N)
            N_0 = c_N_all[jj]
            
            c_C_0_th_all.append(C_0)
            c_N_0_th_all.append(N_0)
            c_H_0_th_all.append(H_0)
            
            C, N, H = fsolve(
                ck.analysis_tools.three_component_system, 
                x0=[C_0, N_0, H_0], 
                args=(C_0, N_0, H_0, K_CN, K_CH, K_NH),
                xtol=1e-10)
            
            CN = K_CN*C*N
            CH = K_CH*C*H
            NH  = K_NH*N*H
            
            c_C_th_all.append(C)
            c_N_th_all.append(N)
            c_H_th_all.append(H)
            c_CN_th_all.append(CN)
            c_CH_th_all.append(CH)
            c_NH_th_all.append(NH)
            
            D_th = ck.analysis_tools.effective_diffusivity(
                A=C, 
                B=N, 
                C=H, 
                D_A=D_C, 
                D_AB=D_CN, 
                D_AC=D_CH, 
                K_1=K_CN, 
                K_2=K_CH)
            
            
            D_th_all.append(D_th)
            
            # free volume correction
            c_excl = C + CN + CH + NH
            
            D_corr = ck.analysis_tools.free_volume_correction_exponential(c_excl, D_th)
            
            D_th_no_bond_corr_all.append(D_corr)
            
            c_free = 1 - C - N - H - CN - CH - NH
            
            # correct diffusivity for CN bond
            D_corr += ck.analysis_tools.bond_diffusivity_correction(C, N, CN, C_0, c_free, D_C, K_CN)
            
            # correct for CH bond
            D_corr += ck.analysis_tools.bond_diffusivity_correction(C, H, CH, C_0, c_free, D_C, K_CH)
            
            D_th_corr_all.append(D_corr)
            
            idx_sys_th[kk, jj, ii] = nn
            nn += 1

c_C_0_th_all = np.array(c_C_0_th_all)
c_N_0_th_all = np.array(c_N_0_th_all)
c_H_0_th_all = np.array(c_H_0_th_all)
            
c_C_th_all = np.array(c_C_th_all)
c_N_th_all = np.array(c_N_th_all)
c_H_th_all = np.array(c_H_th_all)
c_CN_th_all = np.array(c_CN_th_all)
c_CH_th_all = np.array(c_CH_th_all)
c_NH_th_all = np.array(c_NH_th_all)

D_th_all = np.array(D_th_all)
D_th_no_bond_corr_all = np.array(D_th_no_bond_corr_all)
D_th_corr_all = np.array(D_th_corr_all)

# Plot the results
# ----------------

# Physical Units
# mass = 15.77   # kDa (CCL5) https://www.rcsb.org/structure/5COY
mass = 26.7   # kDa (CCL5 + tracking molecule)

# approximating the hydrodynamic radius of a protein by average protein 
# density of 1.38 g/cm^3 = 1.38 * 1e3 kg/m^3 for 10kDa protein.

mass_kg = mass * 1.66054e-24 #for given mass in [kDa] calculate mass in [kg]
r_chem_SI = (3 * mass_kg / (1.38 * 1e3 * 4 * np.pi)) ** (1/3) # calculate radius in [m]
# use stokes law D=kBT/gamma with gamma=6*pi*eta*r and water viscosity ~ 1e-3 Pa*s to calculate diffusivity in [m^2/s]
D_chem_SI = 1.38e-23 * 300 / (6 * np.pi * 1e-3 * r_chem_SI) 

D_chem_mums = D_chem_SI * 1e12 # convert to [um^2/s]

scale_concentration = 3e5 # from N/N_grid to nM/L 

# cSI = c * s
# c*s = N/N_grid * |s| * N_A * 1e-9 mol/L
# c*s = N/N_grid * |s| * N_A * 1e-9 * 1e3  * 6.022e23 * 1/m^3
# N/N_grid 1/a^3 =  N/N_grid * |s| * 1e-9 * 1e3  * 6.022e23 * 1/m^3
# a = (1/(|s| * 1e-9 * 1e3  * 6.022e23))**(1/3) m
a_SI = (1/(scale_concentration * 1e-9 * 1e3  * 6.022e23))**(1/3) # in [m]
a_um = a_SI * 1e6 # in [um]

scale_diffusifity = D_chem_mums/D_C # from a^2/dt to [um^2/s] 

# a^2/dt = |D_real_ums/D_sim| * um^2/s
# dt = |D_sim|*|a_um|^2/|D_real_ums| s
delta_t = D_C * a_um**2 / D_chem_mums # in [s]


@mpltex.acs_decorator
def plot_diffusivities():
    
    fig, ax = plt.subplots(1)
    
    fig.set_dpi(300)
    
    ax.set_xscale("log")
    
    colors = ["#00982C", "#FF7532","#FF7532","#FF7532"]
    alphas = [1, 1, 0.66, 0.33]
      
    for idx_H in range(n_H):
        for idx_C in range(n_C):
            
            c_C_0_sim = c_C_0_sim_all[idx_sys[idx_C, :, idx_H]]
            c_N_0_sim = c_N_0_sim_all[idx_sys[idx_C, :, idx_H]]
            
            ax.plot(
                c_N_0_sim/c_C_0_sim, 
                D_all[idx_sys[idx_C, :, idx_H]]/D_C, 
                marker="o", 
                markerfacecolor="none",
                linestyle="none", 
                color=colors[idx_H], 
                alpha=alphas[idx_H]
            )
            
            
            c_N = c_N_0_th_all[idx_sys_th[idx_C, :, idx_H]]
            c_C = c_C_0_th_all[idx_sys_th[idx_C, :, idx_H]]
            
            ls_l = mpltex.linestyles()
            ls = next(ls_l)["linestyle"]
            
            ax.plot(
                c_N/c_C, 
                D_th_corr_all[idx_sys_th[idx_C, :, idx_H]]/D_C,
                color=colors[idx_H], 
                alpha=alphas[idx_H],
                linestyle=ls
                )
            
    
    ax.set_ylim((0, 1.05))        
    
    xl = ax.get_xlim()
    dx = xl[1] - xl[0]
    yl = ax.get_ylim()
    dy = yl[1] - yl[0]
    
    ax.text(xl[1] + 0.25*dx, yl[1]-dy/10, fr"$c_\mathrm{{C}}^0={c_C_0_sim_all[0]*scale_concentration}$ nMol")
    
    ax.hlines(D_C/D_C,  xl[0], xl[1], ls="-.", color="k")
    ax.hlines(D_CN/D_C, xl[0], xl[1], ls=":", color="k")
            
    ax.set_xlabel("$c_\mathrm{N}^0/c_\mathrm{C}^0$")
    ax.set_ylabel(r"$D^\mathrm{eff}_\mathrm{C} / D^\mathrm{free}_\mathrm{C}$")
    
    leg_handles = [plt.Line2D([0], [0], color=colors[idx], alpha=alphas[idx], linestyle="solid") for idx in range(n_H)]
    leg_labels = [fr"$c_\mathrm{{H}}^0 = {c_H_all[idx]*scale_concentration}$ nMol" for idx in range(n_H)]
    
    leg_handles += [plt.Line2D([0], [0], color="black", linestyle="none", marker="o", markerfacecolor="none")]
    
    ls_l = mpltex.linestyles()
    leg_handles += [plt.Line2D([0], [0],color="black", ls=next(ls_l)["linestyle"])]
    leg_labels += ["simulation", "theory"]

    leg_handles += [plt.Line2D([0], [0], color="black", ls="-.")]
    leg_handles += [plt.Line2D([0], [0], color="black", ls=":")]
    leg_labels += ["$D^\mathrm{free}_\mathrm{C}$"]
    leg_labels += ["$D^\mathrm{free}_\mathrm{CN}$"]
    
    ax.legend(leg_handles, leg_labels, loc="center left", bbox_to_anchor=(1, 0.5))

    fig.savefig(os.path.join(dir_fig, "diffusivity_with_units.pdf"), bbox_inches="tight")
    
    plt.show()


plot_diffusivities()

# plot the concentrations (EXTENDED DATA FIGURES)



@mpltex.acs_decorator
def plot_concentrations(idx_H):
    
    fig, ax = plt.subplots(1)
    
    fig.set_dpi(300)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ls = mpltex.linestyles()
    colors = [next(ls)["color"] for _ in range(6)]
        
    for idx_C in range(n_C):

        c_C_0_sim = c_C_0_sim_all[idx_sys[idx_C, :, idx_H]]
        c_N_0_sim = c_N_0_sim_all[idx_sys[idx_C, :, idx_H]]
        
        c_C_sim  = c_part_all_rel[idx_sys[idx_C, :, idx_H]][:, ptype_C]
        c_N_sim  = c_part_all_rel[idx_sys[idx_C, :, idx_H]][:, ptype_N]
        c_H_sim  = c_part_all_rel[idx_sys[idx_C, :, idx_H]][:, ptype_H]
        c_CN_sim = c_part_all_rel[idx_sys[idx_C, :, idx_H]][:, ptype_CN]
        c_CH_sim = c_part_all_rel[idx_sys[idx_C, :, idx_H]][:, ptype_CH]
        c_NH_sim = c_part_all_rel[idx_sys[idx_C, :, idx_H]][:, ptype_NH]
        
        ax.plot(c_N_0_sim/c_C_0_sim, c_C_sim/c_C_0_sim,  color=colors[0], marker="o", markerfacecolor="none",linestyle="none")
        ax.plot(c_N_0_sim/c_C_0_sim, c_N_sim/c_C_0_sim,  color=colors[1], marker="o", markerfacecolor="none",linestyle="none")
        ax.plot(c_N_0_sim/c_C_0_sim, c_H_sim/c_C_0_sim,  color=colors[2], marker="o", markerfacecolor="none",linestyle="none")
        ax.plot(c_N_0_sim/c_C_0_sim, c_CN_sim/c_C_0_sim, color=colors[3], marker="o", markerfacecolor="none",linestyle="none")
        ax.plot(c_N_0_sim/c_C_0_sim, c_CH_sim/c_C_0_sim, color=colors[4], marker="o", markerfacecolor="none",linestyle="none")
        ax.plot(c_N_0_sim/c_C_0_sim, c_NH_sim/c_C_0_sim, color=colors[5], marker="o", markerfacecolor="none",linestyle="none")
        
        c_N_0_th = c_N_0_th_all[idx_sys_th[idx_C, :, idx_H]]
        c_C_0_th = c_C_0_th_all[idx_sys_th[idx_C, :, idx_H]]
        
        c_C_th = c_C_th_all[idx_sys_th[idx_C, :, idx_H]]
        c_N_th = c_N_th_all[idx_sys_th[idx_C, :, idx_H]]
        c_H_th = c_H_th_all[idx_sys_th[idx_C, :, idx_H]]
        c_CN_th = c_CN_th_all[idx_sys_th[idx_C, :, idx_H]]
        c_CH_th = c_CH_th_all[idx_sys_th[idx_C, :, idx_H]]
        c_NH_th = c_NH_th_all[idx_sys_th[idx_C, :, idx_H]]
        
        ax.plot(c_N_0_th/c_C_0_th, c_C_th/c_C_0_th,  color=colors[0])
        ax.plot(c_N_0_th/c_C_0_th, c_N_th/c_C_0_th,  color=colors[1])
        ax.plot(c_N_0_th/c_C_0_th, c_H_th/c_C_0_th,  color=colors[2])
        ax.plot(c_N_0_th/c_C_0_th, c_CN_th/c_C_0_th, color=colors[3])
        ax.plot(c_N_0_th/c_C_0_th, c_CH_th/c_C_0_th, color=colors[4])
        ax.plot(c_N_0_th/c_C_0_th, c_NH_th/c_C_0_th, color=colors[5])
    
    ax.set_xlabel(r"$c_\mathrm{N}^0/c_\mathrm{C}^0$")
    ax.set_ylabel(r"$c~/~c_\mathrm{C}^0$")
    
    ls_c = mpltex.linestyles()
    leg_handles = [plt.Line2D([0], [0], color=next(ls_c)["color"], linestyle="solid") for idx in range(6)]
    leg_labels  = [
        "$c_\mathrm{C}$", 
        "$c_\mathrm{N}$", 
        "$c_\mathrm{H}$", 
        "$c_\mathrm{CN}$", 
        "$c_\mathrm{CH}$", 
        "$c_\mathrm{NH}$"
    ]
    
    leg_handles += [plt.Line2D([0], [0], color="black", linestyle="none", marker="o", markerfacecolor="none")]
    leg_handles += [plt.Line2D([0], [0],color="black", ls="solid")]
    leg_labels += ["simulation", "theory"]
    
    ax.legend(leg_handles, leg_labels, loc="center left", bbox_to_anchor=(1, 0.5))
    
    fig.savefig(os.path.join(dir_fig, f"concentration_vs_concentration_normalized_c_Hep_{idx_H:d}.pdf"), bbox_inches="tight")
    
    plt.show()

for idx_H in range(n_H):
    plot_concentrations(idx_H)

@mpltex.acs_decorator
def bar_plot_concentrations(idx_H):
    
    fig, ax = plt.subplots(1)
    
    fig.set_dpi(300)
    
    ax.set_xscale("log")
        
    for idx_C in range(n_C):

        c_C_0_sim = c_C_0_sim_all[idx_sys[idx_C, :, idx_H]]
        c_N_0_sim = c_N_0_sim_all[idx_sys[idx_C, :, idx_H]]
        c_H_0_sim = c_H_0_sim_all[idx_sys[idx_C, :, idx_H]]
        
        
        c_C_sim  = c_part_all_rel[idx_sys[idx_C, :, idx_H]][:, ptype_C]
        c_N_sim  = c_part_all_rel[idx_sys[idx_C, :, idx_H]][:, ptype_N]
        c_H_sim  = c_part_all_rel[idx_sys[idx_C, :, idx_H]][:, ptype_H]
        c_CN_sim = c_part_all_rel[idx_sys[idx_C, :, idx_H]][:, ptype_CN]
        c_CH_sim = c_part_all_rel[idx_sys[idx_C, :, idx_H]][:, ptype_CH]
        c_NH_sim = c_part_all_rel[idx_sys[idx_C, :, idx_H]][:, ptype_NH]
        
        c_tot = c_C_sim + c_N_sim + c_H_sim + c_CN_sim + c_CH_sim + c_NH_sim
        x = c_N_0_sim/c_C_0_sim
        s = x[1]/x[0]
        x = np.append(x, x[-1]*s)
        x /= np.sqrt(s)
        width = np.diff(x)
        x = x[:-1]
        
        alpha = 0.5
        bottom = np.zeros_like(c_C_sim)
        ax.bar(x, c_C_sim/c_tot, bottom=bottom, width=width, align="edge", alpha=alpha)
        bottom += c_C_sim/c_tot
        ax.bar(x, c_N_sim/c_tot, bottom=bottom, width=width, align="edge", alpha=alpha)
        bottom+= c_N_sim/c_tot
        ax.bar(x, c_H_sim/c_tot, bottom=bottom, width=width, align="edge", alpha=alpha)
        bottom+= c_H_sim/c_tot
        ax.bar(x, c_CN_sim/c_tot, bottom=bottom, width=width, align="edge", alpha=alpha)
        bottom+= c_CN_sim/c_tot
        ax.bar(x, c_CH_sim/c_tot, bottom=bottom, width=width, align="edge", alpha=alpha)
        bottom+= c_CH_sim/c_tot
        ax.bar(x, c_NH_sim/c_tot, bottom=bottom, width=width, align="edge", alpha=alpha)
        
    
    ax.set_xlim([x[0], x[-1]+width[-1]])
    ax.set_ylim([1e-3, 1])
    
    ax.set_xlabel(r"$c_\mathrm{N}^0/c_\mathrm{C}^0$")
    ax.set_ylabel(r"$c~/~c_\mathrm{total}$")
    
    ls_c = mpltex.linestyles()
    leg_handles = [plt.Line2D([0], [0], color=next(ls_c)["color"], linestyle="solid") for _ in range(6)]
    leg_labels  = [
        "$c_\mathrm{C}$", 
        "$c_\mathrm{N}$",
        "$c_\mathrm{H}$", 
        "$c_\mathrm{CN}$", 
        "$c_\mathrm{CH}$", 
        "$c_\mathrm{NH}$"
    ]
    
    leg_handles += [plt.Line2D([0], [0], color="black", linestyle="none", marker="o", markerfacecolor="none")]
    leg_handles += [plt.Line2D([0], [0],color="black", ls="solid")]
    leg_labels += ["simulation", "theory"]

    
    ax.legend(leg_handles, leg_labels, loc="center left", bbox_to_anchor=(1, 0.5))
    
    fig.savefig(os.path.join(dir_fig, f"concentration_vs_concentration_normalized_bar_plot_c_Hep_{idx_H:d}.pdf"), bbox_inches="tight")
    
    plt.show()

for idx_H in range(n_H):
    bar_plot_concentrations(idx_H)