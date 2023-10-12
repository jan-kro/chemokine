import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from animation import animate


# traj_dirs = ["/local_scratch2/janmak98/chemokine/results/langmuir_ccl5_netrin_homogeneous/trajectories",
#              "/local_scratch2/janmak98/chemokine/results/langmuir_ccl5_netrin_homogeneous_2/trajectories"]
traj_dirs = ["/local_scratch2/janmak98/chemokine/results/langmuir_ccl5_netrin_all_logscale/trajectories",
             "/local_scratch2/janmak98/chemokine/results/langmuir_ccl5_netrin_only_logscale/trajectories"]

# traj_dirs = ["/local_scratch2/janmak98/chemokine/results/langmuir_ccl5_netrin_only_logscale_eq/trajectories",
#              "/local_scratch2/janmak98/chemokine/results/langmuir_ccl5_netrin_only_logscale_eq/trajectories"]

traj_dirs = ["/local_scratch2/janmak98/chemokine/results/langmuir_ccl5_netrin_only_logscale_eq_3/trajectories",
             "/local_scratch2/janmak98/chemokine/results/langmuir_ccl5_netrin_only_logscale_eq_3/trajectories"]

traj_dirs = ["/local_scratch2/janmak98/chemokine/results/langmuir_ccl5_netrin_only_logscale_eq_long_inverse/trajectories",
             "/local_scratch2/janmak98/chemokine/results/langmuir_ccl5_netrin_only_logscale_eq_long/trajectories"]

system_names = []
for traj_dir in traj_dirs:
    system_names.append(traj_dir.split('/')[-2])

system_paths = []
for traj_dir in traj_dirs:
    system_paths.append(os.path.join(traj_dir, os.pardir))
system_paths = np.array(system_paths)


dir_idx = []
ii = 0
traj_fnames = []
for traj_dir in traj_dirs:
    for fname in os.listdir(traj_dir):
        dir_idx.append(ii)
        traj_fnames.append(os.path.join(traj_dir, fname))
    ii+=1

netrin_concentration = []
for fname in traj_fnames:
    netrin_concentration.append(float(fname.split('cn')[1].split('pct')[0]))

# sort concentrations ascendingly and fnames respectively
idx_fnames = np.argsort(netrin_concentration)
netrin_concentration = np.array(netrin_concentration)[idx_fnames]
traj_fnames = np.array(traj_fnames)[idx_fnames]
dir_idx = np.array(dir_idx)[idx_fnames]


n_netrin             = np.zeros(len(netrin_concentration))
n_chemokine          = np.zeros_like(n_netrin)
n_cn_complex         = np.zeros_like(n_netrin)

n_netrin_initial     = np.zeros_like(n_netrin)
n_chemokine_initial  = np.zeros_like(n_netrin)
n_cn_complex_initial = np.zeros_like(n_netrin)

n_grid_sites         = np.zeros_like(n_netrin)
n_all_initial        = np.zeros_like(n_netrin)

# # animate
for idx_file, conc in zip(idx_fnames, netrin_concentration):
    if conc == 0.006:
        print(f'loading {traj_fnames[idx_file]}')
        print(f'conc = {conc}')
        print(f'idx_file = {idx_file}')
        animate(h5py.File(traj_fnames[idx_file], 'r')['trajectory'], show=True, save=False, fname=None)
        
raise SystemExit

nt = []

# load trajectories with ascending concentration
for idx, traj_fname in enumerate(traj_fnames):
    traj = h5py.File(traj_fname, 'r')['trajectory']
    
    n_frames = traj.shape[0]

    n_chemokine_initial[idx]  += np.sum(traj[0] == 1)
    n_netrin_initial[idx]     += np.sum(traj[0] == 2)
    n_cn_complex_initial[idx] += np.sum(traj[0] == 4)
    
    for i in range(1, 9):
        n_all_initial[idx] += np.sum(traj[0] == i)
    
    n_grid_sites[idx] = traj[0].shape[0]*traj[0].shape[1]
    
    nt.append(np.zeros((3, traj.shape[0])))
        
    # get average number of particles for every conccentraion
    for idx_frame in range(n_frames):
        n_chemokine[idx]  += np.sum(traj[idx_frame] == 1)
        n_netrin[idx]     += np.sum(traj[idx_frame] == 2)
        n_cn_complex[idx] += np.sum(traj[idx_frame] == 4)

        nt[idx][0][idx_frame] = np.sum(traj[idx_frame] == 1)
        nt[idx][1][idx_frame] = np.sum(traj[idx_frame] == 2)
        nt[idx][2][idx_frame] = np.sum(traj[idx_frame] == 4)
        
    # mean
    n_chemokine[idx] /= n_frames
    n_netrin[idx] /= n_frames
    n_cn_complex[idx] /= n_frames



nt = np.array(nt)

print("initial number of particles")
print(n_chemokine_initial)
print(n_netrin_initial)
print(n_cn_complex_initial+n_netrin_initial)

# plt.figure(figsize=(16, 9))
# plt.xscale('log')
# for idx, conc in enumerate(netrin_concentration[dir_idx==0]):
#     plt.plot(nt[dir_idx==0][idx][0], label=f'Nc {conc*100:.2f} %')
#     plt.plot(nt[dir_idx==0][idx][1], label=f'Nn {conc*100:.2f} %')
#     plt.plot(nt[dir_idx==0][idx][2], label=f'Ncn {conc*100:.2f} %')

# plt.show()



# plt.figure()
# plt.plot(netrin_concentration, n_chemokine, 'o-', label='chemokine')
# plt.plot(netrin_concentration, n_netrin, 'o-', label='netrin')
# plt.plot(netrin_concentration, n_cn_complex, 'o-', label='complex')
# plt.legend()
# plt.savefig('figures/test.png')

# LANGMUIR
langmuir = n_cn_complex / (n_chemokine + n_cn_complex)
langmuir_2 = n_cn_complex / (n_netrin + n_cn_complex)



print("\nNetrin concentration [%]  Theta  Chemokine [n]  Netrin [n]  Complex [n]")
print("-----------------------------------------------------------------------")
for idx in range(len(netrin_concentration)):
    print(f'{netrin_concentration[idx]*100:^24.2f}{langmuir[idx]:^7.2f}{n_chemokine[idx]:^15.2f}{n_netrin[idx]:^12.2f}{n_cn_complex[idx]:^13.2f}')


binding_probability = np.zeros((9, 9))
binding_probability[1, 2] = 6 # chemokine binds to netrin
binding_probability[2, 1] = 6
binding_probability[2, 6] = 9 # netrin binds to collagen site 
binding_probability[6, 2] = 9 # netrin binds to collagen site 
binding_probability[1, 3] = 3 # chemokine binds to heparansulfate
binding_probability[3, 1] = 3
binding_probability[4, 6] = 9 # chemokine-netrin binds to collagen_site
binding_probability[6, 4] = 9 # chemokine-netrin binds to collagen_site
binding_probability[:, 0] = 1 # all particles bind to empty grid site
binding_probability[0, :] = 1 # all particles bind to empty grid site

def langmuir_analytical(concentration, bond_energy):
    lgmr = concentration*np.exp(bond_energy)/(1 + concentration*np.exp(bond_energy))
    return lgmr
    
def langmuir_analytical_2(netrin_concentration, chemokine_concentration, bond_energy):
    ratio = netrin_concentration/(netrin_concentration+chemokine_concentration)
    lgmr = ratio*np.exp(bond_energy)/(1 + ratio*np.exp(bond_energy))
    return lgmr

def langmuir_analytical_3(n_netrin, n_total, bond_energy):
    lgmr = n_netrin*np.exp(bond_energy)/((n_total-n_netrin) + n_netrin*np.exp(bond_energy))
    return lgmr

def langmuir_analytical_4(n_netrin, n_total, bond_energy):
    lgmr = np.exp(bond_energy)/((n_total/n_netrin-1) + np.exp(bond_energy))
    return lgmr


plt.figure()
plt.plot(n_chemokine[dir_idx==1], langmuir_2[dir_idx==1], 'o-', label='only netrin and chemokine')
plt.plot(n_chemokine[dir_idx==1], langmuir_analytical_3(n_chemokine[dir_idx==1], n_netrin[dir_idx==1]+n_chemokine[dir_idx==1]+n_cn_complex[dir_idx==1], np.log(binding_probability[2, 1])), label='N1 = CCL5+Net1+CN (eq)')
plt.plot(n_chemokine[dir_idx==1], langmuir_analytical_3(n_chemokine[dir_idx==1], n_netrin[dir_idx==1]+n_chemokine[dir_idx==1]+n_cn_complex[dir_idx==1], np.log(69.07)), label='N1 = CCL5+Net1+CN (eq)')
plt.show()

plt.figure()

# plt.plot(n_netrin_initial[dir_idx==0], langmuir[dir_idx==0], 'o-', label='all particle types')
# plt.plot(n_netrin_initial[dir_idx==1], langmuir[dir_idx==1], 'o-', label='only netrin and chemokine')

#plt.plot(n_netrin[dir_idx==0], langmuir[dir_idx==0], 'o-', label='all particle types')
plt.plot(n_netrin[dir_idx==1], langmuir[dir_idx==1], 'o-', label='only netrin and chemokine')

des = np.linspace(-2, 2, 9)
# plt.plot(n_netrin_initial[dir_idx==0], langmuir_analytical_3(n_netrin_initial[dir_idx==0], n_grid_sites[dir_idx==0], np.log(binding_probability[2, 1])), label='N = grid sites')
#plt.plot(n_netrin_initial[dir_idx==0], langmuir_analytical_3(n_netrin_initial[dir_idx==0], n_netrin_initial[dir_idx==0]+n_chemokine_initial[dir_idx==0], np.log(binding_probability[2, 1])), label='N = CCL5+Net1')
# plt.plot(n_netrin_initial[dir_idx==0], langmuir_analytical_3(n_netrin_initial[dir_idx==0], n_all_initial[dir_idx==0], np.log(binding_probability[2, 1])), label='N = all particles')

# plt.plot(n_netrin[dir_idx==0], langmuir_analytical_3(n_netrin[dir_idx==0], n_grid_sites[dir_idx==0], np.log(binding_probability[2, 1])), label='N = grid sites (avg)')
# plt.plot(n_netrin[dir_idx==1], langmuir_analytical_3(n_netrin[dir_idx==1], n_netrin_initial[dir_idx==1]+n_chemokine_initial[dir_idx==1], np.log(binding_probability[2, 1])), label='N = CCL5+Net1 (avg)')
# plt.plot(n_netrin[dir_idx==1], langmuir_analytical_3(n_netrin[dir_idx==1], n_netrin[dir_idx==1]+n_chemokine[dir_idx==1], np.log(binding_probability[2, 1])), label='N1 = CCL5+Net1 (eq)')
# plt.plot(n_netrin[dir_idx==0], langmuir_analytical_3(n_netrin[dir_idx==0], n_netrin[dir_idx==0]+n_chemokine[dir_idx==0], np.log(binding_probability[2, 1])), label='N0 = CCL5+Net1 (eq)')
# plt.plot(n_netrin[dir_idx==1], langmuir_analytical_3(n_netrin[dir_idx==1], n_netrin_initial[dir_idx==1]+n_chemokine_initial[dir_idx==1], np.log(binding_probability[2, 1])), label='N1 = CCL5+Net1 (init)')
#for de in des:

bp = binding_probability[2, 1]
bp = 69.07
plt.plot(n_netrin[dir_idx==1], langmuir_analytical_3(n_netrin[dir_idx==1], n_netrin[dir_idx==1]+n_chemokine[dir_idx==1]+n_cn_complex[dir_idx==1], np.log(bp)), label=f'pb = {bp:.1f} N1 = CCL5+Net1+CN (eq)')
#plt.plot(n_netrin[dir_idx==1], langmuir_analytical_3(n_netrin[dir_idx==1], n_netrin_initial[dir_idx==1]+n_chemokine_initial[dir_idx==1], np.log(bp+de)), label=f'pb = {bp}+{de} N1 = CCL5+Net1+CN (eq) 2')
#plt.plot(n_netrin_initial[dir_idx==1], langmuir_analytical_3(n_netrin_initial[dir_idx==1], n_netrin_initial[dir_idx==1]+n_chemokine_initial[dir_idx==1], np.log(bp+de)), label=f'pb = {bp}+{de} N1 = CCL5+Net1+CN (eq) 3')
plt.plot(n_netrin[dir_idx==1], langmuir_analytical(n_netrin[dir_idx==1]/n_grid_sites[dir_idx==1], np.log(bp+de)), label=f'pb = {bp}+{de} N1 = CCL5+Net1+CN (eq) 3')
#plt.plot(n_netrin[dir_idx==1], langmuir_analytical_3(n_netrin[dir_idx==1], n_netrin[dir_idx==1]+n_chemokine[dir_idx==1], np.log(binding_probability[2, 1])), label='N1 = CCL5+Net1 (eq)')
# plt.plot(n_netrin[dir_idx==1], langmuir_analytical_3(n_netrin[dir_idx==1], n_netrin[dir_idx==1]+n_chemokine[dir_idx==1]+2*n_cn_complex[dir_idx==1], np.log(binding_probability[2, 1])), label='N1 = CCL5+Net1+2*CN (eq)')
#plt.plot(n_netrin[dir_idx==1], langmuir_analytical_3(n_netrin_initial[dir_idx==1], n_netrin_initial[dir_idx==1]+n_chemokine_initial[dir_idx==1], np.log(binding_probability[2, 1])), label='N = CCL5+Net1 (init)')
# plt.plot(n_netrin[dir_idx==0], langmuir_analytical_3(n_netrin[dir_idx==0], n_netrin_initial[dir_idx==0], np.log(binding_probability[2, 1])), label='N = Net1 (avg)')
# plt.plot(n_netrin[dir_idx==0], langmuir_analytical_3(n_netrin[dir_idx==0], n_all_initial[dir_idx==0], np.log(binding_probability[2, 1])), label='N = all particles (avg)')


# plt.plot(n_netrin_initial[dir_idx==1], langmuir_analytical_3(n_netrin_initial[dir_idx==1], n_grid_sites[dir_idx==1], np.log(binding_probability[2, 1])), label='N = grid sites')
# plt.plot(n_netrin_initial[dir_idx==1], langmuir_analytical_3(n_netrin_initial[dir_idx==1], n_netrin_initial[dir_idx==1]+n_chemokine_initial[dir_idx==1], np.log(binding_probability[2, 1])), label='N = CCL5+Net1')
# plt.plot(n_netrin_initial[dir_idx==1], langmuir_analytical_3(n_netrin_initial[dir_idx==1], n_all_initial[dir_idx==1], np.log(binding_probability[2, 1])), label='N = all particles')

# plt.plot(n_netrin[dir_idx==1], langmuir_analytical_3(n_netrin[dir_idx==1], n_grid_sites[dir_idx==1], np.log(binding_probability[2, 1])), label='N = grid sites (avg)')
# plt.plot(n_netrin[dir_idx==1], langmuir_analytical_3(n_netrin[dir_idx==1], n_netrin_initial[dir_idx==1]+n_chemokine_initial[dir_idx==1], np.log(binding_probability[2, 1])), label='N = CCL5+Net1 (avg)')
# plt.plot(n_netrin[dir_idx==1], langmuir_analytical_3(n_netrin[dir_idx==1], n_netrin_initial[dir_idx==1], np.log(binding_probability[2, 1])), label='N = Net1 (avg)')
# plt.plot(n_netrin[dir_idx==1], langmuir_analytical_3(n_netrin[dir_idx==1], n_all_initial[dir_idx==1], np.log(binding_probability[2, 1])), label='N = all particles (avg)')

plt.legend()
plt.xscale('log')
#plt.xlabel('Netrin concentration [%]')
plt.xlabel('Netrin concentration [%]')
plt.ylabel("Langmuir's adsorption isotherm")
plt.ylim(0, 1)
plt.show()
plt.savefig('figures/langmuir_analytical.png')


print('\ndone')



# traj = h5py.File(fname_traj, 'r')['trajectory'][:-1]

# fname_movie = '/net/storage/janmak98/chemokine/output/movies/sim_test.mp4'
# animate(traj, show=False, save=True, fname=fname_movie)
