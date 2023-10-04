import h5py
from animation import animate

fname_traj = '/home/janmak98/Documents/masterthesis/chemokine/code/monte_carlo_simulation/chemokine/trajectories/traj_2023-10-04_12h-19m-11s.hdf5'
traj = h5py.File(fname_traj, 'r')['trajectory'][:-1]

fname_movie = '/net/storage/janmak98/chemokine/output/movies/sim_test.mp4'
animate(traj, show=False, save=True, fname=fname_movie)
