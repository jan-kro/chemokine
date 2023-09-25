from matplotlib.animation import FuncAnimation
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

def animate(traj_chemokine:                np.ndarray,
            traj_netrin:                   np.ndarray,
            traj_heparansulfate:           np.ndarray,
            traj_chemokine_netrin:         np.ndarray,
            traj_chemokine_heparansulfate: np.ndarray,
            grid_size:                     np.ndarray,
            show:                          bool        = True,
            save:                          bool        = False,
            fname:                         str         = None,
            fps:                           int         = 15,
            collagen_y:                    int         = None,
            marker_scale:                  float       = 1.0,
            plot_edge:                     int         = 5):
    """
    Animates the 2D trajectories of the particles.
    
    Parameters
    ----------
    traj_<particle_type> : np.ndarray
        Trajectory of the particles of shape (n_steps, n_particles, 2).
    grid_size : np.ndarray
        Size of the grid with shape (2,).
    show : bool, optional
        Show the animation, by default True
    save : bool, optional
        Save the animation, by default False
    fname : str, optional
        Filename of the animation, by default file gets saved in ./figures with timestamp
    fps : int, optional
        Frames per second, by default 15
    collagen_y : int, optional
        y-value of the collagen fiber, by default center of the grid
    marker_scale : float, optional
        scales the markers in case they are to big or to small
    plot_edge : int, optional
        edge around the grid, that is included in the plot, by default 5 grid steps
    """
    
    pos_0 = traj_chemokine
    pos_1 = traj_netrin
    pos_2 = traj_chemokine_netrin
    pos_3 = traj_heparansulfate
    pos_4 = traj_chemokine_heparansulfate
    
    n_steps = len(pos_0)
    
    if collagen_y is None:
        # if y-vallue of the collagen fiber is not given, set it to the middle of the grid
        collagen_y = np.round(grid_size[1]/2)

    fig, ax = plt.subplots()
    fig.set_figwidth(grid_size[0]/max(grid_size)*20)
    fig.set_figheight(grid_size[1]/max(grid_size)*20)

    box = ax.get_position()

    def update(i):
        
        #clear the frame
        ax.clear()
        
        ax.axhspan(collagen_y-1, collagen_y+1, alpha=0.3, color="red", label="Collagen fiber")
        
        ax.scatter(*pos_3[i].T, label="Heparansulfate",           color="grey",   marker=r"$\sim$",    s=150*marker_scale)
        ax.scatter(*pos_4[i].T, label="Chemokine-Heparansulfate", color="purple", marker="+",          s=80*marker_scale)
        ax.scatter(*pos_0[i].T, label="Chemokine",                color="blue",                        s=80*marker_scale)
        ax.scatter(*pos_1[i].T, label="Netrin",                   color="red",    marker=r"--$\cdot$", s=250*marker_scale)
        ax.scatter(*pos_2[i].T, label="Chemokine-Netrin",         color="green",                       s=90*marker_scale)
        

        ax.set_xlim(-plot_edge, grid_size[0]+plot_edge)    
        ax.set_ylim(-plot_edge, grid_size[1]+plot_edge)
        
        ax.plot([0, grid_size[0]], [0, 0],                       "--", color="black")
        ax.plot([grid_size[0], 0], [grid_size[1], grid_size[1]], "--", color="black")
        ax.plot([grid_size[0],  grid_size[0]], [0, grid_size[1]],      color="black")
        ax.plot([0, 0], [grid_size[1], 0],                             color="black")
        
        
        # draw legend outside of the plot
        
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #draw frame
        plt.draw()
        #pause to show the frame
        #plt.pause(0.05)
        
    anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)

    if show:
        plt.show()

    if save:
        if fname is None:
            now = datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
            fname = os.path.join(os.getcwd(), 'movies', 'sim_'+now+'.mp4')
        print(f'saving animation (fps = {fps:d}) ...')
        anim.save(fname, fps=fps)
        print(f'animation saved at {fname}.')