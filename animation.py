from matplotlib.animation import FuncAnimation
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

def animate(trajectory:                    np.ndarray,
            show:                          bool        = True,
            save:                          bool        = False,
            fname:                         str         = None,
            fps:                           int         = 15,
            marker_scale:                  float       = 1.0,
            plot_edge:                     int         = 5):
    """
    Animates the 2D trajectories of the particles.
    
    Parameters
    ----------
    traj : np.ndarray
        Trajectory of the particles of shape (n_steps, L_x, L_y).
    show : bool, optional 
        Show the animation, by default True
    save : bool, optional
        Save the animation, by default False
    fname : str, optional
        Filename of the animation, by default file gets saved in ./figures with timestamp
    fps : int, optional
        Frames per second, by default 15
    marker_scale : float, optional
        scales the markers in case they are to big or to small
    plot_edge : int, optional
        edge around the grid, that is included in the plot, by default 5 grid steps
    """
    n_steps = trajectory.shape[0]
    grid_size = trajectory[0].shape

    fig, ax = plt.subplots()
    fig.set_figwidth(grid_size[0]/max(grid_size)*20)
    fig.set_figheight(grid_size[1]/max(grid_size)*20)

    box = ax.get_position()

    def update(i):
        
        #clear the frame
        ax.clear()
        
        ax.scatter(*np.where(trajectory[i] == 3), label="Heparansulfate",           color="grey",   marker=r"$\sim$",    s=150*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 5), label="Chemokine-Heparansulfate", color="purple", marker="+",          s=80*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 1), label="Chemokine",                color="blue",                        s=80*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 2), label="Netrin",                   color="red",    marker=r"--$\cdot$", s=250*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 4), label="Chemokine-Netrin",         color="green",                       s=90*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 6), label="Collagen Sites",           color="k",      marker="x",          s=90*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 7), label="CN Collagen Sites",           color="k",      marker="o",          s=90*marker_scale)
        ax.scatter(*np.where(trajectory[i] == 8), label="N Collagen Sites",           color="k",      marker="*",          s=90*marker_scale)
        

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