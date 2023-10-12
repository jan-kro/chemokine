import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_grid(positions,
              show: bool = True,
              save: bool = True,
              fname: str = None):
    
    
    # import background image of collagen fiber
    # img = plt.imread("collagen_fiber.png")
    
    grid_size = positions.shape
        
    fig, ax = plt.subplots()
    fig.set_figwidth(grid_size[0]/max(grid_size)*20)
    fig.set_figheight(grid_size[1]/max(grid_size)*20)

    box = ax.get_position()

    marker_scale = 1
 
    plot_edge = 4

    ax.clear()

    #ax.imshow(img, extent=[0, grid_size[0], 0, grid_size[1])

    ax.scatter(*np.where(positions == 3), label="Heparansulfate",           color="grey",   marker=r"$\sim$",    s=150*marker_scale)
    ax.scatter(*np.where(positions == 5), label="Chemokine-Heparansulfate", color="purple", marker="+",          s=80*marker_scale)
    ax.scatter(*np.where(positions == 1), label="Chemokine",                color="blue",                        s=80*marker_scale)
    ax.scatter(*np.where(positions == 2), label="Netrin",                   color="red",    marker=r"--$\cdot$", s=250*marker_scale)
    ax.scatter(*np.where(positions == 4), label="Chemokine-Netrin",         color="green",                       s=90*marker_scale)
    ax.scatter(*np.where(positions == 6), label="Collagen Sites",           color="k",      marker="x",          s=90*marker_scale)
    ax.scatter(*np.where(positions == 7), label="CN Collagen Sites",           color="k",      marker="o",          s=90*marker_scale)
    ax.scatter(*np.where(positions == 8), label="N Collagen Sites",           color="k",      marker="*",          s=90*marker_scale)


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
    if show:
        plt.show()
    if save:
        if fname is None:
            now = datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
            fname = os.path.join(os.getcwd(), 'plots', 'plot_'+now+'.png')
        plt.savefig(fname)
        print(f'plot saved at {fname}.')