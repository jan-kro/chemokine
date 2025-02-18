import chemokine as ck 
import sys 


#* This script will use the input from the command line to create a movie 
#* of the simulation data, for every system given in the directory.

#* run the script with the following command:
#* python create_movie.py /path/to/config_directory

# decide how many simulateion frames to skip, per animation frame
STRIDE = 1

#! this scales the length scale of the system to micrometers  
a_scale = 0.01769

# handle path names
dirs_sys = ck.analysis_tools.validate_dir_sys_input(sys.argv)

# get config filenames
fnames_config = ck.analysis_tools.list_fnames(dirs_sys, return_dirs=False, return_names=False) 

for k, fname_cfg in enumerate(fnames_config):

    cfg = ck.utils.load_config_dict(fname_cfg)
    print("Rendering animation for system", cfg["name_sys"], "...")
    
    ck.utils.animate_ffmpeg_from_config(
        config=cfg,
        stride=STRIDE,
        delete_frames=True,
        marker_scale=2.0,
        plot_edge=4.0*a_scale,
        a_scale=a_scale,
        axis_labels=["$x\,/\,\mathrm{\mu m}$","$y\,/\,\mathrm{\mu m}$"],
        )
    
    fname = ck.utils.get_fname(cfg["dir_sys"], cfg["name_sys"], ftype="animation")
    print("Saved animation to", fname)