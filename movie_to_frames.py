# This program saves frames of a movie as individual .png files to facilitate analysis
import os
import numpy as np
# Install moviepy with: conda install -c conda-forge moviepy
from moviepy.editor import VideoFileClip

time_step_seconds = 0.5 # time step between frames to save
# Define directory of interest
base_dir = os.path.join(os.path.expanduser('~'),'Desktop','User Data','Jack','190110_TPBi')
movie_file = 'mat=TPBi_sub=Si_T=150C_Mag=50x_Polarized=0_t=45nm-000066.avi'
# Or select interactively:
#root = tk.Tk()
#root.withdraw()
#movie_file = filedialog.askopenfilename()

# open video clip
clip = VideoFileClip(os.path.join(base_dir,movie_file))
# Export frames
print('Total clip duration = ' + str(clip.duration) + 's')
# Make directory for frames, if doesn't already exist
frames_dir = os.path.join(base_dir,'Frames')
if not os.path.isdir(frames_dir):
    os.mkdir(frames_dir)
# Save frames
for t in np.arange(0,clip.duration,time_step_seconds):
    clip.save_frame(os.path.join(frames_dir,"time="+str(t)+"s.png"), t=t)