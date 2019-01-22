import os
# Install moviepy with: conda install -c conda-forge moviepy
from moviepy.editor import VideoFileClip

time_step_seconds = 0.25 # time step between frames to save
# Define directory of interest
base_dir = os.path.join('2018-05-25_TPBi_Thickness','165C_50nm_timeseries')
movie_file = 'something.avi'
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
    clip.save_frame(os.path.join(frames_dir,"t="+str(t)+".png"), t=t)