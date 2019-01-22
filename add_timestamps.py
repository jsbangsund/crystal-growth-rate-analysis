# This program allows timestamps to be added to microscope image filenames
# Takes timestamp from last modified time, so be careful that the modified time reflects
# when the image was taken, not when they were uploaded to the cloud (for example)
# This is needed for certain programs, such as Lumera Infinity, which don't allows
# timestamps to be added to a filename
import tkinter as tk
from tkinter import filedialog
import glob
import os
from datetime import datetime
still_picking=True
while still_picking:
    # Pick files to add timestamps to
    root = tk.Tk()
    root.withdraw()
    image_files = filedialog.askopenfilenames()
    if image_files=='':
        still_picking = False
    # Extract date modified in seconds
    raw_times = [os.path.getmtime(f) for f in image_files]
    times = [t - min(raw_times) for t in raw_times]
    for idx,f in enumerate(image_files):
        f_path = os.path.dirname(f)
        old_name = os.path.basename(f)
        file_extension = '.' + old_name.split('.')[-1]
        old_name = old_name[:-len(file_extension)] # remove file extension
        time_stamp = datetime.fromtimestamp(raw_times[idx]).strftime('%Y-%m-%d %H_%M_%S')
        new_name = 'time=' + '{:.1f}'.format(times[idx]) + 's_' + old_name + '_' + time_stamp + file_extension
        os.rename(f,os.path.join(f_path,new_name))