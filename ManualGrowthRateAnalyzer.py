# Improvements to consider:
# give option for time identifier string in case time=*s wasn't followed
# To ensure extracting perpendicular to growth front, could draw tangent line and specify
    # point where the tangent touches as the center point for a 90 degree rotation
# Other options:
# http://bigwww.epfl.ch/thevenaz/pointpicker/
###################################################################
# Imports
import os
import glob
import pickle
from collections import OrderedDict
import pandas as pd
import numpy as np
from sys import platform as sys_pf
import matplotlib
if sys_pf == 'darwin':
    matplotlib.use("TkAgg") # This fixes crashes on mac
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
# sci-kit image
from skimage import exposure
from skimage.color import rgb2gray
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.measure import profile_line
# GUI imports
import os
import csv
import time
import datetime
import tkinter as tk
from tkinter import LEFT, RIGHT, W, E, N, S, INSERT, END, BOTH
from tkinter.filedialog import (askopenfilename, askdirectory,
                             askopenfilenames)
from tkinter.ttk import Style,Treeview, Scrollbar, Checkbutton
import tkinter.ttk as ttk
# Plotting specifics
# UI
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                         NavigationToolbar2Tk)
from matplotlib.figure import Figure
# Scalebar
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
# Misc.
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import SpanSelector
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib.patches import Rectangle
# Colors
from palettable.tableau import Tableau_10, Tableau_20
from palettable.colorbrewer.qualitative import Set1_9
matplotlib.rc("savefig",dpi=100)
################################################################################

def setNiceTicks(ax,Nx=4,Ny=4,yminor=2,xminor=2,
               tick_loc=('both','both'),logx=False,logy=False):
    # If one of the axes is log, just use defaults
    # Things get screwy on log scales with these locators
    # tick_loc = (x,y) where x is 'both','top',or'bottom'
    #             y is 'both','left','right'
    if not logx:
        ax.xaxis.set_minor_locator(AutoMinorLocator(xminor))
        ax.xaxis.set_major_locator(MaxNLocator(Nx))

    if not logy:
        ax.yaxis.set_minor_locator(AutoMinorLocator(yminor))
        ax.yaxis.set_major_locator(MaxNLocator(Ny))

    # set tick length/width if desired
    ax.tick_params(grid_alpha=0)
    #ax.tick_params(direction='in', length=5, width=1.5, colors='k',
    #           grid_color='gray', grid_alpha=0.5)
    #ax.tick_params(direction='in', length=2.5, width=1, colors='k',which='minor')
    # set ticks at top and right of plot if desired
    if tick_loc:
        ax.xaxis.set_ticks_position(tick_loc[0])
        ax.yaxis.set_ticks_position(tick_loc[1])
    else:
        ax.set_xticks([])
        ax.set_yticks([])

def set_new_im_data(ax,im_data,new_img):
    # Change data extent to match new image
    im_data.set_extent((0, new_img.shape[1], new_img.shape[0], 0))
    # Reset axes limits
    ax.set_xlim(0,new_img.shape[1])
    ax.set_ylim(new_img.shape[0],0)
    # Now set the data
    im_data.set_data(new_img)

# Calibration values for Nikon microscope
micron_per_pixel = {'4x':1000/696, '10x':1000/1750,
                  '20x':500/1740, '50x':230/2016}
# This is obtained by multiplying micron_per_pixel by 2048,
# which is the pixel width for images saved by the Lumenera software
image_width_microns = {'4x':  2942.5,
                         '20x':  588.5,
                         '10x': 1170.3,
                         '50x':  233.7}
    
def get_line_length(line,mag,unit='um',length_per_pixel=None):
    '''
    ax = axis handle
    length = length of scalebar in 'unit'
    unit = unit of length, mm for millimeter or um for microns
    mag = magnification of microscope, '4x','10x','20x',or '50x'
    length_per_pixel = conversion from pixel to length
        default is None, using calibration factors for the Nikon
    height = height of scalebar
    loc = location specifier of scalebar
    '''
    if unit == 'um':
        factor = 1
    if unit == 'mm':
        factor = 1e-3
    # calibration distances for the Nikon microscope
    micron_per_pixel = {'4x':1000/696, '10x':1000/1750,
                       '20x':500/1740, '50x':230/2016}
    if not length_per_pixel:
        length_per_pixel = micron_per_pixel[mag]
    x,y = zip(*line)
    pixels = np.sqrt( (x[1]-x[0])**2 + (y[1]-y[0])**2 )
    length = pixels * length_per_pixel * factor
    return length



class GrowthRateAnalyzer(ttk.Frame):
    def __init__(self,parent):
        ttk.Frame.__init__(self, parent)
        self.parent = parent
        self.root = ttk.Frame
        # Initialization booleans
        self.last_img_process_settings = {}
        self.crop_initialized = False
        self.axes_ranges_initialized = False
        self.threshold_initialized = False
        self.save_initialized = False
        self.df_file = None
        self.base_dir = os.getcwd()
        # Default dir for troubleshooting purposes
        self.base_dir = os.path.join(
            os.path.expanduser('~'),'Google Drive','Research','Data','Gratings')
        self.base_dir = os.path.join(
            os.path.expanduser('~'),'Google Drive','Research','Data','Gratings',
            '2019-01-09_Capped TPBi','TPBi_30nm_Alq3','190C','timeseries_10x')
        self.time_files = glob.glob(os.path.join(self.base_dir,'*.tif'))
        # initialize dataframe save location
        self.df_dir = os.path.join(os.getcwd(),'dataframes')
        if not os.path.isdir(self.df_dir):
            os.mkdir(self.df_dir)
        self.configure_gui()
    def configure_gui(self):
        # Master Window
        self.parent.title("Manual Growth Rate Analysis")
        #self.style = Style()
        #self.style.theme_use("default")
        # Set ttk style
        self.style = ttk.Style(self.parent)
        self.style.theme_use('clam')
        self.style.configure('.', background='#eeeeee')
        # Lay out all the Frames
        file_container = ttk.Frame(self.parent)
        file_container.pack()
        # Create a frame to hold sample properties and the plotter
        sample_props_and_plot_container = ttk.Frame(self.parent)
        sample_props_and_plot_container.pack()
        # For entries of sample properties
        self.sample_props_container = ttk.Frame(sample_props_and_plot_container)
        self.sample_props_container.pack(side=LEFT)
        # For various function buttons
        button_container = ttk.Frame(sample_props_and_plot_container)
        button_container.pack(side=LEFT)
        # For the plot of the image crop
        image_container = ttk.Frame(sample_props_and_plot_container)
        image_container.pack(side=LEFT)
        # For the treeview of tabular data
        #tree_container = ttk.Frame(sample_props_and_plot_container)
        #tree_container.pack(side=LEFT)#fill=BOTH, expand=True
        # For plotting radius vs. time
        self.plot_container = ttk.Frame(sample_props_and_plot_container)
        self.plot_container.pack(side=LEFT)
        # For entry of contrast or other image manipulation
        #self.contrast_container = ttk.Frame(self.parent)
        #self.contrast_container.pack()
        
        # Open directory prompt
        self.l_file_directory = ttk.Label(file_container,
                                 text='Time Series Directory')
        self.l_file_directory.grid(row=0, column=0, sticky=W)
        self.t_file_dir = tk.Text(file_container)
        self.t_file_dir.configure(height = 1, width=70)
        self.t_file_dir.grid(row=0, column=1, sticky=W)
        # This button is no longer needed
        #self.b_getDir = tk.ttk.Button(file_container, command=self.get_directory_click)
        #self.b_getDir.configure(text="Open Dir.")
        #self.b_getDir.grid(row=0, column=2, sticky=W)
        # Open file
        self.b_getFile = ttk.Button(file_container, command=self.open_images_click)
        self.b_getFile.configure(text="Open Files")
        self.b_getFile.grid(row=0, column=3, sticky=W)
        # Select how times should be extracted
        ttk.Label(file_container,text="Get time from:").grid(row=0,column=4)
        self.s_time_source = tk.StringVar()
        self.s_time_source.set('Filename (time=*s)')
        self.e_time_source = ttk.OptionMenu(file_container,self.s_time_source,'Filename (time=*s)',
                                *['Date Modified','Filename (time=*s)'])
        self.e_time_source.grid(row=0,column=5)
        self.e_time_source.config(width=17)
        
        # Set-up sample properties:
        self.configure_sample_props()
        
        # Image display figure
        fig_height = 6
        self.image_fig, self.image_ax = plt.subplots(figsize=(2048/1536 * fig_height,fig_height))
        #self.image_fig.subplots_adjust(wspace=0.29,left=0.01,bottom=0.17,top=.95,right=0.75)
        self.image_canvas = FigureCanvasTkAgg(self.image_fig, master=image_container)
        self.image_canvas.draw()
        self.image_canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.image_canvas, image_container)
        self.toolbar.update()
        
        # Radius vs. time figure
        self.plot_fig, self.plot_ax = plt.subplots(figsize=(fig_height,fig_height))
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=self.plot_container)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.plot_ax.set_xlabel('Time (s)')
        self.plot_ax.set_ylabel('Grain Radius ($\mu$m)')
        
        # Crop Region buttons
        #self.fileDirLabel = ttk.Label(file_container, text = 'Pick image files')
        #self.fileDirLabel.grid(row=1, column=0, sticky=W)
        b_width = 19
        self.b_plotCrop = ttk.Button(button_container, command=self.pick_crop_region)
        self.b_plotCrop.configure(text="Pick Crop")
        self.b_plotCrop.grid(row=0, column=0, sticky=W)
        self.b_plotCrop.config(width=b_width)

        self.b_draw_lines = ttk.Button(button_container,command=self.draw_line_segments)
        self.b_draw_lines.configure(text="Pick Direction")
        self.b_draw_lines.grid(row=1, column=0, sticky=W)
        self.b_draw_lines.config(width=b_width)

        # self.b_get_lines = ttk.Button(button_container, command=self.get_line_segments)
        # self.b_get_lines.configure(text="Get Direction")
        # self.b_get_lines.grid(row=2, column=0, sticky=E)
        # self.b_get_lines.config(width=b_width)
        
        self.b_check_edge = ttk.Button(button_container, command=self.start_edge_selection)
        self.b_check_edge.configure(text="Start Edge Selection")
        self.b_check_edge.grid(row=2, column=0, sticky=W)
        self.b_check_edge.config(width=b_width)
        
        self.b_next_frame = ttk.Button(button_container, command=self.forward_frame)
        self.b_next_frame.configure(text="Forward ->")
        self.b_next_frame.grid(row=3, column=0, sticky=E)
        self.b_next_frame.config(width=b_width)
        # Bind the arrow keys
        self.parent.bind('<Right>', self.forward_frame)
        
        self.b_reverse_frame = ttk.Button(button_container, command=self.reverse_frame)
        self.b_reverse_frame.configure(text="<- Reverse")
        self.b_reverse_frame.grid(row=4, column=0, sticky=E)
        self.b_reverse_frame.config(width=b_width)
        # Bind the arrow keys
        self.parent.bind('<Left>', self.reverse_frame)
        
        self.b_reset_crop = ttk.Button(button_container, command=self.reset_crop)
        self.b_reset_crop.configure(text="Reset Crop")
        self.b_reset_crop.grid(row=5, column=0, sticky=W)
        self.b_reset_crop.config(width=b_width)
        
        self.b_fit = ttk.Button(button_container, command=self.fit_growth_rate)
        self.b_fit.configure(text="Fit Growth Rate")
        self.b_fit.grid(row=6, column=0, sticky=W)
        self.b_fit.config(width=b_width)
        
        # self.b_pick_df = ttk.Button(button_container, command=self.pick_df)
        # self.b_pick_df.configure(text="Pick DF")
        # self.b_pick_df.grid(row=5, column=0, sticky=W)
        # self.b_pick_df.config(width=b_width)
        
        self.b_save_results = ttk.Button(button_container, command=self.save_results)
        self.b_save_results.configure(text="Save Results")
        self.b_save_results.grid(row=7, column=0, sticky=W)
        self.b_save_results.config(width=b_width)

        self.pack(fill=BOTH, expand=1)
    
    def configure_sample_props(self):
        # Sample properties
        # Used as metadata in save dataframe
        self.sample_props =  OrderedDict([
                           ('growth_date',
                             {'label':'Growth Date:',
                              'default_val':'yyyy-m-dd',
                              'type':'Entry',
                              'dtype':'string'}),
                           ('material',
                             {'label':'Material (Sep by /):',
                              'default_val':'TPBi',
                              'type':'Entry',
                              'dtype':'string'}),
                           ('thickness_nm',
                             {'label':'Thickness (nm) (Sep by /):',
                              'default_val':'30',
                              'type':'Entry',
                              'dtype':'float'}),
                           ('deposition_rate_aps',
                             {'label':'Deposition Rate (A/s):',
                              'default_val':'1',
                              'type':'Entry',
                              'dtype':'float'}),
                           ('deposition_temp_c',
                             {'label':'Deposition Temp (C):',
                              'default_val':'25',
                              'type':'Entry',
                              'dtype':'float'}),
                           ('anneal_temp_c',
                             {'label':'Anneal Temp (C):',
                              'default_val':'165',
                              'type':'Entry',
                              'dtype':'float'}),
                           ('substrate',
                             {'label':'Substrate:',
                              'default_val':'Si',
                              'type':'Entry',
                              'dtype':'string'}),
                           ('note',
                             {'label':'Note:',
                              'default_val':'None',
                              'type':'Entry',
                              'dtype':'string'})
                           ])
        self.s_sample_props={}
        self.e_sample_props = ['']*len(self.sample_props)
        row_idx=0
        for key,input_dict in self.sample_props.items():
            ttk.Label(self.sample_props_container,text=input_dict['label']).grid(row=row_idx,column=0)
            self.s_sample_props[key] = tk.StringVar()
            self.s_sample_props[key].set(input_dict['default_val'])
            if input_dict['type']=='Entry':
                self.e_sample_props[row_idx] = ttk.Entry(self.sample_props_container,
                                             textvariable=self.s_sample_props[key],width=10)
            elif input_dict['type']=='OptionMenu':
                self.e_sample_props[row_idx]=tk.OptionMenu(
                                        self.sample_props_container,
                                        self.s_sample_props[key],
                                        *self.sample_props[key]['options'])
            self.e_sample_props[row_idx].grid(row=row_idx,column=1)
            row_idx+=1
        # Add drop down for image magnification:
        # This isn't included in self.sample_props because that would have involved more changes
        ttk.Label(self.sample_props_container,text="Image Mag").grid(row=row_idx,column=0)
        self.s_mag=tk.StringVar()
        self.s_mag.set('10x')
        self.e_mag=ttk.OptionMenu(self.sample_props_container,self.s_mag,'10x',
                           *['4x','10x','20x','50x'])
        self.e_mag.grid(row=row_idx,column=1)
        self.e_mag.config(width=5)
    # TODO - update with contrast options
    def configure_contrast(self):
        # Destroy previous elements in these frames
        for child in self.contrast_container.winfo_children():
            child.destroy()
        # Contrast enhancement methods
        ttk.Label(self.contrast_container,text="Contrast Method:").grid(row=0,column=0)
        self.s_contrast_method=tk.StringVar()
        self.e_contrast_method=ttk.OptionMenu(
            self.contrast_container,self.s_contrast_method,'Adapt. Eq. Hist.',
            *['Adapt. Eq. Hist.','Eq. Hist.','Linear'],command=self.set_contrast_method)
        self.e_contrast_method.grid(row=1,column=0)
        
        ttk.Label(self.contrast_container,text="Inv. Thresh?").grid(row=0,column=5)
        ttk.Label(self.contrast_container,text="Multi Ranges?").grid(row=0,column=6)
        ttk.Label(self.contrast_container,text="Eq. Hist?").grid(row=0,column=7)
        ttk.Label(self.contrast_container,text="Clip Limit").grid(row=0,column=8)
        
        if self.s_contrast_method.get() == 'Linear':
            ttk.Label(self.contrast_container,text="Threshold").grid(row=1,column=1)
            ttk.Label(self.contrast_container,text="Lower").grid(row=0,column=2)
            ttk.Label(self.contrast_container,text="Upper").grid(row=0,column=3)

        self.s_threshold_lower=tk.StringVar()
        self.s_threshold_lower.set('60')
        self.e_s_threshold_lower=ttk.Entry(self.threshold_container,
                                   textvariable=self.s_threshold_lower,width=5)
        self.e_s_threshold_lower.grid(row=1,column=2)

        self.s_threshold_upper=tk.StringVar()
        self.s_threshold_upper.set('70')
        self.e_s_threshold_upper=ttk.Entry(self.threshold_container,
                                   textvariable=self.s_threshold_upper,width=5)
        self.e_s_threshold_upper.grid(row=1,column=3)

        self.s_clip_limit = tk.StringVar()
        self.s_clip_limit.set('0.05')
        self.last_clip_limit=self.s_clip_limit.get()
        self.e_clip_limit = ttk.Entry(self.contrast_container,textvariable=self.s_clip_limit,width=5)
        self.e_clip_limit.grid(row=1,column=8)

        self.b_update_contrast = ttk.Button(self.contrast_container,
                                    command=self.update_contrast)
        self.b_update_contrast.configure(text="Update")
        self.b_update_contrast.grid(row=1, column=10, sticky=W)
        
    def get_directory_click(self):
        self.t_file_dir.delete("1.0",END)
        self.base_dir = askdirectory(initialdir=self.base_dir)
        self.t_file_dir.insert(INSERT, self.base_dir +'/')
    def pick_df(self):
        # Pick save dataframe
        self.df_file = askopenfilename(
                            initialdir=self.df_dir,
                            title='Choose DataFrame .pkl'
                            )
        #if self.df_file:
            #self.b_pick_df.configure(bg='green')
    def open_images_click(self):
        files = askopenfilenames(
                  initialdir=self.base_dir,title='Choose files',
                  filetypes=(("all files","*.*"),
								("png files",".png"),
                            ("tif files",".tif")))
        # If the prompt is canceled, returns empty string. Exit in this case.
        if files == '':
            return
        # Reset initialization for other functions
        self.crop_initialized = False
        self.threshold_initialized = False
        # Store filenames
        self.time_files = list(files)
        if len(self.time_files)==1:
            raise Exception('Please select more than one image file')
        # Load images
        # Commenting this out as slow for large number of images
        #self.full_images=['']*len(self.time_files)
        #for i,f in enumerate(self.time_files):
            #self.full_images[i] = misc.imread(f,mode='L')
        # Try to find magnification and other metadata, if base_dir has changed
        new_base_dir = os.path.dirname(self.time_files[0])
        if not new_base_dir==self.base_dir:
            base_file = os.path.basename(self.time_files[0])
            base_file = base_file.replace('-','_')
            splits = base_file.split('_')
            for split in splits:
                split2 = split.split('=')
                if 'mag' in split.lower():
                    # looking for _mag=##x_ or _magnification=##x_
                    if len(split2)>1:
                        self.s_mag.set(split2[1])
                        print(self.s_mag.get())
                elif 'sub' in split.lower():
                     # looking for _sub=*_ or _substrate=*_ in filename
                    if len(split2)>1:
                        self.s_sample_props['substrate'].set(split2[1])
                elif any(x in split2[0] for x in ['Ta','Tanneal','T']):
                    # looking for _T=*C_ or _Tanneal=*C_ or _Ta=*C_
                      if len(split2)>1:
                        self.s_sample_props['anneal_temp_c'].set(split2[1][:-1])
                elif any(x in split2[0] for x in ['t','thick','thickness']) and 'nm' in split:
                    # looking for _t=*nm_ or _thick=*nm_, etc.
                      if len(split2)>1:
                        self.s_sample_props['thickness_nm'].set(split2[1][:-2])
                      else:
                        self.s_sample_props['thickness_nm'].set(str(split2))
                elif 'mat=' in split:
                    self.s_sample_props['material'].set(split[4:])
                elif any(x in split for x in ['Td','Tgrowth','Tdep']):
                      if len(split2)>1:
                        self.s_sample_props['deposition_temp_c'].set(split2[1][:-1])

            # Try to find growth date from folder (move up a level up to 4 times)
            split = os.path.split(new_base_dir)
            for i in range(4):
                # check if the length matches yyyy-mm-dd
                if len(split[1].split('_')[0])==10: 
                    # check whether string is numeric at right positions for yyyy-mm-dd
                    if all(split[1][idx].isnumeric for idx in [0,1,2,3,5,6,8,9]):
                        self.s_sample_props['growth_date'].set(split[1].split('_')[0])
                        break
                else:
                    split = os.path.split(split[0])
        # Update directory:
        self.base_dir = new_base_dir
        self.t_file_dir.delete("1.0",END)
        self.t_file_dir.insert(INSERT, self.base_dir +'/')
        # Sort files by time
        self.extract_times_and_sort() # saves self.times and self.sort_indices
        self.t0 = self.times[self.sort_indices[0]]
        self.sorted_times = np.array(self.times)[self.sort_indices]-self.t0
        self.time_files = np.array(self.time_files)[self.sort_indices]
    def pick_crop_region(self,delete_line=False): 
        self.reset_image_display(reset_crop=False)
        # Zoom to region of interest in image. This will select crop region below
        # Pick the last time so the whole grain is contained within the crop region
        img=misc.imread(os.path.join(self.base_dir,self.time_files[-1]))#,mode='L'
        self.full_last_frame = img # store last frame 
        #img = exposure.rescale_intensity(img,in_range='image')
        #img = exposure.equalize_adapthist(img,clip_limit=0.05)
        if not self.crop_initialized:
            self.cropData = self.image_ax.imshow(img)
            self.image_canvas.draw()
        else:
            # Set new data
            set_new_im_data(self.image_ax,self.cropData,img)
            self.image_canvas.draw()
        self.crop_initialized = True
        self.axes_ranges_initialized = False
    def get_axes_ranges(self):
        x1,x2=self.image_ax.get_xlim()
        y2,y1=self.image_ax.get_ylim()
        self.x1=int(x1)
        self.x2=int(x2)
        self.y2=int(y2)
        self.y1=int(y1)
        string = ('x1 = ' + str(self.x1)+', x2 = ' + str(self.x2) +
                ', y1 = ' + str(self.y1) + ', y2 = ' + str(self.y2))
        print(string)
        self.image_ax.set_title(string)
        self.axes_ranges_initialized = True
        #self.l_crop_range.config(text=('x1 = ' + str(self.x1) +
        #                            ', x2 = ' + str(self.x2) +
        #                            ', y1 = ' + str(self.y1) +
        #                            ', y2 = ' + str(self.y2)))
        #print(x1,x2,y1,y2)
    def load_frame(self,frame_index):
        # Update crop range
        self.get_axes_ranges()
        # load image
        img=misc.imread(os.path.join(self.base_dir,self.time_files[frame_index]))
        # Set data
        self.cropData.set_data(img)
        self.image_canvas.draw()
        # crop
        #self.current_frame=img[self.y1:self.y2,self.x1:self.x2]
        # TODO enter contrast processing here
        #self.update_image_display(self.current_frame)
    def update_image_display(self,new_image):
        # Change data extent to match new cropped image
        self.cropData.set_extent((0, new_image.shape[1], new_image.shape[0], 0))
        # Reset axes limits
        self.image_ax.set_xlim(0,new_image.shape[1])
        self.image_ax.set_ylim(new_image.shape[0],0)
        # Now set the data
        self.cropData.set_data(new_image)
        self.image_ax.relim()
        self.image_canvas.draw()
    def reset_crop(self):
        self.image_ax.set_xlim(0,self.full_last_frame.shape[1])
        self.image_ax.set_ylim(self.full_last_frame.shape[0],0)
        self.image_canvas.draw()
    def reset_image_display(self,reset_crop=True,delete_line=True):
        if reset_crop:
            # Reset axes limits
            self.image_ax.set_xlim(0,self.full_last_frame.shape[1])
            self.image_ax.set_ylim(self.full_last_frame.shape[0],0)
        # Delete lines
        if delete_line:
            # Remove old lines
            for line in self.image_ax.lines:
                line.remove()
        # Delete picker points and Reset mpl picker
        try:
            self.pick_points.set_data([],[])
            self.image_canvas.mpl_disconnect(self.picker_cid)
        except:
            pass
        self.image_canvas.draw()
    def draw_line_segments(self):
        # Update crop range
        if not self.axes_ranges_initialized:
            self.get_axes_ranges()
        # Remove old lines
        for line in self.image_ax.lines:
            line.remove()
        for line in self.plot_ax.lines:
            line.remove()
        self.image_ax.set_title('Click to draw line endpoints')
        # Build lines
        self.line, = self.image_ax.plot([], [],'-or',ms=2,alpha=0.5)  # empty line
        self.linebuilder = LineBuilder(self.line)
        #self.ax[0].axis('off')
        self.image_canvas.draw()
    def get_line_segments(self):
        # Get line coordinates
        # One line drawn
        self.lines=[]
        if len(self.linebuilder.xs)==2:
            linex1,linex2 = self.linebuilder.xs
            liney1,liney2 = self.linebuilder.ys
            self.lines.append([(linex1,liney1),(linex2,liney2)])
        # Multiple lines drawn
        elif len(self.linebuilder.xs)%2<0.1:
            for i in np.arange(0,len(self.linebuilder.xs),2):
                linex1,linex2 = self.linebuilder.xs[i:i+2]
                liney1,liney2 = self.linebuilder.ys[i:i+2]
                self.lines.append([(linex1,liney1),(linex2,liney2)])
        # Incorrect number of points
        elif len(self.linebuilder.xs)%2==1:
            print('Incorrect number of points, draw start and endpoint for each direction of interest')
        #print(self.lines)
    def start_edge_selection(self):
        # Initialize current frame position
        self.current_frame_index = 0
        # Get line segments
        self.get_line_segments()
        # Initialize coordinates
        self.growth_edge_x = np.zeros(len(self.time_files))
        self.growth_edge_y = np.zeros(len(self.time_files))
        self.distances = np.zeros(len(self.time_files))
        # Stop the LineBuilder
        self.image_canvas.mpl_disconnect(self.linebuilder.cid)
        # Update crop range
        if not self.axes_ranges_initialized:
            self.get_axes_ranges()
        # Check if images dimensions are as expected. If not use image width
        # Not very robust yet
        img = misc.imread(os.path.join(self.base_dir,self.time_files[0]))
        if img.shape[1]==2048:
            length_per_pixel = micron_per_pixel[self.s_mag.get()]
        else:
            length_per_pixel = image_width_microns[self.s_mag.get()]/img.shape[1]
        # Get time from filenames, and sort by time
        self.extract_times_and_sort() # saves self.times and self.sort_indices
        sort_idx = self.sort_indices[self.current_frame_index]
        timeFile = self.time_files[sort_idx]
        self.t0 = self.times[sort_idx]
        self.sorted_times = np.array(self.times)[self.sort_indices]-self.t0
        self.reset_image_display(reset_crop=False,delete_line=False)
        self.load_frame(frame_index=sort_idx)
        self.image_ax.set_title('Frame #' + str(self.current_frame_index))
        
        self.pick_points, = self.image_ax.plot([], [],'ob',ms=2,alpha=0.7)  # empty line
        self.distances_line, = self.plot_ax.plot([], [],'o')
        #self.pointselector = PointSelector(self.pick_point)
        
        # To allow zoom to occur during, distinguish click and zoom
        # see https://stackoverflow.com/questions/48446351/distinguish-button-press-event-from-drag-and-zoom-clicks-in-matplotlib
        # MAX_CLICK_LENGTH = 0.1 # in seconds; anything longer is a drag motion

        # def onclick(event, ax):
            # self.image_ax.time_onclick = time.time()

        # def onrelease(event):
            # # Only clicks inside this axis are valid.
            # if event.inaxes == self.image_ax:
                # if event.button == 1 and ((time.time() - ax.time_onclick) < MAX_CLICK_LENGTH):
                    # print(event.xdata, event.ydata)
                    # # Draw the click just made
                    # ax.scatter(event.xdata, event.ydata)
                    # ax.figure.canvas.draw()
                # elif event.button == 2:
                    # print("scroll click")
                # elif event.button == 3:
                    # print("right click")
                # else:
                    # pass
        def on_pick(event):
            # Act on right click
            if event.button == 3:
                x = event.xdata
                y = event.ydata
                self.growth_edge_x[self.current_frame_index] = x
                self.growth_edge_y[self.current_frame_index] = y
                # Could just plot to current frame [:self.current_frame_index]
                self.pick_points.set_data(self.growth_edge_x,self.growth_edge_y)
                self.image_canvas.draw()
                self.get_distance()
                # If supporting multiple lines, remove this:
                self.forward_frame()
            #print(x,y)
            
            #print(self.growth_edge_x_coord)
        self.picker_cid = self.image_canvas.mpl_connect('button_press_event',on_pick)
        
    def get_closest_point_on_line(self,p1,p2,nearby_point):
        # see https://stackoverflow.com/questions/47177493/python-point-on-a-line-closest-to-third-point
        # When user picks a point, we want to find the closest point that lies on the drawn line
        # Using the function from the above stackexchange, we can get the closest point
        # which lies on a line defined by endpoints p1 and p2 to the outside point "nearby_point"
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = nearby_point
        dx, dy = x2-x1, y2-y1
        det = dx*dx + dy*dy
        a = (dy*(y3-y1)+dx*(x3-x1))/det
        return x1+a*dx, y1+a*dy
    
    def get_distance(self,nearby_point=None):
        # Check if images dimensions are as expected. If not use image width
        # Not very robust yet
        img = self.full_last_frame
        if img.shape[1]==2048:
            length_per_pixel = micron_per_pixel[self.s_mag.get()]
        else:
            length_per_pixel = image_width_microns[self.s_mag.get()]/img.shape[1]
        # Get endpoints of line
        line = self.lines[0]
        
        p1 = (line[0][0],line[0][1])
        p2 = (line[1][0],line[1][1])
        if nearby_point is None:
            nearby_point = (self.growth_edge_x[self.current_frame_index],
                          self.growth_edge_y[self.current_frame_index])
        point_on_line = self.get_closest_point_on_line(p1,p2,nearby_point)
        # line is list of tuples of start and endpoint
            # e.g. [(x1,y1),(x2,y2)]
        line_to_growth_edge = [p1,point_on_line]
        
        # self.distances[line_idx][idx] = get_growth_edge(
                        # denoised,self.lines[line_idx],
                        # length_per_pixel=length_per_pixel
                        # )
        total_line_length = get_line_length(
            line_to_growth_edge,mag=None,unit='um',length_per_pixel=length_per_pixel)

        self.distances[self.current_frame_index] = total_line_length
        self.distances_line.set_data(self.sorted_times,self.distances)
        self.plot_ax.axis([0,np.amax(self.times)*1.1,0,np.amax(self.distances)*1.1])
        self.plot_canvas.draw()
    # This function is connected to the right arrow key
    def forward_frame(self,_event=None):
        # Get coordinates from last frame
        #self.growth_edge_x_coord[self.current_frame_index] = self.pointselector.xs[0]
        #self.growth_edge_y_coord[self.current_frame_index] = self.pointselector.ys[0]
        # Update current frame position
        if self.current_frame_index < len(self.time_files)-1:
            self.current_frame_index += 1
            self.load_frame(frame_index=self.sort_indices[self.current_frame_index])
            self.image_ax.set_title('Frame #' + str(self.current_frame_index))
        else:
            print('last frame reached')
    # This function is connected to the left arrow key
    def reverse_frame(self,_event=None):
        # Get coordinates from last frame
        #self.growth_edge_x_coord[self.current_frame_index] = self.pointselector.xs[0]
        #self.growth_edge_y_coord[self.current_frame_index] = self.pointselector.ys[0]
        # Update current frame position
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.load_frame(frame_index=self.sort_indices[self.current_frame_index])
            self.image_ax.set_title('Frame #' + str(self.current_frame_index))
        else:
            print('first frame reached')
    def fit_growth_rate(self):
        self.growth_rates=[]
        self.growth_rates_string=[]
        self.growth_lines_fit=['']*len(self.lines)
        params = np.polyfit(self.sorted_times, self.distances, 1)
        line1,=self.plot_ax.plot(self.sorted_times,self.sorted_times*params[0]+params[1],'--',
            color='k',linewidth=1.5) # Tableau_10.mpl_colors[c_idx]
        self.growth_lines_fit.append(line1)
        print('{:.2f}'.format(params[0])+' micron/sec')
        self.growth_rates_string.append('{:.2f}'.format(params[0])+' micron/sec')
        self.growth_rates.append(params[0])
        self.plot_ax.set_title('{:.2f}'.format(params[0])+' $\mu$m/s')
        self.plot_canvas.draw()
    def get_img_process_settings(self):
        if self.s_edge_method.get()=='Threshold Grain':
            if self.bool_multi_ranges:
                threshold_lower = [float(x) for x in
                                self.s_threshold_lower.get().split(',')]
                threshold_upper = [float(x) for x in
                                self.s_threshold_upper.get().split(',')]
            else:
                threshold_lower = float(self.s_threshold_lower.get())
                threshold_upper = float(self.s_threshold_upper.get())
            multiple_ranges = self.bool_multi_ranges.get()
            threshold_out = self.bool_threshold_out.get()
        elif self.s_edge_method.get()=='Subtract Images':
            if self.bool_threshold_on.get():
                threshold_lower = float(self.s_threshold_lower.get())
            else:
                threshold_lower = None
            # Variables that aren't used in this method:
            threshold_upper = None
            threshold_out = None
            multiple_ranges = None
        return {'method':self.s_edge_method.get(),'disk':int(self.s_disk.get()),
                'threshold_lower':threshold_lower,'threshold_upper':threshold_upper,
                'crop_region':(self.x1,self.x2,self.y1,self.y2),
                'time_files':self.time_files,'equalize_hist':self.bool_eq_hist.get(),
                'clip_limit':float(self.s_clip_limit.get()),
                'threshold_out':threshold_out,'multiple_ranges':multiple_ranges}
    def extract_times_and_sort(self):
        if self.s_time_source.get()=='Date Modified':
            # Get time from last modified time
            t0 = datetime.datetime.fromtimestamp(os.path.getmtime(self.time_files[0]))
            self.times=[0]*len(self.time_files)
            for idx,timeFile in enumerate(self.time_files):
                ti = datetime.datetime.fromtimestamp(os.path.getmtime(self.time_files[idx]))
                self.times[idx] = (ti-t0).total_seconds()
        elif self.s_time_source.get()=='Filename (time=*s)':
            # Get time from filename'
            self.times=[0]*len(self.time_files)
            for idx,timeFile in enumerate(self.time_files):
                # Split first by "time=" then by "s" to get the numbers in between
                self.times[idx] = float((timeFile.split('time=')[1]).split('s')[0])
                #self.times[idx] = float((timeFile.split('t=')[1]).split('.png')[0])    
        # Loop through files
        self.sort_indices = sorted(range(len(self.times)), key=lambda k: self.times[k])
    def save_results(self):
        # Make save directory
        self.save_dir = os.path.join(self.base_dir,'analysis_results')
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        # Save growth rate file
        # header = 'time (s), radius (micron), x, y'
        # for i in range(1,self.distances.shape[0]+1):
            # header += 'line#'+str(i)+'(micron)'
            # if not i == self.distances.shape[0]:
                # header += ','
            # else:
                # header += '\n'
        # for i,growth_rate in enumerate(self.growth_rates_string):
            # header += growth_rate
            # if not i == len(self.growth_rates_string):
                # header += ','
        export_data = np.array([self.sorted_times,self.distances,
                             self.growth_edge_x,self.growth_edge_y])
        savename = self.increment_save_name(self.save_dir,'radius_vs_time','.csv')
        np.savetxt(os.path.join(self.save_dir,savename+'.csv'),
                 np.transpose(export_data),
                 delimiter=',',header='time (s),radius (micron),x,y')
        # savename = self.increment_save_name(self.save_dir,'growthrates','.csv')
        # with open(os.path.join(self.save_dir,savename+'.csv'), 'a') as f:
            # f.write('Line#,Growth Rate (micron/sec) \n')
            # for idx,growthRate in enumerate(self.growth_rates):
                # line = str(idx+1) + ',' + str(growthRate) + '\n'
                # f.write(line)
        # Save figures
        savename = self.increment_save_name(self.save_dir,'image_with_points','.png')
        self.image_fig.savefig(os.path.join(self.save_dir,savename+'.png'),dpi=200,bbox_inches='tight')
        savename = self.increment_save_name(self.save_dir,'radius_vs_time_plot','.png')
        self.plot_fig.savefig(os.path.join(self.save_dir,savename+'.png'),dpi=200,bbox_inches='tight')
        #savename = self.increment_save_name(self.save_dir,'threshold_plot','.png')
        #self.threshold_fig.savefig(os.path.join(self.save_dir,savename+'.png'),dpi=200,bbox_inches='tight')
        # Save dataframe
        # If no file was selected make new filename and df
        # if not self.df_file:
            # self.df_file = (self.increment_save_name(
                # self.df_dir,str(datetime.date.today()) + '_df','.pkl')
                # +'.pkl'
                # )
            # self.df = pd.DataFrame()
        # # otherwise, load df
        # else:
            # self.df = pd.read_pickle(os.path.join(self.df_dir,self.df_file))
        # # Also save a local df pkl file in the 'Analysis' folder, in case user is stupid and overwrites elsewhere
        # df_local_file = os.path.join(self.save_dir,'df.pkl')
        # if os.path.isfile(df_local_file):
            # df_local = pd.read_pickle(df_local_file)
        # else:
            # df_local = pd.DataFrame()
        # # Now append data
        # data_dict_list=[]
        # img_process_settings = self.get_img_process_settings()
        # for line_idx,growth_rate in enumerate(self.growth_rates):
            # temp_dict = {'growth_rate_umps':growth_rate,
                        # 'line':self.lines[line_idx],
                        # 'x1,x2,y1,y2':(self.x1,self.x2,self.y1,self.y2),
                        # 'image_files':[os.path.basename(x) for x in self.time_files],
                        # 'image_dir':os.path.split(self.time_files[0])[0],
                        # 'threshold_lower':img_process_settings['threshold_lower'],
                        # 'threshold_upper':img_process_settings['threshold_upper'],
                        # 'disk':int(self.s_disk.get()),
                        # 'histogram_equalization':self.bool_eq_hist.get(),
                        # 'edge_find_method':self.s_edge_method.get()}
            # for key,input_dict in self.sample_props.items():
                # if input_dict['dtype']=='string':
                    # temp_string = self.s_sample_props[key].get()
                    # if '/' in temp_string:
                        # temp_list = []
                        # for substring in temp_string.split('/'):
                            # temp_list.append(substring)
                        # temp_dict[key]=temp_list
                    # else:
                        # temp_dict[key]=temp_string
                # elif input_dict['dtype']=='float':
                    # # Try separating by '/' for layer stacks
                    # temp_string = self.s_sample_props[key].get()
                    # if '/' in temp_string:
                        # temp_list = []
                        # for sublayer in temp_string.split('/'):
                            # temp_list.append(float(sublayer))
                        # temp_dict[key]=temp_list
                    # else:
                        # temp_dict[key]=float(self.s_sample_props[key].get())
            # data_dict_list.append(temp_dict)
        # #print(data_dict_list)
        # self.df = self.df.append(data_dict_list,ignore_index=True)
        # df_local = df_local.append(data_dict_list,ignore_index=True)
        # #print(self.df)
        # # Pickle the dataframe
        # self.df.to_pickle(os.path.join(self.df_dir,self.df_file))
        # df_local.to_pickle(df_local_file)
        # # Save csv of data
        # #savename = self.increment_save_name(self.save_dir,'growth_rates_data','.csv')+'.csv'
        # df_local.to_csv(os.path.join(self.save_dir,'growth_rates_data.csv'))
        # print('results saved to ' + self.df_dir + self.df_file)

    def increment_save_name(self,path,savename,extension):
        name_hold = savename
        if name_hold.endswith(extension):
            name_hold = name_hold.rstrip(extension)
        if os.path.isfile(os.path.join(path,name_hold+extension)):
            for add_i in range(1,10):
                name_temp = name_hold + '_' + str(add_i)
                if not os.path.isfile(os.path.join(path,name_temp + extension)):
                    name_hold = name_temp
                    break
        return name_hold

    def label_lines(self):
        # Remove old text, if exists
        try:
            for txt in self.line_texts:
                txt.remove()
            self.line_texts=[]
        except:
            pass
        for line in self.image_ax[0].lines:
            line.remove()
        #img=misc.imread(os.path.join(self.base_dir,self.time_files[-1]))#,mode='L')
        #img = exposure.equalize_adapthist(img,clip_limit=0.05)
        #self.cropData.set_data(img[self.y1:self.y2,self.x1:self.x2])
        line, = self.image_ax[0].plot([], [], '-or')
        # Draw lines on image
        line.set_data(self.linebuilder.xs, self.linebuilder.ys)
        keyStr = '' #string to display growth rates
        self.line_texts=[]
        for idx in range(0, len(self.lines)):    #add labels to lines
            txt = self.image_ax[0].text(self.lines[idx][1][0],
                                self.lines[idx][1][1]-25,
                                idx+1, color = 'red',
                                size = 'x-large', weight = 'bold')
            self.line_texts.append(txt)
            keyStr = keyStr + str(idx+1) + ': ' + str(self.growth_rates_string[idx]) #add growth rate to keyStr
            if idx != len(self.lines)-1:
                keyStr = keyStr + '\n'
        #self.image_ax[0].text(1.05, 0.5, keyStr, transform=self.image_ax[0].transAxes, size = 'x-large', bbox=dict(facecolor='white', alpha=1))
        self.image_ax[0].axis('off')
        self.image_ax[0].set_title('')
        self.image_ax[0].relim()
        # Store the figure so it can be saved later
        self.labeledLinesFig = plt.gcf()
        self.image_canvas.draw()


# Interactively draw line
# You can draw multiple lines as well
class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        #print('click', event)
        if event.inaxes!=self.line.axes: return
        #first click sets values of nucleation site, creates point at n-site
        if len(self.xs) == 0:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.xs.append(self.x1)
            self.ys.append(self.y1)
        #second click connects n-site to click point
        elif len(self.xs) == 1:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
        #following clicks each create a line from n-site to click point
        else:
            self.xs.append(self.x1)
            self.ys.append(self.y1)
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)


        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()
        
# Interactively draw line
# You can draw multiple lines as well
class PointSelector:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        #print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs = [event.xdata]
        self.ys = [event.ydata]
        self.line.set_data([self.x], [self.y])
        self.line.figure.canvas.draw()

def interactive_legend(ax=None,lines1=None,lines2=None,data=None):
    if ax is None:
        ax = plt.gca()
    if ax.legend_ is None:
        ax.legend()

    return InteractiveLegend(ax.legend_,lines1,lines2,data)
# Class for interactive legend
# Clicking on legend items removes them from the plot and from the data
# This allows growth rate fits to be manually removed by user if they are poor quality
class InteractiveLegend(object):
    def __init__(self,legend,lines1,lines2,data):
        self.legend = legend
        self.fig = legend.axes.figure
        self.ax = plt.gca()
        self.lines1 = lines1
        self.lines2 = lines2
        self.data = data

        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()

        self.update_legend()

    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legendHandles:
            artist.set_picker(10) # 10 points tolerance
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

    def _build_lookups(self, legend):
        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))

        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist

        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))

        return lookup_artist, lookup_handle

    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:
            artist = self.lookup_artist[handle]
            if not self.lines1 is None:
                line_idx = self.lines1.index(artist)
                self.data = np.delete(self.data,line_idx,axis=0) # remove data from array
                self.lines1[line_idx].remove()
            if not self.lines2 is None: # remove companion line
                self.lines2[line_idx].remove()
            self.update_legend() # remove line from legend
    
    def update_legend(self):
        import matplotlib as mpl

        l = self.legend

        defaults = dict(
            loc = l._loc,
            numpoints = l.numpoints,
            markerscale = l.markerscale,
            scatterpoints = l.scatterpoints,
            scatteryoffsets = l._scatteryoffsets,
            prop = l.prop,
            # fontsize = None,
            borderpad = l.borderpad,
            labelspacing = l.labelspacing,
            handlelength = l.handlelength,
            handleheight = l.handleheight,
            handletextpad = l.handletextpad,
            borderaxespad = l.borderaxespad,
            columnspacing = l.columnspacing,
            ncol = l._ncol,
            mode = l._mode,
            fancybox = type(l.legendPatch.get_boxstyle())==mpl.patches.BoxStyle.Round,
            shadow = l.shadow,
            title = l.get_title().get_text() if l._legend_title_box.get_visible() else None,
            framealpha = l.get_frame().get_alpha(),
            bbox_to_anchor = l.get_bbox_to_anchor()._bbox,
            bbox_transform = l.get_bbox_to_anchor()._transform,
            frameon = l._drawFrame,
            handler_map = l._custom_handler_map,
        )

        mpl.pyplot.legend(**defaults)
        self.legend = self.ax.legend_
        self.lookup_artist, self.lookup_handle = self._build_lookups(self.legend)
        self._setup_connections()
        self.fig.canvas.draw()

    def show(self):
        plt.show()
        
def main():
    root = tk.Tk()
    app = GrowthRateAnalyzer(root)
    root.mainloop()
    #app.arduino.close()

if __name__ == '__main__':
    main()
