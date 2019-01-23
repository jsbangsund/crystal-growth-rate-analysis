# Improvements to consider:
# give option for time identifier string in case time=*s wasn't followed
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

def threshold_crop_denoise(img_file,x1,x2,y1,y2,threshold_lower,threshold_upper,
                        d,rescale=None,img=None,equalize_hist=False,
                        threshold_out=False,multiple_ranges=False,clip_limit=0.05):
    ''' threshold_crop_denoise
    This function crops an image, then thresholds and denoises it
    x1,x2,y1,y2 define crop indices
    image_file is an image filename to be loaded
    Alternatively, a numpy array of a gray scale image can be passed in as img
    rescale and equalize_hist allow contrast enhancement to be performed before
        thresholding
    threshold_out inverts the threshold so that pixel values outside the region
        are set to True
    multiple_ranges allows for multiple pixel ranges to be threshold (logical or)
        If this is selected, threshold_lower and _upper must be lists of equal length
    '''
    # Read image in gray scale (mode='L'), unless a pre-loaded image is passed
    if img is None:
        img=misc.imread(img_file,mode='L')
    if rescale:
        img = exposure.rescale_intensity(img,in_range=rescale)
    if equalize_hist:
        img = exposure.equalize_adapthist(img,clip_limit=clip_limit)
        img = 255 * img
    # Crop
    cropped = img[y1:y2,x1:x2]
    # Threshold above or below given pixel intensity
    # This converts image to black and white
    if not multiple_ranges:
        if not threshold_out:
            thresholded = np.logical_and(cropped>threshold_lower,cropped<threshold_upper)
        else:
            thresholded = np.logical_or(cropped<threshold_lower,cropped>threshold_upper)
    else:
        if not threshold_out:
            thresholded = np.logical_and(cropped>threshold_lower[0],cropped<threshold_upper[0])
            for r_idx in range(1,len(threshold_lower)):
                temp = np.logical_and(cropped>threshold_lower[r_idx],cropped<threshold_upper[r_idx])
                thresholded = np.logical_or(thresholded,temp)
        else:
            thresholded = np.logical_or(cropped<threshold_lower[0],cropped>threshold_upper[0])
            for r_idx in range(1,len(threshold_lower)):
                temp = np.logical_or(cropped<threshold_lower[r_idx],cropped>threshold_upper[r_idx])
                thresholded = np.logical_and(thresholded,temp)


    # Despeckle with disk size d
    denoised = median(thresholded, disk(d))
    return denoised,thresholded,cropped

def subtract_and_denoise(img_file1,img_file2,x1,x2,y1,y2,d,threshold=None,
                         rescale=None,img1=None,img2=None,equalize_hist=False,clip_limit=0.05):
    # Read image in gray scale (mode='L'), unless a pre-loaded image is passed
    if img1 is None:
        img1=misc.imread(img_file1,mode='L')
    if img2 is None:
        img2=misc.imread(img_file2,mode='L')
    if rescale:
        img1 = exposure.rescale_intensity(img1,in_range=rescale)
        img2 = exposure.rescale_intensity(img2,in_range=rescale)
    if equalize_hist:
        img1 = exposure.equalize_adapthist(img1,clip_limit=clip_limit)
        img1 = 255 * img1
        img2 = exposure.equalize_adapthist(img2,clip_limit=clip_limit)
        img2 = 255 * img2
    cropped1 = img1[y1:y2,x1:x2]
    cropped2 = img2[y1:y2,x1:x2]
    # Subtract and take absolute value. Convert to float so that negative values are possible
    subtract=np.abs(cropped2.astype(np.float32)-cropped1.astype(np.float32))
    # Normalize from 0 to 1
    subtract_norm = (subtract-subtract.min())/(subtract.max()-subtract.min())
    # Threshold, if lower threshold is given:
    if threshold is None:
        thresholded = subtract_norm
    else:
        thresholded = subtract_norm > threshold
    denoised = median(thresholded, disk(d))
    return denoised,thresholded,subtract_norm,cropped2

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
def get_growth_edge(img,line,length_per_pixel):
    # Get line profile
    profile = profile_line(img,
                           (line[0][1],line[0][0]),
                           (line[1][1],line[1][0]))
    # Find last point on grain (where image is still saturated)
    growth_front_endpoint = np.where(profile==np.amax(profile))[0][-1]
    line_endpoint = profile.shape[0]
    # Get total line length
    total_line_length = get_line_length(
        line,mag=None,unit='um',length_per_pixel=length_per_pixel)
    # Distance to growth front is the fraction of the line up to the last point
    distance_to_growth_front = (total_line_length
                             * (growth_front_endpoint+1) # +1 accounts for index starting at 0
                             / line_endpoint)
    return distance_to_growth_front
    
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
        self.base_dir = os.path.join(os.path.expanduser('~'),'Google Drive',
                                  'Research','Data','Gratings')
        # initialize dataframe save location
        self.df_dir = os.path.join(os.getcwd(),'dataframes')
        if not os.path.isdir(self.df_dir):
            os.mkdir(self.df_dir)
        self.configure_gui()
    def configure_gui(self):
        # Master Window
        self.parent.title("Growth Rate Analysis")
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
        self.sample_props_container = ttk.Frame(sample_props_and_plot_container)
        self.sample_props_container.pack(side=LEFT)
        crop_container = ttk.Frame(sample_props_and_plot_container)
        crop_container.pack(side=LEFT)
        plotContainer = ttk.Frame(sample_props_and_plot_container)
        plotContainer.pack(side=LEFT)#fill=BOTH, expand=True
        
        self.threshold_plot_container = ttk.Frame(self.parent)
        self.threshold_plot_container.pack(fill=BOTH, expand=True)
        self.threshold_container = ttk.Frame(self.parent)
        self.threshold_container.pack()
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
        # Crop region plot
        self.fig, self.ax = plt.subplots(ncols=2,figsize=(7,2.5),
                                     gridspec_kw = {'width_ratios':[1, 1.1]})
        self.fig.subplots_adjust(wspace=0.29,left=0.01,bottom=0.17,top=.95,right=0.75)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plotContainer)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plotContainer)
        self.toolbar.update()

        # Crop Region buttons
        #self.fileDirLabel = ttk.Label(file_container, text = 'Pick image files')
        #self.fileDirLabel.grid(row=1, column=0, sticky=W)
        b_width = 19
        self.b_plotCrop = ttk.Button(crop_container, command=self.pick_crop_region)
        self.b_plotCrop.configure(text="Pick Crop")
        self.b_plotCrop.grid(row=0, column=0, sticky=W)
        self.b_plotCrop.config(width=b_width)
        #self.b_getRanges = ttk.Button(crop_container, command=self.get_axes_ranges)
        #self.b_getRanges.configure(text="Get Crop Region")
        #self.b_getRanges.grid(row=0, column=1, sticky=E)
        #self.l_crop_range = ttk.Label(crop_container, text = 'x1=?, x2=?, y1=?, y2=?')
        #self.l_crop_range.grid(row=0, column=2, sticky=W)

        self.b_draw_lines = ttk.Button(crop_container,
                                command=self.draw_line_segments)
        self.b_draw_lines.configure(text="Pick Directions")
        self.b_draw_lines.grid(row=1, column=0, sticky=W)
        self.b_draw_lines.config(width=b_width)

        self.b_get_lines = ttk.Button(crop_container, command=self.get_line_segments)
        self.b_get_lines.configure(text="Get Directions")
        self.b_get_lines.grid(row=2, column=0, sticky=E)
        self.b_get_lines.config(width=b_width)
        
        self.b_check_edge = ttk.Button(crop_container, command=self.check_edge_detection)
        self.b_check_edge.configure(text="Check Edge Detection")
        self.b_check_edge.grid(row=3, column=0, sticky=W)
        self.b_check_edge.config(width=b_width)
        
        self.b_extract_growth = ttk.Button(crop_container, command=self.extract_growth_rates)
        self.b_extract_growth.configure(text="Extract Growth Rates")
        self.b_extract_growth.grid(row=4, column=0, sticky=E)
        self.b_extract_growth.config(width=b_width)
        
        self.b_pick_df = ttk.Button(crop_container, command=self.pick_df)
        self.b_pick_df.configure(text="Pick DF")
        self.b_pick_df.grid(row=5, column=0, sticky=W)
        self.b_pick_df.config(width=b_width)
        
        self.b_save_results = ttk.Button(crop_container, command=self.save_results)
        self.b_save_results.configure(text="Save Results")
        self.b_save_results.grid(row=6, column=0, sticky=W)
        self.b_save_results.config(width=b_width)
        
        self.configure_subtract_fig()

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
                              'dtype':'string'}),
                           # ('objective_mag',
                             # {'label':'Image Mag.:',
                              # 'default_val':'10x',
                              # 'type':'OptionMenu',
                              # 'dtype':'string',
                              # 'options':['4x','10x','20x','50x']})
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
        # self.l_sample_props = ['']*len(labels)
        # self.e_sample_props = ['']*len(labels)
        # self.s_sample_props = ['']*len(labels)
        # for index,label in enumerate(labels):
            # self.l_sample_props[index] = ttk.Label(self.sample_props_container, text=label)
            # self.l_sample_props[index].grid(row=index, column=0, sticky=W)
            # self.s_sample_props[index] = tk.StringVar()
            # self.s_sample_props[index].set(defaults[index])
            # self.e_sample_props[index] = ttk.Entry(
                                        # self.sample_props_container,
                                        # textvariable=self.s_sample_props[index])
            # self.e_sample_props[index].grid(row=index,column=1)
    def configure_threshold_fig(self):
        # Destroy previous elements in these frames
        for child in self.threshold_container.winfo_children():
            child.destroy()
        for child in self.threshold_plot_container.winfo_children():
            child.destroy()
        # Threshold figure
        self.threshold_fig,self.threshold_ax = plt.subplots(ncols=4,
                                                      figsize=(10,2.5))
        self.threshold_fig.subplots_adjust(wspace=0.3,top=0.8)
        self.threshold_canvas = FigureCanvasTkAgg(
                                    self.threshold_fig,
                                    master=self.threshold_plot_container)
        self.threshold_canvas.draw()
        self.threshold_canvas.get_tk_widget().pack(fill = BOTH, expand = True)
        #self.toolbar = NavigationToolbar2Tk(self.threshold_canvas, self.threshold_plot_container)
        #self.toolbar.update()
        ttk.Label(self.threshold_container,text="Threshold").grid(row=1,column=1)
        ttk.Label(self.threshold_container,text="Lower").grid(row=0,column=2)
        ttk.Label(self.threshold_container,text="Upper").grid(row=0,column=3)
        ttk.Label(self.threshold_container,text="Disk").grid(row=0,column=4)
        ttk.Label(self.threshold_container,text="Inv. Thresh?").grid(row=0,column=5)
        ttk.Label(self.threshold_container,text="Multi Ranges?").grid(row=0,column=6)
        ttk.Label(self.threshold_container,text="Eq. Hist?").grid(row=0,column=7)
        ttk.Label(self.threshold_container,text="Clip Limit").grid(row=0,column=8)
        # Edge detection method options
        ttk.Label(self.threshold_container,text="Edge Det. Method:").grid(row=0,column=0)
        self.s_edge_method=tk.StringVar()
        #self.s_edge_method.set('Threshold Grain')
        self.e_edge_method=ttk.OptionMenu(self.threshold_container,self.s_edge_method,'Threshold Grain',
                           *['Subtract Images','Threshold Grain'],
                           command=self.set_edge_method)
        self.e_edge_method.grid(row=1,column=0)
        

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

        self.s_disk=tk.StringVar()
        self.s_disk.set('5')
        self.e_disk=ttk.Entry(self.threshold_container,textvariable=self.s_disk,width=5)
        self.e_disk.grid(row=1,column=4)

        self.bool_threshold_out = tk.BooleanVar()
        self.bool_threshold_out.set(False)
        self.e_threshold_out = tk.Checkbutton(
                                self.threshold_container,
                                variable=self.bool_threshold_out,
                                onvalue=True,offvalue=False)
        self.e_threshold_out.grid(row=1,column=5)

        self.bool_multi_ranges = tk.BooleanVar()
        self.bool_multi_ranges.set(False)
        self.e_multi_ranges = tk.Checkbutton(
                                self.threshold_container,
                                variable=self.bool_multi_ranges,
                                onvalue=True,offvalue=False)
        self.e_multi_ranges.grid(row=1,column=6)

        self.bool_eq_hist = tk.BooleanVar()
        self.bool_eq_hist.set(True)
        self.e_eq_hist = tk.Checkbutton(self.threshold_container,variable=self.bool_eq_hist,
                                  onvalue = True, offvalue = False,
                                  command=self.eq_hist_cb_command)
        self.e_eq_hist.grid(row=1,column=7)

        self.s_clip_limit = tk.StringVar()
        self.s_clip_limit.set('0.05')
        self.last_clip_limit=self.s_clip_limit.get()
        self.e_clip_limit = ttk.Entry(self.threshold_container,textvariable=self.s_clip_limit,width=5)
        self.e_clip_limit.grid(row=1,column=8)

        self.b_clear_ranges = ttk.Button(self.threshold_container,
                                    command=self.clear_threshold_ranges)
        self.b_clear_ranges.configure(text="Clear Ranges")
        self.b_clear_ranges.grid(row=1, column=9, sticky=W)

        self.b_check_threshold = ttk.Button(self.threshold_container,
                                    command=self.check_threshold)
        self.b_check_threshold.configure(text="Check Threshold")
        self.b_check_threshold.grid(row=1, column=10, sticky=W)

        self.rectangles=[]
        self.rectangle = Rectangle(
                            xy=(int(self.s_threshold_lower.get()),0),
                            width=(int(self.s_threshold_lower.get())
                                 -int(self.s_threshold_upper.get())),
                            height=5000,
                            alpha=0.5,facecolor=(55/255,126/255,184/255),
                            edgecolor=None)
        self.threshold_ax[3].add_patch(self.rectangle)
        def onselect(xmin, xmax):
            rect = Rectangle(
                            xy=(xmin,0),width=(xmax-xmin),
                            height=self.threshold_ax[3].get_ylim()[1],
                            alpha=0.5,facecolor=(55/255,126/255,184/255),
                            edgecolor=None)

            if not self.bool_multi_ranges.get():
                # Remove old rectangles
                try:
                    self.rectangle.remove()
                except:
                    pass
                self.rectangle = rect
                self.s_threshold_lower.set(str(int(xmin)))
                self.s_threshold_upper.set(str(int(xmax)))
                self.threshold_ax[3].add_patch(self.rectangle)
            else:
                self.rectangles.append(rect)
                self.threshold_ax[3].add_patch(self.rectangles[-1])
                if self.s_threshold_lower.get()=='':
                    insert_string_1 = str(int(xmin))
                    insert_string_2 = str(int(xmax))
                else:
                    insert_string_1 = self.s_threshold_lower.get() + ',' + str(int(xmin))
                    insert_string_2 = self.s_threshold_upper.get() + ',' + str(int(xmax))
                self.s_threshold_lower.set(insert_string_1)
                self.s_threshold_upper.set(insert_string_2)

            # Make rectangle showing selected area
            #self.threshold_ax[3].add_patch(rect)
            self.check_threshold()

        # Span selector for threshold range
        self.span = SpanSelector(self.threshold_ax[3], onselect, 'horizontal', useblit=True,
                            rectprops=dict(alpha=0.5, facecolor=(55/255,126/255,184/255)))
    
    def configure_subtract_fig(self):
        # Destroy previous elements in these frames
        for child in self.threshold_container.winfo_children():
            child.destroy()
        for child in self.threshold_plot_container.winfo_children():
            child.destroy()
        #Figure
        self.threshold_fig,self.threshold_ax = plt.subplots(ncols=4,
                                                      figsize=(10,2.5))
        self.threshold_fig.subplots_adjust(wspace=0.3,top=0.8)
        self.threshold_canvas = FigureCanvasTkAgg(
                                    self.threshold_fig,
                                    master=self.threshold_plot_container)
        self.threshold_canvas.draw()
        self.threshold_canvas.get_tk_widget().pack(fill = BOTH, expand = True)
        #Subtraction parameters
        ttk.Label(self.threshold_container,text="Threshold?").grid(row=0,column=1)
        ttk.Label(self.threshold_container,text="Lower").grid(row=0,column=2)
        ttk.Label(self.threshold_container,text="Disk").grid(row=0,column=3)
        ttk.Label(self.threshold_container,text="Eq. Hist?").grid(row=0,column=4)
        ttk.Label(self.threshold_container,text="Clip Limit").grid(row=0,column=5)
        
        # Edge detection method options
        ttk.Label(self.threshold_container,text="Edge Det. Method:").grid(row=0,column=0)
        self.s_edge_method=tk.StringVar()
        self.s_edge_method.set('Subtract Images')
        self.e_edge_method=ttk.OptionMenu(self.threshold_container,self.s_edge_method,'Subtract Images',
                           *['Subtract Images','Threshold Grain'],
                           command=self.set_edge_method)
        self.e_edge_method.grid(row=1,column=0)

        self.bool_threshold_on = tk.BooleanVar()
        self.bool_threshold_on.set(True)
        self.e_threshold_on = tk.Checkbutton(
                                self.threshold_container,
                                variable=self.bool_threshold_on,
                                onvalue=True,offvalue=False)
        self.e_threshold_on.grid(row=1,column=1)
        
        self.s_threshold_lower=tk.StringVar()
        self.s_threshold_lower.set('0.05')
        self.e_s_threshold_lower=ttk.Entry(self.threshold_container,
                                   textvariable=self.s_threshold_lower,width=5)
        self.e_s_threshold_lower.grid(row=1,column=2)

        self.s_disk=tk.StringVar()
        self.s_disk.set('8')
        self.e_disk=ttk.Entry(self.threshold_container,textvariable=self.s_disk,width=5)
        self.e_disk.grid(row=1,column=3)
        
        self.bool_eq_hist = tk.BooleanVar()
        self.bool_eq_hist.set(False)
        self.e_eq_hist = ttk.Checkbutton(self.threshold_container,variable=self.bool_eq_hist,
                                  onvalue = True, offvalue = False,
                                  command=self.eq_hist_cb_command)
        self.e_eq_hist.grid(row=1,column=4)

        self.s_clip_limit = tk.StringVar()
        self.s_clip_limit.set('0.05')
        self.last_clip_limit=self.s_clip_limit.get()
        self.e_clip_limit = ttk.Entry(self.threshold_container,textvariable=self.s_clip_limit,width=5)
        self.e_clip_limit.grid(row=1,column=5)

        self.b_check_threshold = ttk.Button(self.threshold_container,
                                    command=self.check_subtraction)
        self.b_check_threshold.configure(text="Check Subtraction")
        self.b_check_threshold.grid(row=1, column=6, sticky=W)
        
    def set_edge_method(self,val):
        if self.s_edge_method.get()=='Subtract Images':
            self.configure_subtract_fig()
        elif self.s_edge_method.get()=='Threshold Grain':
            self.configure_threshold_fig()
        self.threshold_initialized=False
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
        self.full_images=['']*len(self.time_files)
        for i,f in enumerate(self.time_files):
            self.full_images[i] = misc.imread(f,mode='L')
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

    def pick_crop_region(self):
        # Zoom to region of interest in image. This will select crop region below
        # Pick the last time so the whole grain is contained within the crop region
        img=misc.imread(os.path.join(self.base_dir,self.time_files[-1]),
                      mode='L')
        #img = exposure.rescale_intensity(img,in_range='image')
        img = exposure.equalize_adapthist(img,clip_limit=0.05)
        #fig,self.crop_ax=plt.subplots()
        #self.crop_fig, self.crop_ax = plt.subplots()
        if not self.crop_initialized:
            self.cropData = self.ax[0].imshow(img)
            self.canvas.draw()
        else:
            # Remove old text, if exists
            try:
                for txt in self.line_texts:
                    txt.remove()
                self.line_texts=[]
                # remove old lines as well
                for line in self.ax[0].lines:
                    line.remove()
            except:
                pass
            # Set new data
            set_new_im_data(self.ax[0],self.cropData,img)
            self.canvas.draw()
        self.crop_initialized = True
        self.axes_ranges_initialized = False
        # Reset threshold
        self.threshold_initialized = False
        #plt.show()
        #self.canvas.draw()
    def get_axes_ranges(self):
        x1,x2=self.ax[0].get_xlim()
        y2,y1=self.ax[0].get_ylim()
        self.x1=int(x1)
        self.x2=int(x2)
        self.y2=int(y2)
        self.y1=int(y1)
        string = ('x1 = ' + str(self.x1)+', x2 = ' + str(self.x2) +
                ', y1 = ' + str(self.y1) + ', y2 = ' + str(self.y2))
        print(string)
        self.ax[0].set_title(string)
        self.axes_ranges_initialized = True
        #self.l_crop_range.config(text=('x1 = ' + str(self.x1) +
        #                            ', x2 = ' + str(self.x2) +
        #                            ', y1 = ' + str(self.y1) +
        #                            ', y2 = ' + str(self.y2)))
        #print(x1,x2,y1,y2)
    def eq_hist_cb_command(self):
        self.threshold_initialized=False
    def clear_threshold_ranges(self):
        # Remove all rectangles
        [p.remove() for p in reversed(self.rectangles)]
        self.rectangles=[]
        try:
            self.rectangle.remove()
        except:
            pass
        # Reset threshold variables
        self.s_threshold_lower.set('')
        self.s_threshold_upper.set('')
        self.threshold_canvas.draw()
    def check_threshold(self):
        # Update crop range
        if not self.axes_ranges_initialized:
            self.get_axes_ranges()
        if not self.s_clip_limit.get() == self.last_clip_limit:
            self.threshold_initialized=False
            [b.remove() for b in self.threshold_plot_data[3][2]]
        if not self.threshold_initialized:
            self.original_image=misc.imread(os.path.join(self.base_dir,
                                                  self.time_files[-1]),
                                      mode='L')
            try:
                [b.remove() for b in self.threshold_plot_data[3][2]]
            except:
                pass
        if self.bool_multi_ranges.get():
            threshold_lower = [float(x) for x in
                            self.s_threshold_lower.get().split(',')]
            threshold_upper = [float(x) for x in
                            self.s_threshold_upper.get().split(',')]
        else:
            threshold_lower = float(self.s_threshold_lower.get())
            threshold_upper = float(self.s_threshold_upper.get())
        denoised,thresholded,cropped = threshold_crop_denoise(
                                      self.time_files[-1],
                                      self.x1,self.x2,self.y1,self.y2,
                                      threshold_lower,
                                      threshold_upper,
                                      int(self.s_disk.get()),
                                      equalize_hist=self.bool_eq_hist.get(),
                                      multiple_ranges=self.bool_multi_ranges.get(),
                                      threshold_out=self.bool_threshold_out.get(),
                                      clip_limit=float(self.s_clip_limit.get())
                                      )

        if not self.threshold_initialized:
            self.threshold_plot_data = ['']*4
            self.threshold_plot_data[0] = self.threshold_ax[0].imshow(cropped,cmap=plt.get_cmap('gray'))
            self.threshold_plot_data[1] = self.threshold_ax[1].imshow(thresholded,cmap=plt.get_cmap('gray'))
            self.threshold_plot_data[2] = self.threshold_ax[2].imshow(denoised,cmap=plt.get_cmap('gray'))
            # Plot the histogram so we can select a good threshold for the grains
            self.threshold_plot_data[3]=self.threshold_ax[3].hist(
                    cropped.ravel(),bins=256,alpha=0.8,
                    color=(228/255,26/255,28/255))
            self.threshold_ax[3].autoscale()
            # Set subplot titles
            self.threshold_ax[0].set_title('Original Image')
            self.threshold_ax[2].set_title('Despeckled')
            self.threshold_ax[3].set_title('Click and Drag \n to Select Threshold')
        elif self.threshold_initialized:
            set_new_im_data(self.threshold_ax[1],self.threshold_plot_data[1],thresholded)
            set_new_im_data(self.threshold_ax[2],self.threshold_plot_data[2],denoised)
        self.threshold_ax[1].set_title('Thresholded Between \n' + self.s_threshold_lower.get() + ' and ' + self.s_threshold_upper.get())
        self.threshold_canvas.draw()
        self.threshold_initialized = True
        self.last_clip_limit=self.s_clip_limit.get()

    def check_subtraction(self):
        # Update crop range
        if not self.axes_ranges_initialized:
            self.get_axes_ranges()
        if not self.s_clip_limit.get() == self.last_clip_limit:
            self.threshold_initialized=False
            [b.remove() for b in self.threshold_plot_data[3][2]]
        if not self.threshold_initialized:
            self.original_image=misc.imread(os.path.join(self.base_dir,
                                                  self.time_files[-1]),
                                      mode='L')
            try:
                [b.remove() for b in self.threshold_plot_data[3][2]]
            except:
                pass
        if self.bool_threshold_on.get():
            threshold_lower = float(self.s_threshold_lower.get())
        else:
            threshold_lower = None
        denoised,thresholded,subtraction,cropped = subtract_and_denoise(
                                        self.time_files[-2],
                                        self.time_files[-1],
                                        self.x1,self.x2,self.y1,self.y2,
                                        int(self.s_disk.get()),
                                        threshold=threshold_lower,
                                        equalize_hist=self.bool_eq_hist.get(),
                                        clip_limit=float(self.s_clip_limit.get()))

        if not self.threshold_initialized:
            self.threshold_plot_data = ['']*4
            self.threshold_plot_data[0] = self.threshold_ax[0].imshow(cropped,cmap=plt.get_cmap('gray'))
            self.threshold_plot_data[1] = self.threshold_ax[1].imshow(thresholded,cmap=plt.get_cmap('gray'))
            self.threshold_plot_data[2] = self.threshold_ax[2].imshow(denoised,cmap=plt.get_cmap('gray'))
            # Plot the histogram so we can select a good threshold for the grains
            self.threshold_plot_data[3]=self.threshold_ax[3].hist(
                    subtraction.ravel(),bins=100,alpha=0.8,
                    color=(228/255,26/255,28/255))
            self.threshold_ax[3].set_yscale('log')
            # Set subplot titles
            self.threshold_ax[0].set_title('Original Image')
            self.threshold_ax[2].set_title('Despeckled')
            self.threshold_ax[3].set_title('Subtraction Histogram')
        elif self.threshold_initialized:
            set_new_im_data(self.threshold_ax[1],self.threshold_plot_data[1],thresholded)
            set_new_im_data(self.threshold_ax[2],self.threshold_plot_data[2],denoised)
        if self.bool_threshold_on.get():
            self.threshold_ax[1].set_title('Thresholded above \n' + self.s_threshold_lower.get())
        else:
            self.threshold_ax[1].set_title('Subtraction')
        self.threshold_canvas.draw()
        self.threshold_initialized = True
        self.last_clip_limit=self.s_clip_limit.get()
    
    def draw_line_segments(self):
        # Update crop range
        if not self.axes_ranges_initialized:
            self.get_axes_ranges()
        if self.s_edge_method.get() == "Threshold Grain":
            if self.bool_multi_ranges:
                threshold_lower = [float(x) for x in
                                self.s_threshold_lower.get().split(',')]
                threshold_upper = [float(x) for x in
                                self.s_threshold_upper.get().split(',')]
            else:
                threshold_lower = float(self.s_threshold_lower.get())
                threshold_upper = float(self.s_threshold_upper.get())
            denoised = threshold_crop_denoise(
                                          self.time_files[-1],
                                          self.x1,self.x2,self.y1,self.y2,
                                          threshold_lower,
                                          threshold_upper,
                                          int(self.s_disk.get()),
                                          equalize_hist=self.bool_eq_hist.get(),
                                          multiple_ranges=self.bool_multi_ranges.get(),
                                          threshold_out=self.bool_threshold_out.get(),
                                          clip_limit=float(self.s_clip_limit.get())
                                          )[0]
        elif self.s_edge_method.get() == "Subtract Images":
            if self.bool_threshold_on.get():
                threshold_lower = float(self.s_threshold_lower.get())
            else:
                threshold_lower = None
            denoised = subtract_and_denoise(
                                            self.time_files[-2],
                                            self.time_files[-1],
                                            self.x1,self.x2,self.y1,self.y2,
                                            int(self.s_disk.get()),
                                            threshold=threshold_lower,
                                            equalize_hist=self.bool_eq_hist.get(),
                                            clip_limit=float(self.s_clip_limit.get()))[0]
        # Remove old lines
        for line in self.ax[0].lines:
            line.remove()
        # Remove old text, if exists
        try:
            for txt in self.line_texts:
                txt.remove()
            self.line_texts=[]
        except:
            pass
        self.ax[0].set_title('click to build line segments')
        # Change data extent to match new cropped image
        self.cropData.set_extent((0, denoised.shape[1], denoised.shape[0], 0))
        # Reset axes limits
        self.ax[0].set_xlim(0,denoised.shape[1])
        self.ax[0].set_ylim(denoised.shape[0],0)
        # Now set the data
        self.cropData.set_data(denoised)
        self.ax[0].relim()
        #self.ax[0].imshow(denoised)
        # Alternatively
        #img=misc.imread(os.path.join(self.base_dir,self.time_files[-1]),mode='L')
        # ax.imshow(img[self.y1:self.y2,self.x1:self.x2])
        line, = self.ax[0].plot([], [],'-or')  # empty line
        self.linebuilder = LineBuilder(line)
        #self.ax[0].axis('off')

        self.canvas.draw()
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
    def check_edge_detection(self):
        # TODO this function is outdated and doesn't consider substraction method
        # Remove old lines
        for line in self.ax[1].lines:
            line.remove()
        img=misc.imread(os.path.join(self.base_dir,self.time_files[-1]),mode='L')
        #ax[0].imshow(img)
        if self.bool_multi_ranges:
            threshold_lower = [float(x) for x in
                            self.s_threshold_lower.get().split(',')]
            threshold_upper = [float(x) for x in
                            self.s_threshold_upper.get().split(',')]
        else:
            threshold_lower = float(self.s_threshold_lower.get())
            threshold_upper = float(self.s_threshold_upper.get())
        denoised = threshold_crop_denoise(
                                      self.time_files[-1],
                                      self.x1,self.x2,self.y1,self.y2,
                                      threshold_lower,
                                      threshold_upper,
                                      int(self.s_disk.get()),
                                      equalize_hist=self.bool_eq_hist.get(),
                                      multiple_ranges=self.bool_multi_ranges.get(),
                                      threshold_out=self.bool_threshold_out.get(),
                                      clip_limit=float(self.s_clip_limit.get())
                                      )[0]
        # Get edge of growth front from image profile
        line = self.lines[0]
        profile = profile_line(denoised,
                            (line[0][1],line[0][0]),
                            (line[1][1],line[1][0]))
        self.ax[1].plot(profile)
        # Find last point on grain (where image is saturated)
        last_idx = np.where(profile>255*.5)[0][-1]
        self.ax[1].plot(last_idx,255,'o')
        
        theta = np.arctan2((line[1][1]-line[0][1]),(line[1][0]-line[0][0]))
        total_length = np.sqrt((line[1][1]-line[0][1])**2 + (line[1][0]-line[0][0])**2)
        x_edge = line[0][0] + np.cos(theta)*((last_idx+1)/profile.shape[0])*total_length
        y_edge = line[0][1] + np.sin(theta)*((last_idx+1)/profile.shape[0])*total_length
        #self.ax[0].plot(line[0][0],line[0][1],'ob')
        #self.ax[0].plot(line[1][0],line[1][1],'ob')
        self.ax[0].plot(x_edge,y_edge,'ob')
        self.canvas.draw()

    def extract_growth_rates(self):
        # Update crop range
        if not self.axes_ranges_initialized:
            self.get_axes_ranges()
        # Remove old data
        self.ax[1].clear()
        # Check if images dimensions are as expected. If not use image width
        # Not very robust yet
        img = self.full_images[-1]
        if img.shape[1]==2048:
            length_per_pixel = micron_per_pixel[self.s_mag.get()]
        else:
            length_per_pixel = image_width_microns[self.s_mag.get()]/img.shape[1]
        # Check whether image processing settings have changed
        # If not, used stored copies of processed images
        current_img_process_settings = self.get_img_process_settings()
        # Get time from filenames, and sort by time
        if not current_img_process_settings == self.last_img_process_settings:
            self.extract_times_and_sort() # saves self.times and self.sort_indices
        # Initialize distances array
        self.distances = np.zeros((len(self.lines),len(self.times)))
        # Now process images if needed
        if not current_img_process_settings == self.last_img_process_settings:
            self.denoised_images = ['']*len(self.times)
            for idx in range(0,len(self.times)):
                sort_idx = self.sort_indices[idx]
                timeFile = self.time_files[sort_idx]
                if idx==0:
                    t0=self.times[sort_idx]
                    ti=0
                else:
                    ti = self.times[sort_idx]-t0
                if self.s_edge_method.get()=='Threshold Grain':
                    denoised = threshold_crop_denoise(self.time_files[sort_idx],
                                                  self.x1,self.x2,self.y1,self.y2,
                                                  current_img_process_settings['threshold_lower'],
                                                  current_img_process_settings['threshold_upper'],
                                                  int(self.s_disk.get()),
                                                  img = self.full_images[sort_idx],
                                                  equalize_hist=self.bool_eq_hist.get(),
                                                  multiple_ranges=current_img_process_settings['multiple_ranges'],
                                                  threshold_out=current_img_process_settings['threshold_out'],
                                                  clip_limit=float(self.s_clip_limit.get())
                                                  )[0]
                elif self.s_edge_method.get()=='Subtract Images':
                    denoised = subtract_and_denoise(
                                            self.time_files[sort_idx],
                                            self.time_files[self.sort_indices[idx+1]],
                                            self.x1,self.x2,self.y1,self.y2,
                                            int(self.s_disk.get()),
                                            img1=self.full_images[sort_idx],
                                            img2=self.full_images[self.sort_indices[idx+1]],
                                            threshold=current_img_process_settings['threshold_lower'],
                                            equalize_hist=self.bool_eq_hist.get(),
                                            clip_limit=float(self.s_clip_limit.get()))[0]
                self.denoised_images[idx] = denoised # save for speed if re-analyzing same area
            self.last_img_process_settings = current_img_process_settings
        # Now extract growth front at each time step
        for idx,denoised in enumerate(self.denoised_images):
            for line_idx in range(0,len(self.lines)):
                self.distances[line_idx][idx] = get_growth_edge(
                    denoised,self.lines[line_idx],
                    length_per_pixel=length_per_pixel
                    )
        
        # Could break this into separate function, for updating plot
        self.growth_rates=[]
        self.growth_rates_string=[]
        self.growth_lines_fit=['']*len(self.lines)
        self.growth_lines=['']*len(self.lines)
        c_idx=-1
        for line_idx,line in enumerate(self.lines):
            c_idx +=1
            if c_idx>9:
                c_idx=0
            # print(self.times)
            # print(self.distances)
            # x,y,filterIdx=cleanSignal_curvature(self.times,self.distances[line_idx],
                               # curvature_threshold = 0.004,
                               # return_index=True,remove_less_than=1)
            # print(filterIdx)
            # Filter out points that are:
                # less than 80% of the median of the first three points, 
                # 5% greater than the median of the last three points
            # old routine filtering less than 10% of the mean
                # self.distances[line_idx]>np.mean(self.distances[line_idx])*0.1 
            # This filtering could be smarter. Could add derivative filtering, or could
            # iteratively fit and reject points with large MSE
            # logical_and.reduce((condition1,condition2,...,conditionN)) is used so that
            # more than two conditions can be added (logical_and only accepts one arg)
            # See https://stackoverflow.com/questions/20528328/numpy-logical-or-for-more-than-two-arguments
            filterIdx = np.where(np.logical_and.reduce(
                (self.distances[line_idx]>np.median(self.distances[line_idx][0:3])*0.8,
                self.distances[line_idx]<np.median(self.distances[line_idx][-3:])*1.05,
                )))[0]
            self.times = np.array(self.times)
            # Fit the data with a line
            params = np.polyfit(self.times[filterIdx], self.distances[line_idx][filterIdx], 1)
            line1,=self.ax[1].plot(self.times,np.array(self.times)*params[0]+params[1],'--',
                color=Tableau_10.mpl_colors[c_idx],linewidth=1.5)
            self.growth_lines_fit.append(line1)
            self.ax[1].set_xlabel('Time (s)')
            self.ax[1].set_ylabel('Grain Radius ($\mu$m)')
            print('{:.2f}'.format(params[0])+' micron/sec')
            self.growth_rates_string.append('{:.2f}'.format(params[0])+' micron/sec')
            self.growth_rates.append(params[0])
            # Make legend label, decide units based on size of value
            if params[0]>10:
                label_string = '#' + str(line_idx+1) + ', ' + '{:.1f}'.format(params[0])+' $\mu$m/s'
            elif params[0]>0.1:
                label_string = '#' + str(line_idx+1) + ', ' + '{:.2f}'.format(params[0])+' $\mu$m/s'
            else:
                label_string = '#' + str(line_idx+1) + ', ' + '{:.1f}'.format(params[0]*1e3)+' nm/s'
            self.growth_lines[line_idx],=self.ax[1].plot(self.times,self.distances[line_idx],'o',color=Tableau_10.mpl_colors[c_idx],
                label=label_string)
        self.legend = self.ax[1].legend(bbox_to_anchor=(1.0, 1.0),
                    title='click line to remove')
        # Need to figure out best way to modify data between the two classes
        # self.interactive_legend = interactive_legend(
            # ax=self.ax[1],lines1=self.growth_lines,lines2=self.growth_lines_fit,
            # data=self.distances)
        # Re-evaluate limits
        self.ax[1].relim()
        self.canvas.draw()
        self.label_lines()

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
        if self.s_edge_method.get()=='Subtract Images':
            # Make sort_indices and times arrays smaller in length by one element,
            # since subtraction reduces the number of datapoints by one
            self.times.remove(max(self.times))
            #self.sort_indices = self.sort_indices[:-1]
    def save_results(self):
        # Make save directory
        self.save_dir = os.path.join(self.base_dir,'analysis_results')
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        # Save growth rate file
        # header = 'time(s),'
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
        # savename = self.increment_save_name(self.save_dir,'radius_vs_time','.csv')
        # np.savetxt(os.path.join(self.save_dir,savename+'.csv'),np.transpose(np.insert(self.distances,0,self.times,axis=0)),
                    # delimiter=',',header=header)
        # savename = self.increment_save_name(self.save_dir,'growthrates','.csv')
        # with open(os.path.join(self.save_dir,savename+'.csv'), 'a') as f:
            # f.write('Line#,Growth Rate (micron/sec) \n')
            # for idx,growthRate in enumerate(self.growth_rates):
                # line = str(idx+1) + ',' + str(growthRate) + '\n'
                # f.write(line)
        # Save figures
        savename = self.increment_save_name(self.save_dir,'growth_rates_plot','.png')
        self.fig.savefig(os.path.join(self.save_dir,savename+'.png'),dpi=200,bbox_inches='tight')
        savename = self.increment_save_name(self.save_dir,'threshold_plot','.png')
        self.threshold_fig.savefig(os.path.join(self.save_dir,savename+'.png'),dpi=200,bbox_inches='tight')
        # Save dataframe
        # If no file was selected make new filename and df
        if not self.df_file:
            self.df_file = (self.increment_save_name(
                self.df_dir,str(datetime.date.today()) + '_df','.pkl')
                +'.pkl'
                )
            self.df = pd.DataFrame()
        # otherwise, load df
        else:
            self.df = pd.read_pickle(os.path.join(self.df_dir,self.df_file))
        # Also save a local df pkl file in the 'Analysis' folder, in case user is stupid and overwrites elsewhere
        df_local_file = os.path.join(self.save_dir,'df.pkl')
        if os.path.isfile(df_local_file):
            df_local = pd.read_pickle(df_local_file)
        else:
            df_local = pd.DataFrame()
        # Now append data
        data_dict_list=[]
        img_process_settings = self.get_img_process_settings()
        for line_idx,growth_rate in enumerate(self.growth_rates):
            temp_dict = {'growth_rate_umps':growth_rate,
                        'line':self.lines[line_idx],
                        'x1,x2,y1,y2':(self.x1,self.x2,self.y1,self.y2),
                        'image_files':[os.path.basename(x) for x in self.time_files],
                        'image_dir':os.path.split(self.time_files[0])[0],
                        'threshold_lower':img_process_settings['threshold_lower'],
                        'threshold_upper':img_process_settings['threshold_upper'],
                        'disk':int(self.s_disk.get()),
                        'histogram_equalization':self.bool_eq_hist.get(),
                        'edge_find_method':self.s_edge_method.get()}
            for key,input_dict in self.sample_props.items():
                if input_dict['dtype']=='string':
                    temp_string = self.s_sample_props[key].get()
                    if '/' in temp_string:
                        temp_list = []
                        for substring in temp_string.split('/'):
                            temp_list.append(substring)
                        temp_dict[key]=temp_list
                    else:
                        temp_dict[key]=temp_string
                elif input_dict['dtype']=='float':
                    # Try separating by '/' for layer stacks
                    temp_string = self.s_sample_props[key].get()
                    if '/' in temp_string:
                        temp_list = []
                        for sublayer in temp_string.split('/'):
                            temp_list.append(float(sublayer))
                        temp_dict[key]=temp_list
                    else:
                        temp_dict[key]=float(self.s_sample_props[key].get())
            data_dict_list.append(temp_dict)
        #print(data_dict_list)
        self.df = self.df.append(data_dict_list,ignore_index=True)
        df_local = df_local.append(data_dict_list,ignore_index=True)
        #print(self.df)
        # Pickle the dataframe
        self.df.to_pickle(os.path.join(self.df_dir,self.df_file))
        df_local.to_pickle(df_local_file)
        # Save csv of data
        #savename = self.increment_save_name(self.save_dir,'growth_rates_data','.csv')+'.csv'
        df_local.to_csv(os.path.join(self.save_dir,'growth_rates_data.csv'))
        print('results saved to ' + self.df_dir + self.df_file)

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
        for line in self.ax[0].lines:
            line.remove()
        #img=misc.imread(os.path.join(self.base_dir,self.time_files[-1]))#,mode='L')
        #img = exposure.equalize_adapthist(img,clip_limit=0.05)
        #self.cropData.set_data(img[self.y1:self.y2,self.x1:self.x2])
        line, = self.ax[0].plot([], [], '-or')
        # Draw lines on image
        line.set_data(self.linebuilder.xs, self.linebuilder.ys)
        keyStr = '' #string to display growth rates
        self.line_texts=[]
        for idx in range(0, len(self.lines)):    #add labels to lines
            txt = self.ax[0].text(self.lines[idx][1][0],
                                self.lines[idx][1][1]-25,
                                idx+1, color = 'red',
                                size = 'x-large', weight = 'bold')
            self.line_texts.append(txt)
            keyStr = keyStr + str(idx+1) + ': ' + str(self.growth_rates_string[idx]) #add growth rate to keyStr
            if idx != len(self.lines)-1:
                keyStr = keyStr + '\n'
        #self.ax[0].text(1.05, 0.5, keyStr, transform=self.ax[0].transAxes, size = 'x-large', bbox=dict(facecolor='white', alpha=1))
        self.ax[0].axis('off')
        self.ax[0].set_title('')
        self.ax[0].relim()
        # Store the figure so it can be saved later
        self.labeledLinesFig = plt.gcf()
        self.canvas.draw()


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
