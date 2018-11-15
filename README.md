# crystal-growth-rate-analysis
This package provides an interactive GUI for measuring crystal growth rates from timeseries polarized optical microscopy images.
The GUI allows the user to load a series of images and customize various image processing routines to improve edge detection. Two edge detection routines are included:

1. Grain thresholding: Grains which are brighter or darker than the amorphous background are thresholded, and the edge is detected as the last point which is above a certain threshold.
2. Image subtraction: Subsequent images are subtracted to yield a thresholded image of the regions which changed most between two moments in time. If the parameters are tuned properly, a continuous band around each crystalline grain can be obtained.

See below for example usage of the application:
![Example usage of application](https://github.com/jsbangsund/crystal-growth-rate-analysis/blob/master/example_usage.gif)

(screen capture is made with https://www.screentogif.com/)

Example crystal growth:
![Example crystal growth](https://github.com/jsbangsund/crystal-growth-rate-analysis/blob/master/example_crystal_growth.gif)
