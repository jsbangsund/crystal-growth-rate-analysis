# Crystal Growth Rate Analysis
## Summary
This package provides an interactive GUI for measuring crystal growth rates from timeseries polarized optical microscopy images.
The GUI allows the user to load a series of images and customize various image processing routines to improve edge detection. Two edge detection routines are included:

1. Grain thresholding: Grains which are brighter or darker than the amorphous background are thresholded, and the edge is detected as the last point which is above a certain threshold.
2. Image subtraction: Subsequent images are subtracted to yield a thresholded image of the regions which changed most between two moments in time. If the parameters are tuned properly, a continuous band around each crystalline grain can be obtained.

See below for example usage of the application:
![Example usage of application](https://github.com/jsbangsund/crystal-growth-rate-analysis/blob/master/example_usage.gif)

(screen capture is made with https://www.screentogif.com/)

Example crystal growth:

![Example crystal growth](https://github.com/jsbangsund/crystal-growth-rate-analysis/blob/master/example_crystal_growth.gif)

## Running the software
To ensure that the GUI runs properly, it is recommended that you create a virtual environment using Anaconda (see docs [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually))
Install the environment from the environment.yml file in this repository:
    conda env create -f environment.yml
Then activate this environment:
    conda activate myenv
Then, run the program:
    python GrowthRateAnalyzer.py
    
## Other details
ManualGrowthRateAnalyzer.py allows user to manually pick out edge points.
This is useful for very low contrast images where it is difficult to automate edge detection, where high accuracy is needed, or for situations where a user wants to evaluate the accuracy of measurements based on edge detection.
