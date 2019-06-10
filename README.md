# Crystal Growth Rate Analysis
This code was developed to analyze the images in our paper on spontaneous pattern formation in organic semiconductors. If you find this code useful, please cite:

Formation of aligned periodic patterns during the crystallization of organic semiconductor thin films. Nature Materials 1 (2019). doi: [10.1038/s41563-019-0379-3](https://doi.org/10.1038/s41563-019-0379-3).

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

Install the environment from the environment.yml file in this repository (you can change the name of the environment that will be created in the first line of environment.yml):

    conda env create -f environment.yml

Then activate this environment:

    conda activate py35_image
    
Then, run the program:

    python GrowthRateAnalyzer.py
    
To delete an environment:

    conda env remove -n ENV_NAME
    
To update the environment.yml file:
    
    conda env export > environment.yml
    
## Other details
ManualGrowthRateAnalyzer.py allows user to manually pick out edge points.
This is useful for very low contrast images where it is difficult to automate edge detection, where high accuracy is needed, or for situations where a user wants to evaluate the accuracy of measurements based on edge detection.
