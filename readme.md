### Temporal Brightness Management for Immersive Content

This repository contains the offline version of the method described in [Temporal Brightness Management for Immersive Content](https://diglib.eg.org/handle/10.2312/sr20251183) (Surace et al., 2025).

The code might slightly differ from the pipeline showed in the paper. It takes an input video sequence and outputs a brightness reduction factor \[0-1] for each timestamp of the video. The technique aims to maintain a constant loss of visible contrast throughout the video sequence, for ensuring a consistent perceptual experience. The resulting brightness factor should be applied in the luminance domain before any post-processing of the frames.

This code is calibrated with the characteristics of a Varjo XR-3 headset for the perceptual modeling. The simulation of the power for a custom hardware based on Raspberry 7" LCD is printed in the console as output. The contrast sensitivity model used is based on the work of [Barten et al.](https://pure.tue.nl/ws/files/1613279/9901043.pdf)



##### Usage



###### Arguments



`-i`	the input video file

`-tb`	the target average brightness factor for energy savings

`-c`	the strategy for computation of the contrast loss \[standard, xor] where *standard* is based on the ratio of visible contrast magnitudes of original and dimmed frames, *xor* computes a symmetric differences between the visible contrast maps of 	original and dimmed frames, then applies average pooling. The two strategies have similar results

`-l`	compute the contrast loss for different bright multipliers, otherwise it loads data from file

`-sc`	the maximum slope of the contrast loss between subsequent frames

`-p`	shows the brightness - contrast loss plots

`-h`	shows the summary of the arguments on the terminal



###### Examples


`python temporal_brightness_management.py -i basement.mp4 -tb 0.4 -c 'standard' -l -sc -p`

runs the technique with a target brightness equal to 40% of the full brightness, using the standard strategy with constrained optimization, and shows the plot.

`python temporal_brightness_management.py -i basement.mp4 -tb 0.8 -c 'xor' -l -p`

runs the technique with a target brightness equal to 80% of the full brightness, using the xor strategy, and shows the plot.

##### Notes

This software is released under the license [CC-BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

