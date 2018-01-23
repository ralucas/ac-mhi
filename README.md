Activity Classification using MHI
---
Author: Richard Lucas  

*Writeup and Blog Post can be found here <http://blog.richardalucas.com/2017/12/03/Activity-Classification-using-MHI-Techniques/>*

## Introduction

## Getting Started
The following dependencies will need to be installed:
- OpenCV for Python 2.x
- Numpy
- SciKit Learn
- Matplotlib

## Setup
This creates the necessary folders and downloads all the needed files:
`$ python setup.py`

It typically takes about 30 minutes to get everything.

## How to use

#### Run the experiment
To run the experiment and recreate the figures in the report:  
`$ python experiment.py`

It will output all figures in `png` format into the `output/` directory
It will output the multiple activity videos to the `output_videos` directory

#### Classify a video
If you just want to classify and create a video run:
`$ python ac.py [video_file] [output_name]`  

It will output an `mp4` classified video in the `output_videos/` directory

Example:  
`$ python ac.py Test1.mp4 test1`  

==> `output_videos/classified_test1.mp4`  

## Links
- Video Showcase: <https://youtu.be/Fo-wd0yMZVg>
- Multiple Activity Videos:
    - Handclapping & Boxing Video: <https://youtu.be/dEtfgAYwCg4>
    - Multitude activity Video: <https://youtu.be/x_hFKZzc0jw>
    
