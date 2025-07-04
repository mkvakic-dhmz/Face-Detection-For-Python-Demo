#!/usr/bin/bash

# Define images directory variable
images_directory='WIDER_test/images/2--Demonstration/'

# Run the detection algorithm on a directory
time apptainer exec \
    face.sif \
        python3 face-optimized.py ${images_directory}
