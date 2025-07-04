#!/usr/bin/bash

# Get image paths as an array
image_paths=($(find UFDD_val/images -type f -and -not -wholename '*labelled*'))

# Run the detection algorithm on the whole list
time apptainer exec \
    face.sif \
        python3 face-optimized.py ${image_paths[@]:0:1000}
