#!/usr/bin/bash

# Get image paths as an array
image_paths=(UFDD_val/images/focus/*jpg)

# Run the detection algorithm on each of them
for image_path in ${image_paths[@]:0:5};
do
    ./face.sif ${image_path}
done
