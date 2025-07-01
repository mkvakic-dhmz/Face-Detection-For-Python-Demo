#!/usr/bin/bash

# Get image paths as an array
image_paths=($(find UFDD_val/images/snow/*jpg -not -wholename '*labelled*'))

# Run the detection algorithm on each of them
for image_path in ${image_paths[@]:0:30};
do
    ./face.sif ${image_path}
done
