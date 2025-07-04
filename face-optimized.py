#!/usr/bin/env python3
from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image

import PIL
import typer
import pathlib

import onnxruntime as ort

from typing import List
from functools import partial
from concurrent.futures import ThreadPoolExecutor

def label_faces(image_filename: pathlib.Path, face_detector: FaceDetection):

    # Detect faces & log output
    image = PIL.Image.open(image_filename)
    faces = face_detector(image_filename)
    print(f"Found {len(faces)} faces in {image_filename}")

def mark_faces_parallel(images_directory: pathlib.Path):
    """Mark all faces recognized in the image"""

    # Create a single thread session
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Create a face detector
    face_detector = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)
    face_detector.session = ort.InferenceSession(face_detector.model_path, sess_options=sess_options)

    # Get images from the directory
    image_filenames = images_directory.rglob('*.jpg')

    # Create an executor context
    with ThreadPoolExecutor(max_workers=8) as executor:

        # Label faces concurrently
        label_faces_partial = partial(label_faces, face_detector=face_detector)
        executor.map(label_faces_partial, image_filenames)

if __name__ == "__main__":
    typer.run(mark_faces_parallel)
