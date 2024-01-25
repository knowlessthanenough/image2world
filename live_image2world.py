import pyrealsense2 as rs
import numpy as np
import cv2
import time
from depth2world import transform_entire_image_with_depth

def align_depth_to_color(frameset):
    """
    Align the depth frame to the color frame within a frameset.

    Parameters:
    frameset (rs.frameset): The set of frames from the RealSense camera, including both depth and color frames.

    Returns:
    rs.frame: The aligned depth frame.
    """
    # Create alignment primitive with color as its target stream
    align = rs.align(rs.stream.color)

    # Perform the alignment
    aligned_frameset = align.process(frameset)

    # Get the aligned depth frame
    aligned_depth_frame = aligned_frameset.get_depth_frame()

    return aligned_depth_frame

def get_frames_from_camera():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        # Wait for a coherent pair of frames: depth and color
        frameset = pipeline.wait_for_frames()

        if not frameset:
            return None

        return frameset
    finally:
        pipeline.stop()

def capture_image():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline.start(config)

    try:
        frameset = pipeline.wait_for_frames()
        if frameset:
            aligned_depth_frame = align_depth_to_color(frameset)
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_frame = frameset.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

    finally:
        pipeline.stop()

    return color_image, depth_image

def show_aligned_images(RGB_intrinsic_matrix, RGB_rotation_vector, RGB_translation_vector):
    while True:
        color_image, depth_image = capture_image()
        # processed_depth_image
        world_coords_3d = transform_entire_image_with_depth(depth_image, RGB_intrinsic_matrix, RGB_rotation_vector, RGB_translation_vector)

        time.sleep(300)  # Wait for 5 minutes before capturing the next image


if __name__ == '__main__':
    intrinsic_matrix = np.load('data/rgb_intrinsic_matrix.npy')
    rotation_vector = np.load('data/rgb_rotation_vector.npy')
    translation_vector = np.load('data/rgb_translation_vector.npy')
    show_aligned_images(intrinsic_matrix, rotation_vector, translation_vector)


