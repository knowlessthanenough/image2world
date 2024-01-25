import pyrealsense2 as rs
import numpy as np
import cv2

def create_transformation_matrix(rotation_vector, translation_vector):
    if rotation_vector.shape == (9,):
        rotation_matrix = rotation_vector.reshape(3, 3)
    else:
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    transformation_matrix = np.vstack((np.hstack((rotation_matrix, translation_vector.reshape(3, 1))), [0, 0, 0, 1]))
    return transformation_matrix

def matrix_to_rot_vec_and_trans(matrix):
    rot_vec, _ = cv2.Rodrigues(matrix[:3, :3])
    trans_vec = matrix[:3, 3]
    return rot_vec, trans_vec

if __name__ == '__main__':
    # Start the RealSense pipeline  
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable both the depth and color streams
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    # Start the pipeline with the above configuration
    pipeline.start(config)

    try:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Could not acquire depth or color frames.")

        # Retrieve the extrinsic parameters of the depth frame relative to the color frame
        depth_to_color_extrinsic = depth_frame.get_profile().get_extrinsics_to(color_frame.get_profile())
        print("Depth camera to color camera extrinsic parameters: \n", depth_to_color_extrinsic.rotation, "\n", depth_to_color_extrinsic.translation)
        np.save('data/depth_to_rgb_rotation_matrix.npy', depth_to_color_extrinsic.rotation)
        np.save('data/depth_to_rgb_translation_vector.npy', depth_to_color_extrinsic.translation)

        depth_to_color_matrix = create_transformation_matrix(np.array(depth_to_color_extrinsic.rotation), np.array(depth_to_color_extrinsic.translation))

        # Your RGB camera extrinsic parameters (rotation vector and translation vector)
        # Replace these with your actual data
        rgb_rotation_vector = np.load('data/rgb_rotation_vector.npy')  # Replace with your actual path
        print(rgb_rotation_vector)
        rgb_translation_vector = np.load('data/rgb_translation_vector.npy')  # Replace with your actual path
        print(rgb_translation_vector)
        rgb_extrinsic_matrix = create_transformation_matrix(rgb_rotation_vector, rgb_translation_vector)

        # Combine the transformations
        combined_matrix = np.dot(rgb_extrinsic_matrix, depth_to_color_matrix)

        # Convert back to rotation vector and translation vector
        depth_rotation_vector, depth_translation_vector = matrix_to_rot_vec_and_trans(combined_matrix)

        # Save the final extrinsic parameters
        np.save('data/depth_rotation_vector.npy', depth_rotation_vector)
        np.save('data/depth_translation_vector.npy', depth_translation_vector)

        print("Final Depth Camera Rotation Vector:", depth_rotation_vector)
        print("Final Depth Camera Translation Vector:", depth_translation_vector)

    finally:
        pipeline.stop()
