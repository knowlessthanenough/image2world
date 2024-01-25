import pyrealsense2 as rs
import numpy as np

# Configure the pipeline to stream only color (RGB) data
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

try:
    # Wait for a coherent pair of frames
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("Could not get a color frame.")

    # Get the intrinsic parameters of the color stream
    intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()

    # Convert intrinsics to a numpy array
    intrinsic_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                 [0, intrinsics.fy, intrinsics.ppy],
                                 [0, 0, 1]])

    # Print and save the intrinsic matrix
    print("Color camera intrinsics: \n", intrinsic_matrix)
    np.save('rgb_camera_intrinsics.npy', intrinsic_matrix)

finally:
    pipeline.stop()
