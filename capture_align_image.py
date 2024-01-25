import pyrealsense2 as rs
import numpy as np
import cv2

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

def show_aligned_images():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        frame_count = 0
        while True:
            # Wait for a coherent pair of frames: depth and color
            frameset = pipeline.wait_for_frames()
            if not frameset:
                continue

            # Align the depth frame to color frame
            aligned_depth_frame = align_depth_to_color(frameset)

            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_frame = frameset.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Resize images to half the width and height for display
            resized_color_image = cv2.resize(color_image, (640, 360))
            resized_depth_colormap = cv2.resize(depth_colormap, (640, 360))

            # Stack both images horizontally
            images = np.hstack((resized_color_image, resized_depth_colormap))

            # Show images
            cv2.namedWindow('Aligned Depth & Color', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Aligned Depth & Color', images)

            # Break the loop when 'q' is pressed
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('c'):
                # Save color image and depth map
                color_img_filename = f"data/test_color_{frame_count}.png"
                # depth_img_filename = f"data/test_depth_{frame_count}.png"
                depth_map_filename = f"data/depth_map_{frame_count}.npy"

                cv2.imwrite(color_img_filename, color_image)
                # normalized_depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                # cv2.imwrite(depth_img_filename, normalized_depth_image)
                np.save(depth_map_filename, depth_image)
                print(depth_image.shape)

                print(f"Saved {color_img_filename} and {depth_map_filename}")
                frame_count += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    show_aligned_images()
