import numpy as np
import cv2

def read_depth_image(file_path, width=1280, height=720):
    """
    Read a depth image from a raw file captured by Intel RealSense D435i.

    Args:
    file_path (str): Path to the raw depth image file.
    width (int): Width of the depth image. Default is 1280 pixels.
    height (int): Height of the depth image. Default is 720 pixels.

    Returns:
    numpy.ndarray: The depth image.
    """
    # Calculate the total number of bytes in the file
    num_bytes = width * height * 2  # 2 bytes (16 bits) per pixel

    # Read the file as a 1D array of 16-bit unsigned integers
    with open(file_path, 'rb') as file:
        depth_data = np.fromfile(file, dtype=np.uint16, count=num_bytes)

    # Reshape the 1D array into a 2D array of the specified dimensions
    depth_image = depth_data.reshape((height, width))

    # # for show
    # # Step 1: Rescale to 0-1
    # depth_image_rescaled = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))

    # # Step 2: Apply histogram equalization
    # depth_image_equalized = cv2.equalizeHist(np.uint8(depth_image_rescaled * 255))

    # # depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow('Depth Image', depth_image_rescaled)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return depth_image

def image_to_world(i,depth_map, M, r_vec, T):
    """
    Convert image coordinate (u, v) and z to 3D world coordinate (X, Y, Z) using camera calibration parameters.

    :param i: Image coordinate [u,v] (2x1)
    :param depth_map: Depth map (720x1280)
    :param M: Camera intrinsic matrix (3x3)
    :param D: Distortion coefficients (1x5 or 1x4)
    :param r_vec: Rotation vector (3x1)
    :param T: Translation vector (3x1)
    :return w: World coordinate (X, Y)
    """
    i_vector = np.append(i,1).reshape(3, 1)	
    z = depth_map[i[1], i[0]]	
    M_inv = np.linalg.inv(M)
    rot_mat, _ = cv2.Rodrigues(r_vec) # Convert rotation vector to rotation matrix
    rot_mat_inv = np.linalg.inv(rot_mat)
    left_side = np.matmul(rot_mat_inv, np.matmul(M_inv,i_vector)) #(3x1)
    right_side = np.matmul(rot_mat_inv , T) #(3x1)
    s = (z + right_side[2, 0]) / left_side[2, 0] # calculate the scalar
    # w = (s * left_side - right_side) # unit is how many box(need to multiply by the length of the box)
    w = np.matmul(rot_mat_inv, (s * np.matmul(M_inv, i_vector)) - T) # unit is how many box(need to multiply by the length of the box)
    return w
    
def image_to_world_vectorized(coords, depth_map, M_inv, rot_mat_inv, T):
    u, v = coords[..., 0], coords[..., 1]
    z = depth_map[v, u]  # Extract depth values for each coordinate
    i_vector = np.stack([u, v, np.ones_like(u)], axis=-1)  # Convert to homogeneous coordinates

    i_vector_transformed = i_vector @ M_inv.T
    left_side = rot_mat_inv @ i_vector_transformed[..., np.newaxis]  # Shape: [height, width, 3, 1]

    # Ensure 'right_side' is broadcastable with 'left_side'
    right_side = rot_mat_inv @ T.reshape(-1, 1)  # Shape: [3, 1]
    right_side = right_side.reshape(1, 1, 3, 1)  # Reshaped to [1, 1, 3, 1]

    # Reshape 's' for broadcasting
    s = z[..., np.newaxis, np.newaxis]  # Shape becomes [height, width, 1, 1]
    w = s * left_side - right_side
    w = w.squeeze(-1)  # Remove the last dimension, shape: [height, width, 3]

    return w

def transform_entire_image_with_depth(depth_map, M, r_vec, T):
    M_inv = np.linalg.inv(M)
    rot_mat, _ = cv2.Rodrigues(r_vec)
    rot_mat_inv = np.linalg.inv(rot_mat)

    height, width = depth_map.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack([u, v], axis=-1)

    world_coords_3d = image_to_world_vectorized(coords, depth_map, M_inv, rot_mat_inv, T)
    return world_coords_3d

# def depth_to_camera_coordinates(depth_image, intrinsic_matrix):
#     # Get the shape of the depth image
#     height, width = depth_image.shape

#     # Create a grid of coordinates corresponding to the indices of the depth image
#     x, y = np.meshgrid(np.arange(width), np.arange(height))

#     # Normalize x and y to camera space
#     x_norm = (x - intrinsic_matrix[0, 2]) / intrinsic_matrix[0, 0]
#     y_norm = (y - intrinsic_matrix[1, 2]) / intrinsic_matrix[1, 1]

#     # Reproject to 3D space
#     z = depth_image
#     x = np.multiply(x_norm, z)
#     y = np.multiply(y_norm, z)

#     # Stack the coordinates in a 3D array
#     camera_coordinates = np.stack((x, y, z), axis=-1)

#     return camera_coordinates

def show_z_channel(camera_coordinates):
    # Extract the Z channel
    z_channel = camera_coordinates[:, :, 2]

    # Normalize the Z channel for better visualization
    z_norm = cv2.normalize(z_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite("data/world_coordinates.png", z_norm)
    # Display the Z channel
    cv2.imshow('Z Channel', z_norm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def camera_to_world_coordinates(camera_coordinates, rotation_vector, translation_vector):
#     # Convert the rotation vector to a rotation matrix
#     rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

#     # Transform each point
#     world_coordinates = np.dot(camera_coordinates, rotation_matrix.T) + translation_vector
#     print("camera_coordinates.shape", camera_coordinates.shape)

#     return world_coordinates

if __name__ == '__main__':
    # Usage example
    # depth_image = read_depth_image('data/test_depth.raw')
    depth_image = np.load('data/depth_map.npy')
    print(depth_image.shape) #(720, 1280)
    intrinsic_matrix = np.load('data/rgb_intrinsic_matrix.npy')

    # # Convert the depth image to camera coordinates
    # camera_coordinates = depth_to_camera_coordinates(depth_image, intrinsic_matrix)
    # print(camera_coordinates.shape) #(720, 1280, 3)
    # np.save('data/camera_coordinates.npy', camera_coordinates)

    # # Display the Z channel
    # show_z_channel(camera_coordinates)

    # # Load the rotation and translation vectors
    # rotation_vector = np.load('data/depth_rotation_vector.npy')
    # print(rotation_vector.shape)
    # translation_vector = np.load('data/depth_translation_vector.npy')
    # #turn the translation vector into a (3,) matrix
    # translation_vector = translation_vector.reshape(3,)
    # print(translation_vector.shape)

    # # Convert the camera coordinates to world coordinates
    # world_coordinates = camera_to_world_coordinates(camera_coordinates, rotation_vector, translation_vector)
    # print(world_coordinates.shape)
    # print(world_coordinates[0][0])

    world_coordinates = transform_entire_image_with_depth(depth_image, intrinsic_matrix, np.load('data/rgb_rotation_vector.npy'), np.load('data/rgb_translation_vector.npy'))

    # Save the world coordinates
    np.save('data/world_coordinates.npy', world_coordinates)

    # Display the Z channel
    show_z_channel(world_coordinates)

