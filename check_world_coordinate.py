import numpy as np
import cv2
import os



def draw_axes(img, M, r_vec, T, length=130):
    """
    Draw XYZ axes on the image.

    :param img: Input image
    :param M: Camera intrinsic matrix (3x3)
    :param r_vec: Rotation vector (3x1)
    :param T: Translation vector (3x1)
    :param length: Length of the axes (default: 1( 2.5cm ))
    """
    # Define the axes' endpoints in world coordinates
    origin = np.array([0, 0, 0, 1]).reshape(4, 1)
    x_axis = np.array([length, 0, 0, 1]).reshape(4, 1)
    y_axis = np.array([0, length, 0, 1]).reshape(4, 1)
    z_axis = np.array([0, 0, length, 1]).reshape(4, 1)

    # Transform world coordinates to image coordinates
    rot_mat, _ = cv2.Rodrigues(r_vec)
    transform = np.column_stack((rot_mat, T))
    img_origin = np.dot(M, np.dot(transform, origin))
    img_x_axis = np.dot(M, np.dot(transform, x_axis))
    img_y_axis = np.dot(M, np.dot(transform, y_axis))
    img_z_axis = np.dot(M, np.dot(transform, z_axis))

    # Normalize the coordinates
    img_origin = (img_origin / img_origin[2]).astype(int)
    img_x_axis = (img_x_axis / img_x_axis[2]).astype(int)
    img_y_axis = (img_y_axis / img_y_axis[2]).astype(int)
    img_z_axis = (img_z_axis / img_z_axis[2]).astype(int)

    # Draw the axes
    img = cv2.line(img, tuple(img_origin[:2].ravel()), tuple(img_z_axis[:2].ravel()), (0, 0, 255), 2)  # Z - blue
    cv2.putText(img, 'Z', tuple(img_z_axis[:2].ravel()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    img = cv2.line(img, tuple(img_origin[:2].ravel()), tuple(img_x_axis[:2].ravel()), (255, 0, 0), 2)  # X - red
    cv2.putText(img, 'X', tuple(img_x_axis[:2].ravel()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    img = cv2.line(img, tuple(img_origin[:2].ravel()), tuple(img_y_axis[:2].ravel()), (0, 255, 0), 2)  # Y - green
    cv2.putText(img, 'Y', tuple(img_y_axis[:2].ravel()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.clear()  # Clear the list to store only the latest point
        param.append((x, y))

if __name__ == '__main__':

    # actual_height = 2300
    scale_ratio = 1
    # folder_path = "/home/nvidia/Documents/queenMary/test_lab"

    depth_path = "data/world_coordinates.npy"
    image_path = "data/test_color.png"
    depth_map = np.load(depth_path)
    
    # Example usage
    # depth = depth_map[point[0][1]*2][point[0][0]*2]

    image = cv2.imread(image_path)  # Replace "image.jpg" with your image file path
    print(image.shape)

    # Resize the image to half its size
    resized_image = cv2.resize(image, (0, 0), fx=1/scale_ratio, fy=1/scale_ratio)
    resized_image = draw_axes(resized_image, np.load('data/rgb_intrinsic_matrix.npy'), np.load('data/rgb_rotation_vector.npy'), np.load('data/rgb_translation_vector.npy'))
    # cv2.imwrite("data/aligned_depth_with_axes.png", resized_image)
    print(resized_image.shape)

    cv2.namedWindow("Image with Dot")
    point = []  # List to store the clicked points
    cv2.setMouseCallback("Image with Dot", mouse_callback, point)

    while True:
        image_with_dot = resized_image.copy()

        # Draw the newest dot on the image and set text with coordinates
        if point:
            dot_color = (255, 0, 0)  # Green color for the dot
            dot_radius = 2
            cv2.circle(image_with_dot, point[0], dot_radius, dot_color, -1)
            # depth = depth_map[point[0][0], point[0][1]]
            depth = depth_map[point[0][1]*scale_ratio][point[0][0]*scale_ratio]
            depth = [round(d,2) for d in depth]
            # text = f"({point[0][0]*scale_ratio}, {point[0][1]*scale_ratio}): " + str(depth)
            text = str(depth)
            print(f"({point[0][0]*scale_ratio}, {point[0][1]*scale_ratio}): " + str(depth))
            cv2.putText(image_with_dot, text, (point[0][0] + 5, point[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 1)

        cv2.imshow("Image with Dot", image_with_dot)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    print(depth_map[0][0])