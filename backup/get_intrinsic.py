import csv
import numpy as np

def extract_intrinsic_from_csv(file_path):
    intrinsic_params = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if 'Fx' in row:
                intrinsic_params['Fx'] = float(row[1])
            elif 'Fy' in row:
                intrinsic_params['Fy'] = float(row[1])
            elif 'PPx' in row:
                intrinsic_params['PPx'] = float(row[1])
            elif 'PPy' in row:
                intrinsic_params['PPy'] = float(row[1])

    # Construct the camera intrinsic matrix
    K = np.array([[intrinsic_params['Fx'], 0, intrinsic_params['PPx']],
                  [0, intrinsic_params['Fy'], intrinsic_params['PPy']],
                  [0, 0, 1]])

    return K

# Example usage
file_path = '1_Depth_metadata.csv'  # Replace with your CSV file path
intrinsic_matrix = extract_intrinsic_from_csv(file_path)
print(intrinsic_matrix)
np.save('intrinsic_matrix.npy', intrinsic_matrix)