import os
import csv
import numpy

camera_index = 2
scans_path = '/data/ZItong/MatterPort/v1/scans'

import numpy as np

for dir in os.listdir(scans_path):      
    cameras_transformations = []
    camera_pos_location = os.path.join(scans_path, dir, "matterport_camera_poses",dir, "matterport_camera_poses")
    for file in os.listdir(camera_pos_location):
        if file.endswith("_1_2.txt"):
            with open(os.path.join(camera_pos_location, file), 'r') as f:
                matrix = []
                for line in f:
                    row = list(map(float, line.strip().split()))
                    matrix.append(row)

                dat = {
                    'camera': file.split("_")[0],
                    'position': [ matrix[0][3], matrix[1][3], matrix[2][3] ], #position
                    'rotation': #rotation matrix
                [ [matrix[0][0], matrix[0][1], matrix[0][2]],
                    [-matrix[1][0], -matrix[1][1], -matrix[1][2]],
                    [-matrix[2][0], -matrix[2][1], -matrix[2][2]] ]
                }

                #Correcting the rotation matrix
                rot_numpy = np.array(dat['rotation']) 
                rotation_z_90 = numpy.array([
                    [numpy.cos(numpy.pi / 2), -numpy.sin(numpy.pi / 2), 0],
                    [numpy.sin(numpy.pi / 2), numpy.cos(numpy.pi / 2), 0],
                    [0, 0, 1]
                ])
                rot_numpy = np.dot( rotation_z_90,rot_numpy)                
                dat['rotation'] = rot_numpy.tolist()

                cameras_transformations.append(dat)

                

    # Define the output files
    positions_csv = os.path.join(scans_path,dir, 'camera_positions.csv')

    # Save positions to CSV
    with open(positions_csv, 'w', newline='') as csvfile:
        for camera in cameras_transformations:
            writer = csv.writer(csvfile)
            writer.writerow([camera['camera'], camera['position'][0], camera['position'][1], camera['position'][2]] + [item for sublist in camera['rotation'] for item in sublist])
    print("Done for", positions_csv)
    
            
    