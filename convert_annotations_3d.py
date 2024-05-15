import json
import numpy as np
import cv2
import os
import json
import numpy as np

image_dir = "depth_images"

def calculate_3d_point(depth_image, point_2d, occluded):
    if occluded:
        # Apply an alternative method to estimate the 3D position
        x, y = int(point_2d[0]), int(point_2d[1])
        z = np.mean(depth_image)  # Basic estimation, can be improved
        return (x, y, z)
    else:
        x, y = int(point_2d[0]), int(point_2d[1])
        z = depth_image[y, x]
        return (x, y, z)

def calculate_rotation_vector(depth_image, point_2d, direction_2d, occluded):
    point_3d = calculate_3d_point(depth_image, point_2d, occluded)
    direction_3d = calculate_3d_point(depth_image, direction_2d, occluded)
    vector_3d = np.array(direction_3d) - np.array(point_3d)
    return vector_3d.tolist()  # Convert the NumPy array to a Python list

def load_depth_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return img

with open('annotations.json', 'r') as f:
    annotations = json.load(f)

results = {}
for filename, data in annotations.items():
    depth_image = load_depth_image(os.path.join(image_dir, filename))
    points_3d = {key: calculate_3d_point(depth_image, value[:2], value[2]) for key, value in data['points'].items()}
    rotations_3d = {key: calculate_rotation_vector(depth_image, data['points'][key][:2], value[:2], value[2]) for key, value in data['rotations'].items()}
    
    #print("Points 3D:", points_3d)
    #print("Rotations 3D:", rotations_3d)
    
    results[filename] = {'points_3d': points_3d, 'rotations_3d': rotations_3d}

#print("Results:", results)

def convert_uint8_to_int(value):
    if isinstance(value, np.uint8):
        return int(value)
    return value

# Weird ass looking json converter cause json.dump was not working
with open('results.json', 'w') as f:
    f.write("{\n")
    for idx, (filename, data) in enumerate(results.items()):
        f.write(f'  "{filename}": {{\n')
        f.write(f'    "points_3d": {{\n')
        for key, value in data['points_3d'].items():
            converted_value = [convert_uint8_to_int(v) for v in value]
            f.write(f'      "{key}": {converted_value}')
            if key != list(data['points_3d'].keys())[-1]:  # Check if it's not the last value
                f.write(",")  # Add a comma to all values except the last
            f.write("\n")
        f.write("    },\n")
        
        f.write(f'    "rotations_3d": {{\n')
        for key, value in data['rotations_3d'].items():
            converted_value = [convert_uint8_to_int(v) for v in value]
            f.write(f'      "{key}": {converted_value}')
            if key != list(data['rotations_3d'].keys())[-1]:  # Check if it's not the last value
                f.write(",")  # Add a comma to all values except the last
            f.write("\n")
        f.write("    }\n")
        
        f.write("  }")
        if idx < len(results) - 1:  # Check if it's not the last entry
            f.write(",")  # Add a comma to all entries except the last
        f.write("\n")
    f.write("}\n")