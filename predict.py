# Just a weird predict 3D point calculator

def calculate_3d_point(depth_image, point_2d):
    x, y = int(point_2d[0]), int(point_2d[1])
    z = depth_image[y, x]
    return (x, y, z)

def predict_keypoints_and_rotations(depth_frame, model):
    normalized_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    input_tensor = torch.tensor(normalized_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    output = model(input_tensor).detach().numpy().reshape(-1, 2)
    keypoints_2d = output[:5]
    rotations_2d = output[5:]
    return keypoints_2d, rotations_2d

depth_frame = capture_depth_frame()
keypoints_2d, rotations_2d = predict_keypoints_and_rotations(depth_frame, model)

points_3d = {key: calculate_3d_point(depth_frame, keypoints_2d[i]) for i, key in enumerate(['footLeft', 'footRight', 'kneeLeft', 'kneeRight', 'hips'])}
rotations_3d = {key: calculate_3d_point(depth_frame, rotations_2d[i]) for i, key in enumerate(['footLeft', 'footRight', 'kneeLeft', 'kneeRight'])}
print("Predicted Points 3D:", points_3d)
print("Predicted Rotations 3D:", rotations_3d)