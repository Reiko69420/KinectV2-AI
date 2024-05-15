import cv2
import os
import json

# Define point to get
POINTS = ['footLeft', 'footRight', 'kneeLeft', 'kneeRight', 'hips']
ROTATIONS = ['footLeft', 'footRight', 'kneeLeft', 'kneeRight']

annotations = {}

current_point = 0
points = {}
rotations = {}
occlusion_flag = False
annotation_finished = False

def mouse_event(event, x, y, flags, params):
    global current_point, points, rotations, occlusion_flag, annotation_finished
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_point < len(POINTS):
            points[POINTS[current_point]] = (x, y, occlusion_flag)
            current_point += 1
        elif current_point < len(POINTS) + len(ROTATIONS):
            index = current_point - len(POINTS)
            rotations[ROTATIONS[index]] = (x, y, occlusion_flag)
            current_point += 1
        if current_point == len(POINTS) + len(ROTATIONS):
            annotation_finished = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        occlusion_flag = not occlusion_flag

def annotate_image(image_path):
    global current_point, points, rotations, occlusion_flag, annotation_finished
    points = {}
    rotations = {}
    current_point = 0
    occlusion_flag = False
    annotation_finished = False

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Image", img_color)
    cv2.setMouseCallback("Image", mouse_event)

    while not annotation_finished:
        img_copy = img_color.copy()
        if current_point < len(POINTS):
            annotation_text = f"Annotate {POINTS[current_point]}"
        else:
            annotation_text = f"Annotate Rotation for {ROTATIONS[current_point - len(POINTS)]}"
        
        occlusion_text = "Occlusion: ON" if occlusion_flag else "Occlusion: OFF"
        
        # See selected point
        for point, coord in points.items():
            cv2.circle(img_copy, (coord[0], coord[1]), 5, (0, 255, 0), -1)
        for rotation, coord in rotations.items():
            cv2.circle(img_copy, (coord[0], coord[1]), 5, (255, 0, 0), -1)
        
        cv2.putText(img_copy, annotation_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_copy, occlusion_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_copy, "Press 'r' to reset annotations", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img_copy, "Press 's' to skip to next image", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img_copy, "'Right Click' if bone is occluded", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow("Image", img_copy)
        key = cv2.waitKey(1)
        
        if key == ord('r'):
            points = {}
            rotations = {}
            current_point = 0
            occlusion_flag = False
            annotation_finished = False
        
        if key == ord('s'):
            annotation_finished = True

    cv2.destroyAllWindows()
    return points, rotations

def annotate_all_images(image_dir):
    annotations = {}
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            print(f"Annotating {filename}")
            points, rotations = annotate_image(image_path)
            annotations[filename] = {'points': points, 'rotations': rotations}
    return annotations

image_dir = "depth_images"
annotations = annotate_all_images(image_dir)

with open('annotations.json', 'w') as f:
    json.dump(annotations, f)