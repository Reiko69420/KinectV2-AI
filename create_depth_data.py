import os
import time
import numpy as np
from pykinect import runtime
from pykinect import com
import cv2

output_dir = "depth_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

kinect = runtime.PyKinectRuntime(com.FrameSourceTypes_Depth)

def capture_depth_frame():
    if kinect.has_new_depth_frame():
        frame = kinect.get_last_depth_frame()
        frame = frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width))
        return frame
    return None

def save_depth_frame_as_image(frame, index):
    # Normalize depth for grayscale image, close = 0 far = 255, maybe finding another way to Normalize can give more information on the persons body?
    normalized_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    filename = os.path.join(output_dir, f"depth_{index}.png")
    cv2.imwrite(filename, normalized_frame)

index = 0
while True:
    depth_frame = capture_depth_frame()
    if depth_frame is not None:
        save_depth_frame_as_image(depth_frame, index)
        index += 1
    time.sleep(1)


kinect.close()