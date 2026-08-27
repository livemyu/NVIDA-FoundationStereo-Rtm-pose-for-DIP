import os
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def extract_sample_stereo_pair(bag_path):
    print("Extracting sample stereo pair from bag...")
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)
    
    bridge = CvBridge()
    left_img, right_img = None, None
    
    # Read messages until we find a pair around 30 seconds in
    msg_count = 0
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic == '/camera/left/image_raw' and left_img is None:
            msg_count += 1
            if msg_count > 300: # skip initial frames
                msg = deserialize_message(data, Image)
                left_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        elif topic == '/camera/right/image_raw' and right_img is None and left_img is not None:
            msg = deserialize_message(data, Image)
            right_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
        if left_img is not None and right_img is not None:
            break
            
    return left_img, right_img

def compute_sgbm_disparity(left, right):
    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    
    window_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disp = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
    disp[disp <= 0] = 0
    return disp

def compute_foundationstereo_disparity(left, right, model_path):
    print("Running FoundationStereo ONNX inference...")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # FoundationStereo ONNX input shape: 320 x 736
    target_h, target_w = 320, 736
    h_orig, w_orig = left.shape[:2]
    
    # Preprocess
    l_resized = cv2.resize(left, (target_w, target_h))
    r_resized = cv2.resize(right, (target_w, target_h))
    
    l_tensor = (l_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    r_tensor = (r_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    
    input_names = [inp.name for inp in session.get_inputs()]
    inputs = {input_names[0]: l_tensor, input_names[1]: r_tensor}
    
    outputs = session.run(None, inputs)
    disp = outputs[0][0, 0] # 320 x 736
    
    # Scale disparity back to original image size
    disp_resized = cv2.resize(disp, (w_orig, h_orig)) * (w_orig / target_w)
    return disp_resized

def main():
    bag_path = 'my_dataset_20260806_113813'
    model_path = 'foundationstereo_320x736.onnx'
    out_img = 'output/stereo_depth_comparison.png'
    
    left, right = extract_sample_stereo_pair(bag_path)
    if left is None or right is None:
        print("Error: Could not extract stereo pair from dataset.")
        return
        
    print("Computing traditional ROS SGBM disparity...")
    disp_sgbm = compute_sgbm_disparity(left, right)
    
    print("Computing FoundationStereo Deep Neural Disparity...")
    disp_foundation = compute_foundationstereo_disparity(left, right, model_path)
    
    # Plot side-by-side comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), dpi=150)
    
    # 1. Original Left RGB Image
    axes[0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Camera Left Image (White Wall Corridor)", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Traditional ROS SGBM Disparity Map
    im1 = axes[1].imshow(disp_sgbm, cmap='jet', vmin=0, vmax=128)
    axes[1].set_title("ROS 2 Traditional SGBM (High Noise & Black Holes in Textureless White Walls)", fontsize=12, fontweight='bold', color='crimson')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.04, label='Disparity (pixels)')
    
    # 3. FoundationStereo Deep Neural Disparity Map
    im2 = axes[2].imshow(disp_foundation, cmap='jet', vmin=0, vmax=128)
    axes[2].set_title("NVIDIA FoundationStereo Deep Model (Smooth, Dense & Zero-Hole Wall Depth)", fontsize=12, fontweight='bold', color='forestgreen')
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.03, pad=0.04, label='Disparity (pixels)')
    
    plt.tight_layout()
    plt.savefig(out_img, bbox_inches='tight')
    plt.close()
    print(f"Saved side-by-side depth comparison rendering to {out_img}")

if __name__ == '__main__':
    main()
