import os
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def extract_multi_stereo_pairs(bag_path, sample_indices=[100, 500, 900, 1400]):
    print(f"Extracting sample stereo pairs at frame indices {sample_indices} from bag...")
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)
    
    bridge = CvBridge()
    pairs = {}
    left_buf = {}
    
    msg_idx = 0
    max_needed = max(sample_indices) + 20
    
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic == '/camera/left/image_raw':
            msg_idx += 1
            if msg_idx in sample_indices:
                msg = deserialize_message(data, Image)
                left_buf[msg_idx] = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        elif topic == '/camera/right/image_raw':
            for idx in list(left_buf.keys()):
                if idx not in pairs:
                    msg = deserialize_message(data, Image)
                    right_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    pairs[idx] = (left_buf[idx], right_img)
        if msg_idx > max_needed:
            break
            
    return [pairs[idx] for idx in sample_indices if idx in pairs]

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

def compute_foundationstereo_disparity(session, left, right):
    target_h, target_w = 320, 736
    h_orig, w_orig = left.shape[:2]
    
    l_resized = cv2.resize(left, (target_w, target_h))
    r_resized = cv2.resize(right, (target_w, target_h))
    
    l_tensor = (l_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    r_tensor = (r_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    
    input_names = [inp.name for inp in session.get_inputs()]
    inputs = {input_names[0]: l_tensor, input_names[1]: r_tensor}
    
    outputs = session.run(None, inputs)
    disp = outputs[0][0, 0]
    
    disp_resized = cv2.resize(disp, (w_orig, h_orig)) * (w_orig / target_w)
    return disp_resized

def main():
    bag_path = 'my_dataset_20260806_113813'
    model_path = 'foundationstereo_320x736.onnx'
    out_img = 'output/stereo_multi_frame_comparison.png'
    out_mp4 = 'output/stereo_depth_video_demo.mp4'
    
    sample_pairs = extract_multi_stereo_pairs(bag_path, sample_indices=[150, 600, 1000, 1500])
    print(f"Successfully extracted {len(sample_pairs)} stereo pairs across dataset.")
    
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # 1. Generate 4-frame matrix plot
    fig, axes = plt.subplots(len(sample_pairs), 3, figsize=(15, 4 * len(sample_pairs)), dpi=150)
    
    for i, (left, right) in enumerate(sample_pairs):
        disp_sgbm = compute_sgbm_disparity(left, right)
        disp_found = compute_foundationstereo_disparity(session, left, right)
        
        # Column 1: Left RGB
        axes[i, 0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Frame {i+1}: Original RGB Image", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Column 2: SGBM
        axes[i, 1].imshow(disp_sgbm, cmap='jet', vmin=0, vmax=128)
        axes[i, 1].set_title(f"Frame {i+1}: ROS SGBM (High Noise)", fontsize=10, fontweight='bold', color='crimson')
        axes[i, 1].axis('off')
        
        # Column 3: FoundationStereo
        im = axes[i, 2].imshow(disp_found, cmap='jet', vmin=0, vmax=128)
        axes[i, 2].set_title(f"Frame {i+1}: FoundationStereo (Smooth Wall)", fontsize=10, fontweight='bold', color='forestgreen')
        axes[i, 2].axis('off')
        
    plt.suptitle("Multi-Frame Side-by-Side Depth Comparison across White Wall Corridors", fontsize=14, fontweight='bold', y=1.002)
    plt.tight_layout()
    plt.savefig(out_img, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-frame depth comparison matrix to {out_img}")
    
    # 2. Render 30-frame side-by-side video (RGB vs SGBM vs FoundationStereo)
    print("Generating 30-frame side-by-side depth video MP4...")
    video_pairs = extract_multi_stereo_pairs(bag_path, sample_indices=list(range(200, 230)))
    
    h, w = video_pairs[0][0].shape[:2]
    out_video_w = w * 3
    out_video_h = h
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_avi = 'output/stereo_depth_video_demo.avi'
    writer = cv2.VideoWriter(out_avi, fourcc, 10.0, (out_video_w, out_video_h))
    
    for i, (left, right) in enumerate(video_pairs):
        disp_sgbm = compute_sgbm_disparity(left, right)
        disp_found = compute_foundationstereo_disparity(session, left, right)
        
        # Colorize disparities using Jet colormap
        disp_sgbm_norm = cv2.normalize(disp_sgbm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_sgbm_color = cv2.applyColorMap(disp_sgbm_norm, cv2.COLORMAP_JET)
        
        disp_found_norm = cv2.normalize(disp_found, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_found_color = cv2.applyColorMap(disp_found_norm, cv2.COLORMAP_JET)
        
        # Put labels
        cv2.putText(left, "Left RGB Camera", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(disp_sgbm_color, "ROS SGBM Depth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(disp_found_color, "FoundationStereo AI Depth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Concatenate horizontally
        frame_combined = np.hstack((left, disp_sgbm_color, disp_found_color))
        writer.write(frame_combined)
        
    writer.release()
    print(f"Saved side-by-side depth comparison AVI video to {out_avi}")
    os.system(f"ffmpeg -y -i {out_avi} -c:v libx264 -pix_fmt yuv420p output/stereo_depth_video_demo.mp4 >/dev/null 2>&1")


if __name__ == '__main__':
    main()
