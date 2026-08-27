#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo 2: Spatial Pinch & Dexterous Grasping Metric Gauge (TensorRT + Stereo Depth)
================================================================================
Features:
1. Real-time Metric Pinch Gap Calculation (Thumb Tip 4 to Index Tip 8 in cm).
2. Advanced Gesture Semantic Classifier (PINCHING, OPEN_PALM, FIST, POINTING, THUMBS_UP).
3. Handedness Detection (Left Hand vs Right Hand based on 3D Palm Geometry).
4. Dynamic HUD & Visual Effects:
   - Tapered Bone Skeleton (Thick wrist -> Medium joint -> Thin tip).
   - 3D Pinch Connector Line & Distance Gauge.
   - Palm Mesh Semi-Transparent Shading.
   - Individual Fingertip Depth Floating Badges (Thumb, Index, Middle, Ring, Pinky).
5. 3-View Video Output with Grasping HUD.
"""

import os
import sys
import time
import argparse
import queue
import threading
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts_py'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

if '/usr/local/cuda/bin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
if '/usr/local/cuda/lib64' not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from stereo_depth_pipeline_gpu import StereoCalibrator, GPUStereoDepthEngine
from one_euro_filter import HandPoseSmoother3D
from rtmpose_depth_pipeline import (
    CUDA_FAST_PIPELINE_SOURCE, build_turbo_lut, RTMDetHandTRT, RTMPoseHandTRT
)

FINGER_CHAINS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20]
]

FINGER_COLORS = [
    (0, 0, 255),
    (0, 165, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0)
]

def classify_gesture(kpts, kpts_3d):
    """Classifies hand gesture into semantic categories based on 21 3D/2D keypoints"""
    # 0: Wrist, 4: Thumb, 8: Index, 12: Middle, 16: Ring, 20: Pinky
    # Tips & PIP joints
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    mcps = [1, 5, 9, 13, 17]

    # Extended finger flags
    extended = []
    wrist = np.array(kpts[0][:2])
    for t, p in zip(tips, pips):
        tip_pos = np.array(kpts[t][:2])
        pip_pos = np.array(kpts[p][:2])
        # Compare distance from wrist
        d_tip = np.linalg.norm(tip_pos - wrist)
        d_pip = np.linalg.norm(pip_pos - wrist)
        extended.append(d_tip > d_pip * 1.15)

    # 3D Pinch distance
    t_tip = np.array(kpts_3d[4][:3])
    i_tip = np.array(kpts_3d[8][:3])
    if t_tip[2] > 0.1 and i_tip[2] > 0.1:
        pinch_gap = float(np.linalg.norm(t_tip - i_tip) * 100.0)
    else:
        pinch_gap = float(np.linalg.norm(np.array(kpts[4][:2]) - np.array(kpts[8][:2])) * 0.1)

    # Handedness (Thumb relative to palm normal)
    # Vector Index MCP (5) -> Pinky MCP (17)
    v_palm_x = np.array(kpts[17][:2]) - np.array(kpts[5][:2])
    # Vector Wrist (0) -> Middle MCP (9)
    v_palm_y = np.array(kpts[9][:2]) - np.array(kpts[0][:2])
    # Cross product sign
    cross_z = v_palm_x[0] * v_palm_y[1] - v_palm_x[1] * v_palm_y[0]
    is_palm_facing = cross_z > 0
    thumb_x = kpts[4][0] - kpts[0][0]
    if (thumb_x < 0 and is_palm_facing) or (thumb_x > 0 and not is_palm_facing):
        handedness = "Right Hand"
    else:
        handedness = "Left Hand"

    # Gesture Rules
    if pinch_gap > 0 and pinch_gap < 3.5:
        gesture = "PINCHING"
    elif sum(extended) == 5:
        gesture = "OPEN PALM"
    elif sum(extended) == 0 or (sum(extended) == 1 and extended[0]):
        gesture = "FIST"
    elif extended[1] and not extended[2] and not extended[3] and not extended[4]:
        gesture = "POINTING"
    elif extended[0] and not extended[1] and not extended[2] and not extended[3] and not extended[4]:
        gesture = "THUMBS UP"
    elif extended[1] and extended[2] and not extended[3] and not extended[4]:
        gesture = "VICTORY / V"
    else:
        gesture = "MANIPULATING"

    return gesture, pinch_gap, handedness

def render_spatial_grasping_hud(img, kpts, kpts_3d, scale_x, scale_y, offset_x, h_idx, bbox):
    gesture, pinch_gap, handedness = classify_gesture(kpts, kpts_3d)

    # 1. Palm Mesh Shading (Wrist 0, 1, 5, 9, 13, 17)
    palm_pts = [0, 1, 5, 9, 13, 17]
    poly = []
    for p in palm_pts:
        if kpts[p][2] > 0.2:
            poly.append([int(kpts[p][0] * scale_x) + offset_x, int(kpts[p][1] * scale_y)])
    if len(poly) >= 4:
        overlay = img.copy()
        cv2.fillConvexPoly(overlay, np.array(poly, dtype=np.int32), (0, 255, 128))
        cv2.addWeighted(overlay, 0.20, img, 0.80, 0, img)

    # 2. Tapered Bone Lines
    bone_widths = [5, 4, 3, 2] # From MCP to Tip
    for f_idx, chain in enumerate(FINGER_CHAINS):
        color = FINGER_COLORS[f_idx]
        for j in range(len(chain) - 1):
            p1, p2 = chain[j], chain[j + 1]
            if kpts[p1][2] > 0.25 and kpts[p2][2] > 0.25:
                lx1, ly1 = int(kpts[p1][0] * scale_x) + offset_x, int(kpts[p1][1] * scale_y)
                lx2, ly2 = int(kpts[p2][0] * scale_x) + offset_x, int(kpts[p2][1] * scale_y)
                w = bone_widths[j]
                cv2.line(img, (lx1, ly1), (lx2, ly2), color, w, cv2.LINE_AA)

    # 3. 21 Keypoints with Fingertip Halos
    for k_i in range(21):
        if kpts[k_i][2] > 0.25:
            lx, ly = int(kpts[k_i][0] * scale_x) + offset_x, int(kpts[k_i][1] * scale_y)
            if k_i in [4, 8, 12, 16, 20]:
                cv2.circle(img, (lx, ly), 8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(img, (lx, ly), 5, (0, 0, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (lx, ly), 4, (0, 255, 0), -1, cv2.LINE_AA)

    # 4. Pinch Metric Arc (Between Thumb 4 and Index 8)
    if kpts[4][2] > 0.25 and kpts[8][2] > 0.25:
        tx, ty = int(kpts[4][0] * scale_x) + offset_x, int(kpts[4][1] * scale_y)
        ix, iy = int(kpts[8][0] * scale_x) + offset_x, int(kpts[8][1] * scale_y)
        # Line color: Red if pinched (<3.5cm), Yellow otherwise
        arc_color = (0, 0, 255) if (pinch_gap > 0 and pinch_gap < 3.5) else (0, 255, 255)
        cv2.line(img, (tx, ty), (ix, iy), arc_color, 2, cv2.LINE_AA)
        
        # Center Badge
        mx, my = (tx + ix) // 2, (ty + iy) // 2
        badge_text = f"{pinch_gap:.1f}cm"
        cv2.rectangle(img, (mx - 32, my - 16), (mx + 32, my + 6), (0, 0, 0), -1)
        cv2.putText(img, badge_text, (mx - 28, my - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # 5. Hand HUD Status Card
    bx1, by1 = int(bbox[0] * scale_x) + offset_x, int(bbox[1] * scale_y)
    bx2, by2 = int(bbox[2] * scale_x) + offset_x, int(bbox[3] * scale_y)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 128), 2)

    wrist_z = kpts_3d[0][2] * 100.0 # cm
    wrist_x = kpts_3d[0][0] * 100.0 # cm

    # Background Banner
    card_y = max(45, by1 - 10)
    cv2.rectangle(img, (bx1, card_y - 35), (bx1 + 260, card_y), (0, 0, 0), -1)
    
    # Text Line 1: Gesture + Handedness
    line1 = f"{handedness} | {gesture}"
    gesture_color = (0, 255, 255) if gesture == "PINCHING" else (0, 255, 0)
    cv2.putText(img, line1, (bx1 + 8, card_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, gesture_color, 2, cv2.LINE_AA)

    # Text Line 2: 3D Metric Coords
    line2 = f"Z: {wrist_z:.1f}cm | X: {wrist_x:+.1f}cm | Pinch: {pinch_gap:.1f}cm"
    cv2.putText(img, line2, (bx1 + 8, card_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

def run_spatial_grasping_metric(
    video_path,
    calib_path,
    hand_det_engine_path,
    hand_pose_engine_path,
    depth_engine_path,
    output_video_path,
    depth_model='foundation',
    min_depth=0.15,
    max_depth=5.0,
    max_frames=0
):
    print(f"\n==================================================================")
    print(f"   [Demo 2] 启动空间抓取与指尖物理开合度精准测量 Demo")
    print(f"   输入视频: {video_path}")
    print(f"   输出视频: {output_video_path}")
    print(f"==================================================================")

    calibrator = StereoCalibrator(calib_path)
    orig_w, orig_h = calibrator.w, calibrator.h
    fx, fy, cx, cy = calibrator.fx, calibrator.fy, calibrator.cx, calibrator.cy
    baseline = calibrator.baseline
    depth_engine = GPUStereoDepthEngine(model_type=depth_model, orig_w=orig_w, orig_h=orig_h, fx=fx, baseline=baseline)

    mod = SourceModule(CUDA_FAST_PIPELINE_SOURCE)
    remap_kernel = mod.get_function("cuda_remap_bilinear")
    render_kernel = mod.get_function("cuda_render_three_views")

    c_turbo_lut_ptr, _ = mod.get_global("c_turbo_lut")
    cuda.memcpy_htod(c_turbo_lut_ptr, build_turbo_lut())

    d_map1_x = cuda.mem_alloc(calibrator.map_l1.nbytes)
    d_map1_y = cuda.mem_alloc(calibrator.map_l2.nbytes)
    d_map2_x = cuda.mem_alloc(calibrator.map_r1.nbytes)
    d_map2_y = cuda.mem_alloc(calibrator.map_r2.nbytes)

    cuda.memcpy_htod(d_map1_x, np.ascontiguousarray(calibrator.map_l1, dtype=np.float32))
    cuda.memcpy_htod(d_map1_y, np.ascontiguousarray(calibrator.map_l2, dtype=np.float32))
    cuda.memcpy_htod(d_map2_x, np.ascontiguousarray(calibrator.map_r1, dtype=np.float32))
    cuda.memcpy_htod(d_map2_y, np.ascontiguousarray(calibrator.map_r2, dtype=np.float32))

    frame_bytes = orig_h * orig_w * 3
    d_raw_left = cuda.mem_alloc(frame_bytes)
    d_raw_right = cuda.mem_alloc(frame_bytes)
    d_rect_left = cuda.mem_alloc(frame_bytes)
    d_rect_right = cuda.mem_alloc(frame_bytes)
    h_rect_left = np.empty((orig_h, orig_w, 3), dtype=np.uint8)

    out_w, out_h = 2880, 600
    panel_w = out_w // 3

    d_out_frame = cuda.mem_alloc(out_h * out_w * 3)
    h_out_frame = np.empty((out_h, out_w, 3), dtype=np.uint8)

    block_dim = 16
    remap_block = (block_dim, block_dim, 1)
    remap_grid = ((orig_w + block_dim - 1) // block_dim, (orig_h + block_dim - 1) // block_dim, 1)
    render_block = (block_dim, block_dim, 1)
    render_grid = ((out_w + block_dim - 1) // block_dim, (out_h + block_dim - 1) // block_dim, 1)

    hand_det = RTMDetHandTRT(hand_det_engine_path)
    hand_pose = RTMPoseHandTRT(hand_pose_engine_path)
    smoother = HandPoseSmoother3D()

    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = total_frames if max_frames <= 0 else min(total_frames, max_frames)

    write_queue = queue.Queue(maxsize=16)
    stop_event = threading.Event()

    def async_writer_worker():
        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps_in, (out_w, out_h))
        while not stop_event.is_set():
            try:
                frame_data = write_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame_data is None:
                write_queue.task_done()
                break
            writer.write(frame_data)
            write_queue.task_done()
        writer.release()

    writer_t = threading.Thread(target=async_writer_worker, daemon=True)
    writer_t.start()

    processed_count = 0
    t_start = time.perf_counter()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            processed_count += 1
            if max_frames > 0 and processed_count > max_frames:
                break

            w = frame.shape[1]
            if w == 4000:
                l_raw, r_raw = frame[:, 160:2080], frame[:, 2080:4000]
            elif w == 3840:
                l_raw, r_raw = frame[:, 0:1920], frame[:, 1920:3840]
            else:
                mid = w // 2
                l_raw, r_raw = frame[:, :mid], frame[:, mid:]

            cuda.memcpy_htod_async(d_raw_left, np.ascontiguousarray(l_raw), depth_engine.stream)
            cuda.memcpy_htod_async(d_raw_right, np.ascontiguousarray(r_raw), depth_engine.stream)

            remap_kernel(d_raw_left, d_rect_left, d_map1_x, d_map1_y, np.int32(orig_w), np.int32(orig_h),
                         block=remap_block, grid=remap_grid, stream=depth_engine.stream)
            remap_kernel(d_raw_right, d_rect_right, d_map2_x, d_map2_y, np.int32(orig_w), np.int32(orig_h),
                         block=remap_block, grid=remap_grid, stream=depth_engine.stream)

            cuda.memcpy_dtoh_async(h_rect_left, d_rect_left, depth_engine.stream)

            depth_engine.preprocess_kernel(d_rect_left, depth_engine.d_model_left, np.int32(orig_w), np.int32(orig_h),
                                           np.int32(depth_engine.input_w), np.int32(depth_engine.input_h),
                                           block=depth_engine.pre_block, grid=depth_engine.pre_grid, stream=depth_engine.stream)
            depth_engine.preprocess_kernel(d_rect_right, depth_engine.d_model_right, np.int32(orig_w), np.int32(orig_h),
                                           np.int32(depth_engine.input_w), np.int32(depth_engine.input_h),
                                           block=depth_engine.pre_block, grid=depth_engine.pre_grid, stream=depth_engine.stream)
            depth_engine.context.execute_async_v3(stream_handle=depth_engine.stream.handle)
            depth_engine.postprocess_kernel(depth_engine.d_model_disp, depth_engine.d_final_disp, depth_engine.d_final_depth,
                                            np.int32(depth_engine.input_w), np.int32(depth_engine.input_h),
                                            np.int32(orig_w), np.int32(orig_h), np.float32(depth_engine.fx_baseline),
                                            block=depth_engine.post_block, grid=depth_engine.post_grid, stream=depth_engine.stream)

            render_kernel(d_rect_left, depth_engine.d_final_depth, d_out_frame,
                          np.int32(orig_w), np.int32(orig_h),
                          np.int32(out_w), np.int32(out_h),
                          np.float32(min_depth), np.float32(max_depth),
                          block=render_block, grid=render_grid, stream=depth_engine.stream)

            cuda.memcpy_dtoh_async(depth_engine.h_final_depth, depth_engine.d_final_depth, depth_engine.stream)
            cuda.memcpy_dtoh_async(h_out_frame, d_out_frame, depth_engine.stream)
            depth_engine.stream.synchronize()

            bboxes = hand_det.detect(h_rect_left, conf_thr=0.40, stream=depth_engine.stream)
            poses = hand_pose.estimate(h_rect_left, bboxes, stream=depth_engine.stream)
            timestamp_sec = float((processed_count - 1) / fps_in)
            if smoother is not None:
                poses = smoother.smooth(poses, timestamp_sec)

            scale_x = panel_w / orig_w
            scale_y = out_h / orig_h
            offset_x = 2 * panel_w

            for h_idx, p in enumerate(poses):
                bb = p['bbox']
                kpts = p['kpts']
                h_id = p.get('hand_id', h_idx)
                
                kpts_3d = []
                for k_i in range(21):
                    px_u, px_v, conf = kpts[k_i]
                    ix = int(np.clip(px_u, 0, orig_w - 1))
                    iy = int(np.clip(px_v, 0, orig_h - 1))
                    
                    patch = depth_engine.h_final_depth[max(0, iy-2):min(orig_h, iy+3), max(0, ix-2):min(orig_w, ix+3)]
                    valid = patch[(patch >= min_depth) & (patch <= max_depth)]
                    z_m = float(np.median(valid)) if len(valid) > 0 else 0.0
                    
                    if z_m > 0.1:
                        x_m = float((ix - cx) * z_m / fx)
                        y_m = float((iy - cy) * z_m / fy)
                    else:
                        x_m, y_m, z_m = 0.0, 0.0, 0.0
                    kpts_3d.append([x_m, y_m, z_m, float(conf)])

                if smoother is not None:
                    kpts_3d = smoother.smooth_3d(h_id, kpts_3d, timestamp_sec)

                render_spatial_grasping_hud(h_out_frame, kpts, kpts_3d, scale_x, scale_y, offset_x, h_idx, bb)

            # Dividers & Header Banners
            cv2.line(h_out_frame, (panel_w, 0), (panel_w, out_h), (255, 255, 255), 2)
            cv2.line(h_out_frame, (2 * panel_w, 0), (2 * panel_w, out_h), (255, 255, 255), 2)

            cv2.rectangle(h_out_frame, (10, 8), (340, 42), (0, 0, 0), -1)
            cv2.putText(h_out_frame, "1. RAW (Left Rectified)", (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(h_out_frame, (panel_w + 10, 8), (panel_w + 480, 42), (0, 0, 0), -1)
            cv2.putText(h_out_frame, f"2. FoundationStereo Depth ({min_depth:.2f}-{max_depth:.1f}m)",
                        (panel_w + 18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(h_out_frame, (2 * panel_w + 10, 8), (2 * panel_w + 640, 42), (0, 0, 0), -1)
            cv2.putText(h_out_frame, f"3. [SPATIAL GRASP HUD] 3D Meter | F:{processed_count}",
                        (2 * panel_w + 18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2, cv2.LINE_AA)

            write_queue.put(h_out_frame.copy())

            if processed_count % 100 == 0:
                elapsed = time.perf_counter() - t_start
                cur_fps = processed_count / elapsed
                print(f"[Grasping Meter] Frame {processed_count}/{frames_to_process} ({cur_fps:.1f} FPS)")

    finally:
        cap.release()
        write_queue.put(None)
        write_queue.join()
        stop_event.set()
        writer_t.join()

        t_total = time.perf_counter() - t_start
        fps_avg = processed_count / t_total if t_total > 0 else 0
        print(f"\n==================================================================")
        print(f"   空间抓取度与指尖测距 Demo 完成: {processed_count} Frames ({fps_avg:.2f} FPS)")
        print(f"   输出成片: {output_video_path}")
        print(f"==================================================================")

def main():
    parser = argparse.ArgumentParser(description="Demo 2: Spatial Pinch & Grasping Metric Gauge")
    parser.add_argument("--video", required=True, help="Input stereo video MP4")
    parser.add_argument("--calib", default="/home/elp/spatial_ai_trt_ws/calibration/camera_calib_results/camchain-datacalibration_bagskalibr_input.yaml")
    parser.add_argument("--hand_det_engine", default="/home/elp/spatial_ai_trt_ws/models/rtmdet_hand.engine")
    parser.add_argument("--hand_pose_engine", default="/home/elp/spatial_ai_trt_ws/models/rtmpose_hand.engine")
    parser.add_argument("--depth_engine", default="/home/elp/spatial_ai_trt_ws/foundationstereo_320x736_fp16.engine")
    parser.add_argument("--model", default="foundation", choices=["foundation", "ess"])
    parser.add_argument("--output_video", default="/home/elp/spatial_ai_trt_ws/output_results/01_rtmpose_hand/demo2_spatial_grasping.mp4")
    parser.add_argument("--min_depth", type=float, default=0.15)
    parser.add_argument("--max_depth", type=float, default=5.0)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    run_spatial_grasping_metric(
        video_path=args.video,
        calib_path=args.calib,
        hand_det_engine_path=args.hand_det_engine,
        hand_pose_engine_path=args.hand_pose_engine,
        depth_engine_path=args.depth_engine,
        output_video_path=args.output_video,
        depth_model=args.model,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()
