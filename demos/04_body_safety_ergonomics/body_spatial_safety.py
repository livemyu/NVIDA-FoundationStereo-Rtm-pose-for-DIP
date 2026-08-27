#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo 4: Human 3D Spatial Safety & Ergonomics Analysis (TensorRT + Stereo Depth)
==============================================================================
Features:
1. RTMPose-Body (COCO 17 Keypoints) + FoundationStereo 3D Metric Depth.
2. Real-time Ergonomics & Posture Metrics:
   - Spine / Torso Inclination Angle (calculates bending degree for lifting safety).
   - Arm Elevation & Waving Detection (detects emergency waving gesture).
   - Hand-to-Workplace Working Height Metric.
3. 3D Spatial Safety Zoning & Collision Warning:
   - Green Zone (Z > 2.0m): [SAFE / NORMAL OPERATION]
   - Yellow Zone (1.0m < Z <= 2.0m): [CAUTION / DECELERATE ROBOT]
   - Red Zone (Z <= 1.0m): [EMERGENCY / PROXIMITY HAZARD]
4. Dynamic 3-View Video Output with Ergonomics HUD.
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
from one_euro_filter import BodyPoseSmoother3D
from rtmpose_body_pipeline import (
    CUDA_FAST_PIPELINE_SOURCE, build_turbo_lut, RTMDetPersonTRT, RTMPoseBodyTRT,
    COCO_BODY_CONNECTIONS, get_limb_color
)

def analyze_body_ergonomics_and_safety(kpts, kpts_3d):
    """Computes spine angle, distance, posture and safety status for human body"""
    # Keypoint indices: 5: L_Shoulder, 6: R_Shoulder, 11: L_Hip, 12: R_Hip, 9: L_Wrist, 10: R_Wrist
    # 1. Torso Center & Distance
    if kpts[5][2] > 0.3 and kpts[6][2] > 0.3:
        sh_center_3d = (np.array(kpts_3d[5][:3]) + np.array(kpts_3d[6][:3])) * 0.5
    else:
        sh_center_3d = np.array(kpts_3d[0][:3])

    if kpts[11][2] > 0.3 and kpts[12][2] > 0.3:
        hip_center_3d = (np.array(kpts_3d[11][:3]) + np.array(kpts_3d[12][:3])) * 0.5
    else:
        hip_center_3d = sh_center_3d + np.array([0.0, 0.4, 0.0])

    dist_z = float(sh_center_3d[2])

    # 2. Spine Inclination Angle relative to vertical Y axis
    spine_vec = sh_center_3d - hip_center_3d
    norm_s = np.linalg.norm(spine_vec)
    if norm_s > 1e-4:
        # vertical unit vector [0, -1, 0] (upwards)
        cos_ang = np.dot(spine_vec, np.array([0, -1, 0])) / norm_s
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        spine_angle_deg = float(np.degrees(np.arccos(cos_ang)))
    else:
        spine_angle_deg = 0.0

    # 3. Posture Label
    if spine_angle_deg > 40.0:
        posture = "BENDING FORWARD (HEAVY LOAD HAZARD)"
        posture_status = "WARNING"
    elif spine_angle_deg > 20.0:
        posture = "SLIGHTLY INCLINED"
        posture_status = "NORMAL"
    else:
        posture = "UPRIGHT / STANDING"
        posture_status = "GOOD"

    # 4. Safety Zone
    if dist_z > 2.0:
        safety_zone = "GREEN (SAFE ZONE > 2.0m)"
        safety_color = (0, 255, 0)
    elif dist_z > 1.0:
        safety_zone = "YELLOW (CAUTION ZONE 1.0-2.0m)"
        safety_color = (0, 255, 255)
    elif dist_z > 0.1:
        safety_zone = "RED (COLLISION HAZARD < 1.0m)"
        safety_color = (0, 0, 255)
    else:
        safety_zone = "UNKNOWN"
        safety_color = (150, 150, 150)

    # 5. Check if hands are raised (wrists above shoulders)
    hands_raised = False
    if kpts[9][2] > 0.3 and kpts[5][2] > 0.3 and kpts[9][1] < kpts[5][1]:
        hands_raised = True
    if kpts[10][2] > 0.3 and kpts[6][2] > 0.3 and kpts[10][1] < kpts[6][1]:
        hands_raised = True

    return {
        "dist_z_m": dist_z,
        "sh_center_3d": sh_center_3d,
        "spine_angle_deg": spine_angle_deg,
        "posture": posture,
        "posture_status": posture_status,
        "safety_zone": safety_zone,
        "safety_color": safety_color,
        "hands_raised": hands_raised
    }

def render_body_safety_hud(img, kpts, kpts_3d, scale_x, scale_y, offset_x, p_idx, bbox):
    analysis = analyze_body_ergonomics_and_safety(kpts, kpts_3d)

    # 1. Render Skeleton
    for (p1, p2) in COCO_BODY_CONNECTIONS:
        x_a, y_a, c_a = kpts[p1]
        x_b, y_b, c_b = kpts[p2]
        if c_a > 0.25 and c_b > 0.25:
            color = get_limb_color(p1, p2)
            lx1, ly1 = int(x_a * scale_x) + offset_x, int(y_a * scale_y)
            lx2, ly2 = int(x_b * scale_x) + offset_x, int(y_b * scale_y)
            cv2.line(img, (lx1, ly1), (lx2, ly2), color, 3, cv2.LINE_AA)

    for i in range(17):
        kx, ky, kc = kpts[i]
        if kc > 0.25:
            lx, ly = int(kx * scale_x) + offset_x, int(ky * scale_y)
            cv2.circle(img, (lx, ly), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (lx, ly), 6, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Bounding Box with Safety Zone Color
    bx1, by1 = int(bbox[0] * scale_x) + offset_x, int(bbox[1] * scale_y)
    bx2, by2 = int(bbox[2] * scale_x) + offset_x, int(bbox[3] * scale_y)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), analysis["safety_color"], 2)

    # 3. Status HUD Card
    card_y = max(55, by1 - 10)
    card_w = 340
    cv2.rectangle(img, (bx1, card_y - 50), (bx1 + card_w, card_y), (0, 0, 0), -1)
    cv2.rectangle(img, (bx1, card_y - 50), (bx1 + card_w, card_y), analysis["safety_color"], 2)

    line1 = f"Person #{p_idx+1} | {analysis['safety_zone'].split('(')[0].strip()}"
    if analysis["hands_raised"]:
        line1 += " [WAVING]"
    cv2.putText(img, line1, (bx1 + 8, card_y - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, analysis["safety_color"], 2, cv2.LINE_AA)

    line2 = f"Dist: {analysis['dist_z_m']:.2f}m | Spine Angle: {analysis['spine_angle_deg']:.1f} deg"
    cv2.putText(img, line2, (bx1 + 8, card_y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    line3 = f"Posture: {analysis['posture']}"
    cv2.putText(img, line3, (bx1 + 8, card_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

def run_body_safety_pipeline(
    video_path,
    calib_path,
    rtmdet_engine_path,
    rtmpose_engine_path,
    depth_engine_path,
    output_video_path,
    depth_model='foundation',
    min_depth=0.15,
    max_depth=5.0,
    max_frames=0
):
    print(f"\n==================================================================")
    print(f"   [Demo 4] 启动人体 3D 空间安全警戒与工序姿态分析 Demo")
    print(f"   输入视频: {video_path}")
    print(f"   成片视频: {output_video_path}")
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

    person_det = RTMDetPersonTRT(rtmdet_engine_path)
    body_pose = RTMPoseBodyTRT(rtmpose_engine_path)
    smoother = BodyPoseSmoother3D()

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

            bboxes = person_det.detect(h_rect_left, conf_thr=0.35, stream=depth_engine.stream)
            poses = body_pose.estimate(h_rect_left, bboxes, stream=depth_engine.stream)
            timestamp_sec = float((processed_count - 1) / fps_in)
            if smoother is not None:
                poses = smoother.smooth(poses, timestamp_sec)

            scale_x = panel_w / orig_w
            scale_y = out_h / orig_h
            offset_x = 2 * panel_w

            for p_idx, p in enumerate(poses):
                bb = p['bbox']
                kpts = p['kpts']
                p_id = p.get('person_id', p_idx)
                
                kpts_3d = []
                for k_i in range(17):
                    px_u, px_v, conf = kpts[k_i]
                    ix = int(np.clip(px_u, 0, orig_w - 1))
                    iy = int(np.clip(px_v, 0, orig_h - 1))
                    
                    patch = depth_engine.h_final_depth[max(0, iy-3):min(orig_h, iy+4), max(0, ix-3):min(orig_w, ix+4)]
                    valid = patch[(patch >= min_depth) & (patch <= max_depth)]
                    z_m = float(np.median(valid)) if len(valid) > 0 else 0.0
                    
                    if z_m > 0.1:
                        x_m = float((ix - cx) * z_m / fx)
                        y_m = float((iy - cy) * z_m / fy)
                    else:
                        x_m, y_m, z_m = 0.0, 0.0, 0.0
                    kpts_3d.append([x_m, y_m, z_m, float(conf)])

                if smoother is not None:
                    kpts_3d = smoother.smooth_3d(p_id, kpts_3d, timestamp_sec)

                render_body_safety_hud(h_out_frame, kpts, kpts_3d, scale_x, scale_y, offset_x, p_idx, bb)

            cv2.line(h_out_frame, (panel_w, 0), (panel_w, out_h), (255, 255, 255), 2)
            cv2.line(h_out_frame, (2 * panel_w, 0), (2 * panel_w, out_h), (255, 255, 255), 2)

            cv2.rectangle(h_out_frame, (10, 8), (340, 42), (0, 0, 0), -1)
            cv2.putText(h_out_frame, "1. RAW (Left Rectified)", (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(h_out_frame, (panel_w + 10, 8), (panel_w + 480, 42), (0, 0, 0), -1)
            cv2.putText(h_out_frame, f"2. FoundationStereo Depth ({min_depth:.2f}-{max_depth:.1f}m)",
                        (panel_w + 18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(h_out_frame, (2 * panel_w + 10, 8), (2 * panel_w + 640, 42), (0, 0, 0), -1)
            cv2.putText(h_out_frame, f"3. [3D SAFETY & POSTURE] Ergonomics | F:{processed_count}",
                        (2 * panel_w + 18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

            write_queue.put(h_out_frame.copy())

            if processed_count % 100 == 0:
                elapsed = time.perf_counter() - t_start
                cur_fps = processed_count / elapsed
                print(f"[Body Safety HUD] Frame {processed_count}/{frames_to_process} ({cur_fps:.1f} FPS)")

    finally:
        cap.release()
        write_queue.put(None)
        write_queue.join()
        stop_event.set()
        writer_t.join()

        t_total = time.perf_counter() - t_start
        fps_avg = processed_count / t_total if t_total > 0 else 0
        print(f"\n==================================================================")
        print(f"   人体 3D 安全与工效学分析 Demo 完成: {processed_count} Frames ({fps_avg:.2f} FPS)")
        print(f"   输出成片: {output_video_path}")
        print(f"==================================================================")

def main():
    parser = argparse.ArgumentParser(description="Demo 4: Human 3D Spatial Safety & Ergonomics Analysis")
    parser.add_argument("--video", required=True, help="Input stereo video MP4")
    parser.add_argument("--calib", default="/home/elp/spatial_ai_trt_ws/calibration/camera_calib_results/camchain-datacalibration_bagskalibr_input.yaml")
    parser.add_argument("--person_det_engine", default="/home/elp/spatial_ai_trt_ws/models/rtmdet_person.engine")
    parser.add_argument("--body_pose_engine", default="/home/elp/spatial_ai_trt_ws/models/rtmpose_body.engine")
    parser.add_argument("--depth_engine", default="/home/elp/spatial_ai_trt_ws/foundationstereo_320x736_fp16.engine")
    parser.add_argument("--model", default="foundation", choices=["foundation", "ess"])
    parser.add_argument("--output_video", default="/home/elp/spatial_ai_trt_ws/output_results/01_rtmpose_hand/demo4_body_safety.mp4")
    parser.add_argument("--min_depth", type=float, default=0.15)
    parser.add_argument("--max_depth", type=float, default=5.0)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    run_body_safety_pipeline(
        video_path=args.video,
        calib_path=args.calib,
        rtmdet_engine_path=args.person_det_engine,
        rtmpose_engine_path=args.body_pose_engine,
        depth_engine_path=args.depth_engine,
        output_video_path=args.output_video,
        depth_model=args.model,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()
