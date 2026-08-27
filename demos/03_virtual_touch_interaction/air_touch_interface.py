#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo 3: Spatial Virtual Touch & Gesture Air-Interface (TensorRT + Stereo Depth)
==============================================================================
Features:
1. Defines a 3D Virtual Touch Interaction Plane at Z = 35 cm in front of the camera.
2. 3 Virtual Floating 3D Buttons:
   - Button 1: [RECORD / GRASP]
   - Button 2: [SNAPSHOT / CAPTURE]
   - Button 3: [EMERGENCY STOP]
3. Tracks Index Fingertip (8) 3D Spatial Position (X, Y, Z).
4. Proximity-aware Visual Feedback:
   - Distance Bar: Shows remaining distance to touch plane (e.g. `dZ = +4.2 cm`).
   - Touch Penetration Event: When Z <= 35 cm inside a button box, triggers Touch Event
     with animated expanding ripple rings and button color flash.
5. 3-View Video Output with 3D Spatial Touch UI.
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

class VirtualTouchInterface:
    def __init__(self, panel_w, out_h, offset_x, touch_plane_z=0.35):
        self.panel_w = panel_w
        self.out_h = out_h
        self.offset_x = offset_x
        self.touch_plane_z = touch_plane_z # 35 cm
        
        # Define 3 Virtual 3D Buttons (x1, y1, x2, y2, label) in Panel 3 coordinate
        btn_w, btn_h = 240, 60
        start_y = 100
        spacing = 80
        
        self.buttons = [
            {"id": 1, "name": "1. RECORD DEMO", "rect": (offset_x + 30, start_y, offset_x + 30 + btn_w, start_y + btn_h), "color": (0, 165, 255), "active_timer": 0},
            {"id": 2, "name": "2. SNAPSHOT 3D", "rect": (offset_x + 30, start_y + spacing, offset_x + 30 + btn_w, start_y + spacing + btn_h), "color": (255, 255, 0), "active_timer": 0},
            {"id": 3, "name": "3. EMERGENCY STOP", "rect": (offset_x + 30, start_y + 2 * spacing, offset_x + 30 + btn_w, start_y + 2 * spacing + btn_h), "color": (0, 0, 255), "active_timer": 0}
        ]
        self.ripples = []

    def update_and_render(self, img, index_tip_3d, index_tip_2d_panel):
        # Decay button active timers
        for b in self.buttons:
            if b["active_timer"] > 0:
                b["active_timer"] -= 1

        tip_x_m, tip_y_m, tip_z_m = index_tip_3d
        px, py = index_tip_2d_panel

        is_touched = False
        touched_btn_name = ""

        # Check Touch Event if Z <= touch_plane_z
        if tip_z_m > 0.1 and tip_z_m <= self.touch_plane_z:
            for b in self.buttons:
                x1, y1, x2, y2 = b["rect"]
                if x1 <= px <= x2 and y1 <= py <= y2:
                    b["active_timer"] = 15 # active for 15 frames
                    is_touched = True
                    touched_btn_name = b["name"]
                    self.ripples.append({"x": px, "y": py, "r": 5, "max_r": 45, "life": 12})
                    break

        # Render Virtual Buttons
        for b in self.buttons:
            x1, y1, x2, y2 = b["rect"]
            is_active = b["active_timer"] > 0
            
            # Button Body
            bg_color = (0, 255, 0) if is_active else (40, 40, 40)
            border_color = (255, 255, 255) if is_active else b["color"]
            
            # Translucent background
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
            cv2.addWeighted(overlay, 0.45 if is_active else 0.25, img, 0.55 if is_active else 0.75, 0, img)
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 3 if is_active else 2)

            # Button Text
            text = f"{b['name']} [CLICK!]" if is_active else b["name"]
            text_color = (0, 0, 0) if is_active else (255, 255, 255)
            cv2.putText(img, text, (x1 + 12, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)

        # Render Ripples
        new_ripples = []
        for r in self.ripples:
            if r["life"] > 0:
                cv2.circle(img, (r["x"], r["y"]), int(r["r"]), (0, 255, 255), 2, cv2.LINE_AA)
                r["r"] += (r["max_r"] - r["r"]) * 0.35 + 2
                r["life"] -= 1
                new_ripples.append(r)
        self.ripples = new_ripples

        # Render Interaction Status Card
        card_x, card_y = self.offset_x + 30, self.out_h - 100
        cv2.rectangle(img, (card_x, card_y), (card_x + 400, card_y + 80), (0, 0, 0), -1)
        cv2.rectangle(img, (card_x, card_y), (card_x + 400, card_y + 80), (0, 255, 255), 1)

        plane_cm = self.touch_plane_z * 100.0
        cur_z_cm = tip_z_m * 100.0
        delta_z = (tip_z_m - self.touch_plane_z) * 100.0 if tip_z_m > 0.1 else 999.0

        cv2.putText(img, f"Air Touch Plane: Z = {plane_cm:.0f} cm", (card_x + 12, card_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

        if tip_z_m > 0.1:
            status_text = f"Index Tip: Z = {cur_z_cm:.1f} cm (dZ: {delta_z:+.1f} cm)"
            color_z = (0, 255, 0) if delta_z <= 0 else (0, 255, 255)
            cv2.putText(img, status_text, (card_x + 12, card_y + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color_z, 2, cv2.LINE_AA)
            if is_touched:
                cv2.putText(img, f"EVENT: {touched_btn_name} TRIGGERED!", (card_x + 12, card_y + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "Index Tip: No Hand Detected", (card_x + 12, card_y + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1, cv2.LINE_AA)

def run_virtual_touch_demo(
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
    print(f"   [Demo 3] 启动空间无接触虚拟悬浮触控交互 Demo")
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

    hand_det = RTMDetHandTRT(hand_det_engine_path)
    hand_pose = RTMPoseHandTRT(hand_pose_engine_path)
    smoother = HandPoseSmoother3D()

    touch_ui = VirtualTouchInterface(panel_w=panel_w, out_h=out_h, offset_x=2*panel_w, touch_plane_z=0.35)

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

            index_tip_3d = [0.0, 0.0, 0.0]
            index_tip_2d_panel = (0, 0)

            for h_idx, p in enumerate(poses):
                bb = p['bbox']
                kpts = p['kpts']
                h_id = p.get('hand_id', h_idx)
                
                # Draw skeleton
                for f_idx, chain in enumerate(FINGER_CHAINS):
                    color = FINGER_COLORS[f_idx]
                    for j in range(len(chain) - 1):
                        p1, p2 = chain[j], chain[j + 1]
                        if kpts[p1][2] > 0.25 and kpts[p2][2] > 0.25:
                            lx1, ly1 = int(kpts[p1][0] * scale_x) + offset_x, int(kpts[p1][1] * scale_y)
                            lx2, ly2 = int(kpts[p2][0] * scale_x) + offset_x, int(kpts[p2][1] * scale_y)
                            cv2.line(h_out_frame, (lx1, ly1), (lx2, ly2), color, 3, cv2.LINE_AA)

                for k_i in range(21):
                    if kpts[k_i][2] > 0.25:
                        lx, ly = int(kpts[k_i][0] * scale_x) + offset_x, int(kpts[k_i][1] * scale_y)
                        cv2.circle(h_out_frame, (lx, ly), 5 if k_i == 8 else 3, (0, 0, 255) if k_i == 8 else (0, 255, 0), -1)

                # Index tip 3D sampling (8)
                if kpts[8][2] > 0.25:
                    ix = int(np.clip(kpts[8][0], 0, orig_w - 1))
                    iy = int(np.clip(kpts[8][1], 0, orig_h - 1))
                    patch = depth_engine.h_final_depth[max(0, iy-2):min(orig_h, iy+3), max(0, ix-2):min(orig_w, ix+3)]
                    valid = patch[(patch >= min_depth) & (patch <= max_depth)]
                    z_m = float(np.median(valid)) if len(valid) > 0 else 0.0
                    if z_m > 0.1:
                        raw_tip_3d = [(ix - cx) * z_m / fx, (iy - cy) * z_m / fy, z_m]
                        if smoother is not None:
                            # Smooth 3D
                            kpts_3d_dummy = [[0,0,0,0]] * 21
                            kpts_3d_dummy[8] = [raw_tip_3d[0], raw_tip_3d[1], raw_tip_3d[2], 1.0]
                            kpts_3d_smooth = smoother.smooth_3d(h_id, kpts_3d_dummy, timestamp_sec)
                            index_tip_3d = kpts_3d_smooth[8][:3]
                        else:
                            index_tip_3d = raw_tip_3d
                        index_tip_2d_panel = (int(kpts[8][0] * scale_x) + offset_x, int(kpts[8][1] * scale_y))

            # Render Virtual Touch UI
            touch_ui.update_and_render(h_out_frame, index_tip_3d, index_tip_2d_panel)

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
            cv2.putText(h_out_frame, f"3. [3D AIR TOUCH UI] Interactive | F:{processed_count}",
                        (2 * panel_w + 18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2, cv2.LINE_AA)

            write_queue.put(h_out_frame.copy())

            if processed_count % 100 == 0:
                elapsed = time.perf_counter() - t_start
                cur_fps = processed_count / elapsed
                print(f"[Air Touch] Frame {processed_count}/{frames_to_process} ({cur_fps:.1f} FPS)")

    finally:
        cap.release()
        write_queue.put(None)
        write_queue.join()
        stop_event.set()
        writer_t.join()

        t_total = time.perf_counter() - t_start
        fps_avg = processed_count / t_total if t_total > 0 else 0
        print(f"\n==================================================================")
        print(f"   空间无接触虚拟触控 Demo 完成: {processed_count} Frames ({fps_avg:.2f} FPS)")
        print(f"   输出成片: {output_video_path}")
        print(f"==================================================================")

def main():
    parser = argparse.ArgumentParser(description="Demo 3: Spatial Virtual Touch & Air Interface")
    parser.add_argument("--video", required=True, help="Input stereo video MP4")
    parser.add_argument("--calib", default="/home/elp/spatial_ai_trt_ws/calibration/camera_calib_results/camchain-datacalibration_bagskalibr_input.yaml")
    parser.add_argument("--hand_det_engine", default="/home/elp/spatial_ai_trt_ws/models/rtmdet_hand.engine")
    parser.add_argument("--hand_pose_engine", default="/home/elp/spatial_ai_trt_ws/models/rtmpose_hand.engine")
    parser.add_argument("--depth_engine", default="/home/elp/spatial_ai_trt_ws/foundationstereo_320x736_fp16.engine")
    parser.add_argument("--model", default="foundation", choices=["foundation", "ess"])
    parser.add_argument("--output_video", default="/home/elp/spatial_ai_trt_ws/output_results/01_rtmpose_hand/demo3_air_touch.mp4")
    parser.add_argument("--min_depth", type=float, default=0.15)
    parser.add_argument("--max_depth", type=float, default=5.0)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    run_virtual_touch_demo(
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
