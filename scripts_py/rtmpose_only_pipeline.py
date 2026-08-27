#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Speed Pure RTMPose-M Hand Pose Estimation Pipeline (TensorRT FP16)
Features:
1. RTMDet-Nano Hand Detector (TensorRT FP16, ~2.0 ms)
2. RTMPose-M Hand 21-Keypoint SimCC Estimator (TensorRT FP16, ~3.2 ms)
3. Direct full-resolution left frame rendering (1920x1200) with anatomical bone colors
4. High-throughput multi-threaded asynchronous video writer (150+ FPS)
"""

import os
import sys
import time
import argparse
import queue
import threading
import numpy as np
import cv2

if '/usr/local/cuda/bin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
if '/usr/local/cuda/lib64' not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Standard 21 Hand Skeleton Connections
RTMPOSE_SKELETON_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm arch
    (5, 9), (9, 13), (13, 17)
]

# Anatomical Colors for Hand Skeleton Bones (BGR)
FINGER_COLORS = {
    'thumb': (0, 165, 255),    # Orange
    'index': (0, 255, 255),    # Yellow
    'middle': (0, 255, 0),     # Green
    'ring': (255, 255, 0),     # Cyan
    'pinky': (255, 0, 255),    # Magenta
    'palm': (220, 220, 220)    # White/Gray
}

def get_bone_color(p1, p2):
    if p1 in [1, 2, 3, 4] and p2 in [0, 1, 2, 3, 4]:
        return FINGER_COLORS['thumb']
    elif p1 in [5, 6, 7, 8] and p2 in [0, 5, 6, 7, 8]:
        return FINGER_COLORS['index']
    elif p1 in [9, 10, 11, 12] and p2 in [0, 9, 10, 11, 12]:
        return FINGER_COLORS['middle']
    elif p1 in [13, 14, 15, 16] and p2 in [0, 13, 14, 15, 16]:
        return FINGER_COLORS['ring']
    elif p1 in [17, 18, 19, 20] and p2 in [0, 17, 18, 19, 20]:
        return FINGER_COLORS['pinky']
    else:
        return FINGER_COLORS['palm']

def get_affine_transform(center, scale, output_size, rot=0):
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    
    rot_rad = np.pi * rot / 180
    src_dir = np.array([0, src_w * -0.5], np.float32)
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_dir = np.array([src_dir[0] * cs - src_dir[1] * sn, src_dir[0] * sn + src_dir[1] * cs])
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    
    src[0, :] = center
    src[1, :] = center + src_dir
    src[2, :] = src[1, :] + np.array([-src_dir[1], src_dir[0]])
    
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = dst[1, :] + np.array([-dst_dir[1], dst_dir[0]])
    
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))

def bbox_to_center_scale(bbox, padding=1.25):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    center = np.array([x1 + w * 0.5, y1 + h * 0.5], dtype=np.float32)
    size = max(w, h) * padding
    scale = np.array([size, size], dtype=np.float32)
    return center, scale

class OutputAllocator(trt.IOutputAllocator):
    def __init__(self):
        super().__init__()
        self.buffers = {}
        self.shapes = {}

    def reallocate_output(self, tensor_name, memory, size, alignment):
        ptr = cuda.mem_alloc(size)
        self.buffers[tensor_name] = ptr
        return int(ptr)

    def notify_shape(self, tensor_name, shape):
        self.shapes[tensor_name] = tuple(shape)

class RTMDetHandTRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_shape = (1, 3, 320, 320)
        self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape) * 4))
        self.det_mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.det_std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.allocator = OutputAllocator()
        self.context.set_output_allocator("dets", self.allocator)
        self.context.set_output_allocator("labels", self.allocator)

    def detect(self, img_bgr, conf_thr=0.25, stream=None):
        shape = img_bgr.shape[:2]
        r = min(320.0 / shape[0], 320.0 / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = 320 - new_unpad[0], 320 - new_unpad[1]
        
        im_resized = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
        im_padded = cv2.copyMakeBorder(im_resized, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        norm_img = (im_padded.astype(np.float32) - self.det_mean) / self.det_std
        inp = np.ascontiguousarray(np.transpose(norm_img, (2, 0, 1))[None, ...], dtype=np.float32)
        
        cuda.memcpy_htod_async(self.d_input, inp, stream)
        
        self.context.set_input_shape("input", self.input_shape)
        self.context.set_tensor_address("input", int(self.d_input))
        
        self.context.execute_async_v3(stream.handle if stream else 0)
        if stream: stream.synchronize()
        
        dets_shape = self.allocator.shapes.get("dets", (1, 0, 5))
        if dets_shape[1] == 0:
            return []
            
        h_dets = np.empty(dets_shape, dtype=np.float32)
        cuda.memcpy_dtoh(h_dets, self.allocator.buffers["dets"])
        
        dets = h_dets[0]
        valid_bboxes = []
        for det in dets:
            score = det[4]
            if score > conf_thr:
                x1 = det[0] / r
                y1 = det[1] / r
                x2 = det[2] / r
                y2 = det[3] / r
                valid_bboxes.append([x1, y1, x2, y2, score])
        return valid_bboxes

class RTMPoseHandTRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.pose_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
        self.pose_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)
        self.d_input = None
        self.d_simcc_x = None
        self.d_simcc_y = None
        self.curr_batch = 0

    def estimate(self, img_bgr, bboxes, stream=None):
        if len(bboxes) == 0:
            return []
        
        B = min(len(bboxes), 8)
        bboxes = bboxes[:B]
        
        batch_crops = []
        inv_trans_list = []
        
        for bbox in bboxes:
            center, scale = bbox_to_center_scale(bbox[:4], padding=1.25)
            trans = get_affine_transform(center, scale, (256, 256))
            inv_trans = cv2.invertAffineTransform(trans)
            
            crop = cv2.warpAffine(img_bgr, trans, (256, 256), flags=cv2.INTER_LINEAR)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            norm_crop = (crop_rgb.astype(np.float32) - self.pose_mean) / self.pose_std
            inp_crop = np.transpose(norm_crop, (2, 0, 1))
            
            batch_crops.append(inp_crop)
            inv_trans_list.append(inv_trans)
            
        inp = np.ascontiguousarray(np.stack(batch_crops, axis=0), dtype=np.float32)
        
        if self.curr_batch != B:
            self.d_input = cuda.mem_alloc(int(B * 3 * 256 * 256 * 4))
            self.d_simcc_x = cuda.mem_alloc(int(B * 21 * 512 * 4))
            self.d_simcc_y = cuda.mem_alloc(int(B * 21 * 512 * 4))
            self.curr_batch = B
            
        cuda.memcpy_htod_async(self.d_input, inp, stream)
        
        self.context.set_input_shape("input", (B, 3, 256, 256))
        self.context.set_tensor_address("input", int(self.d_input))
        self.context.set_tensor_address("simcc_x", int(self.d_simcc_x))
        self.context.set_tensor_address("simcc_y", int(self.d_simcc_y))
        
        self.context.execute_async_v3(stream.handle if stream else 0)
        
        h_simcc_x = np.empty((B, 21, 512), dtype=np.float32)
        h_simcc_y = np.empty((B, 21, 512), dtype=np.float32)
        
        cuda.memcpy_dtoh_async(h_simcc_x, self.d_simcc_x, stream)
        cuda.memcpy_dtoh_async(h_simcc_y, self.d_simcc_y, stream)
        if stream: stream.synchronize()
        
        results = []
        for i in range(B):
            sx = h_simcc_x[i]
            sy = h_simcc_y[i]
            
            loc_x = np.argmax(sx, axis=-1) / 2.0
            loc_y = np.argmax(sy, axis=-1) / 2.0
            
            scores = np.maximum(0.0, np.minimum(np.max(sx, axis=-1), np.max(sy, axis=-1)))
            
            kpts_256 = np.stack([loc_x, loc_y, np.ones(21)], axis=1)
            kpts_orig = np.dot(kpts_256, inv_trans_list[i].T)
            
            kpts = np.concatenate([kpts_orig, scores[:, None]], axis=1)
            results.append({
                'bbox': bboxes[i],
                'kpts': kpts
            })
        return results

def run_rtmpose_only_pipeline(
    video_path,
    rtmdet_engine_path,
    rtmpose_engine_path,
    output_path,
    conf_thr=0.25,
    max_frames=0
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(rtmdet_engine_path):
        raise FileNotFoundError(f"RTMDet engine not found: {rtmdet_engine_path}")
    if not os.path.exists(rtmpose_engine_path):
        raise FileNotFoundError(f"RTMPose engine not found: {rtmpose_engine_path}")

    print(f"\n==================================================================")
    print(f"   启动纯 RTMPose-M 两阶段超高速手势姿态估计")
    print(f"   手部检测: [{rtmdet_engine_path}]")
    print(f"   姿态估计: [{rtmpose_engine_path}]")
    print(f"   输入视频: {video_path}")
    print(f"   输出视频: {output_path}")
    print(f"==================================================================")

    stream = cuda.Stream()

    # Load Engines
    hand_det = RTMDetHandTRT(rtmdet_engine_path)
    hand_pose = RTMPoseHandTRT(rtmpose_engine_path)

    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or np.isnan(fps_in):
        fps_in = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = total_frames if max_frames <= 0 else min(total_frames, max_frames)
    
    out_w, out_h = 1920, 1200
    print(f"[Video Info] Total Frames: {total_frames}, FPS: {fps_in:.1f} | Output Resolution: {out_w}x{out_h}")

    # Async Disk Writer
    write_queue = queue.Queue(maxsize=32)
    stop_event = threading.Event()

    def async_writer_worker():
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (out_w, out_h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(output_path, fourcc, fps_in, (out_w, out_h))

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
                l_raw = frame[:, 160:2080]
            elif w == 3840:
                l_raw = frame[:, 0:1920]
            else:
                mid = w // 2
                l_raw = frame[:, :mid]

            if l_raw.shape[1] != out_w or l_raw.shape[0] != out_h:
                l_raw = cv2.resize(l_raw, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            canvas = l_raw.copy()

            # 1. Hand Detection & 21-Keypoint Pose
            bboxes = hand_det.detect(l_raw, conf_thr=conf_thr, stream=stream)
            poses = hand_pose.estimate(l_raw, bboxes, stream=stream)

            # 2. Render Hand BBoxes and 21 Keypoints
            for hand_idx, p in enumerate(poses):
                bb = p['bbox']
                kpts = p['kpts']
                
                bx1, by1, bx2, by2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (255, 180, 0), 2)
                
                tag_text = f"Hand #{hand_idx+1} ({bb[4]:.2f})"
                cv2.putText(canvas, tag_text, (bx1, max(28, by1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                # Draw Skeleton Lines
                for (p1, p2) in RTMPOSE_SKELETON_CONNECTIONS:
                    x_a, y_a, c_a = kpts[p1]
                    x_b, y_b, c_b = kpts[p2]
                    if c_a > 0.2 and c_b > 0.2:
                        color = get_bone_color(p1, p2)
                        cv2.line(canvas, (int(x_a), int(y_a)), (int(x_b), int(y_b)), color, 3, cv2.LINE_AA)

                # Draw 21 Keypoints
                for i in range(21):
                    kx, ky, kc = kpts[i]
                    if kc > 0.2:
                        cv2.circle(canvas, (int(kx), int(ky)), 5, (0, 0, 255), -1, cv2.LINE_AA)
                        cv2.circle(canvas, (int(kx), int(ky)), 6, (255, 255, 255), 1, cv2.LINE_AA)

            # Header info
            cv2.rectangle(canvas, (10, 10), (520, 50), (0, 0, 0), -1)
            cv2.putText(canvas, f"RTMPose-M Hand | Frame: {processed_count}/{frames_to_process}",
                        (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)

            write_queue.put(canvas)

            if processed_count % 200 == 0:
                elapsed = time.perf_counter() - t_start
                cur_fps = processed_count / elapsed
                print(f"[RTMPose Pure] Frame {processed_count}/{frames_to_process} ({cur_fps:.1f} FPS | {(1000/cur_fps):.1f} ms/frame)")

    except Exception as e:
        print(f"[Error in pipeline] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        write_queue.put(None)
        write_queue.join()
        stop_event.set()
        writer_t.join()

        t_total = time.perf_counter() - t_start
        fps_avg = processed_count / t_total if t_total > 0 else 0
        print(f"\n==================================================================")
        print(f"   纯 RTMPose-M 手势视频生成完成")
        print(f"   处理帧数: {processed_count} Frames")
        print(f"   总耗时: {t_total:.2f} 秒")
        print(f"   平均速度: {fps_avg:.2f} FPS (单帧耗时: {(1000/fps_avg):.2f} ms)")
        print(f"   输出成片: {output_path}")
        print(f"==================================================================")

def main():
    parser = argparse.ArgumentParser(description="Pure RTMPose Hand Pose Estimation Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video MP4")
    parser.add_argument("--rtmdet_engine", default="/home/elp/spatial_ai_trt_ws/models/rtmdet_hand.engine")
    parser.add_argument("--rtmpose_engine", default="/home/elp/spatial_ai_trt_ws/models/rtmpose_hand.engine")
    parser.add_argument("--output", required=True, help="Path to output video MP4")
    parser.add_argument("--conf_thr", type=float, default=0.25)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    run_rtmpose_only_pipeline(
        video_path=args.video,
        rtmdet_engine_path=args.rtmdet_engine,
        rtmpose_engine_path=args.rtmpose_engine,
        output_path=args.output,
        conf_thr=args.conf_thr,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()
