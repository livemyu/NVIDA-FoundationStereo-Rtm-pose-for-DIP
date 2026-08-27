#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTMPose Two-Stage High-Precision Body Pose (COCO 17 Keypoints) & Stereo Depth Pipeline (TensorRT)
Features:
1. Fast GPU CUDA Stereo Remap (StereoCalibrator)
2. FoundationStereo / ESS Stereo Depth TensorRT Engine (3D depth estimation)
3. RTMDet-Nano Person Detector (TensorRT FP16, 416x416 or 320x320)
4. RTMPose-M Body 17-Keypoint SimCC Estimator (TensorRT FP16, 256x192)
5. 3D Spatial Skeleton Coordinate Calculation (Head, Shoulders, Wrists, Torso in meters/cm)
6. 3-View Video Output: [Raw Left | Stereo Depth | 17-Keypoint 3D Pose]
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
from pycuda.compiler import SourceModule

from stereo_depth_pipeline_gpu import StereoCalibrator, GPUStereoDepthEngine
from one_euro_filter import BodyPoseSmoother3D

# COCO 17 Body Skeleton Connections
COCO_BODY_CONNECTIONS = [
    # Head & Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Upper Body / Arms
    (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Lower Body / Legs
    (11, 13), (13, 15), (12, 14), (14, 16)
]

BODY_LIMB_COLORS = {
    'head': (0, 255, 255),    # Yellow
    'l_arm': (0, 165, 255),   # Orange
    'r_arm': (255, 255, 0),   # Cyan
    'torso': (0, 255, 0),     # Green
    'l_leg': (255, 0, 255),   # Magenta
    'r_leg': (255, 0, 0)      # Blue
}

def get_limb_color(p1, p2):
    if p1 in [0, 1, 2, 3, 4] and p2 in [0, 1, 2, 3, 4]:
        return BODY_LIMB_COLORS['head']
    elif (p1 in [5, 7, 9] and p2 in [5, 7, 9]):
        return BODY_LIMB_COLORS['l_arm']
    elif (p1 in [6, 8, 10] and p2 in [6, 8, 10]):
        return BODY_LIMB_COLORS['r_arm']
    elif (p1 in [5, 6, 11, 12] and p2 in [5, 6, 11, 12]):
        return BODY_LIMB_COLORS['torso']
    elif (p1 in [11, 13, 15] and p2 in [11, 13, 15]):
        return BODY_LIMB_COLORS['l_leg']
    else:
        return BODY_LIMB_COLORS['r_leg']

CUDA_FAST_PIPELINE_SOURCE = r"""
__constant__ unsigned char c_turbo_lut[256 * 3];

extern "C" __global__ void cuda_remap_bilinear(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    const float* __restrict__ map_x,
    const float* __restrict__ map_y,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float src_x = map_x[idx];
    float src_y = map_y[idx];

    if (src_x < 0.0f || src_x >= (float)(width - 1) || src_y < 0.0f || src_y >= (float)(height - 1)) {
        dst[idx * 3 + 0] = 0;
        dst[idx * 3 + 1] = 0;
        dst[idx * 3 + 2] = 0;
        return;
    }

    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float dx = src_x - (float)x0;
    float dy = src_y - (float)y0;

    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    int idx00 = (y0 * width + x0) * 3;
    int idx01 = (y0 * width + x1) * 3;
    int idx10 = (y1 * width + x0) * 3;
    int idx11 = (y1 * width + x1) * 3;

    int dst_idx = idx * 3;
    #pragma unroll
    for (int c = 0; c < 3; c++) {
        float val = w00 * (float)src[idx00 + c] +
                    w01 * (float)src[idx01 + c] +
                    w10 * (float)src[idx10 + c] +
                    w11 * (float)src[idx11 + c];
        dst[dst_idx + c] = (unsigned char)fminf(fmaxf(val, 0.0f), 255.0f);
    }
}

extern "C" __global__ void cuda_render_three_views(
    const unsigned char* __restrict__ left_rect,
    const float* __restrict__ depth_metric,
    unsigned char* __restrict__ out_frame,
    int orig_w, int orig_h,
    int out_w, int out_h,
    float min_d, float max_d
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    int panel_w = out_w / 3;
    int panel_idx = x / panel_w;
    int px = x % panel_w;

    int sx = px * orig_w / panel_w;
    int sy = y * orig_h / out_h;
    if (sx >= orig_w) sx = orig_w - 1;
    if (sy >= orig_h) sy = orig_h - 1;

    int out_idx = (y * out_w + x) * 3;

    if (panel_idx == 0 || panel_idx == 2) {
        int left_idx = (sy * orig_w + sx) * 3;
        out_frame[out_idx + 0] = left_rect[left_idx + 0];
        out_frame[out_idx + 1] = left_rect[left_idx + 1];
        out_frame[out_idx + 2] = left_rect[left_idx + 2];
    } else {
        float z = depth_metric[sy * orig_w + sx];

        if (z <= 0.0f || isnan(z) || z < min_d) {
            out_frame[out_idx + 0] = 0;
            out_frame[out_idx + 1] = 0;
            out_frame[out_idx + 2] = 0;
        } else if (z >= max_d) {
            out_frame[out_idx + 0] = c_turbo_lut[0 * 3 + 0];
            out_frame[out_idx + 1] = c_turbo_lut[0 * 3 + 1];
            out_frame[out_idx + 2] = c_turbo_lut[0 * 3 + 2];
        } else {
            float norm = 1.0f - (z - min_d) / (max_d - min_d);
            int lut_i = (int)(norm * 255.0f);
            if (lut_i < 0) lut_i = 0;
            if (lut_i > 255) lut_i = 255;
            out_frame[out_idx + 0] = c_turbo_lut[lut_i * 3 + 0];
            out_frame[out_idx + 1] = c_turbo_lut[lut_i * 3 + 1];
            out_frame[out_idx + 2] = c_turbo_lut[lut_i * 3 + 2];
        }
    }
}
"""

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

class RTMDetPersonTRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_shape = (1, 3, 416, 416)
        self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape) * 4))
        self.det_mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.det_std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.allocator = OutputAllocator()
        self.context.set_output_allocator("dets", self.allocator)
        self.context.set_output_allocator("labels", self.allocator)

    def detect(self, img_bgr, conf_thr=0.35, stream=None):
        shape = img_bgr.shape[:2]
        r = min(416.0 / shape[0], 416.0 / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = 416 - new_unpad[0], 416 - new_unpad[1]
        
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

class RTMPoseBodyTRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.pose_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
        self.pose_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)
        
        self.d_input = cuda.mem_alloc(int(1 * 3 * 256 * 192 * 4))
        self.d_simcc_x = cuda.mem_alloc(int(1 * 17 * 384 * 4))
        self.d_simcc_y = cuda.mem_alloc(int(1 * 17 * 512 * 4))
        self.h_simcc_x = np.empty((1, 17, 384), dtype=np.float32)
        self.h_simcc_y = np.empty((1, 17, 512), dtype=np.float32)

    def estimate(self, img_bgr, bboxes, stream=None):
        if len(bboxes) == 0:
            return []
        
        results = []
        for bbox in bboxes:
            center, scale = bbox_to_center_scale(bbox[:4], padding=1.25)
            trans = get_affine_transform(center, scale, (192, 256))
            inv_trans = cv2.invertAffineTransform(trans)
            
            crop = cv2.warpAffine(img_bgr, trans, (192, 256), flags=cv2.INTER_LINEAR)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            norm_crop = (crop_rgb.astype(np.float32) - self.pose_mean) / self.pose_std
            inp = np.ascontiguousarray(np.transpose(norm_crop, (2, 0, 1))[None, ...], dtype=np.float32)
            
            cuda.memcpy_htod_async(self.d_input, inp, stream)
            
            self.context.set_input_shape("input", (1, 3, 256, 192))
            self.context.set_tensor_address("input", int(self.d_input))
            self.context.set_tensor_address("simcc_x", int(self.d_simcc_x))
            self.context.set_tensor_address("simcc_y", int(self.d_simcc_y))
            
            self.context.execute_async_v3(stream.handle if stream else 0)
            
            cuda.memcpy_dtoh_async(self.h_simcc_x, self.d_simcc_x, stream)
            cuda.memcpy_dtoh_async(self.h_simcc_y, self.d_simcc_y, stream)
            if stream: stream.synchronize()
            
            sx = self.h_simcc_x[0]
            sy = self.h_simcc_y[0]
            
            loc_x = np.argmax(sx, axis=-1) / 2.0
            loc_y = np.argmax(sy, axis=-1) / 2.0
            
            scores = np.maximum(0.0, np.minimum(np.max(sx, axis=-1), np.max(sy, axis=-1)))
            
            kpts_256 = np.stack([loc_x, loc_y, np.ones(17)], axis=1)
            kpts_orig = np.dot(kpts_256, inv_trans.T)
            
            kpts = np.concatenate([kpts_orig, scores[:, None]], axis=1)
            results.append({
                'bbox': bbox,
                'kpts': kpts
            })
        return results

def build_turbo_lut():
    return cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_TURBO).reshape(-1)

def run_rtmpose_body_pipeline(
    video_path,
    calib_path,
    rtmdet_engine_path,
    rtmpose_engine_path,
    depth_engine_path,
    output_path,
    depth_model='foundation',
    min_depth=0.15,
    max_depth=5.0,
    max_frames=0
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(rtmdet_engine_path):
        raise FileNotFoundError(f"RTMDet engine not found: {rtmdet_engine_path}")
    if not os.path.exists(rtmpose_engine_path):
        raise FileNotFoundError(f"RTMPose engine not found: {rtmpose_engine_path}")
    if not os.path.exists(depth_engine_path):
        raise FileNotFoundError(f"Depth engine not found: {depth_engine_path}")

    # Locate calibration file
    if not os.path.exists(calib_path):
        possible_paths = [
            calib_path,
            '/home/elp/spatial_ai_trt_ws/calibration/camera_calib_results/camchain-datacalibration_bagskalibr_input.yaml',
            '/home/elp/picture_resize_recording_NVIDA/calibration/camera_calib_results/camchain-datacalibration_bagskalibr_input.yaml'
        ]
        for p in possible_paths:
            if os.path.exists(p):
                calib_path = p
                break

    print(f"\n==================================================================")
    print(f"   启动 RTMPose-Body 人体姿态估计 + 3D空间深度全流程")
    print(f"   深度模型: [{depth_model}] ({depth_engine_path})")
    print(f"   人体检测: [{rtmdet_engine_path}]")
    print(f"   姿态估计: [{rtmpose_engine_path}] (COCO 17 关键点)")
    print(f"   输入视频: {video_path}")
    print(f"   输出视频: {output_path}")
    print(f"==================================================================")

    # 1. Load Calibration & Depth Engine
    calibrator = StereoCalibrator(calib_path)
    orig_w, orig_h = calibrator.w, calibrator.h
    fx, fy, cx, cy = calibrator.fx, calibrator.fy, calibrator.cx, calibrator.cy
    baseline = calibrator.baseline
    depth_engine = GPUStereoDepthEngine(model_type=depth_model, orig_w=orig_w, orig_h=orig_h, fx=fx, baseline=baseline)

    # 2. CUDA Kernels & Memory
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

    # Output Resolution: 3 views (2880 x 600) -> 960 x 600 each
    out_w, out_h = 2880, 600
    panel_w = out_w // 3

    d_out_frame = cuda.mem_alloc(out_h * out_w * 3)
    h_out_frame = np.empty((out_h, out_w, 3), dtype=np.uint8)

    block_dim = 16
    remap_block = (block_dim, block_dim, 1)
    remap_grid = ((orig_w + block_dim - 1) // block_dim, (orig_h + block_dim - 1) // block_dim, 1)
    render_block = (block_dim, block_dim, 1)
    render_grid = ((out_w + block_dim - 1) // block_dim, (out_h + block_dim - 1) // block_dim, 1)

    # 3. Load RTMPose-Body Engines
    person_det = RTMDetPersonTRT(rtmdet_engine_path)
    body_pose = RTMPoseBodyTRT(rtmpose_engine_path)
    smoother = BodyPoseSmoother3D()

    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or np.isnan(fps_in):
        fps_in = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = total_frames if max_frames <= 0 else min(total_frames, max_frames)
    print(f"[Video Info] Total Frames: {total_frames}, FPS: {fps_in:.1f} | Will Process: {frames_to_process} Frames")

    # Async Disk Writer
    write_queue = queue.Queue(maxsize=16)
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
                r_raw = frame[:, 2080:4000]
            elif w == 3840:
                l_raw = frame[:, 0:1920]
                r_raw = frame[:, 1920:3840]
            else:
                mid = w // 2
                l_raw = frame[:, :mid]
                r_raw = frame[:, mid:]

            # 1. GPU CUDA Stereo Rectification
            cuda.memcpy_htod_async(d_raw_left, np.ascontiguousarray(l_raw), depth_engine.stream)
            cuda.memcpy_htod_async(d_raw_right, np.ascontiguousarray(r_raw), depth_engine.stream)

            remap_kernel(d_raw_left, d_rect_left, d_map1_x, d_map1_y, np.int32(orig_w), np.int32(orig_h),
                         block=remap_block, grid=remap_grid, stream=depth_engine.stream)
            remap_kernel(d_raw_right, d_rect_right, d_map2_x, d_map2_y, np.int32(orig_w), np.int32(orig_h),
                         block=remap_block, grid=remap_grid, stream=depth_engine.stream)

            cuda.memcpy_dtoh_async(h_rect_left, d_rect_left, depth_engine.stream)

            # 2. Stereo Depth Inference
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

            # 3. CUDA Three-Views Render
            render_kernel(d_rect_left, depth_engine.d_final_depth, d_out_frame,
                          np.int32(orig_w), np.int32(orig_h),
                          np.int32(out_w), np.int32(out_h),
                          np.float32(min_depth), np.float32(max_depth),
                          block=render_block, grid=render_grid, stream=depth_engine.stream)

            cuda.memcpy_dtoh_async(depth_engine.h_final_depth, depth_engine.d_final_depth, depth_engine.stream)
            cuda.memcpy_dtoh_async(h_out_frame, d_out_frame, depth_engine.stream)
            depth_engine.stream.synchronize()

            # 4. Two-Stage RTMPose Person Detection & 17-Keypoint Body Pose
            bboxes = person_det.detect(h_rect_left, conf_thr=0.35, stream=depth_engine.stream)
            poses = body_pose.estimate(h_rect_left, bboxes, stream=depth_engine.stream)
            timestamp_sec = float((processed_count - 1) / fps_in)
            if smoother is not None:
                poses = smoother.smooth(poses, timestamp_sec)

            # 5. Overlay Skeleton & 3D Spatial Positions on Panel 2
            scale_x = panel_w / orig_w
            scale_y = out_h / orig_h
            offset_x = 2 * panel_w

            for p_idx, p in enumerate(poses):
                bb = p['bbox']
                kpts = p['kpts']
                
                # Scaled bbox
                bx1, by1 = int(bb[0] * scale_x) + offset_x, int(bb[1] * scale_y)
                bx2, by2 = int(bb[2] * scale_x) + offset_x, int(bb[3] * scale_y)
                cv2.rectangle(h_out_frame, (bx1, by1), (bx2, by2), (0, 255, 128), 2)

                # Sample 3D depth at chest center (between shoulders 5 & 6)
                if kpts[5][2] > 0.3 and kpts[6][2] > 0.3:
                    cx_px = int((kpts[5][0] + kpts[6][0]) * 0.5)
                    cy_px = int((kpts[5][1] + kpts[6][1]) * 0.5)
                else:
                    cx_px = int(np.clip(kpts[0][0], 0, orig_w - 1))
                    cy_px = int(np.clip(kpts[0][1], 0, orig_h - 1))

                ix, iy = int(np.clip(cx_px, 0, orig_w - 1)), int(np.clip(cy_px, 0, orig_h - 1))
                patch = depth_engine.h_final_depth[max(0, iy-5):min(orig_h, iy+6), max(0, ix-5):min(orig_w, ix+6)]
                valid_patch = patch[(patch >= min_depth) & (patch <= max_depth)]
                
                body_z = float(np.median(valid_patch)) if len(valid_patch) > 0 else 0.0
                if body_z > 0.1:
                    body_xm = (ix - cx) * body_z / fx
                    body_ym = (iy - cy) * body_z / fy
                    tag_text = f"Person #{p_idx+1} | Z:{body_z:.2f}m (X:{body_xm:.2f}m)"
                else:
                    tag_text = f"Person #{p_idx+1} ({bb[4]:.2f})"

                cv2.putText(h_out_frame, tag_text, (bx1, max(25, by1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

                # Draw Skeleton Lines
                for (p1, p2) in COCO_BODY_CONNECTIONS:
                    x_a, y_a, c_a = kpts[p1]
                    x_b, y_b, c_b = kpts[p2]
                    if c_a > 0.25 and c_b > 0.25:
                        color = get_limb_color(p1, p2)
                        lx1, ly1 = int(x_a * scale_x) + offset_x, int(y_a * scale_y)
                        lx2, ly2 = int(x_b * scale_x) + offset_x, int(y_b * scale_y)
                        cv2.line(h_out_frame, (lx1, ly1), (lx2, ly2), color, 3, cv2.LINE_AA)

                # Draw 17 Keypoints
                for i in range(17):
                    kx, ky, kc = kpts[i]
                    if kc > 0.25:
                        lx, ly = int(kx * scale_x) + offset_x, int(ky * scale_y)
                        cv2.circle(h_out_frame, (lx, ly), 5, (0, 0, 255), -1, cv2.LINE_AA)
                        cv2.circle(h_out_frame, (lx, ly), 6, (255, 255, 255), 1, cv2.LINE_AA)

            # Panel Titles & Dividers
            cv2.line(h_out_frame, (panel_w, 0), (panel_w, out_h), (255, 255, 255), 2)
            cv2.line(h_out_frame, (2 * panel_w, 0), (2 * panel_w, out_h), (255, 255, 255), 2)

            cv2.rectangle(h_out_frame, (10, 8), (340, 42), (0, 0, 0), -1)
            cv2.putText(h_out_frame, "1. RAW (Left Rectified)", (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(h_out_frame, (panel_w + 10, 8), (panel_w + 480, 42), (0, 0, 0), -1)
            cv2.putText(h_out_frame, f"2. FoundationStereo Depth ({min_depth:.2f}-{max_depth:.1f}m)",
                        (panel_w + 18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(h_out_frame, (2 * panel_w + 10, 8), (2 * panel_w + 480, 42), (0, 0, 0), -1)
            cv2.putText(h_out_frame, f"3. RTMPose-Body 17-Kpts 3D | F:{processed_count}",
                        (2 * panel_w + 18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2, cv2.LINE_AA)

            write_queue.put(h_out_frame.copy())

            if processed_count % 100 == 0:
                elapsed = time.perf_counter() - t_start
                cur_fps = processed_count / elapsed
                print(f"[RTMPose-Body 3-View] Frame {processed_count}/{frames_to_process} ({cur_fps:.1f} FPS | {(1000/cur_fps):.1f} ms/frame)")

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
        print(f"   RTMPose-Body 人体姿态与双目深度处理完成")
        print(f"   处理帧数: {processed_count} Frames")
        print(f"   总耗时: {t_total:.2f} 秒")
        print(f"   平均速度: {fps_avg:.2f} FPS (单帧耗时: {(1000/fps_avg):.2f} ms)")
        print(f"   输出成片: {output_path}")
        print(f"==================================================================")

def main():
    parser = argparse.ArgumentParser(description="RTMPose-Body + Stereo Depth TensorRT Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video MP4")
    parser.add_argument("--calib", required=True, help="Path to camera calibration YAML")
    parser.add_argument("--person_det_engine", default="/home/elp/spatial_ai_trt_ws/models/rtmdet_person.engine")
    parser.add_argument("--body_pose_engine", default="/home/elp/spatial_ai_trt_ws/models/rtmpose_body.engine")
    parser.add_argument("--depth_engine", default="/home/elp/spatial_ai_trt_ws/foundationstereo_320x736_fp16.engine")
    parser.add_argument("--model", default="foundation", choices=["foundation", "ess"])
    parser.add_argument("--output", required=True, help="Path to output video MP4")
    parser.add_argument("--min_depth", type=float, default=0.15)
    parser.add_argument("--max_depth", type=float, default=5.0)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    run_rtmpose_body_pipeline(
        video_path=args.video,
        calib_path=args.calib,
        rtmdet_engine_path=args.person_det_engine,
        rtmpose_engine_path=args.body_pose_engine,
        depth_engine_path=args.depth_engine,
        depth_model=args.model,
        output_path=args.output,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()
