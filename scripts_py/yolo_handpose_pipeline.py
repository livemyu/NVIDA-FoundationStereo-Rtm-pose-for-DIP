import os
import sys
import time
import argparse
import queue
import threading
import numpy as np
import cv2
import yaml

if '/usr/local/cuda/bin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from stereo_depth_pipeline_gpu import StereoCalibrator

# 21 手部关键点骨骼连接拓扑定义 (MediaPipe / YOLO-Pose 标准)
HAND_SKELETON_CONNECTIONS = [
    # 拇指
    (0, 1), (1, 2), (2, 3), (3, 4),
    # 食指
    (0, 5), (5, 6), (6, 7), (7, 8),
    # 中指
    (0, 9), (9, 10), (10, 11), (11, 12),
    # 无名指
    (0, 13), (13, 14), (14, 15), (15, 16),
    # 小指
    (0, 17), (17, 18), (18, 19), (19, 20),
    # 掌心连接
    (5, 9), (9, 13), (13, 17)
]

# 手指关键点颜色盘 (BGR)
HAND_COLORS = [
    (0, 255, 255),  # 0 腕关节 (黄)
    (255, 0, 0), (255, 50, 0), (255, 100, 0), (255, 150, 0),       # 拇指 (蓝)
    (0, 255, 0), (0, 255, 50), (0, 255, 100), (0, 255, 150),       # 食指 (绿)
    (0, 0, 255), (50, 0, 255), (100, 0, 255), (150, 0, 255),       # 中指 (红)
    (255, 0, 255), (255, 50, 255), (255, 100, 255), (255, 150, 255), # 无名指 (品红)
    (0, 165, 255), (50, 165, 255), (100, 165, 255), (150, 165, 255)  # 小指 (橙)
]

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

extern "C" __global__ void cuda_render_side_by_side(
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

    int half_w = out_w / 2;
    int sx = (x % half_w) * orig_w / half_w;
    int sy = y * orig_h / out_h;
    if (sx >= orig_w) sx = orig_w - 1;
    if (sy >= orig_h) sy = orig_h - 1;

    int out_idx = (y * out_w + x) * 3;

    if (x < half_w) {
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


class YOLOHandPoseEngine:
    def __init__(self, engine_path, input_size=(640, 640), conf_thresh=0.35):
        self.input_w, self.input_h = input_size
        self.conf_thresh = conf_thresh

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.h_input = cuda.pagelocked_empty((1, 3, self.input_h, self.input_w), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)

        out_shape = self.engine.get_tensor_shape(self.engine.get_tensor_name(1))
        self.out_shape = tuple(out_shape)
        self.h_output = cuda.pagelocked_empty(self.out_shape, dtype=np.float32)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        self.context.set_tensor_address(self.engine.get_tensor_name(0), int(self.d_input))
        self.context.set_tensor_address(self.engine.get_tensor_name(1), int(self.d_output))

    def infer(self, img_bgr):
        h, w = img_bgr.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        top = (self.input_h - nh) // 2
        bottom = self.input_h - nh - top
        left = (self.input_w - nw) // 2
        right = self.input_w - nw - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        np.copyto(self.h_input[0], chw)

        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        out = self.h_output[0] # [300, 69]
        detected_hands = []

        for det in out:
            score = float(det[4])
            if score < self.conf_thresh:
                continue

            x1 = (float(det[0]) - left) / scale
            y1 = (float(det[1]) - top) / scale
            x2 = (float(det[2]) - left) / scale
            y2 = (float(det[3]) - top) / scale

            kpts = []
            kpt_raw = det[6:69].reshape(21, 3)
            for kx, ky, kc in kpt_raw:
                orig_kx = (float(kx) - left) / scale
                orig_ky = (float(ky) - top) / scale
                kpts.append((orig_kx, orig_ky, float(kc)))

            detected_hands.append({
                'bbox': (x1, y1, x2, y2),
                'score': score,
                'kpts': kpts
            })

        return detected_hands

    def draw_hands(self, canvas, hands, depth_map=None, fx=1304.25, fy=1304.25, cx=959.75, cy=643.68):
        for hand_idx, hand in enumerate(hands):
            x1, y1, x2, y2 = [int(v) for v in hand['bbox']]
            score = hand['score']
            kpts = hand['kpts']

            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

            wrist_x, wrist_y, wrist_conf = kpts[0]
            depth_text = f"Hand #{hand_idx+1} ({score:.2f})"

            if depth_map is not None:
                ix, iy = int(np.clip(wrist_x, 0, depth_map.shape[1]-1)), int(np.clip(wrist_y, 0, depth_map.shape[0]-1))
                patch = depth_map[max(0, iy-2):min(depth_map.shape[0], iy+3), max(0, ix-2):min(depth_map.shape[1], ix+3)]
                valid_patch = patch[patch > 0]
                if len(valid_patch) > 0:
                    z_m = float(np.median(valid_patch))
                    x_m = (wrist_x - cx) * z_m / fx
                    y_m = (wrist_y - cy) * z_m / fy
                    depth_text += f" | Z: {z_m*100:.1f}cm (X:{x_m*100:.0f}, Y:{y_m*100:.0f})"

            cv2.putText(canvas, depth_text, (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            for (p1, p2) in HAND_SKELETON_CONNECTIONS:
                x_a, y_a, c_a = kpts[p1]
                x_b, y_b, c_b = kpts[p2]
                if c_a > 0.25 and c_b > 0.25:
                    cv2.line(canvas, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (255, 255, 255), 2, cv2.LINE_AA)

            for i, (kx, ky, kc) in enumerate(kpts):
                if kc > 0.25:
                    color = HAND_COLORS[i % len(HAND_COLORS)]
                    cv2.circle(canvas, (int(kx), int(ky)), 4, color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (int(kx), int(ky)), 5, (0, 0, 0), 1, cv2.LINE_AA)


def build_turbo_lut():
    return cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_TURBO).reshape(-1)

def run_fast_yolo_pipeline(video_path, calib_path, yolo_engine_path, output_path, depth_model='foundation', min_depth=0.15, max_depth=5.0, max_frames=0):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(yolo_engine_path):
        raise FileNotFoundError(f"YOLO engine not found: {yolo_engine_path}")

    # 自动定位校准文件
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
    print(f"   启动 YOLO 手部关键点姿态估计 + 3D空间深度全流程: [{depth_model}] + [handpose.engine]")
    print(f"   输入视频: {video_path}")
    print(f"   校准文件: {calib_path}")
    print(f"   模型文件: {yolo_engine_path}")
    print(f"   深度量程: {min_depth:.2f}m - {max_depth:.2f}m")
    print(f"   输出文件: {output_path}")
    print(f"==================================================================")

    # 载入标定
    calibrator = StereoCalibrator(calib_path)
    orig_w, orig_h = calibrator.w, calibrator.h
    fx, fy, cx, cy = calibrator.fx, calibrator.fy, calibrator.cx, calibrator.cy
    baseline = calibrator.baseline

    # 初始化 CUDA 核函数
    mod = SourceModule(CUDA_FAST_PIPELINE_SOURCE)
    remap_kernel = mod.get_function("cuda_remap_bilinear")
    render_kernel = mod.get_function("cuda_render_side_by_side")

    turbo_lut = build_turbo_lut()
    c_lut_ptr, _ = mod.get_global("c_turbo_lut")
    cuda.memcpy_htod(c_lut_ptr, turbo_lut)

    d_map1_x = cuda.mem_alloc(calibrator.map_l1.nbytes)
    d_map1_y = cuda.mem_alloc(calibrator.map_l2.nbytes)
    d_map2_x = cuda.mem_alloc(calibrator.map_r1.nbytes)
    d_map2_y = cuda.mem_alloc(calibrator.map_r2.nbytes)
    cuda.memcpy_htod(d_map1_x, np.ascontiguousarray(calibrator.map_l1, dtype=np.float32))
    cuda.memcpy_htod(d_map1_y, np.ascontiguousarray(calibrator.map_l2, dtype=np.float32))
    cuda.memcpy_htod(d_map2_x, np.ascontiguousarray(calibrator.map_r1, dtype=np.float32))
    cuda.memcpy_htod(d_map2_y, np.ascontiguousarray(calibrator.map_r2, dtype=np.float32))

    # 初始化深度引擎
    from stereo_depth_pipeline_gpu import GPUStereoDepthEngine
    depth_engine = GPUStereoDepthEngine(model_type=depth_model, orig_w=orig_w, orig_h=orig_h, fx=fx, baseline=baseline)

    # 初始化 YOLO 引擎
    yolo = YOLOHandPoseEngine(engine_path=yolo_engine_path, input_size=(640, 640), conf_thresh=0.35)

    # 显存缓冲区
    d_raw_left = cuda.mem_alloc(orig_w * orig_h * 3)
    d_raw_right = cuda.mem_alloc(orig_w * orig_h * 3)
    d_rect_left = cuda.mem_alloc(orig_w * orig_h * 3)
    d_rect_right = cuda.mem_alloc(orig_w * orig_h * 3)

    h_rect_left = cuda.pagelocked_empty((orig_h, orig_w, 3), dtype=np.uint8)

    out_w, out_h = 1920, 600
    d_out_frame = cuda.mem_alloc(out_w * out_h * 3)
    h_out_frame = cuda.pagelocked_empty((out_h, out_w, 3), dtype=np.uint8)

    remap_block = (32, 16, 1)
    remap_grid = ((orig_w + 31) // 32, (orig_h + 15) // 16, 1)
    render_block = (32, 16, 1)
    render_grid = ((out_w + 31) // 32, (out_h + 15) // 16, 1)

    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or np.isnan(fps_in):
        fps_in = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = total_frames if max_frames <= 0 else min(total_frames, max_frames)
    print(f"[Video Info] Total Frames: {total_frames}, FPS: {fps_in:.1f} | Will Process: {frames_to_process} Frames")

    # 异步写盘线程
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

            # 1. GPU CUDA 极速双目校正 (0.9 ms)
            cuda.memcpy_htod_async(d_raw_left, np.ascontiguousarray(l_raw), depth_engine.stream)
            cuda.memcpy_htod_async(d_raw_right, np.ascontiguousarray(r_raw), depth_engine.stream)

            remap_kernel(d_raw_left, d_rect_left, d_map1_x, d_map1_y, np.int32(orig_w), np.int32(orig_h),
                         block=remap_block, grid=remap_grid, stream=depth_engine.stream)
            remap_kernel(d_raw_right, d_rect_right, d_map2_x, d_map2_y, np.int32(orig_w), np.int32(orig_h),
                         block=remap_block, grid=remap_grid, stream=depth_engine.stream)

            # 下载校正后的左目图像供 YOLO 推理
            cuda.memcpy_dtoh_async(h_rect_left, d_rect_left, depth_engine.stream)

            # 2. ESS 深度推理 (19.6 ms)
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

            # 3. CUDA 渲染左目 + Turbo深度热力图 (0.2 ms)
            render_kernel(d_rect_left, depth_engine.d_final_depth, d_out_frame,
                          np.int32(orig_w), np.int32(orig_h),
                          np.int32(out_w), np.int32(out_h),
                          np.float32(min_depth), np.float32(max_depth),
                          block=render_block, grid=render_grid, stream=depth_engine.stream)

            # 下载深度与合成图
            cuda.memcpy_dtoh_async(depth_engine.h_final_depth, depth_engine.d_final_depth, depth_engine.stream)
            cuda.memcpy_dtoh_async(h_out_frame, d_out_frame, depth_engine.stream)
            depth_engine.stream.synchronize()

            # 4. YOLO 手势关键点推理 (4.8 ms)
            hands = yolo.infer(h_rect_left)

            # 5. 在左半屏绘制 21 手部关键点及 3D 深度测距
            # 将检测到的坐标缩放到左半屏 (960x600)
            if len(hands) > 0:
                scale_x = (out_w // 2) / orig_w
                scale_y = out_h / orig_h
                scaled_hands = []
                for h_item in hands:
                    bx1, by1, bx2, by2 = h_item['bbox']
                    s_bx1, s_by1 = bx1 * scale_x, by1 * scale_y
                    s_bx2, s_by2 = bx2 * scale_x, by2 * scale_y
                    s_kpts = [(kx * scale_x, ky * scale_y, kc) for (kx, ky, kc) in h_item['kpts']]
                    scaled_hands.append({
                        'bbox': (s_bx1, s_by1, s_bx2, s_by2),
                        'score': h_item['score'],
                        'kpts': s_kpts,
                        'orig_wrist': h_item['kpts'][0]
                    })

                # 在左半侧画布上绘制
                left_view = h_out_frame[:, :out_w//2]
                for h_idx, s_hand in enumerate(scaled_hands):
                    sx1, sy1, sx2, sy2 = [int(v) for v in s_hand['bbox']]
                    cv2.rectangle(left_view, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)

                    # 3D 测距 (从实际深度图采样)
                    ow_x, ow_y, _ = s_hand['orig_wrist']
                    ix = int(np.clip(ow_x, 0, orig_w - 1))
                    iy = int(np.clip(ow_y, 0, orig_h - 1))
                    patch = depth_engine.h_final_depth[max(0, iy-3):min(orig_h, iy+4), max(0, ix-3):min(orig_w, ix+4)]
                    valid_p = patch[patch > 0]
                    
                    depth_label = f"Hand #{h_idx+1} ({s_hand['score']:.2f})"
                    if len(valid_p) > 0:
                        z_val = float(np.median(valid_p))
                        x_val = (ow_x - cx) * z_val / fx
                        y_val = (ow_y - cy) * z_val / fy
                        depth_label += f" | Z:{z_val*100:.1f}cm (X:{x_val*100:.0f},Y:{y_val*100:.0f})"

                    cv2.putText(left_view, depth_label, (sx1, max(22, sy1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

                    for (p1, p2) in HAND_SKELETON_CONNECTIONS:
                        xa, ya, ca = s_hand['kpts'][p1]
                        xb, yb, cb = s_hand['kpts'][p2]
                        if ca > 0.25 and cb > 0.25:
                            cv2.line(left_view, (int(xa), int(ya)), (int(xb), int(yb)), (255, 255, 255), 2, cv2.LINE_AA)

                    for ki, (kx, ky, kc) in enumerate(s_hand['kpts']):
                        if kc > 0.25:
                            col = HAND_COLORS[ki % len(HAND_COLORS)]
                            cv2.circle(left_view, (int(kx), int(ky)), 5, col, -1, cv2.LINE_AA)
                            cv2.circle(left_view, (int(kx), int(ky)), 6, (0, 0, 0), 1, cv2.LINE_AA)

            # 异步写盘
            write_queue.put(h_out_frame.copy())

            if processed_count % 20 == 0 or processed_count == frames_to_process:
                elapsed = time.perf_counter() - t_start
                fps_curr = processed_count / elapsed
                print(f"  Frame [{processed_count:4d}/{frames_to_process:4d}] - Realtime Speed: {fps_curr:5.1f} FPS | Elapsed: {elapsed:5.1f}s | Hands: {len(hands)}", flush=True)

    finally:
        write_queue.put(None)
        writer_t.join()
        cap.release()

    total_time = time.perf_counter() - t_start
    print(f"\n==================================================================")
    print(f"   全 GPU 极速 YOLO 手部姿态检测与 3D 深度测量完成: [{output_path}]")
    print(f"   总处理帧数 : {processed_count} 帧 | 耗时: {total_time:.2f} 秒 ({processed_count/total_time:.2f} FPS)")
    print(f"==================================================================\n")


def main():
    parser = argparse.ArgumentParser(description='Fast YOLO Hand Pose & 3D Spatial Depth Pipeline')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default='foundation', choices=['foundation', 'ess', 'light_ess'], help='Stereo Depth Model (foundation or ess)')
    parser.add_argument('--min_depth', type=float, default=0.15, help='Min depth range in meters (default: 0.15)')
    parser.add_argument('--max_depth', type=float, default=5.0, help='Max depth range in meters (default: 5.0)')
    parser.add_argument('--engine', type=str, default=None, help='Path to YOLO handpose.engine')
    parser.add_argument('--calib', type=str, default=None, help='Path to calibration YAML')
    parser.add_argument('--output', type=str, default=None, help='Path to output MP4')
    parser.add_argument('--max_frames', type=int, default=0, help='Max frames to process')
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    calib_path = args.calib or os.path.join(curr_dir, 'calibration/camera_calib_results/camchain-datacalibration_bagskalibr_input.yaml')
    engine_path = args.engine or os.path.join(curr_dir, 'models/handpose.engine')

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        output_path = os.path.join(curr_dir, f'output_results/yolo_handpose_{base}.mp4')
    else:
        output_path = args.output

    run_fast_yolo_pipeline(
        video_path=args.video,
        calib_path=calib_path,
        yolo_engine_path=engine_path,
        output_path=output_path,
        depth_model=args.model,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()
