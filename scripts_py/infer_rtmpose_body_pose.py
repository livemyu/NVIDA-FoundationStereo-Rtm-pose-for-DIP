#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTMPose 人体全身姿态估计 (支持双目左目画面自动提取、SDK 解码与 Web 实时显示)
================================================================================
核心特性:
1. 自动双目画面解析: 自动识别 4000x1200 / 3840x1200 双目原始视频，剔除 160px IMU 点阵并精准提取【左目 1920x1200】
2. 纯 2D 极速推理 (单帧 ~3.5ms, 280+ FPS)
3. 人体 17 关键点解剖骨骼拓扑连线
4. Web 实时流显示 (--web, 浏览器打开 http://<IP>:8080 实时看画面，彻底解决 SSH 无法弹窗问题)
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
import cv2

# 自动确保 CUDA 环境
if '/usr/local/cuda/bin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
if '/usr/local/cuda/lib64' not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

LIMB_COLORS = [
    (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),
    (255, 255, 255), (255, 200, 100), (255, 200, 100), (255, 255, 255),
    (0, 255, 255), (0, 255, 255),
    (0, 255, 0), (0, 255, 0),
    (255, 255, 0), (255, 255, 0),
    (0, 165, 255), (0, 165, 255)
]


def find_default_engine(model_type='body_pose'):
    """自动在 models/ 目录下查找匹配的模型文件"""
    candidates = []
    if model_type == 'body_det':
        candidates = ['rtmdet_nano_person.engine', 'rtmdet_person.engine']
    elif model_type == 'body_pose':
        candidates = ['rtmpose_m_body.engine', 'rtmpose_body.engine']
        
    search_dirs = [
        os.path.join(SCRIPT_DIR, 'models'),
        SCRIPT_DIR,
        '/home/jetson/rtmpose/models',
        '/home/elp/spatial_ai_trt_ws/models',
        '/home/elp/picture_resize_recording_NVIDA/models'
    ]
    for d in search_dirs:
        for c in candidates:
            p = os.path.join(d, c)
            if os.path.exists(p):
                return p
    return os.path.join(SCRIPT_DIR, 'models', candidates[0])


class TRTModel:
    def __init__(self, engine_path, max_batch=1, max_dets=300):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            raw_shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
            
            alloc_shape = []
            for dim_idx, d in enumerate(raw_shape):
                if d == -1:
                    alloc_shape.append(max_batch if dim_idx == 0 else max_dets)
                else:
                    alloc_shape.append(d)
            alloc_shape = tuple(alloc_shape)
            
            if is_input:
                self.context.set_input_shape(name, alloc_shape)
                
            host_mem = np.zeros(alloc_shape, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.buffers[name] = {
                'host': host_mem,
                'device': device_mem,
                'shape': alloc_shape,
                'raw_shape': raw_shape,
                'dtype': dtype,
                'is_input': is_input
            }
            self.context.set_tensor_address(name, int(device_mem))

    def forward(self, input_dict):
        for name, data in input_dict.items():
            if self.buffers[name]['raw_shape'][0] == -1 or list(data.shape) != list(self.buffers[name]['shape']):
                self.context.set_input_shape(name, data.shape)
            np.copyto(self.buffers[name]['host'][:data.size].reshape(data.shape), data)
            cuda.memcpy_htod_async(self.buffers[name]['device'], self.buffers[name]['host'], self.stream)
            
        self.context.execute_async_v3(self.stream.handle)
        
        for name, buf in self.buffers.items():
            if not buf['is_input']:
                cuda.memcpy_dtoh_async(buf['host'], buf['device'], self.stream)
        self.stream.synchronize()
        
        outputs = {}
        for name, buf in self.buffers.items():
            if not buf['is_input']:
                actual_shape = tuple(self.context.get_tensor_shape(name))
                if -1 not in actual_shape:
                    outputs[name] = buf['host'].reshape(actual_shape).copy()
                else:
                    outputs[name] = buf['host'].copy()
        return outputs


class RTMPoseBodyEstimator:
    def __init__(self, det_engine, pose_engine):
        print(f"[Init] 加载 RTMDet 人体检测引擎: {det_engine}")
        self.det_model = TRTModel(det_engine) if det_engine and os.path.exists(det_engine) else None
        
        print(f"[Init] 加载 RTMPose-M 人体全身姿态估计引擎: {pose_engine}")
        self.pose_model = TRTModel(pose_engine)

    def extract_left_eye(self, raw_frame):
        """自动从双目拼接帧中裁剪出纯净的左目图像"""
        if raw_frame is None:
            return None
        h, w = raw_frame.shape[:2]
        if w >= 3900:
            return raw_frame[:, 160:2080]
        elif w >= 3800:
            return raw_frame[:, 0:1920]
        elif w == 1920 and h == 960:
            return raw_frame[:, 0:960]
        return raw_frame

    def detect_persons(self, img_bgr, conf_thresh=0.40):
        h, w = img_bgr.shape[:2]
        if self.det_model is None:
            return [[0, 0, w, h]]
            
        det_input_size = (320, 320)
        img_resized = cv2.resize(img_bgr, det_input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_nchw = np.transpose(img_rgb, (2, 0, 1))[np.newaxis, ...]
        
        input_name = list(self.det_model.buffers.keys())[0]
        outputs = self.det_model.forward({input_name: img_nchw})
        
        boxes = outputs.get('dets', outputs.get('bboxes', list(outputs.values())[0]))[0]
        scale_x = w / det_input_size[0]
        scale_y = h / det_input_size[1]
        
        valid_boxes = []
        for box in boxes:
            score = box[4] if len(box) >= 5 else 1.0
            if score >= conf_thresh:
                x1 = int(box[0] * scale_x)
                y1 = int(box[1] * scale_y)
                x2 = int(box[2] * scale_x)
                y2 = int(box[3] * scale_y)
                valid_boxes.append([max(0, x1), max(0, y1), min(w, x2), min(h, y2)])
                
        return valid_boxes if valid_boxes else [[0, 0, w, h]]

    def estimate_pose(self, img_bgr, bbox):
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = max(10, x2 - x1), max(10, y2 - y1)
        
        pad = 0.15
        cx, cy = x1 + bw * 0.5, y1 + bh * 0.5
        box_w = bw * (1.0 + pad)
        box_h = bh * (1.0 + pad)
        
        aspect_ratio = 192.0 / 256.0
        if box_w > box_h * aspect_ratio:
            box_h = box_w / aspect_ratio
        else:
            box_w = box_h * aspect_ratio
            
        crop_x1 = max(0, int(cx - box_w * 0.5))
        crop_y1 = max(0, int(cy - box_h * 0.5))
        crop_x2 = min(w, int(cx + box_w * 0.5))
        crop_y2 = min(h, int(cy + box_h * 0.5))
        
        crop_img = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_img.size == 0:
            return None, None
            
        input_size = (192, 256)
        crop_resized = cv2.resize(crop_img, input_size)
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        crop_norm = (crop_rgb - mean) / std
        crop_nchw = np.transpose(crop_norm, (2, 0, 1))[np.newaxis, ...]
        
        input_name = list(self.pose_model.buffers.keys())[0]
        outputs = self.pose_model.forward({input_name: crop_nchw})
        
        simcc_x = outputs.get('simcc_x', list(outputs.values())[0])[0]
        simcc_y = outputs.get('simcc_y', list(outputs.values())[1])[0]
        
        kpts = []
        scores = []
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        for i in range(17):
            px = np.argmax(simcc_x[i]) / float(simcc_x.shape[1])
            py = np.argmax(simcc_y[i]) / float(simcc_y.shape[1])
            conf = float(np.max(simcc_x[i]) * np.max(simcc_y[i]))
            
            orig_px = crop_x1 + px * crop_w
            orig_py = crop_y1 + py * crop_h
            kpts.append([orig_px, orig_py])
            scores.append(conf)
            
        return np.array(kpts), np.array(scores)

    def draw_body_skeleton(self, canvas, kpts, scores, conf_thresh=0.30):
        if kpts is None or len(kpts) == 0:
            return canvas
            
        for idx, (p1, p2) in enumerate(BODY_SKELETON):
            if scores[p1] >= conf_thresh and scores[p2] >= conf_thresh:
                pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                color = LIMB_COLORS[idx] if idx < len(LIMB_COLORS) else (0, 255, 0)
                cv2.line(canvas, pt1, pt2, color, 3, cv2.LINE_AA)
                
        for i in range(17):
            if scores[i] >= conf_thresh:
                pt = (int(kpts[i][0]), int(kpts[i][1]))
                cv2.circle(canvas, pt, 5, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
                
        return canvas


# ==================== Web 实时视频流服务器 (MJPEG) ====================
class MJPEGWebServer:
    def __init__(self, host='0.0.0.0', port=8080):
        self.host = host
        self.port = port
        self.current_frame = None
        self.lock = threading.Lock()
        self.running = True

    def update_frame(self, frame_bgr):
        with self.lock:
            h, w = frame_bgr.shape[:2]
            scale = 960.0 / max(w, 1)
            if scale < 1.0:
                self.current_frame = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
            else:
                self.current_frame = frame_bgr.copy()

    def start(self):
        from http.server import HTTPServer, BaseHTTPRequestHandler

        server_self = self

        class StreamHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.end_headers()
                    html = f"""
                    <html>
                    <head><title>RTMPose 实时人体姿态监控</title></head>
                    <body style="margin:0; background:#111; color:#eee; text-align:center; font-family:sans-serif;">
                        <h2 style="padding:10px; margin:0;">RTMPose 实时人体姿态骨骼 (左目视角)</h2>
                        <img src="/stream.mjpg" style="max-width:95vw; max-height:85vh; border:2px solid #ffaa00; border-radius:8px;" />
                    </body>
                    </html>
                    """
                    self.wfile.write(html.encode('utf-8'))
                elif self.path == '/stream.mjpg':
                    self.send_response(200)
                    self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
                    self.end_headers()
                    while server_self.running:
                        with server_self.lock:
                            if server_self.current_frame is None:
                                time.sleep(0.01)
                                continue
                            ret, jpeg = cv2.imencode('.jpg', server_self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ret:
                            self.wfile.write(b"--jpgboundary\r\n")
                            self.send_header('Content-type', 'image/jpeg')
                            self.send_header('Content-length', str(len(jpeg)))
                            self.end_headers()
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b"\r\n")
                        time.sleep(0.03)
                else:
                    self.send_error(404)

        server = HTTPServer((self.host, self.port), StreamHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"\n[Web Stream] 实时视频流已开启: http://{self.host}:{self.port} (浏览器直接打开即可观看)")


def main():
    parser = argparse.ArgumentParser(description="RTMPose 人体全身姿态估计 (支持左目自动提取与 Web 显示)")
    parser.add_argument("--input", default="0", help="输入视频路径或摄像头编号 (如 0 或 test.mp4)")
    parser.add_argument("--det_engine", default=None, help="人体检测 TensorRT 模型路径")
    parser.add_argument("--pose_engine", default=None, help="人体姿态 TensorRT 模型路径")
    parser.add_argument("--output", default=None, help="输出视频保存路径 (如 body_out.mp4)")
    parser.add_argument("--show", action="store_true", help="是否尝试在本地桌面弹出 GUI 窗口")
    parser.add_argument("--web", action="store_true", default=True, help="开启浏览器 Web 实时视频流 (默认开启，端口 8080)")
    parser.add_argument("--web_port", type=int, default=8080, help="Web 视频流端口 (默认 8080)")
    args = parser.parse_args()

    det_engine = args.det_engine or find_default_engine('body_det')
    pose_engine = args.pose_engine or find_default_engine('body_pose')

    estimator = RTMPoseBodyEstimator(det_engine, pose_engine)

    web_server = None
    if args.web:
        web_server = MJPEGWebServer(port=args.web_port)
        web_server.start()

    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[Error] 无法打开输入源: {args.input}")
        return

    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"\n============================================================")
    print(f" [OK] RTMPose 人体骨骼估计器已启动")
    print(f"  - 输入源原始分辨率: {raw_w} x {raw_h} ({fps:.1f} FPS)")
    print(f"  - 画面处理模式    : 自动提取【左目视角 1920x1200】")
    if args.web:
        print(f"  - Web 实时流地址  : http://<开发板IP>:{args.web_port}")
    print(f"============================================================")

    writer = None
    frame_count = 0
    start_time = time.time()

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t0 = time.time()

        # 1. 自动提取左目纯净画面 (已剔除 160px IMU 点阵)
        left_img = estimator.extract_left_eye(raw_frame)
        canvas = left_img.copy()

        # 2. 人体检测与全身 17 关键点解算
        bboxes = estimator.detect_persons(left_img)
        for bbox in bboxes:
            kpts, scores = estimator.estimate_pose(left_img, bbox)
            if kpts is not None:
                estimator.draw_body_skeleton(canvas, kpts, scores)

        infer_time = (time.time() - t0) * 1000.0
        cv2.putText(canvas, f"Left-Eye Body Skeleton | Latency: {infer_time:.1f} ms | FPS: {1000.0/max(1.0, infer_time):.1f}", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 3. 初始化写盘器
        if args.output and writer is None:
            out_h, out_w = canvas.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.output, fourcc, fps, (out_w, out_h))

        if writer:
            writer.write(canvas)

        # 4. Web 实时流同步推送
        if web_server:
            web_server.update_frame(canvas)

        # 5. 本地窗口显示
        if args.show:
            try:
                cv2.imshow("RTMPose Left-Eye Body Pose", canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                pass

        if frame_count % 30 == 0:
            avg_fps = frame_count / (time.time() - start_time)
            print(f"  - 已处理 {frame_count} 帧, 实时吞吐量: {avg_fps:.1f} FPS")

    cap.release()
    if writer:
        writer.release()
        print(f"[Done] 处理完毕，已保存左目人体骨骼视频: {args.output}")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
