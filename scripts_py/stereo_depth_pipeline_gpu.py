import os
import sys
import time
import argparse
import numpy as np
import cv2
import yaml

# 自动确保 nvcc 在 PATH 中
if '/usr/local/cuda/bin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

CUDA_KERNELS = """
extern "C" {

__global__ void preprocess_bgr_to_rgb_planar(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    float scale_x = (float)src_w / (float)dst_w;
    float scale_y = (float)src_h / (float)dst_h;

    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;

    src_x = fmaxf(0.0f, fminf(src_x, (float)(src_w - 1)));
    src_y = fmaxf(0.0f, fminf(src_y, (float)(src_h - 1)));

    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float dx = src_x - x0;
    float dy = src_y - y0;

    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    int idx00 = (y0 * src_w + x0) * 3;
    int idx01 = (y0 * src_w + x1) * 3;
    int idx10 = (y1 * src_w + x0) * 3;
    int idx11 = (y1 * src_w + x1) * 3;

    // BGR in src -> RGB in dst
    float b = w00 * src[idx00 + 0] + w01 * src[idx01 + 0] + w10 * src[idx10 + 0] + w11 * src[idx11 + 0];
    float g = w00 * src[idx00 + 1] + w01 * src[idx01 + 1] + w10 * src[idx10 + 1] + w11 * src[idx11 + 1];
    float r = w00 * src[idx00 + 2] + w01 * src[idx01 + 2] + w10 * src[idx10 + 2] + w11 * src[idx11 + 2];

    int plane_size = dst_w * dst_h;
    int dst_idx = y * dst_w + x;

    dst[0 * plane_size + dst_idx] = r / 255.0f;
    dst[1 * plane_size + dst_idx] = g / 255.0f;
    dst[2 * plane_size + dst_idx] = b / 255.0f;
}

__global__ void postprocess_disparity_to_depth(
    const float* __restrict__ src_disp,
    float* __restrict__ dst_disp,
    float* __restrict__ dst_depth,
    int disp_w, int disp_h,
    int orig_w, int orig_h,
    float fx_baseline
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= orig_w || y >= orig_h) return;

    float scale_x = (float)disp_w / (float)orig_w;
    float scale_y = (float)disp_h / (float)orig_h;

    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;

    src_x = fmaxf(0.0f, fminf(src_x, (float)(disp_w - 1)));
    src_y = fmaxf(0.0f, fminf(src_y, (float)(disp_h - 1)));

    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, disp_w - 1);
    int y1 = min(y0 + 1, disp_h - 1);

    float dx = src_x - x0;
    float dy = src_y - y0;

    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    float disp_raw = w00 * src_disp[y0 * disp_w + x0] +
                     w01 * src_disp[y0 * disp_w + x1] +
                     w10 * src_disp[y1 * disp_w + x0] +
                     w11 * src_disp[y1 * disp_w + x1];

    float scale_back = (float)orig_w / (float)disp_w;
    float disp_val = fmaxf(0.0f, disp_raw * scale_back);

    int dst_idx = y * orig_w + x;
    if (dst_disp != 0) {
        dst_disp[dst_idx] = disp_val;
    }
    if (dst_depth != 0) {
        dst_depth[dst_idx] = (disp_val > 0.01f) ? (fx_baseline / disp_val) : 0.0f;
    }
}

}
"""

class StereoCalibrator:
    def __init__(self, calib_yaml_path):
        if not os.path.exists(calib_yaml_path):
            raise FileNotFoundError(f"Calibration file not found: {calib_yaml_path}")
            
        with open(calib_yaml_path, 'r') as f:
            calib = yaml.safe_load(f)
            
        cam0 = calib['cam0']
        cam1 = calib['cam1']
        
        self.w, self.h = cam0['resolution']
        self.K1 = np.array([[cam0['intrinsics'][0], 0, cam0['intrinsics'][2]],
                            [0, cam0['intrinsics'][1], cam0['intrinsics'][3]],
                            [0, 0, 1]], dtype=np.float64)
        self.D1 = np.array(cam0['distortion_coeffs'], dtype=np.float64)
        
        self.K2 = np.array([[cam1['intrinsics'][0], 0, cam1['intrinsics'][2]],
                            [0, cam1['intrinsics'][1], cam1['intrinsics'][3]],
                            [0, 0, 1]], dtype=np.float64)
        self.D2 = np.array(cam1['distortion_coeffs'], dtype=np.float64)
        
        T_cn_cnm1 = np.array(cam1['T_cn_cnm1'], dtype=np.float64)
        self.R = T_cn_cnm1[:3, :3]
        self.T = T_cn_cnm1[:3, 3].reshape(3, 1)
        
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, (self.w, self.h),
            self.R, self.T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        self.map_l1, self.map_l2 = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1, (self.w, self.h), cv2.CV_32FC1
        )
        self.map_r1, self.map_r2 = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2, (self.w, self.h), cv2.CV_32FC1
        )
        
        self.fx = float(self.P1[0, 0])
        self.fy = float(self.P1[1, 1])
        self.cx = float(self.P1[0, 2])
        self.cy = float(self.P1[1, 2])
        self.baseline = float(abs(self.P2[0, 3] / self.P2[0, 0]))
        
        print(f"[StereoCalibrator] Loaded Calibration ({self.w}x{self.h}):")
        print(f"  - Rectified Focal Length (fx): {self.fx:.2f} px")
        print(f"  - Principal Point (cx, cy)  : ({self.cx:.2f}, {self.cy:.2f})")
        print(f"  - Baseline Length (B)        : {self.baseline * 1000.0:.2f} mm")

    def rectify(self, left_raw, right_raw):
        left_rect = cv2.remap(left_raw, self.map_l1, self.map_l2, interpolation=cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, self.map_r1, self.map_r2, interpolation=cv2.INTER_LINEAR)
        return left_rect, right_rect


class GPUStereoDepthEngine:
    MODEL_CONFIGS = {
        'foundation': {
            'engine_file': 'foundationstereo_320x736_fp16.engine',
            'input_h': 320, 'input_w': 736,
            'input_names': ('left_image', 'right_image'),
            'output_name': 'disparity',
            'needs_plugin': False
        },
        'ess': {
            'engine_file': 'ess_fp16.engine',
            'input_h': 576, 'input_w': 960,
            'input_names': ('input_left', 'input_right'),
            'output_name': 'output_left',
            'needs_plugin': True
        },
        'light_ess': {
            'engine_file': 'light_ess_fp16.engine',
            'input_h': 288, 'input_w': 480,
            'input_names': ('input_left', 'input_right'),
            'output_name': 'output_left',
            'needs_plugin': True
        }
    }

    def __init__(self, model_type='ess', engine_path=None, orig_w=1920, orig_h=1200, fx=1304.75, baseline=0.06321):
        self.model_type = model_type
        if self.model_type not in self.MODEL_CONFIGS:
            raise ValueError(f'Unknown model type: {model_type}')
            
        cfg = self.MODEL_CONFIGS[self.model_type]
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.orig_w, self.orig_h = orig_w, orig_h
        self.fx = fx
        self.baseline = baseline
        self.fx_baseline = float(fx * baseline)
        
        # 1. 编译并加载 CUDA Kernels
        self.mod = SourceModule(CUDA_KERNELS)
        self.preprocess_kernel = self.mod.get_function('preprocess_bgr_to_rgb_planar')
        self.postprocess_kernel = self.mod.get_function('postprocess_disparity_to_depth')
        
        # 2. 插件与 TensorRT 引擎加载
        if cfg['needs_plugin']:
            plugin_path = os.path.join(curr_dir, 'dnn_stereo_disparity_v4.1.0_onnx/plugins/aarch64/ess_plugins.so')
            if os.path.exists(plugin_path):
                import ctypes
                ctypes.CDLL(plugin_path)
                
        if engine_path is None:
            engine_path = os.path.join(curr_dir, cfg['engine_file'])
            
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.input_h, self.input_w = cfg['input_h'], cfg['input_w']
        self.left_input_name, self.right_input_name = cfg['input_names']
        self.disp_output_name = cfg['output_name']
        
        # 3. 分配 GPU 显存
        self.d_raw_left = cuda.mem_alloc(self.orig_w * self.orig_h * 3)
        self.d_raw_right = cuda.mem_alloc(self.orig_w * self.orig_h * 3)
        
        self.d_final_disp = cuda.mem_alloc(self.orig_w * self.orig_h * 4)
        self.d_final_depth = cuda.mem_alloc(self.orig_w * self.orig_h * 4)
        
        self.h_final_disp = cuda.pagelocked_empty((self.orig_h, self.orig_w), dtype=np.float32)
        self.h_final_depth = cuda.pagelocked_empty((self.orig_h, self.orig_w), dtype=np.float32)
        
        self.trt_buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            nbytes = int(np.prod(shape) * 4)
            self.trt_buffers[name] = cuda.mem_alloc(nbytes)
            self.context.set_tensor_address(name, int(self.trt_buffers[name]))
            
        self.d_model_left = self.trt_buffers[self.left_input_name]
        self.d_model_right = self.trt_buffers[self.right_input_name]
        self.d_model_disp = self.trt_buffers[self.disp_output_name]
                
        self.pre_block = (32, 16, 1)
        self.pre_grid = ((self.input_w + 31) // 32, (self.input_h + 15) // 16, 1)
        
        self.post_block = (32, 16, 1)
        self.post_grid = ((self.orig_w + 31) // 32, (self.orig_h + 15) // 16, 1)
        
        self.warmup()
        print(f"[GPUStereoDepthEngine] All-CUDA Pipeline Initialized [{self.model_type}] ({self.input_w}x{self.input_h} -> {self.orig_w}x{self.orig_h}).")

    def warmup(self):
        for _ in range(3):
            self.preprocess_kernel(self.d_raw_left, self.d_model_left, np.int32(self.orig_w), np.int32(self.orig_h), np.int32(self.input_w), np.int32(self.input_h), block=self.pre_block, grid=self.pre_grid, stream=self.stream)
            self.preprocess_kernel(self.d_raw_right, self.d_model_right, np.int32(self.orig_w), np.int32(self.orig_h), np.int32(self.input_w), np.int32(self.input_h), block=self.pre_block, grid=self.pre_grid, stream=self.stream)
            self.context.execute_async_v3(self.stream.handle)
            self.postprocess_kernel(self.d_model_disp, self.d_final_disp, self.d_final_depth, np.int32(self.input_w), np.int32(self.input_h), np.int32(self.orig_w), np.int32(self.orig_h), np.float32(self.fx_baseline), block=self.post_block, grid=self.post_grid, stream=self.stream)
        self.stream.synchronize()

    def infer_depth_gpu(self, left_bgr_h, right_bgr_h, return_disparity=False):
        # 1. 异步拷贝原始 1920x1200 BGR 字节至 GPU
        cuda.memcpy_htod_async(self.d_raw_left, left_bgr_h, self.stream)
        cuda.memcpy_htod_async(self.d_raw_right, right_bgr_h, self.stream)
        
        # 2. GPU CUDA 核函数执行双目双线性缩放、BGR2RGB 与归一化
        self.preprocess_kernel(self.d_raw_left, self.d_model_left, np.int32(self.orig_w), np.int32(self.orig_h), np.int32(self.input_w), np.int32(self.input_h), block=self.pre_block, grid=self.pre_grid, stream=self.stream)
        self.preprocess_kernel(self.d_raw_right, self.d_model_right, np.int32(self.orig_w), np.int32(self.orig_h), np.int32(self.input_w), np.int32(self.input_h), block=self.pre_block, grid=self.pre_grid, stream=self.stream)
        
        # 3. GPU TensorRT 异步推理核心
        self.context.execute_async_v3(self.stream.handle)
        
        # 4. GPU CUDA 核函数执行视差插值还原 + 物理深度计算
        self.postprocess_kernel(self.d_model_disp, self.d_final_disp, self.d_final_depth, np.int32(self.input_w), np.int32(self.input_h), np.int32(self.orig_w), np.int32(self.orig_h), np.float32(self.fx_baseline), block=self.post_block, grid=self.post_grid, stream=self.stream)
        
        # 5. 异步拷贝结果回主机内存
        cuda.memcpy_dtoh_async(self.h_final_depth, self.d_final_depth, self.stream)
        if return_disparity:
            cuda.memcpy_dtoh_async(self.h_final_disp, self.d_final_disp, self.stream)
        self.stream.synchronize()
        
        if return_disparity:
            return self.h_final_depth, self.h_final_disp
        return self.h_final_depth


def colorize_disparity(disp, vmin=0, vmax=128):
    disp_clipped = np.clip(disp, vmin, vmax)
    disp_norm = ((disp_clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)


def colorize_depth(depth, min_d=0.2, max_d=6.0):
    valid = (depth > min_d) & (depth < max_d)
    depth_vis = np.zeros_like(depth, dtype=np.uint8)
    depth_vis[valid] = ((depth[valid] - min_d) / (max_d - min_d) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    color[~valid] = [0, 0, 0]
    return color


def save_pointcloud_ply(filename, depth, rgb_img, fx, fy, cx, cy, max_depth=6.0):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = (depth > 0.2) & (depth < max_depth) & ~np.isnan(depth) & ~np.isinf(depth)
    
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    
    b = rgb_img[valid, 0]
    g = rgb_img[valid, 1]
    r = rgb_img[valid, 2]
    
    num_points = len(z)
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(0, num_points, 2):  # 降采样 1/2 保存以节省空间
            f.write(f"{x[i]:.4f} {y[i]:.4f} {z[i]:.4f} {r[i]} {g[i]} {b[i]}\n")
    print(f"[Pointcloud] Exported {num_points // 2} points to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='All-CUDA Stereo Depth Pipeline for Images & Videos')
    parser.add_argument('--model', type=str, default='ess', choices=['foundation', 'ess', 'light_ess'], help='Model backend')
    parser.add_argument('--image', type=str, default=None, help='Path to 4000x1200 or 3840x1200 stereo image')
    parser.add_argument('--calib', type=str, default=None, help='Path to camchain calibration YAML')
    parser.add_argument('--output_dir', type=str, default='output_results', help='Directory to save output artifacts')
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 标定文件路径
    calib_path = args.calib
    if calib_path is None:
        calib_path = os.path.join(curr_dir, 'calibration/camera_calib_results/camchain-datacalibration_bagskalibr_input.yaml')
        
    calibrator = None
    orig_w, orig_h = 1920, 1200
    fx, fy, cx, cy = 1304.75, 1304.75, 959.75, 643.68
    baseline = 0.06321
    
    if os.path.exists(calib_path):
        calibrator = StereoCalibrator(calib_path)
        orig_w, orig_h = calibrator.w, calibrator.h
        fx, fy, cx, cy = calibrator.fx, calibrator.fy, calibrator.cx, calibrator.cy
        baseline = calibrator.baseline
    else:
        print(f"[Warning] Calib file not found at {calib_path}, using default 1920x1200 pinhole parameters.")

    engine = GPUStereoDepthEngine(
        model_type=args.model,
        orig_w=orig_w, orig_h=orig_h,
        fx=fx, baseline=baseline
    )

    if args.image is not None:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Input image not found: {args.image}")
            
        print(f"\n[Image Inference] Reading input image: {args.image}")
        raw_full = cv2.imread(args.image)
        h, w = raw_full.shape[:2]
        print(f"  - Loaded image resolution: {w} x {h}")
        
        # 裁剪左右目
        if w == 4000:
            left_raw = raw_full[:, 160:2080]
            right_raw = raw_full[:, 2080:4000]
        elif w == 3840:
            left_raw = raw_full[:, 0:1920]
            right_raw = raw_full[:, 1920:3840]
        else:
            mid = w // 2
            left_raw = raw_full[:, :mid]
            right_raw = raw_full[:, mid:]
            
        print(f"  - Left Eye Slice : {left_raw.shape[1]} x {left_raw.shape[0]}")
        print(f"  - Right Eye Slice: {right_raw.shape[1]} x {right_raw.shape[0]}")
        
        # 极线校正
        if calibrator is not None:
            t_rect0 = time.perf_counter()
            left_rect, right_rect = calibrator.rectify(left_raw, right_raw)
            t_rect1 = time.perf_counter()
            print(f"  - Stereo Rectification Time: {(t_rect1 - t_rect0)*1000.0:.2f} ms")
        else:
            left_rect, right_rect = left_raw, right_raw
            
        # GPU 核心推理
        t0 = time.perf_counter()
        depth, disparity = engine.infer_depth_gpu(left_rect, right_rect, return_disparity=True)
        t1 = time.perf_counter()
        gpu_time_ms = (t1 - t0) * 1000.0
        print(f"  - All-CUDA GPU Depth Inference Time: {gpu_time_ms:.2f} ms (FPS: {1000.0/gpu_time_ms:.1f})")
        
        # 保存结果
        os.makedirs(args.output_dir, exist_ok=True)
        model_tag = args.model
        
        rect_left_path = os.path.join(args.output_dir, f"rectified_left.jpg")
        disp_color_path = os.path.join(args.output_dir, f"disparity_{model_tag}.jpg")
        depth_color_path = os.path.join(args.output_dir, f"depth_{model_tag}.jpg")
        side_by_side_path = os.path.join(args.output_dir, f"comparison_{model_tag}.jpg")
        ply_path = os.path.join(args.output_dir, f"pointcloud_{model_tag}.ply")
        
        disp_vis = colorize_disparity(disparity, vmin=0, vmax=128)
        depth_vis = colorize_depth(depth, min_d=0.3, max_d=5.0)
        
        cv2.imwrite(rect_left_path, left_rect)
        cv2.imwrite(disp_color_path, disp_vis)
        cv2.imwrite(depth_color_path, depth_vis)
        
        # 拼接 4 视图对比图 (左目原图, 右目原图, 视差图, 深度图)
        top_row = np.hstack([cv2.resize(left_rect, (960, 600)), cv2.resize(right_rect, (960, 600))])
        bot_row = np.hstack([cv2.resize(disp_vis, (960, 600)), cv2.resize(depth_vis, (960, 600))])
        side_by_side = np.vstack([top_row, bot_row])
        cv2.imwrite(side_by_side_path, side_by_side)
        
        save_pointcloud_ply(ply_path, depth, left_rect, fx, fy, cx, cy, max_depth=5.0)
        
        print(f"\n[Artifacts Saved Successfully in {args.output_dir}/]:")
        print(f"  - 极线校正左目图 : {rect_left_path}")
        print(f"  - 视差彩色热力图 : {disp_color_path}")
        print(f"  - 物理深度距离图 : {depth_color_path}")
        print(f"  - 四视图综合对比 : {side_by_side_path}")
        print(f"  - 3D 稠密点云文件: {ply_path}")
        
    else:
        # Benchmark 模式
        print(f"\n[Benchmark] Running 50-frame ALL-CUDA benchmark ({orig_w}x{orig_h}) for model [{args.model}]...")
        dummy_l = np.random.randint(0, 255, (orig_h, orig_w, 3), dtype=np.uint8)
        dummy_r = np.random.randint(0, 255, (orig_h, orig_w, 3), dtype=np.uint8)
        
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            depth = engine.infer_depth_gpu(dummy_l, dummy_r)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            
        avg_lat = np.mean(latencies)
        print("==================================================================")
        print(f"   全 GPU (All-CUDA) 端到端全链路实测性能 [{args.model}] ({orig_w}x{orig_h})")
        print("==================================================================")
        print(f"端到端总时延 (含显存传输+CUDA前处理+TRT推理+CUDA后处理) : {avg_lat:.2f} ms")
        print(f"最小单帧时延 (Min)                                     : {np.min(latencies):.2f} ms")
        print(f"最大单帧时延 (Max)                                     : {np.max(latencies):.2f} ms")
        print(f"端到端真实吞吐量 (FPS)                                 : {1000.0/avg_lat:.2f} FPS")
        print("==================================================================\n")

if __name__ == '__main__':
    main()
