import os
import sys
import time
import argparse
import ctypes
import numpy as np
import cv2
import csv

# 自动确保 nvcc 在 PATH 中
if '/usr/local/cuda/bin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from stereo_depth_pipeline_gpu import StereoCalibrator, GPUStereoDepthEngine, colorize_depth, colorize_disparity

# ----------------- ctypes SDK 结构体定义 -----------------

SYNC_MAX_VIDEO_COUNT = 20
SYNC_IMU_SAMPLE_COUNT = 11
SYNC_MTT_SAMPLE_COUNT = 5

SYNC_PIX_FMT_NV12 = 0
SYNC_PIX_FMT_RGB = 1


class SyncICM42688Data(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("uTime", ctypes.c_uint64),
        ("fAccData_X", ctypes.c_float),
        ("fAccData_Y", ctypes.c_float),
        ("fAccData_Z", ctypes.c_float),
        ("fGyroData_X", ctypes.c_float),
        ("fGyroData_Y", ctypes.c_float),
        ("fGyroData_Z", ctypes.c_float),
    ]


class SyncAK09940Data(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("uTime", ctypes.c_uint64),
        ("iX", ctypes.c_int32),
        ("iY", ctypes.c_int32),
        ("iZ", ctypes.c_int32),
        ("Temp", ctypes.c_float),
        ("iStatusBit", ctypes.c_int32),
    ]


class SyncVideoFileInfo(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("channelIndex", ctypes.c_uint32),
        ("filePath", ctypes.c_char * 512),
        ("fileName", ctypes.c_char * 256),
        ("durationMs", ctypes.c_int64),
        ("isLoaded", ctypes.c_bool),
    ]


class SyncVideoFrameData(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("bufSize", ctypes.c_int),
        ("format", ctypes.c_int),
        ("startExposureTime", ctypes.c_uint64),
        ("endExposureTime", ctypes.c_uint64),
        ("imu_data", SyncICM42688Data * SYNC_IMU_SAMPLE_COUNT),
        ("mtt_data", SyncAK09940Data * SYNC_MTT_SAMPLE_COUNT),
        ("channelIndex", ctypes.c_int),
        ("isValid", ctypes.c_bool),
    ]


class SyncVideoGroupData(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("frameData", SyncVideoFrameData * SYNC_MAX_VIDEO_COUNT),
        ("videoCount", ctypes.c_int),
    ]


VIDEO_FRAME_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.POINTER(SyncVideoGroupData), ctypes.c_void_p)


class SDKVideoProcessor:
    def __init__(self, sdk_so_path, calib_path, model_type='ess', output_video='output.mp4', output_imu_csv='imu.csv', max_frames=0):
        self.sdk_so_path = sdk_so_path
        self.calib_path = calib_path
        self.model_type = model_type
        self.output_video = output_video
        self.output_imu_csv = output_imu_csv
        self.max_frames = max_frames
        
        self.processed_frames = 0
        self.writer = None
        self.csv_file = None
        self.csv_writer = None
        self.running = True
        self.latencies = []
        self.t_start = None
        self.last_frame_time = time.perf_counter()
        
        # 1. 载入 SDK
        if not os.path.exists(self.sdk_so_path):
            raise FileNotFoundError(f"SDK library not found: {self.sdk_so_path}")
        self.lib = ctypes.CDLL(self.sdk_so_path)
        self.setup_sdk_signatures()
        
        # 2. 载入相机标定
        self.calibrator = None
        self.orig_w, self.orig_h = 1920, 1200
        self.fx, self.fy, self.cx, self.cy = 1304.75, 1304.75, 959.75, 643.68
        self.baseline = 0.06321
        
        if os.path.exists(self.calib_path):
            self.calibrator = StereoCalibrator(self.calib_path)
            self.orig_w, self.orig_h = self.calibrator.w, self.calibrator.h
            self.fx, self.fy, self.cx, self.cy = self.calibrator.fx, self.calibrator.fy, self.calibrator.cx, self.calibrator.cy
            self.baseline = self.calibrator.baseline
            
        # 3. 初始化 All-CUDA 深度引擎
        self.cuda_context = cuda.Context.get_current()
        self.engine = GPUStereoDepthEngine(
            model_type=self.model_type,
            orig_w=self.orig_w, orig_h=self.orig_h,
            fx=self.fx, baseline=self.baseline
        )
        
        # 4. 初始化 IMU CSV
        if self.output_imu_csv:
            os.makedirs(os.path.dirname(os.path.abspath(self.output_imu_csv)), exist_ok=True)
            self.csv_file = open(self.output_imu_csv, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "frame_idx", "exp_start_us", "exp_end_us",
                "sample_idx", "imu_time_us", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"
            ])

    def setup_sdk_signatures(self):
        self.lib.SYNC_Initialize.restype = ctypes.c_int
        self.lib.SYNC_Release.restype = None
        self.lib.SYNC_GetVersion.restype = ctypes.c_char_p
        self.lib.SYNC_LoadVideoFolder.argtypes = [ctypes.c_char_p]
        self.lib.SYNC_LoadVideoFolder.restype = ctypes.c_int
        self.lib.SYNC_SetOutputPixelFormat.argtypes = [ctypes.c_int]
        self.lib.SYNC_SetOutputPixelFormat.restype = ctypes.c_int
        self.lib.SYNC_SetFrameCallback.argtypes = [VIDEO_FRAME_CALLBACK, ctypes.c_void_p]
        self.lib.SYNC_SetFrameCallback.restype = ctypes.c_int
        self.lib.SYNC_StartDecode.argtypes = [ctypes.c_int]
        self.lib.SYNC_StartDecode.restype = ctypes.c_int
        self.lib.SYNC_StopDecode.restype = ctypes.c_int
        self.lib.SYNC_IsDecoding.restype = ctypes.c_bool
        self.lib.SYNC_GetFrameRate.argtypes = [ctypes.c_uint32]
        self.lib.SYNC_GetFrameRate.restype = ctypes.c_float

    def on_frame(self, group_ptr, user_data):
        if not group_ptr or not self.running:
            return
        group = group_ptr.contents
        if group.videoCount <= 0:
            return
            
        frame = group.frameData[0]
        if not frame.isValid or not frame.data:
            return

        self.last_frame_time = time.perf_counter()
        
        # 绑定 CUDA 上下文至回调线程
        self.cuda_context.push()
        try:
            t0 = time.perf_counter()
            self.processed_frames += 1
            
            # 1. 解析 11 点 IMU 数据并写入 CSV
            if self.csv_writer:
                for s in range(SYNC_IMU_SAMPLE_COUNT):
                    imu = frame.imu_data[s]
                    self.csv_writer.writerow([
                        self.processed_frames, frame.startExposureTime, frame.endExposureTime,
                        s, imu.uTime, imu.fAccData_X, imu.fAccData_Y, imu.fAccData_Z,
                        imu.fGyroData_X, imu.fGyroData_Y, imu.fGyroData_Z
                    ])
                    
            # 2. 从 SDK 内存指针获取 RGB 数据 (3840 x 1200 x 3)
            num_bytes = frame.width * frame.height * 3
            raw_ptr = ctypes.cast(frame.data, ctypes.POINTER(ctypes.c_uint8 * num_bytes))
            rgb_full = np.frombuffer(raw_ptr.contents, dtype=np.uint8).reshape((frame.height, frame.width, 3))
            
            # RGB -> BGR
            bgr_full = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR)
            
            # 3. 切分左右目 (各 1920x1200)
            left_raw = bgr_full[:, 0:1920]
            right_raw = bgr_full[:, 1920:3840]
            
            # 4. 标定极线校正
            if self.calibrator is not None:
                left_rect, right_rect = self.calibrator.rectify(left_raw, right_raw)
            else:
                left_rect, right_rect = left_raw, right_raw
                
            # 5. All-CUDA GPU 深度估计
            depth = self.engine.infer_depth_gpu(left_rect, right_rect, return_disparity=False)
            
            # 6. 色彩映射与 1920x600 拼接
            depth_vis = colorize_depth(depth, min_d=0.3, max_d=5.0)
            l_small = cv2.resize(left_rect, (960, 600), interpolation=cv2.INTER_LINEAR)
            d_small = cv2.resize(depth_vis, (960, 600), interpolation=cv2.INTER_LINEAR)
            out_frame = np.hstack([l_small, d_small])
            
            # 7. 写入输出视频
            if self.writer is None:
                os.makedirs(os.path.dirname(os.path.abspath(self.output_video)), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(self.output_video, fourcc, 30.0, (1920, 600))
                if not self.writer.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    self.writer = cv2.VideoWriter(self.output_video, fourcc, 30.0, (1920, 600))
                    
            self.writer.write(out_frame)
            
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            self.latencies.append(lat_ms)
            
            if self.processed_frames % 10 == 0 or (self.max_frames > 0 and self.processed_frames >= self.max_frames):
                recent_fps = 1000.0 / np.mean(self.latencies[-30:])
                elapsed = time.perf_counter() - self.t_start
                print(f"  Frame [{self.processed_frames:4d}{f'/{self.max_frames}' if self.max_frames > 0 else ''}] - Speed: {recent_fps:5.1f} FPS | Elapsed: {elapsed:5.1f}s", flush=True)
                
            if self.max_frames > 0 and self.processed_frames >= self.max_frames:
                self.running = False
        finally:
            self.cuda_context.pop()

    def run(self, video_path):
        print(f"\n[SDK] Initializing SDK: {self.lib.SYNC_GetVersion().decode()}")
        ret = self.lib.SYNC_Initialize()
        if ret != 0:
            raise RuntimeError("SYNC_Initialize failed")
            
        self.lib.SYNC_SetOutputPixelFormat(SYNC_PIX_FMT_RGB)
        
        print(f"[SDK] Loading Video File: {video_path}")
        loaded = self.lib.SYNC_LoadVideoFolder(video_path.encode('utf-8'))
        if loaded <= 0:
            self.lib.SYNC_Release()
            raise RuntimeError(f"Failed to load video: {video_path}")
            
        self.c_callback = VIDEO_FRAME_CALLBACK(self.on_frame)
        self.lib.SYNC_SetFrameCallback(self.c_callback, None)
        
        print(f"[SDK] Starting NVDEC Hardware Decoding...")
        self.t_start = time.perf_counter()
        self.last_frame_time = time.perf_counter()
        self.lib.SYNC_StartDecode(1)
        
        # 监测解码状态与超时自动退出 (当 2 秒内无新帧到达时判定为视频播放结束)
        while self.running and self.lib.SYNC_IsDecoding():
            time.sleep(0.05)
            if self.processed_frames > 0 and (time.perf_counter() - self.last_frame_time > 2.0):
                print("[SDK] Reached End-of-Stream (EOF). Auto-completing...")
                break
            
        print("[SDK] Stopping Decoding & Releasing...")
        self.lib.SYNC_StopDecode()
        self.lib.SYNC_Release()
        
        if self.writer:
            self.writer.release()
        if self.csv_file:
            self.csv_file.close()
            
        total_time = time.perf_counter() - self.t_start
        overall_fps = self.processed_frames / total_time if total_time > 0 else 0
        print("\n==================================================================")
        print(f"   SDK 硬件解码视频深度流生成完成: [{self.output_video}]")
        print(f"   同步 IMU 轨迹文件: [{self.output_imu_csv}]")
        print(f"   总处理帧数 : {self.processed_frames} 帧")
        print(f"   全流程总耗时 : {total_time:.2f} 秒")
        print(f"   端到端综合平均帧率: {overall_fps:.2f} FPS")
        print("==================================================================\n")


def main():
    parser = argparse.ArgumentParser(description='SDK-Accelerated Stereo Video Depth Generation')
    parser.add_argument('--video', type=str, required=True, help='Path to input stereo MP4 video')
    parser.add_argument('--model', type=str, default='ess', choices=['foundation', 'ess', 'light_ess'])
    parser.add_argument('--sdk', type=str, default=None, help='Path to libsyncvideo.so')
    parser.add_argument('--calib', type=str, default=None, help='Path to camchain calibration YAML')
    parser.add_argument('--output_video', type=str, default=None)
    parser.add_argument('--output_imu', type=str, default=None)
    parser.add_argument('--max_frames', type=int, default=0)
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    sdk_so = args.sdk
    if sdk_so is None:
        sdk_so = os.path.join(curr_dir, 'test_Sync/libsyncvideo.so')
        
    calib_yaml = args.calib
    if calib_yaml is None:
        calib_yaml = os.path.join(curr_dir, 'calibration/camera_calib_results/camchain-datacalibration_bagskalibr_input.yaml')
        
    out_video = args.output_video
    base_name = os.path.splitext(os.path.basename(args.video))[0]
    if out_video is None:
        out_video = os.path.join(curr_dir, f"output_results/sdk_depth_{base_name}_{args.model}.mp4")
        
    out_imu = args.output_imu
    if out_imu is None:
        out_imu = os.path.join(curr_dir, f"output_results/sdk_imu_{base_name}.csv")

    processor = SDKVideoProcessor(
        sdk_so_path=sdk_so,
        calib_path=calib_yaml,
        model_type=args.model,
        output_video=out_video,
        output_imu_csv=out_imu,
        max_frames=args.max_frames
    )
    processor.run(args.video)


if __name__ == '__main__':
    main()
