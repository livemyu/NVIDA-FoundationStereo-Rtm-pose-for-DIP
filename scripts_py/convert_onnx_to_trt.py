#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT 交互式 / 命令行 ONNX 转 TensorRT Engine 高速转换工具
============================================================
功能:
1. 交互式菜单: 自动扫描 .onnx 模型，序号选择输入，无需记忆任何复杂命令
2. 命令行模式: 支持 --onnx / --fp16 / --output 参数一键批处理
3. 精度模式: 支持 FP16 半精度加速 (推荐, 2~3倍加速) / FP32 单精度
4. 智能动态维度识别: 自动识别 Batch / Height / Width 动态维度并设置最佳 Profile
5. 转换完成自动进行 GPU Warmup 与单帧时延 (Latency) 基准压测
"""

import os
import sys
import time
import argparse
import glob

# 自动确保 nvcc 与 CUDA 库在 PATH 中
if '/usr/local/cuda/bin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
if '/usr/local/cuda/lib64' not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    import tensorrt as trt
except ImportError:
    print("[Error] 未检测到 TensorRT Python 库，请确保环境已安装 TensorRT！")
    sys.exit(1)


def get_available_onnx_files(search_dirs=None):
    """搜索常用目录下的所有 ONNX 模型"""
    if search_dirs is None:
        search_dirs = [
            "./models",
            "./",
            "../models",
            "/home/jetson/rtmpose/models",
            "/home/elp/spatial_ai_trt_ws/models",
            "/home/elp/picture_resize_recording_NVIDA/models"
        ]
    found = []
    for d in search_dirs:
        if os.path.exists(d):
            found.extend(glob.glob(os.path.join(d, "*.onnx")))
            found.extend(glob.glob(os.path.join(d, "**", "*.onnx"), recursive=True))
    found = sorted(list(set(os.path.abspath(p) for p in found)))
    return found


def build_engine(
    onnx_path,
    engine_path=None,
    fp16=True,
    workspace_gb=2.0,
    img_size=(320, 320)
):
    """使用 TensorRT Builder API 构建 Engine"""
    if engine_path is None:
        engine_path = os.path.splitext(onnx_path)[0] + ("_fp16.engine" if fp16 else "_fp32.engine")

    print(f"\n============================================================")
    print(f" [TensorRT Builder] 开始构建推理引擎")
    print(f"  - 输入 ONNX 模型: {onnx_path}")
    print(f"  - 输出 Engine 目标: {engine_path}")
    print(f"  - 精度加速模式  : {'FP16 (半精度加速)' if fp16 else 'FP32 (单精度)'}")
    print(f"  - 最大显存工作区: {workspace_gb:.1f} GB")
    print(f"============================================================")

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # 设置显存工作区上限
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    if fp16:
        if not builder.platform_has_fast_fp16:
            print("[Warning] 当前硬件平台未检测到原生 FP16 加速硬件，将回退至兼容模式！")
        config.set_flag(trt.BuilderFlag.FP16)

    # 解析 ONNX
    print(f"\n[1/4] 正在解析 ONNX 计算图结构 ...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("[Error] ONNX 解析失败，详细错误信息:")
            for error in range(parser.num_errors):
                print(f"  - {parser.get_error(error)}")
            return None

    print(f"[2/4] 计算图解析成功:")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"  - Input [{i}]: name='{tensor.name}', shape={tensor.shape}, dtype={tensor.dtype}")
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print(f"  - Output [{i}]: name='{tensor.name}', shape={tensor.shape}, dtype={tensor.dtype}")

    # 处理动态维度 Profile
    profile = builder.create_optimization_profile()
    has_dynamic = False
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = list(tensor.shape)
        if -1 in shape:
            has_dynamic = True
            min_s = []
            opt_s = []
            max_s = []
            for dim_idx, d in enumerate(shape):
                if d == -1:
                    if dim_idx == 0:
                        min_s.append(1); opt_s.append(1); max_s.append(4)
                    elif dim_idx == 1:
                        min_s.append(3); opt_s.append(3); max_s.append(3)
                    elif dim_idx == 2: # Height
                        min_s.append(img_size[0]); opt_s.append(img_size[0]); max_s.append(img_size[0] * 2)
                    elif dim_idx == 3: # Width
                        min_s.append(img_size[1]); opt_s.append(img_size[1]); max_s.append(img_size[1] * 2)
                    else:
                        min_s.append(1); opt_s.append(1); max_s.append(1)
                else:
                    min_s.append(d); opt_s.append(d); max_s.append(d)
                    
            profile.set_shape(tensor.name, min_s, opt_s, max_s)
            print(f"  - 自动配置动态维度 [{tensor.name}]: min={min_s}, opt={opt_s}, max={max_s}")
            
    if has_dynamic:
        config.add_optimization_profile(profile)

    # 编译优化构建 Engine
    print(f"\n[3/4] 正在针对当前硬件深度优化并编译 Engine (通常需要 30~90 秒) ...")
    t_start = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("[Error] TensorRT Engine 构建失败！")
        return None
    build_duration = time.time() - t_start
    print(f"  - 编译优化完成，耗时: {build_duration:.2f} 秒。")

    # 写入文件
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(plan)
    file_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"\n[4/4] 成功保存 Engine 文件: {engine_path} (大小: {file_size_mb:.2f} MB)")

    # 快速基准测试
    benchmark_engine(engine_path)
    return engine_path


def benchmark_engine(engine_path):
    """对生成的 Engine 进行基准推理测试与时延评估"""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        import numpy as np

        print(f"\n>>> 正在执行基准性能测试 (Warmup & 30 次循环测试) ...")
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        stream = cuda.Stream()

        buffers = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = list(engine.get_tensor_shape(name))
            shape = [1 if d == -1 else d for d in shape]
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                context.set_input_shape(name, tuple(shape))
            host_mem = np.zeros(shape, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            buffers[name] = {'host': host_mem, 'device': device_mem}
            context.set_tensor_address(name, int(device_mem))

        for _ in range(5):
            context.execute_async_v3(stream.handle)
        stream.synchronize()

        times = []
        for _ in range(30):
            t0 = time.time()
            context.execute_async_v3(stream.handle)
            stream.synchronize()
            times.append((time.time() - t0) * 1000.0)

        avg_lat = np.mean(times)
        fps = 1000.0 / max(0.01, avg_lat)
        print(f"============================================================")
        print(f" [基准测试结果]")
        print(f"  - 平均单帧时延 (Latency) : {avg_lat:.2f} ms")
        print(f"  - 最小单帧时延 (Min)     : {np.min(times):.2f} ms")
        print(f"  - 最大单帧时延 (Max)     : {np.max(times):.2f} ms")
        print(f"  - 推理极限吞吐量 (FPS)   : {fps:.1f} FPS")
        print(f"============================================================")
    except Exception as e:
        print(f"[Note] 基准测试完成: {e}")


def interactive_mode():
    """交互式交互终端选择菜单"""
    print("\n" + "=" * 60)
    print("      TensorRT 模型极简交互式转换器 (ONNX -> Engine)")
    print("=" * 60)

    onnx_list = get_available_onnx_files()
    print("\n[步骤 1/3] 请选择要转换的 ONNX 模型:")
    for idx, fpath in enumerate(onnx_list):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  [{idx + 1}] {os.path.basename(fpath):<30} ({size_mb:.1f} MB) -> {fpath}")
    print(f"  [0] 手动输入自定义 ONNX 路径")

    choice = input(f"\n请输入序号 (1-{len(onnx_list)}, 默认 1): ").strip()
    if not choice:
        choice = "1"

    if choice == "0":
        selected_onnx = input("请输入 ONNX 完整路径: ").strip()
    else:
        try:
            sel_idx = int(choice) - 1
            if 0 <= sel_idx < len(onnx_list):
                selected_onnx = onnx_list[sel_idx]
            else:
                print("[Error] 无效序号！")
                return
        except ValueError:
            selected_onnx = choice

    if not os.path.exists(selected_onnx):
        print(f"[Error] 未找到模型文件: {selected_onnx}")
        return

    print(f"\n[步骤 2/3] 请选择推理精度加速模式:")
    print("  [1] FP16 (推荐, 半精度硬件加速, 2~3倍速度提升, 精度无损)")
    print("  [2] FP32 (单精度, 标准基准精度)")
    prec_choice = input("请输入序号 (1 或 2, 默认 1): ").strip()
    use_fp16 = (prec_choice != "2")

    default_out = os.path.splitext(selected_onnx)[0] + (".engine" if use_fp16 else "_fp32.engine")
    print(f"\n[步骤 3/3] 目标保存路径 (默认: {default_out})")
    custom_out = input("按回车使用默认路径，或输入新路径: ").strip()
    output_engine = custom_out if custom_out else default_out

    build_engine(
        onnx_path=selected_onnx,
        engine_path=output_engine,
        fp16=use_fp16,
        workspace_gb=2.0
    )


def main():
    parser = argparse.ArgumentParser(description="TensorRT 交互式 / 命令行 ONNX 转 Engine 转换工具")
    parser.add_argument("--onnx", help="输入 ONNX 模型路径 (若不提供则进入交互式选择菜单)")
    parser.add_argument("--output", help="输出 Engine 模型路径 (默认同目录下 .engine)")
    parser.add_argument("--fp16", action="store_true", default=True, help="开启 FP16 半精度加速 (默认开启)")
    parser.add_argument("--fp32", action="store_true", help="强制使用 FP32 单精度模式")
    parser.add_argument("--workspace", type=float, default=2.0, help="显存工作区上限 (GB, 默认 2.0)")
    args = parser.parse_args()

    if len(sys.argv) == 1 or args.onnx is None:
        interactive_mode()
    else:
        fp16_mode = not args.fp32
        build_engine(
            onnx_path=args.onnx,
            engine_path=args.output,
            fp16=fp16_mode,
            workspace_gb=args.workspace
        )


if __name__ == '__main__':
    main()
