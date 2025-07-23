# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import cv2
import numpy as np
cimport numpy as cnp
cimport cython
from PIL import Image, ImageDraw, ImageFont
import time
from tqdm import tqdm
import os

# 初始化NumPy数组支持
cnp.import_array()

def create_char_gray_dict(chars, font_path=None, font_size=20):
    """创建字符灰度值字典"""
    char_dict = {}

    if font_path is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, font_size)

    for char in chars:
        img = Image.new('RGB', (font_size, font_size), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, fill=(255, 255, 255), font=font)

        # 转换为灰度并计算平均像素值
        gray_img = img.convert('L')
        np_img = np.array(gray_img)
        mean_value = np.mean(np_img)
        char_dict[char] = mean_value

    return char_dict

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void generate_ascii_frame(
    cnp.uint8_t[:, :] gray_frame,
    cnp.uint8_t[:, :, :] output_frame,
    cnp.int32_t[:] char_indices,
    cnp.uint8_t[:, :, :, :] char_atlas,
    int char_size,
    int output_width_chars,
    int output_height_chars
) noexcept:
    """Cython优化版ASCII帧生成 - 无Python对象"""
    cdef:
        int i, j, y, x
        int char_idx
        int atlas_idx
        int output_y, output_x

    for i in range(output_height_chars):
        for j in range(output_width_chars):
            char_idx = gray_frame[i, j]
            atlas_idx = char_indices[char_idx]

            output_y = i * char_size
            output_x = j * char_size

            # 直接复制预渲染的字符图像块
            for y in range(char_size):
                for x in range(char_size):
                    output_frame[output_y + y, output_x + x, 0] = char_atlas[atlas_idx, y, x, 0]
                    output_frame[output_y + y, output_x + x, 1] = char_atlas[atlas_idx, y, x, 1]
                    output_frame[output_y + y, output_x + x, 2] = char_atlas[atlas_idx, y, x, 2]

def video_to_ascii_video(
        input_video_path,
        output_video_path,
        char_dict,
        font_path=None,
        char_size=12,
        output_width_chars=100,
        fps=None,
        codec='avc1',
        progress=True,
        showCallback = None
):
    """
    高效内存优化版视频转换函数
    """
    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {input_video_path}")

    # 获取视频信息
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置输出帧率
    if fps is None:
        fps = input_fps

    # 计算字符画尺寸
    aspect_ratio = height / width
    output_height_chars = int(output_width_chars * aspect_ratio * 0.75)

    # 输出尺寸
    output_width_px = output_width_chars * char_size
    output_height_px = output_height_chars * char_size

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (output_width_px, output_height_px),
        True
    )

    if not video_writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (output_width_px, output_height_px),
            True
        )
        if not video_writer.isOpened():
            raise ValueError(f"无法创建视频文件: {output_video_path}")

    # 预计算字符查找表
    char_values = np.array(list(char_dict.values()))
    char_keys = list(char_dict.keys())
    lookup_list = []

    # 创建字符索引映射
    for gray_val in range(256):
        min_diff = 1000.0
        idx = 0
        for i in range(len(char_values)):
            diff = abs(char_values[i] - gray_val)
            if diff < min_diff:
                min_diff = diff
                idx = i
        lookup_list.append(char_keys[idx])

    # 加载字体
    if font_path:
        try:
            font = ImageFont.truetype(font_path, char_size)
        except:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # 预渲染字符图集 - 只渲染唯一字符
    unique_chars = set(lookup_list)
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}

    # 创建字符图集 - 四维数组
    char_atlas = np.zeros((len(unique_chars), char_size, char_size, 3), dtype=np.uint8)
    for char, idx in char_to_index.items():
        img = Image.new('RGB', (char_size, char_size), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, fill=(255, 255, 255), font=font)
        char_atlas[idx] = np.array(img)

    # 创建字符索引映射表
    char_indices = np.zeros(256, dtype=np.int32)
    for gray_val in range(256):
        char = lookup_list[gray_val]
        char_indices[gray_val] = char_to_index[char]

    # 预分配内存 - 复用内存缓冲区
    # 输出帧缓冲区
    output_frame = np.zeros((output_height_px, output_width_px, 3), dtype=np.uint8)

    # 灰度帧缓冲区 - 复用内存
    gray_frame_buffer = np.zeros((height, width), dtype=np.uint8)

    # 调整大小后的灰度帧缓冲区 - 复用内存
    resized_gray_buffer = np.zeros((output_height_chars, output_width_chars), dtype=np.uint8)

    # 进度条设置
    if progress:
        pbar = tqdm(total=total_frames, desc="处理视频帧", unit="帧")

    start_time = time.time()

    # 单线程处理但高度优化
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度 - 直接写入缓冲区
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=gray_frame_buffer)

        # 调整大小到字符网格尺寸 - 直接写入缓冲区
        cv2.resize(gray_frame_buffer, (output_width_chars, output_height_chars),
                  dst=resized_gray_buffer, interpolation=cv2.INTER_AREA)

        # 使用优化的Cython函数生成ASCII帧
        generate_ascii_frame(
            resized_gray_buffer,
            output_frame,
            char_indices,
            char_atlas,
            char_size,
            output_width_chars,
            output_height_chars
        )

        # 直接写入视频帧
        video_writer.write(output_frame)
        if showCallback:
            showCallback(output_frame)

        if progress:
            pbar.update(1)

    # 清理资源
    cap.release()
    video_writer.release()

    # 释放大内存对象
    del output_frame, gray_frame_buffer, resized_gray_buffer, char_atlas
    import gc
    gc.collect()

    total_time = time.time() - start_time
    avg_speed = total_frames / total_time if total_time > 0 else 0

    if progress:
        pbar.close()

    print(f"\n视频转换完成! 共处理 {total_frames} 帧, 耗时 {total_time:.2f} 秒")
    print(f"平均速度: {avg_speed:.1f} FPS")
    print(f"输出尺寸: {output_width_px}x{output_height_px} 像素")
    print(f"字符尺寸: {char_size}px, 网格: {output_width_chars}x{output_height_chars}")
    print(f"输出视频: {output_video_path}")

    return output_video_path