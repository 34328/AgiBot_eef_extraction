import subprocess
import tempfile
import glob
import shutil

import base64
import os

from openai import OpenAI
from prompt import genrobot_video_prompt
# ==========================================
# 配置部分 (请手动填写)
# ==========================================
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "") 
MODEL_NAME = os.getenv("MODEL_NAME", "") 

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def extract_frames(video_path, target_fps=2, verbose=True):
    """
    使用 ffmpeg 提取视频帧，支持多种编码格式 (AV1, H.265/HEVC, VP9, H.264 等)
    会依次尝试多个 ffmpeg 路径以确保最佳兼容性
    """
    if not os.path.exists(video_path):
        if verbose:
            print(f"文件不存在: {video_path}")
        return []

    # 创建临时文件夹存储提取的帧
    temp_dir = tempfile.mkdtemp()
    
    # 定义多个 ffmpeg 路径优先级 (系统版本通常编解码器更全)
    ffmpeg_paths = [
        '/usr/bin/ffmpeg',           # 系统 ffmpeg (通常支持更多编解码器)
        '/usr/local/bin/ffmpeg',     # 本地安装的 ffmpeg
        'ffmpeg'                     # PATH 中的默认 ffmpeg
    ]
    
    # 获取干净的环境变量
    clean_env = os.environ.copy()
    if 'LD_LIBRARY_PATH' in clean_env:
        del clean_env['LD_LIBRARY_PATH']
    if 'PYTHONPATH' in clean_env:
        del clean_env['PYTHONPATH']
    
    last_error = None
    for ffmpeg_cmd in ffmpeg_paths:
        try:
            # 检查 ffmpeg 是否存在
            if ffmpeg_cmd != 'ffmpeg' and not os.path.exists(ffmpeg_cmd):
                continue
                
            command = [
                ffmpeg_cmd, '-i', video_path,
                '-vf', f'fps={target_fps},scale=-2:224',
                os.path.join(temp_dir, 'frame_%04d.jpg'),
                '-loglevel', 'error',
                '-y'  # 覆盖输出
            ]
            
            if verbose:
                print(f"正在使用 {ffmpeg_cmd} 提取帧 (目标 FPS: {target_fps})...")
            subprocess.run(command, check=True, env=clean_env, capture_output=True)
            
            # 读取提取的图片并转为 base64
            frame_files = sorted(glob.glob(os.path.join(temp_dir, 'frame_*.jpg')))
            if not frame_files:
                if verbose:
                    print(f"警告: {ffmpeg_cmd} 未能提取任何帧，尝试下一个...")
                continue
                
            base64_frames = []
            for f in frame_files:
                with open(f, "rb") as image_file:
                    base64_frames.append(base64.b64encode(image_file.read()).decode("utf-8"))
            
            if verbose:
                print(f"成功提取了 {len(base64_frames)} 帧")
            return base64_frames
            
        except subprocess.CalledProcessError as e:
            last_error = e
            error_msg = e.stderr.decode() if e.stderr else str(e)
            if verbose:
                print(f"{ffmpeg_cmd} 提取失败: {error_msg.strip()}, 尝试下一个...")
            # 清理已有帧文件，准备重试
            for f in glob.glob(os.path.join(temp_dir, 'frame_*.jpg')):
                os.remove(f)
            continue
        except FileNotFoundError:
            continue
    
    # 所有尝试都失败
    if verbose:
        print(f"所有 ffmpeg 路径都无法提取视频帧: {last_error}")
    shutil.rmtree(temp_dir)
    return []
    
    # 清理临时文件 (正常情况下在 return 之前已处理)
    shutil.rmtree(temp_dir)
    return []

def score_video(video_path, verbose=True):
    """
    提取帧并调用 OpenAI API 进行评分
    """
    # 1. 提取帧
    frames = extract_frames(video_path, verbose=verbose)
    if not frames:
        return None
    
    # 2. 构建提示词

    
    prompt_message = genrobot_video_prompt(frames)

    # 3. 发送请求
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=prompt_message,
            max_tokens=100  # 不需要太长，只要一个数字
        )
        raw_response = response.choices[0].message.content.strip()
        
        # 尝试从响应中提取数字
        import re
        numbers = re.findall(r'\d+', raw_response)
        if numbers:
            score = int(numbers[0])
            # 有时候模型可能会返回 0-10, 0-100 不同量级，这里假设是 0-100
            # 简单的边界检查
            return min(max(score, 0), 100)
        else:
            if verbose:
                print(f"无法从响应中提取分数: {raw_response}")
            return None
    except Exception as e:
        if verbose:
            print(f"调用 API 失败: {e}")
        return None

# if __name__ == "__main__":
#     video_path = "/mnt/raid0/AgiBot2Lerobot/AgiBot_Word_Beta/observations/327/648642/videos/head_color.mp4"
#     print(f"开始测试视频评分: {video_path}")
#     score = score_video(video_path, verbose=True)
#     print(f"==============================")
#     if score is not None:
#         print(f"最终模型返回评分: {score}")
#     else:
#         print(f"评分测试失败。")
#     print(f"==============================")

