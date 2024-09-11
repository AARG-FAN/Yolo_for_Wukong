import cv2
import os

def extract_frames(video_path, output_folder, fps=10):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取视频帧率和总帧数
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频帧率: {video_fps}, 总帧数: {total_frames}")

    frame_interval = int(video_fps / fps)
    frame_count = 0
    extracted_count = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # 每秒提取 10 帧
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{extracted_count:06d}.jpg")
            cv2.imwrite(frame_name, frame)
            print(f"保存帧: {frame_name}")
            extracted_count += 1

        frame_count += 1

    video_capture.release()
    print(f"提取完成，共保存 {extracted_count} 帧图像.")

# 示例使用
video_path = r'C:\Users\Administrator\Desktop\bandicam 2024-09-06 22-50-33-620.mp4'  # 替换为你的视频文件路径
output_folder = r'D:\AI\yolo8\dataset8'    # 替换为你想要保存图片的文件夹路径
extract_frames(video_path, output_folder, fps=10)
