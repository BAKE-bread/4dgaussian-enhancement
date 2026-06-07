import cv2
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import pyiqa

class NovelViewVideoEvaluator:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"初始化评价器，使用设备: {self.device}")
        
        # 支持读取的图片格式
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        # 初始化 NIQE 评价器 (PyIQA 自动下载预训练的统计算法)
        print("正在加载 NIQE 模型...")
        self.niqe_metric = pyiqa.create_metric('niqe', device=self.device, as_loss=False)

    def calculate_niqe(self, frame_rgb):
        """
        计算单帧的 NIQE 分数。
        注意：NIQE 分数越低，代表图像质量越好、越自然。
        """
        # pyiqa 期望输入格式为 Tensor [1, C, H, W]，取值范围 [0, 1]
        img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            score = self.niqe_metric(img_tensor)
        return score.item()

    def calculate_flow_warping_error(self, frame1_rgb, frame2_rgb):
        """
        计算两帧之间的时序光流形变误差 (Warping Error)。
        越低代表时序一致性越好，闪烁越少。
        """
        # 转为灰度图以计算光流
        gray1 = cv2.cvtColor(frame1_rgb, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2_rgb, cv2.COLOR_RGB2GRAY)

        # 1. 计算前向光流 (frame1 -> frame2)
        # 使用 Farneback 稠密光流算法，速度快且对平滑相机运动很鲁棒
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # 2. 将 frame2 根据光流 Warp 回 frame1 的坐标系
        h, w = gray1.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (X + flow[..., 0]).astype(np.float32)
        map_y = (Y + flow[..., 1]).astype(np.float32)

        # 使用双线性插值进行重采样
        warped_frame2 = cv2.remap(frame2_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # 3. 计算 Warping Error (L1 误差)
        # 为了避免相机移动导致边缘产生不可避免的光流越界误差，我们裁剪掉边缘 5% 的像素
        crop_h, crop_w = int(h * 0.05), int(w * 0.05)
        
        # 计算 RGB 通道的绝对误差
        diff = np.abs(frame1_rgb.astype(np.float32) - warped_frame2.astype(np.float32))
        valid_diff = diff[crop_h:-crop_h, crop_w:-crop_w]

        # 返回平均像素误差
        mean_l1_error = np.mean(valid_diff)
        return mean_l1_error

    def evaluate_video(self, video_path):
        """
        读取视频文件并计算全视频的平均 NIQE 和 Flow Error
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"找不到视频文件: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < 2:
            raise ValueError("视频帧数不足，无法计算时序误差！")

        niqe_scores = []
        flow_errors = []

        # 读取第一帧
        ret, prev_frame = cap.read()
        if not ret:
            raise RuntimeError("无法读取视频帧！")
        
        prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
        
        # 计算第一帧的 NIQE
        niqe_scores.append(self.calculate_niqe(prev_frame_rgb))

        print(f"开始评测视频: {os.path.basename(video_path)} (共 {total_frames} 帧)")
        
        for _ in tqdm(range(1, total_frames)):
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            curr_frame_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
            
            # 1. 计算当前帧空间质量 NIQE
            niqe_scores.append(self.calculate_niqe(curr_frame_rgb))
            
            # 2. 计算与上一帧的时序光流误差
            flow_err = self.calculate_flow_warping_error(prev_frame_rgb, curr_frame_rgb)
            flow_errors.append(flow_err)
            
            # 更新 prev_frame
            prev_frame_rgb = curr_frame_rgb

        cap.release()

        avg_niqe = np.mean(niqe_scores)
        avg_flow_err = np.mean(flow_errors)

        self._print_results(os.path.basename(video_path), avg_niqe, avg_flow_err)

        return avg_niqe, avg_flow_err

    def evaluate_frames_folder(self, folder_path):
        """
        从文件夹读取序列帧（图片）并计算平均 NIQE 和 Flow Error
        """
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"找不到文件夹: {folder_path}")

        # 获取所有图片文件
        all_files = os.listdir(folder_path)
        image_files = [f for f in all_files if f.lower().endswith(self.image_extensions)]
        
        if len(image_files) == 0:
            raise ValueError(f"文件夹中没有支持的图片格式: {self.image_extensions}")
        
        # 按文件名排序（通常序列帧按文件名自然顺序）
        image_files.sort()
        image_paths = [os.path.join(folder_path, f) for f in image_files]

        total_frames = len(image_paths)
        if total_frames < 2:
            raise ValueError("图片帧数不足，无法计算时序误差！")

        print(f"开始评测文件夹: {os.path.basename(folder_path)} (共 {total_frames} 帧图片)")

        niqe_scores = []
        flow_errors = []

        # 读取第一帧
        prev_frame_bgr = cv2.imread(image_paths[0])
        if prev_frame_bgr is None:
            raise RuntimeError(f"无法读取图片: {image_paths[0]}")
        prev_frame_rgb = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2RGB)
        niqe_scores.append(self.calculate_niqe(prev_frame_rgb))

        # 循环处理后续帧
        for i in tqdm(range(1, total_frames)):
            curr_frame_bgr = cv2.imread(image_paths[i])
            if curr_frame_bgr is None:
                print(f"警告: 跳过无法读取的图片 {image_paths[i]}")
                continue
            curr_frame_rgb = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2RGB)

            # 1. 当前帧 NIQE
            niqe_scores.append(self.calculate_niqe(curr_frame_rgb))

            # 2. 与前帧的 warping error
            # 检查尺寸是否一致（防止文件夹中混入不同尺寸的图片）
            if prev_frame_rgb.shape != curr_frame_rgb.shape:
                print(f"警告: 帧 {i-1} 和帧 {i} 尺寸不一致，跳过时序误差计算")
                flow_err = np.nan
            else:
                flow_err = self.calculate_flow_warping_error(prev_frame_rgb, curr_frame_rgb)
            flow_errors.append(flow_err)

            prev_frame_rgb = curr_frame_rgb

        # 过滤掉 NaN（尺寸不一致的帧对）
        flow_errors = [e for e in flow_errors if not np.isnan(e)]
        if len(flow_errors) == 0:
            print("警告: 没有有效的时序误差结果（可能是所有帧尺寸不一致）")
            avg_flow_err = float('nan')
        else:
            avg_flow_err = np.mean(flow_errors)

        avg_niqe = np.mean(niqe_scores)

        self._print_results(os.path.basename(folder_path), avg_niqe, avg_flow_err)

        return avg_niqe, avg_flow_err

    def _print_results(self, name, avg_niqe, avg_flow_err):
        """统一打印结果"""
        print("\n" + "="*40)
        print(f"评测结果 - {name}")
        print("="*40)
        print(f"✅ 平均 NIQE (空间质量):     {avg_niqe:.4f}  (越低越好)")
        print(f"✅ 平均 Warping Error (时序):  {avg_flow_err:.4f}  (越低越好)")
        print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="新视角生成视频质量评价")
    parser.add_argument("--video", type=str, default=None, help="输入生成的 mp4 视频路径")
    parser.add_argument("--frames_folder", type=str, default=None, help="输入序列帧文件夹路径（支持jpg/png/bmp等格式）")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备 (cuda 或 cpu)")
    args = parser.parse_args()

    # 检查输入：必须且只能指定其中一个
    if args.video is None and args.frames_folder is None:
        parser.error("必须指定 --video 或 --frames_folder 中的一个")
    if args.video is not None and args.frames_folder is not None:
        parser.error("只能指定 --video 或 --frames_folder 中的一个，不能同时使用")

    evaluator = NovelViewVideoEvaluator(device=args.device)

    if args.video is not None:
        evaluator.evaluate_video(args.video)
    else:  # args.frames_folder is not None
        evaluator.evaluate_frames_folder(args.frames_folder)