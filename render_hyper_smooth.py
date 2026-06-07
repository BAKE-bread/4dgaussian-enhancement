# render_smooth.py
import torch
from scene import Scene
import os
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from utils.general_utils import safe_state
from scene.cameras import Camera
import imageio
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d
import numpy as np

def apply_zoom(cam, zoom_factor=1.2):
    """zoom_factor >1 拉近（放大物体），<1 拉远（广角）"""
    new_cam = copy.deepcopy(cam)  # 注意深拷贝
    new_cam.FoVx = cam.FoVx / zoom_factor   # 原代码中 FoVx 是弧度值，除法使视野变小（长焦）
    new_cam.FoVy = cam.FoVy / zoom_factor
    return new_cam

def add_small_rotation(c2w_mat, angle_x_deg=2.0, angle_y_deg=2.0):
    """c2w_mat: 4x4 世界到相机（C2W）矩阵"""
    rx = R.from_euler('x', angle_x_deg, degrees=True).as_matrix()
    ry = R.from_euler('y', angle_y_deg, degrees=True).as_matrix()
    rot_perturb = rx @ ry   # 先俯仰再偏航
    c2w_rot = c2w_mat[:3, :3]
    c2w_trans = c2w_mat[:3, 3]
    new_rot = c2w_rot @ rot_perturb   # 在原始朝向基础上旋转
    new_c2w = np.eye(4)
    new_c2w[:3, :3] = new_rot
    new_c2w[:3, 3] = c2w_trans
    return new_c2w

def add_small_translation(c2w_mat, right_amp=0.02, up_amp=0.02):
    """位移幅度单位：米（假设训练场景尺度归一化过）"""
    right_vec = c2w_mat[:3, 0]   # 相机右轴 (X)
    up_vec = c2w_mat[:3, 1]      # 相机上轴 (Y)
    pos = c2w_mat[:3, 3]
    # 随时间变化的偏移（需传入当前时间索引）
    t = np.sin(2 * np.pi * idx / total_frames)   # 示例
    new_pos = pos + right_amp * right_vec * t + up_amp * up_vec * t
    new_c2w = c2w_mat.copy()
    new_c2w[:3, 3] = new_pos
    return new_c2w


def get_smooth_path(cameras, num_frames=120, smoothing_window=5, zoom_range=(1.0, 1.5),rot_amp=(2.0, 2.0)):
    """
    Generates a smoothed trajectory based on training cameras.
    Strictly follows the training view distribution to avoid artifacts.
    """
    print("Computing smoothed path from training cameras...")
    
    # 1. 按时间排序相机
    sorted_cams = sorted(cameras, key=lambda x: x.time)
    
    # 2. 去重：相同时间的相机合并为一个关键帧
    time_dict = {}
    for cam in sorted_cams:
        t = cam.time
        if t not in time_dict:
            time_dict[t] = []
        time_dict[t].append(cam)
    
    unique_times = []
    unique_positions = []
    unique_quats = []  # 四元数 (w,x,y,z) 格式，用于平均
    
    for t, cam_list in time_dict.items():
        unique_times.append(t)
        # 计算平均位置
        avg_pos = np.mean([-c.R.T @ c.T for c in cam_list], axis=0)
        unique_positions.append(avg_pos)
        
        # 计算平均旋转（四元数加权平均，这里简单平均后归一化）
        quats = R.from_matrix([c.R.T for c in cam_list]).as_quat()  # (N,4) 格式 (x,y,z,w)
        # 四元数平均：求和后归一化
        avg_quat = np.mean(quats, axis=0)
        avg_quat = avg_quat / np.linalg.norm(avg_quat)
        unique_quats.append(avg_quat)
    
    unique_times = np.array(unique_times)
    unique_positions = np.array(unique_positions)
    unique_quats = np.array(unique_quats)
    
    # 3. 插值时间轴
    render_times = np.linspace(unique_times[0], unique_times[-1], num_frames)
    
    # 位置插值
    kind = 'linear' if len(unique_times) < 4 else 'cubic'
    pos_interp = interp1d(unique_times, unique_positions, axis=0, kind=kind, fill_value="extrapolate")
    smooth_positions = pos_interp(render_times)
    
    # 位置滑动平均平滑（可选）
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        for i in range(3):
            smooth_positions[:, i] = np.convolve(smooth_positions[:, i], kernel, mode='same')
            smooth_positions[0:smoothing_window//2, i] = smooth_positions[smoothing_window//2, i]
            smooth_positions[-smoothing_window//2:, i] = smooth_positions[-smoothing_window//2-1, i]
    
    # 旋转插值（SLERP）
    key_rots = R.from_quat(unique_quats)  # unique_quats 已经是 (x,y,z,w) 格式
    slerp = Slerp(unique_times, key_rots)
    smooth_rots = slerp(render_times)
    
    # 4. 构建渲染相机
    render_cameras = []
    ref_cam = sorted_cams[0]  # 任意参考相机
    fovx = ref_cam.FoVx
    fovy = ref_cam.FoVy
    width = ref_cam.image_width
    height = ref_cam.image_height
    
    for i in range(num_frames):
        t = render_times[i]
        pos = smooth_positions[i]
        rot_c2w = smooth_rots[i].as_matrix()
        
        # 转回 W2C
        R_w2c = rot_c2w.T
        T_w2c = -np.dot(R_w2c, pos)
        
        dummy_image = torch.zeros((3, height, width), dtype=torch.float32)
        
        cam = Camera(
            colmap_id=int(i + 10000),
            R=R_w2c,
            T=T_w2c,
            FoVx=fovx,
            FoVy=fovy,
            image=dummy_image,
            gt_alpha_mask=None,
            image_name=f"smooth_{i:05d}",
            uid=int(i + 10000),
            data_device=ref_cam.data_device,
            time=t
        )
        render_cameras.append(cam)
        
    '''
    final_cameras = []
    total = len(render_cameras)
    
    for idx, cam in enumerate(render_cameras):
        # 深拷贝，避免修改原始对象
        new_cam = deepcopy(cam)
        
        # ---- 变焦 (随时间线性变化) ----
        if zoom_range is not None:
            zoom_min, zoom_max = zoom_range
            alpha = idx / (total - 1) if total > 1 else 0.5
            zoom = zoom_min + (zoom_max - zoom_min) * alpha
            new_cam.FoVx = cam.FoVx / zoom   # 注意：原代码中 FoV 是弧度，除以 zoom 使视野变小（长焦）
            new_cam.FoVy = cam.FoVy / zoom
        
        if rot_amp is not None:
            pitch_amp, yaw_amp = rot_amp
            # 使用正弦波产生平滑变化，频率可调
            pitch = pitch_amp * np.sin(2 * np.pi * idx / 60)   # 周期 60 帧
            yaw   = yaw_amp   * np.cos(2 * np.pi * idx / 50)   # 周期 50 帧
            # 构建 C2W 旋转矩阵
            c2w_rot = new_cam.R.T  # 当前 C2W 旋转
            # 计算扰动旋转（在局部坐标系中：先俯仰绕 X，再偏航绕 Y）
            rot_pitch = R.from_euler('x', pitch, degrees=True).as_matrix()
            rot_yaw   = R.from_euler('y', yaw,   degrees=True).as_matrix()
            rot_perturb = rot_pitch @ rot_yaw  # 注意顺序
            # 应用扰动：新旋转 = 原旋转 @ 扰动（因为扰动是在相机坐标系下）
            new_c2w_rot = c2w_rot @ rot_perturb
            # 更新相机的 R, T
            new_cam.R = new_c2w_rot.T   # 转回 W2C
            new_cam.T = -new_cam.R @ c2w_rot @ new_cam.T   # 保持位置不变？实际上位置也应跟随旋转，但为了简单我们只旋转方向，位置不变。更好的方式是同时旋转位置向量，但幅度很小可忽略。这里我们保持位置不变（即绕相机光心旋转）。

            pos = -cam.R.T @ cam.T
            new_c2w = np.eye(4)
            new_c2w[:3, :3] = new_c2w_rot
            new_c2w[:3, 3] = pos
            new_cam.R = new_c2w[:3, :3].T
            new_cam.T = -new_cam.R @ new_c2w[:3, 3]
            
        final_cameras.append(new_cam)
        '''
    return render_cameras

def render_smooth_sets(dataset : ModelParams, hyperparam : ModelHiddenParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        train_cams = scene.getTestCameras()
        
        # Generate Smooth Path
        # smoothing_window=15 (aggressive smoothing for "Steadicam" look)
        smooth_cameras = get_smooth_path(train_cams, num_frames=330, smoothing_window=15)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        render_path = os.path.join(dataset.model_path, f"smooth_render_{iteration}")
        makedirs(render_path, exist_ok=True)
        
        print(f"Rendering {len(smooth_cameras)} frames to {render_path} ...")
        render_images = []
        
        for idx, view in enumerate(tqdm(smooth_cameras, desc="Rendering Smooth")):
             # 将时间戳转换为 float32 张量（与模型参数类型匹配）
            time_tensor = torch.tensor(view.time, dtype=torch.float32, device='cuda')

            # 临时替换 view.time 为张量（render 内部直接使用此属性）
            original_time = view.time
            view.time = time_tensor

            res = render(view, gaussians, pipeline, background)

            # 恢复原始时间戳（避免影响后续帧或日志）
            view.time = original_time

            rendering = res["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            img_np = (255 * np.clip(rendering.cpu().numpy(), 0, 1)).astype(np.uint8).transpose(1, 2, 0)
            render_images.append(img_np)
        video_path = os.path.join(dataset.model_path, 'smooth_video.mp4')
        imageio.mimwrite(video_path, render_images, fps=30)
        print(f"Video saved to {video_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Render Smooth Path Script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)
    
    args = get_combined_args(parser)
    print("Rendering Smooth Path for model:", args.model_path)
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
        
    safe_state(args.quiet)
    render_smooth_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args))