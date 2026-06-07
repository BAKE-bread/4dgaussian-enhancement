# render_orbit.py
import torch
from scene import Scene
import os
import json
import glob
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov
import imageio
from scene.cameras import Camera 

def load_orbit_cameras(dataset_args, source_path):
    """
    加载并解析 camera-paths/orbit-mild 下的相机参数
    修复坐标系问题：OpenCV -> OpenGL
    """
    cameras = []
    
    # 1. 读取场景归一化参数 (scene.json)
    scene_json_path = os.path.join(source_path, "scene.json")
    if os.path.exists(scene_json_path):
        with open(scene_json_path, 'r') as f:
            scene_data = json.load(f)
        scene_scale = scene_data['scale']
        scene_center = np.array(scene_data['center'])
        print(f"[Orbit Loader] Scene normalization: scale={scene_scale}, center={scene_center}")
    else:
        print("[Orbit Loader] Warning: scene.json not found. Assuming identity scale.")
        scene_center = np.zeros(3)
        scene_scale = 1.0

    # 2. 获取 orbit json 文件
    orbit_path = os.path.join(source_path, "camera-paths", "orbit-mild")
    if not os.path.exists(orbit_path):
        # 尝试查找其他路径
        print(f"Path not found: {orbit_path}, trying to search recursively...")
        json_files = sorted(glob.glob(os.path.join(source_path, "camera-paths", "*", "*.json")))
    else:
        json_files = sorted(glob.glob(os.path.join(orbit_path, "*.json")))
        
    if len(json_files) == 0:
        raise ValueError(f"No JSON files found in {source_path}/camera-paths")

    print(f"[Orbit Loader] Found {len(json_files)} camera poses.")

    # 定义坐标转换矩阵 (OpenCV -> OpenGL)
    Cv2Gl = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)

    for idx, json_file in enumerate(json_files):
        with open(json_file, 'r') as f:
            cam_data = json.load(f)

        # --- 解析内参 ---
        W = int(cam_data['image_size'][0])
        H = int(cam_data['image_size'][1])
        
        focal = cam_data['focal_length']
        if isinstance(focal, list):
            focal = focal[0]
        fl_x = float(focal)
        fl_y = float(focal)
        
        FovY = focal2fov(fl_x, H)
        FovX = focal2fov(fl_y, W)

        # --- 解析外参 ---
        # Position: Camera Center in World
        position = np.array(cam_data['position']) 
        
        # Orientation: Rotation (Camera to World)
        orientation = np.array(cam_data['orientation']) 

        # 空间位置归一化
        position = (position - scene_center) * scene_scale
        
        # 坐标系旋转 (OpenCV -> OpenGL)
        R_c2w = orientation @ Cv2Gl
        
        # 转为 W2C (World to Camera)
        R_w2c = R_c2w.T
        T_w2c = -np.dot(R_w2c, position)

        # --- 时间处理 ---
        # 将整个序列映射到 [0, 1]
        time_val = idx / (len(json_files) - 1) if len(json_files) > 1 else 0.0

        # --- 创建相机对象 ---
        image = torch.zeros((3, H, W), dtype=torch.float32)
        
        cam = Camera(
            colmap_id=idx,
            R=R_w2c,
            T=T_w2c,
            FoVx=FovX,
            FoVy=FovY,
            image=image,
            gt_alpha_mask=None,
            image_name=os.path.basename(json_file),
            uid=idx,
            data_device=dataset_args.data_device,
            time=time_val
        )
        cameras.append(cam)

    return cameras

def render_orbit_sets(dataset : ModelParams, hyperparam : ModelHiddenParams, iteration : int, pipeline : PipelineParams, source_path : str):
    with torch.no_grad():
        # 初始化 GaussianModel
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        
        # 初始化场景
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # === 调试：打印训练相机的第一个位姿 ===
        # 这有助于对比 orbit 相机是否在相似的尺度和方向上
        try:
            train_cam = scene.getTrainCameras()[0]
            print("\n[Debug] First Train Camera Pose:")
            print(f"  Center: {train_cam.camera_center.numpy()}")
            print(f"  R (row 0): {train_cam.R[0, :].numpy()}")
            print("-" * 30)
        except:
            print("[Debug] Could not load train cameras for debug comparison.")
        # ==================================

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 加载 Orbit 相机
        orbit_cameras = load_orbit_cameras(dataset, source_path)
        
        # === 调试：打印 Orbit 相机的第一个位姿 ===
        print("[Debug] First Orbit Camera Pose:")
        print(f"  Center: {orbit_cameras[0].camera_center.numpy()}")
        print(f"  R (row 0): {orbit_cameras[0].R[0, :]}")
        print(f"  Time: {orbit_cameras[0].time}")
        print("-" * 30 + "\n")
        # ==================================

        render_path = os.path.join(dataset.model_path, "orbit_render")
        makedirs(render_path, exist_ok=True)
        
        render_images = []
        print(f"Rendering {len(orbit_cameras)} frames to {render_path} ...")

        for idx, view in enumerate(tqdm(orbit_cameras, desc="Rendering Orbit")):
            rendering = render(view, gaussians, pipeline, background)["render"]
            
            # 保存
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            
            # 收集用于视频
            img_np = (255 * np.clip(rendering.cpu().numpy(), 0, 1)).astype(np.uint8).transpose(1, 2, 0)
            render_images.append(img_np)
            
        # 生成视频
        video_path = os.path.join(dataset.model_path, 'orbit_video.mp4')
        try:
            imageio.mimwrite(video_path, render_images, fps=30)
            print(f"Done. Video saved to {video_path}")
        except Exception as e:
            print(f"Could not save video via imageio: {e}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Orbit Rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)
    
    args = get_combined_args(parser)
    print("Rendering Orbit Path for model:", args.model_path)
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
        
    safe_state(args.quiet)

    render_orbit_sets(
        model.extract(args), 
        hyperparam.extract(args), 
        args.iteration, 
        pipeline.extract(args),
        args.source_path
    )