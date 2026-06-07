import os
import torch
import numpy as np
import copy
import imageio
import torchvision
from tqdm import tqdm
from argparse import ArgumentParser

# 导入 4DGaussians 相关依赖
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

def apply_offset_to_pose(base_cam, pitch_deg, yaw_deg, zoom_factor):
    """
    在原生相机位姿的基础上，施加局部的俯仰、偏航和缩放偏移。
    """
    # 1. 提取安全的 World-to-Camera 矩阵
    w2c = base_cam.world_view_transform.transpose(0, 1).cpu().numpy()
    R_w2c = w2c[:3, :3]
    
    # 2. 提取世界坐标系下的相机光心
    C_world = base_cam.camera_center.cpu().numpy()

    # 3. 计算局部旋转矩阵
    rad_p = np.radians(pitch_deg)
    rad_y = np.radians(yaw_deg)

    # 绕局部 X 轴 (Pitch - 上下俯仰)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rad_p), -np.sin(rad_p)],
        [0, np.sin(rad_p), np.cos(rad_p)]
    ])
    
    # 绕局部 Y 轴 (Yaw - 左右偏航)
    Ry = np.array([
        [np.cos(rad_y), 0, -np.sin(rad_y)],
        [0, 1, 0],
        [np.sin(rad_y), 0, np.cos(rad_y)]
    ])

    # 将局部旋转叠加到原来的 R 上
    R_local = Ry @ Rx
    R_w2c_new = R_local @ R_w2c

    # 4. 保持光心原地旋转，反推新的平移向量 T
    T_w2c_new = -np.dot(R_w2c_new, C_world)

    # 5. 克隆原相机 (完美继承 time, uid, 原始长宽等属性)
    v_cam = copy.copy(base_cam)
    
    # 注意：3DGS 需要传入的 R 是 R_w2c 的转置
    v_cam.R = R_w2c_new.transpose()
    v_cam.T = T_w2c_new
    
    # 缩放处理 (FOV 变小 = Zoom In)
    v_cam.FoVx = base_cam.FoVx / zoom_factor
    v_cam.FoVy = base_cam.FoVy / zoom_factor

    # 6. 重新生成 3DGS 底层强依赖的投影张量
    v_cam.world_view_transform = torch.tensor(getWorld2View2(v_cam.R, v_cam.T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
    v_cam.projection_matrix = getProjectionMatrix(znear=v_cam.znear, zfar=v_cam.zfar, fovX=v_cam.FoVx, fovY=v_cam.FoVy).transpose(0,1).cuda()
    v_cam.full_proj_transform = (v_cam.world_view_transform.unsqueeze(0).bmm(v_cam.projection_matrix.unsqueeze(0))).squeeze(0)
    v_cam.camera_center = v_cam.world_view_transform.inverse()[3, :3]

    return v_cam

def render_temporal_offset_video(dataset, hyper, pipeline, args):
    with torch.no_grad():
        print("Loading Gaussians and 4D Deformation Network...")
        gaussians = GaussianModel(dataset.sh_degree, hyper)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        cam_type = scene.dataset_type

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 优先读取视频轨道的相机，若无则读取测试集相机
        views = scene.getVideoCameras()
        if not views or len(views) == 0:
            views = scene.getTestCameras()
            
        # 确保按时间顺序排列
        if hasattr(views[0], 'time'):
            views = sorted(views, key=lambda x: x.time)

        N = len(views)
        print(f"Total temporal frames found: {N}")

        # 生成连续变化的参数矩阵 (伴随整个时序视频)
        yaw_offsets = np.linspace(args.yaw_start, args.yaw_end, N)
        pitch_offsets = np.linspace(args.pitch_start, args.pitch_end, N)
        zoom_factors = np.linspace(args.zoom_start, args.zoom_end, N)

        # 建立输出目录
        out_name = f"temporal_offset_p{args.pitch_start}to{args.pitch_end}_y{args.yaw_start}to{args.yaw_end}_z{args.zoom_start}to{args.zoom_end}"
        render_path = os.path.join(dataset.model_path, out_name)
        os.makedirs(render_path, exist_ok=True)
        
        render_images = []
        to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

        print("Rendering sequence with dynamic offsets...")
        for i, view in enumerate(tqdm(views)):
            # 拿到当前帧原生的动态 time 和 pose，注入我们计算好的增量偏移
            v_cam = apply_offset_to_pose(
                base_cam=view,
                pitch_deg=pitch_offsets[i],
                yaw_deg=yaw_offsets[i],
                zoom_factor=zoom_factors[i]
            )

            # 渲染当前帧
            rendering = render(v_cam, gaussians, pipeline, background, cam_type=cam_type)["render"]
            
            # 转换为 uint8 格式准备导出视频
            frame_rgb = to8b(rendering).transpose(1, 2, 0)
            render_images.append(frame_rgb)
            
            # 同步保存单帧图像
            torchvision.utils.save_image(rendering, os.path.join(render_path, f"{i:05d}.png"))
            
        # 导出 MP4 视频
        video_path = os.path.join(render_path, "dynamic_offset.mp4")
        imageio.mimwrite(video_path, render_images, fps=30)
        print(f"\nSuccess! Video and images saved to: {render_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Render Temporal Offset Video for 4D Gaussians")
    model = ModelParams(parser, sentinel=True) # sentinel=True 确保读取原始配置
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--quiet", action="store_true")
    
    # 连续视角的起始与终止控制参数
    parser.add_argument("--yaw_start", type=float, default=0.0, help="Start horizontal angle offset")
    parser.add_argument("--yaw_end", type=float, default=0.0, help="End horizontal angle offset")
    parser.add_argument("--pitch_start", type=float, default=0.0, help="Start vertical angle offset")
    parser.add_argument("--pitch_end", type=float, default=0.0, help="End vertical angle offset")
    parser.add_argument("--zoom_start", type=float, default=1.0, help="Start zoom factor")
    parser.add_argument("--zoom_end", type=float, default=1.0, help="End zoom factor")

    args = get_combined_args(parser)
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    if args.quiet:
        import warnings
        warnings.filterwarnings("ignore")

    render_temporal_offset_video(model.extract(args), hyperparam.extract(args), pipeline.extract(args), args)