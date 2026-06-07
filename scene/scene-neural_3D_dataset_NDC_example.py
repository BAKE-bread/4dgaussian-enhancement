"""
场景数据加载器 - Neural 3D Dataset (NDC 归一化设备坐标) 示例。
支持训练/测试集划分、螺旋轨迹和圆弧轨迹生成、掩码与深度图加载。
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import glob

# ==============================================================================
# 超参数集中定义与注释
# ==============================================================================
# 默认图像尺寸（原始分辨率 1352x1014，经过下采样）
ORIGINAL_WIDTH = 1352
ORIGINAL_HEIGHT = 1014
# 近/远平面边界缩放因子（用于螺旋轨迹）
BD_FACTOR = 0.75
# 螺旋轨迹的 N_rots 参数（旋转圈数）
SPIRAL_N_ROTS = 2
# 圆弧轨迹摆动幅度（度）
ARC_ANGLE_DEG = 24.0
# 圆弧轨迹半径缩放因子（1.0 = 原始平均距离）
ARC_RADIUS_SCALE = 2.5
# 圆弧轨迹帧数
ARC_N_VIEWS = 300
# 螺旋轨迹帧数
SPIRAL_N_VIEWS = 300
# 训练 / 测试时每个相机的最大图像数
MAX_IMAGES_PER_CAM = 300
# 相对深度约束参数（用于螺旋轨迹）
SPIRAL_DT = 0.75
# 深度平滑参数
ZDELTA_FACTOR = 0.2
# 随机姿态生成数量（本文件未使用，但保留常量定义）
N_RANDOM_POSE = 1000
# 评估步长（保留原参数）
EVAL_STEP = 1
# 球体缩放（保留原参数）
SPHERE_SCALE = 1.0


class CameraInfo:
    """简易相机信息容器，用于存储单帧数据"""
    def __init__(self, uid, image, R, T, FovX, FovY, time, width, height, mask=None, depth=None):
        self.uid = uid
        self.image = image
        self.R = R
        self.T = T
        self.FovX = FovX
        self.FovY = FovY
        self.time = time
        self.width = width
        self.height = height
        self.mask = mask
        self.depth = depth


def normalize(v):
    """向量归一化，添加小量避免除零"""
    return v / (np.linalg.norm(v) + 1e-10)


def average_poses(poses):
    """计算所有相机位姿的平均中心与平均方向，返回平均位姿矩阵"""
    center = poses[..., 3].mean(0)
    z = normalize(poses[..., 2].mean(0))
    y_ = poses[..., 1].mean(0)
    x = normalize(np.cross(z, y_))
    y = np.cross(x, z)
    pose_avg = np.stack([x, y, z, center], 1)
    return pose_avg


def viewmatrix(z, up, pos):
    """根据视线方向、上向量和相机位置构建 4x4 视图矩阵"""
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=SPIRAL_N_ROTS, N=SPIRAL_N_VIEWS):
    """生成螺旋轨迹，用于 NeRF 风格的可视化"""
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]) * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=SPIRAL_N_VIEWS):
    """根据所有相机位姿生成螺旋轨迹参数，返回位姿数组"""
    c2w = average_poses(c2ws_all)
    up = normalize(c2ws_all[:, :3, 1].sum(0))
    dt = SPIRAL_DT
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
    zdelta = near_fars.min() * ZDELTA_FACTOR
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views)
    return np.stack(render_poses)


def get_arc_path(c2ws, N_views=ARC_N_VIEWS, axis='horizontal', angle_deg=ARC_ANGLE_DEG, radius_scale=ARC_RADIUS_SCALE):
    """
    生成始终指向场景中心的圆弧轨迹（Swing 运动）。
    参数：
        radius_scale: 1.0=原始平均距离，<1.0=拉近，>1.0=拉远
        axis: 'horizontal' 或 'vertical'
    """
    c2w_avg = average_poses(c2ws)
    center = c2w_avg[:3, 3]   # 场景中心
    up = c2w_avg[:3, 1]       # 上方向

    # 计算所有相机到中心的平均距离，并应用缩放
    dists = np.linalg.norm(c2ws[:, :3, 3] - center, axis=1)
    radius = np.mean(dists) * radius_scale

    render_poses = []
    rad = np.deg2rad(angle_deg)
    # 使用 sin 插值生成平滑的往复运动
    t = np.linspace(0, 1, N_views)
    angles = np.sin(t * 2 * np.pi) * rad   # 左右摆动

    R_avg = c2w_avg[:3, :3]

    for theta in angles:
        c = np.cos(theta)
        s = np.sin(theta)

        if axis == 'horizontal':
            local_pos = np.array([radius * s, 0, radius * c])
        elif axis == 'vertical':
            local_pos = np.array([0, radius * s, radius * c])
        else:
            raise ValueError("axis must be 'horizontal' or 'vertical'")

        world_pos = center + R_avg @ local_pos
        z_axis = normalize(world_pos - center)   # 看向中心

        pose = viewmatrix(z_axis, up, world_pos)
        render_poses.append(pose)

    return np.stack(render_poses)


def get_file_map(dir_path):
    """
    扫描目录，返回 {文件名前缀: 完整路径} 的映射。
    支持多种图像格式，自动跳过隐藏文件。
    """
    if not os.path.exists(dir_path):
        return {}
    valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    file_map = {}
    for f in os.listdir(dir_path):
        if f.startswith('.'):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            stem = os.path.splitext(f)[0]
            file_map[stem] = os.path.join(dir_path, f)
    return file_map


class Neural3D_NDC_Dataset(Dataset):
    """
    支持 LLFF 风格 poses_bounds.npy 格式的数据集。
    自动生成螺旋和圆弧两种测试轨迹。
    """
    def __init__(
        self,
        datadir,
        split="train",
        downsample=1.0,
        is_stack=True,
        cal_fine_bbox=False,
        N_vis=-1,
        time_scale=1.0,
        scene_bbox_min=[-1.0, -1.0, -1.0],
        scene_bbox_max=[1.0, 1.0, 1.0],
        N_random_pose=N_RANDOM_POSE,
        bd_factor=BD_FACTOR,
        eval_step=EVAL_STEP,
        eval_index=0,
        sphere_scale=SPHERE_SCALE,
    ):
        # 计算下采样后的图像尺寸
        self.img_wh = (int(ORIGINAL_WIDTH / 2 / downsample), int(ORIGINAL_HEIGHT / 2 / downsample))
        self.root_dir = datadir
        self.split = split
        self.downsample = ORIGINAL_WIDTH / self.img_wh[0]
        self.transform = T.ToTensor()
        self.eval_index = eval_index

        self.near = 0.0
        self.far = 1.0
        self.near_fars = np.array([[0.0, 1.0]])

        self.load_meta()

    def load_meta(self):
        """加载所有相机位姿、内外参，并构建图像路径映射"""
        poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        all_paths = sorted(glob.glob(os.path.join(self.root_dir, "cam*")))
        cam_folders = [p for p in all_paths if os.path.isdir(p) and not p.endswith(".mp4")]

        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        self.near_fars = poses_arr[:, -2:]
        H, W, focal = poses[0, :, -1]
        self.focal = [focal / self.downsample, focal / self.downsample]

        # LLFF 坐标系修正
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        self.poses = poses

        # === 生成两种测试轨迹 ===
        # 1. 螺旋轨迹 (Spiral) - 300帧
        self.val_poses_spiral = get_spiral(poses, self.near_fars, N_views=SPIRAL_N_VIEWS)
        # 2. 圆弧轨迹 (Circle/Arc) - 300帧
        # 可通过调整 ARC_RADIUS_SCALE 和 ARC_ANGLE_DEG 控制远近与摆动幅度
        self.val_poses_circle = get_arc_path(poses,
                                             N_views=ARC_N_VIEWS,
                                             axis='horizontal',
                                             angle_deg=ARC_ANGLE_DEG,
                                             radius_scale=ARC_RADIUS_SCALE)

        self.image_paths = []
        self.mask_paths = []
        self.depth_paths = []
        self.image_poses = []
        self.image_times = []

        for index, cam_folder in enumerate(cam_folders):
            if self.split == "train":
                pass
            elif self.split == "test":
                if index != self.eval_index:
                    continue

            img_dir = os.path.join(cam_folder, "images")
            mask_dir = os.path.join(cam_folder, "masks")
            depth_dir = os.path.join(cam_folder, "depth_maps")

            if not os.path.exists(img_dir):
                continue

            # === 建立文件名映射，忽略后缀差异 ===
            img_map = get_file_map(img_dir)
            mask_map = get_file_map(mask_dir)
            depth_map = get_file_map(depth_dir)

            sorted_stems = sorted(img_map.keys())

            if index == 0:
                print(f"[Dataset] Cam {index}: {len(sorted_stems)} images found.")

            if index >= len(self.poses):
                continue

            pose = self.poses[index]
            R = pose[:3, :3]
            R = -R
            R[:, 0] = -R[:, 0]
            T = -pose[:3, 3].dot(R)

            for i, stem in enumerate(sorted_stems):
                if i >= MAX_IMAGES_PER_CAM:
                    break

                # 图像路径
                self.image_paths.append(img_map[stem])

                # 掩码路径（精确匹配）
                if stem in mask_map:
                    self.mask_paths.append(mask_map[stem])
                else:
                    # 使用特殊标记，防止列表错位
                    self.mask_paths.append("DUMMY_MASK")

                # 深度图路径
                if stem in depth_map:
                    self.depth_paths.append(depth_map[stem])
                else:
                    self.depth_paths.append("DUMMY_DEPTH")

                self.image_poses.append((R, T))
                self.image_times.append(i / MAX_IMAGES_PER_CAM)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """返回 CameraInfo 对象，包含图像、掩码、深度和位姿"""
        print(self.image_paths[index])   # 保留原始调试输出
        img_path = self.image_paths[index]
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img_tensor = self.transform(img)
        except Exception as e:
            print(f"[Error] Failed to load image {img_path}: {e}")
            img_tensor = torch.zeros((3, self.img_wh[1], self.img_wh[0]))

        mask_tensor = None
        if index < len(self.mask_paths) and os.path.exists(self.mask_paths[index]):
            try:
                mask = Image.open(self.mask_paths[index]).convert('L')
                mask = mask.resize(self.img_wh, Image.NEAREST)
                mask_tensor = self.transform(mask)
                mask_tensor = (mask_tensor > 0.5).float()
            except Exception as e:
                # 只有当 mask 确实存在但读取失败时才报错
                if "DUMMY" not in self.mask_paths[index]:
                    print(f"[Error] Failed to load mask {self.mask_paths[index]}: {e}")

        depth_tensor = None
        if index < len(self.depth_paths) and os.path.exists(self.depth_paths[index]):
            try:
                depth = Image.open(self.depth_paths[index]).convert('L')
                depth = depth.resize(self.img_wh, Image.BILINEAR)
                depth_tensor = self.transform(depth)
            except Exception as e:
                if "DUMMY" not in self.depth_paths[index]:
                    print(f"[Error] Failed to load depth {self.depth_paths[index]}: {e}")

        R, T = self.image_poses[index]
        time = self.image_times[index]

        return CameraInfo(
            uid=index, image=img_tensor, R=R, T=T, FovX=None, FovY=None,
            time=time, width=self.img_wh[0], height=self.img_wh[1],
            mask=mask_tensor, depth=depth_tensor
        )