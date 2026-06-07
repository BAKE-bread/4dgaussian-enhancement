"""
场景数据加载器 - HyperNeRF 格式示例
支持加载图像、掩码、深度图，并构建 CameraInfo 对象。
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from torch.utils.data import Dataset
from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov
from utils.pose_utils import smooth_camera_poses
from scene.utils import Camera

# ==============================================================================
# 超参数集中定义与注释
# ==============================================================================
# 默认图像文件扩展名列表，用于 find_valid_path 查找
DEFAULT_IMG_EXTS = ['.png', '.jpg', '.jpeg']
# 视频路径生成时的采样间隔（每隔1帧取一帧）
VIDEO_SAMPLE_INTERVAL = 1
# 视频平滑插值步数
VIDEO_SMOOTH_STEPS = 10
# 视频路径最大长度限制
MAX_VIDEO_FRAMES = 500
# 缩放比例转换：当 ratio=1.0 时对应原图；实际缩放因子 scale = 1/ratio
# ratio 小于 1 时放大图像，大于 1 时缩小图像


class CameraInfo(NamedTuple):
    """
    相机信息结构体，包含图像、位姿、掩码、深度等必要数据。
    """
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time: float
    mask: np.array
    depth: np.array
    mask_path: str
    depth_path: str


def find_valid_path(base_dir, file_id, extensions=DEFAULT_IMG_EXTS):
    """
    在给定目录中查找指定 ID 的文件，支持多种扩展名。
    返回标准化后的绝对路径，若不存在则返回 None。
    """
    for ext in extensions:
        path = os.path.join(base_dir, f"{file_id}{ext}")
        # 核心修复：标准化路径，解决混合斜杠问题
        norm_path = os.path.normpath(path)
        if os.path.exists(norm_path):
            return norm_path
    return None


class Load_hyper_data(Dataset):
    """
    HyperNeRF 格式数据集加载器。
    支持 train / test / video 三种模式。
    """
    def __init__(self,
                 datadir,
                 ratio=1.0,
                 use_bg_points=False,
                 split="train"
                 ):
        datadir = os.path.expanduser(datadir)
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)

        # 从 scene.json 中读取近/远平面、坐标缩放和场景中心
        self.near = scene_json['near']
        self.far = scene_json['far']
        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']

        self.all_img = dataset_json['ids']
        self.val_id = dataset_json['val_ids']
        self.split = split

        # 如果没有 val_ids，则按步长 4 自动划分训练/测试集
        if len(self.val_id) == 0:
            self.i_train = np.array([i for i in np.arange(len(self.all_img)) if (i % 4 == 0)])
            self.i_test = self.i_train + 2
            self.i_test = self.i_test[:-1, ]
        else:
            self.train_id = dataset_json['train_ids']
            self.i_test = []
            self.i_train = []
            for i in range(len(self.all_img)):
                id_ = self.all_img[i]
                if id_ in self.val_id:
                    self.i_test.append(i)
                if id_ in self.train_id:
                    self.i_train.append(i)

        # 相机 ID 和时间归一化
        self.all_cam = [meta_json[i]['camera_id'] for i in self.all_img]
        self.all_time = [meta_json[i]['warp_id'] for i in self.all_img]
        max_time = max(self.all_time)
        self.all_time = [meta_json[i]['warp_id'] / max_time for i in self.all_img]
        self.max_time = max(self.all_time)
        self.min_time = min(self.all_time)

        # 加载所有相机的内参
        self.all_cam_params = []
        for im in self.all_img:
            camera = Camera.from_json(f'{datadir}/camera/{im}.json')
            self.all_cam_params.append(camera)

        self.all_img_origin = self.all_img

        # === 路径构建（使用 os.path.normpath 保证跨平台兼容性）===
        scale = int(1 / ratio)  # ratio=1.0 时 scale=1，ratio=0.5 时 scale=2
        self.img_dir = os.path.normpath(f'{datadir}/rgb/{scale}x/images')
        self.mask_dir = os.path.normpath(f'{datadir}/rgb/{scale}x/masks')
        self.depth_dir = os.path.normpath(f'{datadir}/rgb/{scale}x/depth_maps')

        self.all_img_paths = []
        self.all_mask_paths = []
        self.all_depth_paths = []

        print(f"[HyperLoader] Scanning files in {self.img_dir} ...")

        for i in self.all_img_origin:
            # 1. 图像路径
            p = find_valid_path(self.img_dir, i, DEFAULT_IMG_EXTS)
            if p is None:
                p = os.path.normpath(os.path.join(self.img_dir, f"{i}.png"))
                print(f"[Warning] Image {i} not found! Fallback to {p}")
            self.all_img_paths.append(p)

            # 2. 掩码路径（允许缺失）
            m = find_valid_path(self.mask_dir, i, DEFAULT_IMG_EXTS)
            self.all_mask_paths.append(m)

            # 3. 深度图路径（允许缺失）
            d = find_valid_path(self.depth_dir, i, DEFAULT_IMG_EXTS)
            self.all_depth_paths.append(d)

        # 兼容性：将 self.all_img 覆盖为实际路径列表
        self.all_img = self.all_img_paths

        self.h, self.w = self.all_cam_params[0].image_shape
        self.map = {}  # 缓存已加载的 CameraInfo

        # 尝试加载第一张图像作为尺寸参考
        if os.path.exists(self.all_img[0]):
            self.image_one = Image.open(self.all_img[0])
            self.image_one_torch = PILtoTorch(self.image_one, None).to(torch.float32)
        else:
            self.image_one_torch = torch.zeros((3, int(self.h), int(self.w)))

        self.image_mask = None

    def generate_video_path(self):
        """生成平滑的视频相机路径和时间戳"""
        self.select_video_cams = [item for i, item in enumerate(self.all_cam_params)
                                   if i % VIDEO_SAMPLE_INTERVAL == 0]
        self.video_path, self.video_time = smooth_camera_poses(self.select_video_cams, VIDEO_SMOOTH_STEPS)
        self.video_path = self.video_path[:MAX_VIDEO_FRAMES]
        self.video_time = self.video_time[:MAX_VIDEO_FRAMES]

    def __len__(self):
        if self.split == "train":
            return len(self.i_train)
        elif self.split == "test":
            return len(self.i_test)
        elif self.split == "video":
            return len(self.i_test)

    def __getitem__(self, index):
        if self.split == "train":
            return self.load_raw(self.i_train[index])
        elif self.split == "test":
            return self.load_raw(self.i_test[index])
        elif self.split == "video":
            return self.load_raw(index)

    def load_raw(self, idx):
        """根据索引加载原始数据，返回 CameraInfo 对象"""
        if idx in self.map.keys():
            return self.map[idx]

        camera = self.all_cam_params[idx]
        image_path = self.all_img_paths[idx]
        image_name = os.path.basename(image_path).split(".")[0]

        if os.path.exists(image_path):
            image = Image.open(image_path)
            w, h = image.size
            image = PILtoTorch(image, None).to(torch.float32)[:3, :, :]
        else:
            w, h = self.w, self.h
            image = torch.zeros((3, int(h), int(w)))

        time = self.all_time[idx]
        R = camera.orientation.T
        T = - camera.position @ R
        FovY = focal2fov(camera.focal_length, self.h)
        FovX = focal2fov(camera.focal_length, self.w)

        mask_path = self.all_mask_paths[idx]
        depth_path = self.all_depth_paths[idx]

        caminfo = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                             image_path=image_path, image_name=image_name,
                             width=w, height=h, time=time,
                             mask=None, depth=None,
                             mask_path=mask_path, depth_path=depth_path)
        self.map[idx] = caminfo
        return caminfo


def format_hyper_data(data_class, split):
    """
    将 Load_hyper_data 中的原始数据格式化为 CameraInfo 列表（不加载图像）。
    用于在场景初始化时批量处理相机信息。
    """
    if split == "train":
        data_idx = data_class.i_train
    elif split == "test":
        data_idx = data_class.i_test
    else:
        raise ValueError(f"Unsupported split: {split}")

    cam_infos = []
    for uid, index in tqdm(enumerate(data_idx), desc=f"Formatting {split} data"):
        camera = data_class.all_cam_params[index]
        time = data_class.all_time[index]
        R = camera.orientation.T
        T = - camera.position @ R
        FovY = focal2fov(camera.focal_length, data_class.h)
        FovX = focal2fov(camera.focal_length, data_class.w)

        image_path = data_class.all_img_paths[index]
        image_name = os.path.basename(image_path).split(".")[0]
        mask_path = data_class.all_mask_paths[index]
        depth_path = data_class.all_depth_paths[index]

        # 调试信息：输出第一个样本的路径（保留原调试逻辑）
        if uid == 0:
            print(f"[Format Debug] First Sample Paths (Normalized):")
            print(f"  Img: {image_path}")
            print(f"  Msk: {mask_path}")
            print(f"  Dep: {depth_path}")

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                              image_path=image_path, image_name=image_name,
                              width=int(data_class.w), height=int(data_class.h),
                              time=time, mask=None, depth=None,
                              mask_path=mask_path, depth_path=depth_path)
        cam_infos.append(cam_info)
    return cam_infos