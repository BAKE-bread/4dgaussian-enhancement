from torch.utils.data import Dataset
from scene.cameras import Camera
import torch
import os
import sys
from PIL import Image
from torchvision import transforms as T
import numpy as np
from utils.graphics_utils import focal2fov

class FourDGSdataset(Dataset):
    def __init__(self, dataset, args, dataset_type):
        self.dataset = dataset
        self.args = args
        self.dataset_type = dataset_type
        self.transform = T.ToTensor()
        self.debug_count = 0 

    def __getitem__(self, index):
        if self.dataset_type != "PanopticSports":
            caminfo = self.dataset[index]
            
            image_tensor = None
            mask_tensor = None
            depth_tensor = None

            # ------------------------------------------------------------------
            # 1. 图像处理 (Image)
            # ------------------------------------------------------------------
            if caminfo.image is None:
                # Lazy Loading
                try:
                    img = Image.open(caminfo.image_path).convert('RGB')
                    if img.size[0] != caminfo.width or img.size[1] != caminfo.height:
                        img = img.resize((caminfo.width, caminfo.height), Image.LANCZOS)
                    image_tensor = self.transform(img)
                except Exception as e:
                    print(f"[Error] RGB load failed at {caminfo.image_path}: {e}", flush=True)
                    image_tensor = torch.zeros((3, caminfo.height, caminfo.width))
            else:
                # Already Loaded
                image_tensor = caminfo.image

            # ------------------------------------------------------------------
            # 2. 掩膜处理 (Mask) - 合并两种加载逻辑
            # ------------------------------------------------------------------
            if caminfo.mask is not None:
                mask_tensor = caminfo.mask
            elif caminfo.mask_path and os.path.exists(caminfo.mask_path):
                try:
                    mask = Image.open(caminfo.mask_path).convert('L')
                    mask = mask.resize((caminfo.width, caminfo.height), Image.NEAREST)
                    mask_tensor = self.transform(mask)
                    
                    # 二值化：兼容不同数值范围 (0~1 或 0~255)
                    if mask_tensor.max() <= 1.0:
                        # 已经是 0~1 范围，使用 0.5 阈值
                        mask_tensor = (mask_tensor > 0.5).float()
                    else:
                        # 可能是 0~255 范围，使用 127.5 阈值
                        mask_tensor = (mask_tensor > 127.5).float()
                except Exception as e:
                    print(f"[Warning] Mask load failed: {e} | Path: {caminfo.mask_path}", flush=True)

            # ------------------------------------------------------------------
            # 3. 深度图处理 (Depth) - 独立加载，兼容两种写法
            # ------------------------------------------------------------------
            if caminfo.depth is not None:
                depth_tensor = caminfo.depth
            elif caminfo.depth_path and os.path.exists(caminfo.depth_path):
                try:
                    depth = Image.open(caminfo.depth_path)
                    # 转换为灰度图（支持 24-bit RGB 深度图）
                    depth = depth.convert('L')
                    depth = depth.resize((caminfo.width, caminfo.height), Image.BILINEAR)
                    depth_tensor = self.transform(depth)
                except Exception as e:
                    # 静默失败，不影响主流程
                    pass

            # ------------------------------------------------------------------
            # Debug 信息 (可选，便于调试)
            # ------------------------------------------------------------------
            if self.debug_count < 5:
                if mask_tensor is None and caminfo.mask_path and os.path.exists(caminfo.mask_path):
                    print(f"\n[Dataset Debug] Worker {os.getpid()} - Sample {index} MASK FAILED: {caminfo.mask_path}", flush=True)
                    self.debug_count += 1
                elif mask_tensor is not None and self.debug_count < 3:
                    print(f"[Dataset Debug] Worker {os.getpid()} - Sample {index} MASK LOADED", flush=True)
                    self.debug_count += 1

            # ------------------------------------------------------------------
            # 返回 Camera 对象（保持输出格式）
            # ------------------------------------------------------------------
            return Camera(colmap_id=index, R=caminfo.R, T=caminfo.T, FoVx=caminfo.FovX, FoVy=caminfo.FovY, 
                          image=image_tensor, 
                          gt_alpha_mask=mask_tensor,
                          image_name=caminfo.image_name, uid=index, 
                          data_device=torch.device("cuda"), time=caminfo.time,
                          mask=mask_tensor, depth=depth_tensor)
        else:
            # PanopticSports 数据集直接返回原数据
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)