#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

'''
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", time = 0,
                 mask = None, depth=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        self.original_image = image.clamp(0.0, 1.0)[:3,:,:]
        # breakpoint()
        # .to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
            # .to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))
                                                #   , device=self.data_device)
        self.depth = depth
        self.mask = mask
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        # .cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        # .cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
'''

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", time = 0,
                 mask = None, depth=None): # 确保参数里有 mask 和 depth
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        self.data_device = torch.device(data_device)

        self.original_image = image.clamp(0.0, 1.0)[:3,:,:]
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # 这里的 gt_alpha_mask 是 dataset 传进来的 mask
        if gt_alpha_mask is not None:
            # 将图像背景变黑，这对于 L1 Loss 很重要
            # self.original_image = gt_alpha_mask
            self.gt_alpha_mask = gt_alpha_mask # 存储一份，虽然外面可以通过 mask 访问
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))
            self.gt_alpha_mask = None

        # 存储深度图和Mask供 Loss 使用
        self.depth = depth
        self.mask = mask # 明确存储 Mask
        
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def rotate_view(self, yaw_deg: float, pitch_deg: float):
        """
        旋转相机视角（兼容hyperferf/nerfies格式）
        :param yaw_deg: 水平旋转角度（左右），范围±15度
        :param pitch_deg: 垂直旋转角度（上下），范围±15度
        """
        # 角度转弧度
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)

        # 构建旋转矩阵（yaw: 绕Y轴, pitch: 绕X轴）
        rot_yaw = R.from_euler('y', yaw, degrees=False).as_matrix()
        rot_pitch = R.from_euler('x', pitch, degrees=False).as_matrix()
        rot_total = rot_yaw @ rot_pitch

        # 应用旋转到相机外参
        self.R = self.R @ rot_total
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def zoom_view(self, zoom_factor: float):
        """
        缩放相机视角（zoom in/out）
        :param zoom_factor: 缩放系数，>1为放大，<1为缩小
        """
        # 调整焦距（FoV和焦距成反比）
        self.FoVx = self.FoVx / zoom_factor
        self.FoVy = self.FoVy / zoom_factor
        
        # 更新投影矩阵
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, 
            fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time

