import os
import json
import argparse
import bisect
import numpy as np
from pathlib import Path
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R, Slerp

# ==========================================
# 工具函数与核心数据结构
# ==========================================

def qvec2rotmat(qvec):
    """将 COLMAP 的四元数 [w, x, y, z] 转换为 3x3 旋转矩阵"""
    q0, q1, q2, q3 = qvec
    return np.array([
        [1 - 2 * q2**2 - 2 * q3**2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
        [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1**2 - 2 * q3**2, 2 * q2 * q3 - 2 * q0 * q1],
        [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1**2 - 2 * q2**2]
    ])

class Camera:
    """Nerfies 标准相机类"""
    def __init__(self, orientation, position, focal_length, principal_point, image_size, 
                 pixel_aspect_ratio=1.0, radial_distortion=None, tangential_distortion=None):
        self.orientation = orientation
        self.position = position
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.image_size = image_size
        self.pixel_aspect_ratio = pixel_aspect_ratio
        self.radial_distortion = radial_distortion if radial_distortion is not None else np.zeros(3)
        self.tangential_distortion = tangential_distortion if tangential_distortion is not None else np.zeros(2)

    def to_json(self):
        return {
            'orientation': self.orientation.tolist(),
            'position': self.position.tolist(),
            'focal_length': float(self.focal_length),
            'principal_point': self.principal_point.tolist(),
            'skew': 0.0,
            'pixel_aspect_ratio': float(self.pixel_aspect_ratio),
            'radial_distortion': self.radial_distortion.tolist(),
            'tangential_distortion': self.tangential_distortion.tolist(),
            'image_size': self.image_size.tolist()
        }

# ==========================================
# COLMAP 纯 Python 解析器 
# [踩坑点1] 彻底抛弃 pycolmap 库，避免因为缺失 Track 关联和版本问题导致的崩溃
# ==========================================

def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(tuple(map(float, elems[4:])))
            
            # [踩坑点2] 动态兼容各种相机模型，防止 PINHOLE 没有畸变参数 k1, k2 导致报错
            if model == "SIMPLE_PINHOLE":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
                k1, k2, p1, p2 = 0.0, 0.0, 0.0, 0.0
            elif model == "PINHOLE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                k1, k2, p1, p2 = 0.0, 0.0, 0.0, 0.0
            elif model == "OPENCV":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                k1, k2, p1, p2 = params[4], params[5], params[6], params[7]
            else:
                # 兼容其他模型，默认无畸变
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                k1, k2, p1, p2 = 0.0, 0.0, 0.0, 0.0

            cameras[camera_id] = {
                'width': width, 'height': height,
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2
            }
    return cameras

def read_images_text(path):
    images = {}
    with open(path, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#") or not line:
                i += 1
                continue
            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            camera_id = int(elems[8])
            name = elems[9]
            
            images[image_id] = {'qvec': qvec, 'tvec': tvec, 'camera_id': camera_id, 'name': name}
            
            # 跳过特征点行（即便它是空的也不影响我们的解析）
            i += 2 
    return images

def read_points3D_text(path):
    points = []
    if not os.path.exists(path):
        return np.zeros((0, 3))
    
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            elems = line.split()
            xyz = np.array(tuple(map(float, elems[1:4])))
            points.append(xyz)
    return np.array(points)

# ==========================================
# 核心处理流程
# ==========================================

def process_colmap_to_nerfies(colmap_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("1. 正在解析 COLMAP 数据...")
    cam_data = read_cameras_text(os.path.join(colmap_dir, "cameras.txt"))
    img_data = read_images_text(os.path.join(colmap_dir, "images.txt"))
    points3d = read_points3D_text(os.path.join(colmap_dir, "points3D.txt"))
    
    # 构建 Nerfies Camera 字典
    camera_dict = {}
    image_ids = sorted(list(img_data.keys()))
    
    for img_id in image_ids:
        img = img_data[img_id]
        cam = cam_data[img.camera_id]
        
        # 将 COLMAP (W2C) 转为 相机位姿 (C2W)
        R_w2c = qvec2rotmat(img['qvec'])
        t_w2c = img['tvec']
        position = -(R_w2c.T @ t_w2c)
        orientation = R_w2c
        
        # 提取去掉后缀的文件名作为 key (例如 "000001.png" -> "000001")
        item_id = img['name'].split('.')[0] 
        
        camera_dict[item_id] = Camera(
            orientation=orientation,
            position=position,
            focal_length=cam['fx'],
            pixel_aspect_ratio=cam['fy'] / cam['fx'],
            principal_point=np.array([cam['cx'], cam['cy']]),
            image_size=np.array([cam['width'], cam['height']]),
            radial_distortion=np.array([cam['k1'], cam['k2'], 0.0]),
            tangential_distortion=np.array([cam['p1'], cam['p2']])
        )
    
    item_ids = sorted(list(camera_dict.keys()))
    print(f"✅ 加载成功: {len(item_ids)} 个相机, {len(points3d)} 个 3D 点。")

    # ==========================================
    # 场景计算 (BBox, Scale, Center, Near/Far)
    # ==========================================
    print("2. 正在计算场景边界与缩放...")
    cam_positions = np.array([camera_dict[i].position for i in item_ids])
    
    # 容错处理：如果没有3D点，仅使用相机位置计算边界
    if len(points3d) > 0:
        all_points = np.concatenate([points3d, cam_positions], axis=0)
    else:
        all_points = cam_positions
        
    lower = np.percentile(all_points, 1, axis=0) # 使用 1% 和 99% 剔除极端离群飞点
    upper = np.percentile(all_points, 99, axis=0)
    bbox_corners = np.stack([lower, upper])
    
    scene_center = np.mean(bbox_corners, axis=0)
    scene_scale = 1.0 / np.sqrt(np.sum((upper - lower) ** 2))
    
    # 粗略估算 near 和 far
    # [踩坑点4] 由于没有2D特征，我们用相机到场景中心的距离估算视距
    dist_to_center = np.linalg.norm(cam_positions - scene_center, axis=1)
    avg_dist = np.mean(dist_to_center)
    scene_radius = np.linalg.norm(upper - lower) / 2
    
    near = max(0.01, avg_dist - scene_radius) * scene_scale
    far = (avg_dist + scene_radius * 2) * scene_scale
    
    scene_json = {
        'scale': float(scene_scale),
        'center': scene_center.tolist(),
        'bbox': bbox_corners.tolist(),
        'near': float(near),
        'far': float(far)
    }
    with (out_dir / 'scene.json').open('w') as f:
        json.dump(scene_json, f, indent=2)

    # ==========================================
    # 生成时序电影级漫游轨迹 (Cinematic Walkthrough)
    # [踩坑点3] 彻底剔除 JAX 和 look_at，使用球面插值 SLERP 保证 Roll 轴稳定
    # ==========================================
    print("3. 正在生成平滑漫游轨迹 (Cinematic Smooth Path)...")
    
    origins = cam_positions
    orientations = [camera_dict[i].orientation for i in item_ids]
    
    # 剔除相邻完全重复的帧，防止 B-Spline 或 Slerp 除以零报错
    valid_idx = [0]
    for i in range(1, len(origins)):
        if np.linalg.norm(origins[i] - origins[valid_idx[-1]]) > 1e-5:
            valid_idx.append(i)
            
    v_origins = origins[valid_idx]
    v_orientations = np.array([orientations[i] for i in valid_idx])
    
    # 时间序列
    t_v = np.linspace(0, 1, len(v_origins))
    
    # 1. 位置平滑 (B-Spline)
    smooth_factor = 0.5 * (1.0 / scene_scale) # 根据场景尺度自适应平滑力度
    tck_pos, _ = splprep(v_origins.T, u=t_v, s=smooth_factor)
    
    # 生成新视角的帧数 (例如帧率扩充 1.5 倍)
    num_smooth_frames = int(len(item_ids) * 1.5)
    t_new = np.linspace(0, 1, num=num_smooth_frames)
    smooth_origins = np.array(splev(t_new, tck_pos)).T
    
    # 2. 旋转平滑 (SLERP)
    # Scipy 的 Rotation 使用 scipy 特有的接口，注意 COLMAP 矩阵进，矩阵出
    rots = R.from_matrix(v_orientations)
    slerp_func = Slerp(t_v, rots)
    smooth_rots = slerp_func(t_new).as_matrix()
    
    # 构造虚拟相机集合
    cinematic_cameras = []
    template_cam = camera_dict[item_ids[0]]
    
    for pos, rot in zip(smooth_origins, smooth_rots):
        cam = Camera(
            orientation=rot,
            position=pos,
            focal_length=template_cam.focal_length,
            principal_point=template_cam.principal_point,
            image_size=template_cam.image_size,
            pixel_aspect_ratio=template_cam.pixel_aspect_ratio,
            radial_distortion=template_cam.radial_distortion,
            tangential_distortion=template_cam.tangential_distortion
        )
        cinematic_cameras.append(cam)

    # ==========================================
    # 保存所有元数据与 JSON
    # ==========================================
    print("4. 正在保存所有 Dataset 结构与 JSON 文件...")
    
    # 数据集划分 (每20帧一个验证集)
    val_ids = item_ids[::20]
    train_ids = sorted(list(set(item_ids) - set(val_ids)))
    
    with (out_dir / 'dataset.json').open('w') as f:
        json.dump({
            'count': len(item_ids),
            'num_exemplars': len(train_ids),
            'ids': item_ids,
            'train_ids': train_ids,
            'val_ids': val_ids,
        }, f, indent=2)

    # Metadata
    metadata_json = {}
    for i, img_id in enumerate(train_ids):
        metadata_json[img_id] = {'warp_id': i, 'appearance_id': i, 'camera_id': 0}
    for i, img_id in enumerate(val_ids):
        idx = bisect.bisect_left(train_ids, img_id)
        metadata_json[img_id] = {'warp_id': idx, 'appearance_id': idx, 'camera_id': 0}
    
    with (out_dir / 'metadata.json').open('w') as f:
        json.dump(metadata_json, f, indent=2)

    # 原始相机导出
    camera_dir = out_dir / 'camera'
    camera_dir.mkdir(parents=True, exist_ok=True)
    for item_id, camera in camera_dict.items():
        with (camera_dir / f'{item_id}.json').open('w') as f:
            json.dump(camera.to_json(), f, indent=2)
            
    # 测试相机导出 (用于渲染)
    test_camera_dir = out_dir / 'camera-paths' / 'cinematic-forward'
    test_camera_dir.mkdir(parents=True, exist_ok=True)
    for i, camera in enumerate(cinematic_cameras):
        with (test_camera_dir / f'{i:06d}.json').open('w') as f:
            json.dump(camera.to_json(), f, indent=2)

    print(f"🎉 处理完成！所有数据已成功保存至: {out_dir}")
    print(f"👉 下一步: 请将 render_orbit.py 中的路径指向 'camera-paths/cinematic-forward'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nerfies 数据流纯 Python 处理脚本")
    parser.add_argument("--colmap_dir", type=str, required=True, help="包含 cameras.txt 等的稀疏重建目录")
    parser.add_argument("--out_dir", type=str, required=True, help="Nerfies 数据集输出目录")
    args = parser.parse_args()
    
    process_colmap_to_nerfies(args.colmap_dir, args.out_dir)