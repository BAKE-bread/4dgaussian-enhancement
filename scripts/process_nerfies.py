import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# 确保 nerfies 在你的 PYTHONPATH 中
from nerfies.camera import Camera

def qvec2rotmat(qvec):
    """将四元数转换为旋转矩阵"""
    q0, q1, q2, q3 = qvec
    return np.array([[1 - 2 * q2**2 - 2 * q3**2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],[2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1**2 - 2 * q3**2, 2 * q2 * q3 - 2 * q0 * q1],[2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1**2 - 2 * q2**2]
    ])

class SceneManager:
    """脱离 pycolmap 的纯 Python 解析器，直接读取 cameras.txt 和 images.txt"""
    @classmethod
    def from_colmap_dir(cls, colmap_path, min_track_length=5):
        colmap_path = Path(colmap_path)
        
        cam_txt = colmap_path / "cameras.txt"
        img_txt = colmap_path / "images.txt"
        pts_txt = colmap_path / "points3D.txt"

        if not cam_txt.exists() or not img_txt.exists():
            raise FileNotFoundError(
                f"COLMAP 文本文件未找到：{colmap_path}\n"
                f"请确保你的模型已导出为 .txt 格式 (cameras.txt, images.txt, points3D.txt)。"
            )

        # 1. 解析 cameras.txt
        cameras = {}
        with open(cam_txt, "r") as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.strip().split()
                if not parts: continue
                cam_id = int(parts[0])
                model = parts[1]
                width, height = int(parts[2]), int(parts[3])
                params = [float(x) for x in parts[4:]]
                cameras[cam_id] = {"model": model, "width": width, "height": height, "params": params}

        # 2. 解析 images.txt，保留全部，不做任何删减
        images = {}
        with open(img_txt, "r") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#") or not line:
                i += 1
                continue
            parts = line.split()
            img_id = int(parts[0])
            qvec = np.array([float(x) for x in parts[1:5]])
            tvec = np.array([float(x) for x in parts[5:8]])
            cam_id = int(parts[8])
            name = parts[9]
            images[img_id] = {"qvec": qvec, "tvec": tvec, "camera_id": cam_id, "name": name}
            i += 2  # 跳过 2D points 所在行

        # 3. 解析 points3D.txt
        points =[]
        if pts_txt.exists():
            with open(pts_txt, "r") as f:
                for line in f:
                    if line.startswith("#"): continue
                    parts = line.strip().split()
                    if not parts: continue
                    xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                    # 计算 track 长度：前 8 列为特征数据，之后每两列 (IMAGE_ID, POINT2D_IDX) 代表一个 track
                    track_len = (len(parts) - 8) // 2
                    if track_len >= min_track_length:
                        points.append(xyz)
        
        points = np.array(points) if points else np.zeros((0, 3))

        # 4. 组装为 Nerfies Cameras
        sfm_cameras = {}
        for img_id, img_data in images.items():
            cam_data = cameras[img_data["camera_id"]]
            img_name = img_data["name"].split('.')[0]

            rotmat = qvec2rotmat(img_data["qvec"])
            tvec = img_data["tvec"]
            camera_position = -(tvec @ rotmat)  # 等价于 -R.T @ t

            model = cam_data["model"]
            params = cam_data["params"]
            k1, k2, p1, p2 = 0.0, 0.0, 0.0, 0.0

            # 兼容常见的相机模型
            if model == "SIMPLE_PINHOLE":
                fx, fy = params[0], params[0]
                cx, cy = params[1], params[2]
            elif model == "PINHOLE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model == "SIMPLE_RADIAL":
                fx, fy = params[0], params[0]
                cx, cy = params[1], params[2]
                k1 = params[3]
            elif model == "RADIAL":
                fx, fy = params[0], params[0]
                cx, cy = params[1], params[2]
                k1, k2 = params[3], params[4]
            elif model == "OPENCV":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                k1, k2, p1, p2 = params[4], params[5], params[6], params[7]
            else:
                fx, fy = params[0], params[0]
                cx, cy = cam_data["width"]/2, cam_data["height"]/2

            sfm_cameras[img_name] = Camera(
                orientation=rotmat,
                position=camera_position,
                focal_length=fx,
                pixel_aspect_ratio=fy / fx,
                principal_point=np.array([cx, cy]),
                radial_distortion=np.array([k1, k2, 0.0]),
                tangential_distortion=np.array([p1, p2]),
                skew=0.0,
                image_size=np.array([cam_data["width"], cam_data["height"]])
            )

        return cls(sfm_cameras, points)

    def __init__(self, cameras, points):
        self.camera_dict = cameras
        self.points = points

    def __len__(self):
        return len(self.camera_dict)

    @property
    def image_ids(self):
        return sorted(self.camera_dict.keys())

    @property
    def camera_list(self):
        return [self.camera_dict[i] for i in self.image_ids]

    @property
    def camera_positions(self):
        return np.stack([camera.position for camera in self.camera_list])


def filter_outlier_points(points, inner_percentile):
    if len(points) == 0:
        return points
    outer = 1.0 - inner_percentile
    lower, upper = outer / 2.0, 1.0 - (outer / 2.0)
    centers_min = np.quantile(points, lower, axis=0)
    centers_max = np.quantile(points, upper, axis=0)
    result = points.copy()
    too_near = np.any(result < centers_min[None, :], axis=1)
    too_far = np.any(result > centers_max[None, :], axis=1)
    return result[~(too_near | too_far)]

def estimate_near_far_for_image(scene_manager, image_id):
    points = filter_outlier_points(scene_manager.points, 0.95)
    points = np.concatenate([points, scene_manager.camera_positions], axis=0)
    camera = scene_manager.camera_dict[image_id]
    pixels = camera.project(points)
    depths = camera.points_to_local_points(points)[..., 2]

    in_frustum = (
        (pixels[..., 0] >= 0.0) & (pixels[..., 0] <= camera.image_size_x) &
        (pixels[..., 1] >= 0.0) & (pixels[..., 1] <= camera.image_size_y)
    )
    depths = depths[in_frustum]
    depths = depths[depths > 0]
    
    if len(depths) == 0:
        return 0.1, 10.0
    return np.quantile(depths, 0.001), np.quantile(depths, 0.999)

def estimate_near_far(scene_manager):
    image_ids = scene_manager.image_ids
    rng = np.random.RandomState(0)
    image_ids = rng.choice(image_ids, size=len(scene_manager.camera_list), replace=False)
    
    result =[{'image_id': img_id, **dict(zip(('near', 'far'), estimate_near_far_for_image(scene_manager, img_id)))} 
              for img_id in image_ids]
    return pd.DataFrame.from_records(result)

def get_bbox_corners(points):
    return np.stack([points.min(axis=0), points.max(axis=0)])

def points_bound(points):
    return np.stack((np.min(points, axis=0), np.max(points, axis=0)), axis=1)

def points_bounding_size(points):
    bounds = points_bound(points)
    return np.linalg.norm(bounds[:, 1] - bounds[:, 0])

def look_at(camera, camera_position: np.ndarray, look_at_position: np.ndarray, up_vector: np.ndarray):
    look_at_camera = camera.copy()
    optical_axis = look_at_position - camera_position
    norm = np.linalg.norm(optical_axis)
    optical_axis = optical_axis / norm if norm > 1e-5 else np.array([0, 0, 1.0])
    
    right_vector = np.cross(optical_axis, up_vector)
    norm = np.linalg.norm(right_vector)
    right_vector = right_vector / norm if norm > 1e-5 else np.array([1.0, 0, 0])
        
    camera_rotation = np.identity(3)
    camera_rotation[0, :] = right_vector
    camera_rotation[1, :] = np.cross(optical_axis, right_vector)
    camera_rotation[2, :] = optical_axis
    
    look_at_camera.position = camera_position
    look_at_camera.orientation = camera_rotation
    return look_at_camera

def triangulate_rays_numpy(origins, directions):
    """用纯 Numpy 代替 tensorflow_graphics 的 ray_triangulate。"""
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    A, b = np.zeros((3, 3)), np.zeros(3)
    I = np.eye(3)
    for o, d in zip(origins, directions):
        d = d.reshape((3, 1))
        I_minus_ddT = I - (d @ d.T)
        A += I_minus_ddT
        b += I_minus_ddT @ o
    return np.linalg.solve(A, b)

def generate_nerfies_dataset(sparse_dir: str, out_dir: str, val_interval: int = 20):
    sparse_path = Path(sparse_dir)
    root_dir = Path(out_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading COLMAP scene from {sparse_path}...")
    scene_manager = SceneManager.from_colmap_dir(sparse_path, min_track_length=5)
    
    if len(scene_manager) == 0:
        raise ValueError(f"没有读取到任何相机！请确认 {sparse_path} 内存在正确的 cameras.txt 和 images.txt 文件。")
        
    print(f"Loaded {len(scene_manager)} cameras. (100% Retained, No Blur Filtering applied)")
    
    print("Estimating near/far planes...")
    near_far = estimate_near_far(scene_manager)
    near = near_far['near'].quantile(0.001) / 0.8
    far = near_far['far'].quantile(0.999) * 1.2
    
    print("Computing scene center and scale...")
    points = filter_outlier_points(scene_manager.points, 0.95)
    bbox_corners = get_bbox_corners(np.concatenate([points, scene_manager.camera_positions], axis=0))
    scene_center = np.mean(bbox_corners, axis=0)
    scene_scale = 1.0 / np.sqrt(np.sum((bbox_corners[1] - bbox_corners[0]) ** 2))
    print(f"Computed Scale: {scene_scale}")
    
    print("Generating orbit camera trajectory (Test Set)...")
    ref_cameras = scene_manager.camera_list
    origins = np.array([c.position for c in ref_cameras])
    directions = np.array([c.optical_axis for c in ref_cameras])
    
    look_at_point = triangulate_rays_numpy(origins, directions)
    avg_position = np.mean(origins, axis=0)
    up = -np.mean([c.orientation[..., 1] for c in ref_cameras], axis=0)
    
    # --- 严格按照用户指定的参数生成 test path ---
    bounding_size = points_bounding_size(origins) / 2
    x_scale = 0.7
    y_scale = 0.7
    xs = x_scale * bounding_size
    ys = y_scale * bounding_size
    radius = 0.4
    num_frames = 200
    # ----------------------------------------
    
    orbit_cameras =[]
    angles = np.linspace(0, 2 * math.pi, num=num_frames)
    for angle in angles:
        x = math.cos(angle) * radius * xs
        y = math.sin(angle) * radius * ys
        position = np.array([x, y, 0]) + avg_position
        orbit_cameras.append(look_at(ref_cameras[0], position, look_at_point, up))
    
    print("Saving Dataset Configuration...")
    with (root_dir / 'scene.json').open('w') as f:
        json.dump({
            'scale': float(scene_scale),
            'center': scene_center.tolist(),
            'bbox': bbox_corners.tolist(),
            'near': float(near * scene_scale),
            'far': float(far * scene_scale),
        }, f, indent=2)
    
    # 获取所有的 image ids，不做任何遗漏
    all_ids = scene_manager.image_ids
    train_ids = all_ids
    val_ids   = all_ids
    # val_ids = all_ids[::val_interval]
    # train_ids = sorted(set(all_ids) - set(val_ids))
    with (root_dir / 'dataset.json').open('w') as f:
        json.dump({
            'count': len(scene_manager),
            'num_exemplars': len(train_ids),
            'ids': all_ids,
            'train_ids': train_ids,
            'val_ids': val_ids,
        }, f, indent=2)
    
    import bisect
    metadata_json = {}
    for i, img_id in enumerate(train_ids):
        metadata_json[img_id] = {'warp_id': i, 'appearance_id': i, 'camera_id': 0}
    for i, img_id in enumerate(val_ids):
        metadata_json[img_id] = {'warp_id': bisect.bisect_left(train_ids, img_id), 'appearance_id': bisect.bisect_left(train_ids, img_id), 'camera_id': 0}
    with (root_dir / 'metadata.json').open('w') as f:
        json.dump(metadata_json, f, indent=2)
    
    camera_dir = root_dir / 'camera'
    camera_dir.mkdir(exist_ok=True, parents=True)
    for item_id, camera in scene_manager.camera_dict.items():
        with (camera_dir / f'{item_id}.json').open('w') as f:
            json.dump(camera.to_json(), f, indent=2)
            
    test_camera_dir = root_dir / 'camera-paths' / 'orbit-mild'
    test_camera_dir.mkdir(exist_ok=True, parents=True)
    for i, camera in enumerate(orbit_cameras):
        with (test_camera_dir / f'{i:06d}.json').open('w') as f:
            json.dump(camera.to_json(), f, indent=2)

    print(f"Data Successfully Saved To: {root_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COLMAP Sparse model to Nerfies Format")
    parser.add_argument("--sparse", type=str, required=True, help="Path to COLMAP sparse folder (e.g., ./colmap/sparse/0)")
    parser.add_argument("--out", type=str, required=True, help="Output dataset directory path")
    parser.add_argument("--val_interval", type=int, default=20, help="Train/Val split interval")
    args = parser.parse_args()
    
    generate_nerfies_dataset(args.sparse, args.out, args.val_interval)