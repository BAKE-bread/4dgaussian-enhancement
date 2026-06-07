import argparse
import numpy as np
from collections import defaultdict
import math
import sys

def estimate_voxel_size(points_xyz, target_voxels=15000):
    """
    根据点云范围和目标体素数量估算体素边长。
    points_xyz: (N, 3) numpy array
    target_voxels: 期望的体素数量（约等于降采样后的点数）
    """
    min_xyz = np.min(points_xyz, axis=0)
    max_xyz = np.max(points_xyz, axis=0)
    bbox_size = max_xyz - min_xyz
    volume = np.prod(bbox_size)
    if volume <= 0:
        return 0.01  # 退化情况
    voxel_size = (volume / target_voxels) ** (1/3)
    # 避免过小或过大
    voxel_size = max(voxel_size, 1e-6)
    return voxel_size

def voxel_downsample_txt(input_file, output_file, voxel_size=None, target_points=15000):
    """
    体素降采样txt点云文件。
    input_file : 输入txt路径
    output_file: 输出txt路径
    voxel_size : 体素边长（如果为None，则根据target_points自动估算）
    target_points: 期望的输出点数（仅当voxel_size=None时生效）
    """
    print(f"读取点云文件: {input_file}")
    # 存储所有点信息（原始顺序不重要，按体素分组）
    # 为了节省内存，边读边分组
    voxel_dict = defaultdict(list)  # key: (ix,iy,iz) -> list of tuples (x,y,z,r,g,b,rest_fields)

    # 第一遍读取：收集所有点并计算包围盒（如果自动估算体素大小）
    points_xyz_for_bbox = []
    all_points_data = []  # 存储每行的原始数值列表，用于后续快速分组（若内存紧张可改为边读边分组）
    # 但为了自动估算体素大小，需要先知道所有坐标。400万点存储坐标和颜色约 400万*6*8≈192MB，可接受
    # 更节省的方法：先读一次获得包围盒，再读一次进行分组。但两次IO耗时。这里采用一次读取并存储全部数据。

    # 读取所有行
    with open(input_file, 'r') as fin:
        lines = fin.readlines()

    # 解析
    data_list = []  # 每个元素是 [x,y,z,r,g,b, rest_fields列表]
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue  # 格式错误
        # 前7个: ID, X, Y, Z, R, G, B
        # 实际我们不需要原始ID，但保留其他所有字段
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            r = int(parts[4])
            g = int(parts[5])
            b = int(parts[6])
            rest = parts[7:]  # error 和 track 等
        except ValueError:
            continue
        data_list.append([x, y, z, r, g, b, rest])
        points_xyz_for_bbox.append((x, y, z))

    if not data_list:
        print("未找到有效点数据")
        return

    points_xyz = np.array(points_xyz_for_bbox, dtype=np.float64)
    del points_xyz_for_bbox  # 释放内存

    # 确定体素大小
    if voxel_size is None:
        voxel_size = estimate_voxel_size(points_xyz, target_points)
        print(f"自动估算体素大小: {voxel_size:.6f} (目标点数 {target_points})")
    else:
        print(f"使用指定体素大小: {voxel_size}")

    # 分组：计算体素索引
    print("进行体素分配...")
    for i, (x, y, z, r, g, b, rest) in enumerate(data_list):
        ix = math.floor(x / voxel_size)
        iy = math.floor(y / voxel_size)
        iz = math.floor(z / voxel_size)
        key = (ix, iy, iz)
        voxel_dict[key].append((x, y, z, r, g, b, rest))

    # 对每个体素计算代表点
    print(f"体素数量: {len(voxel_dict)}，期望输出点数 ~ {len(voxel_dict)}")
    output_points = []
    for key, points in voxel_dict.items():
        n = len(points)
        # 计算重心坐标
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        sum_z = sum(p[2] for p in points)
        cx = sum_x / n
        cy = sum_y / n
        cz = sum_z / n
        # 计算RGB均值（四舍五入）
        sum_r = sum(p[3] for p in points)
        sum_g = sum(p[4] for p in points)
        sum_b = sum(p[5] for p in points)
        cr = int(round(sum_r / n))
        cg = int(round(sum_g / n))
        cb = int(round(sum_b / n))
        # 其他字段：取第一个点的 rest（error, track...）
        # 注意：如果希望 error 取平均，可以自行修改，但 track 通常保持原样或置0
        first_rest = points[0][6]
        output_points.append((cx, cy, cz, cr, cg, cb, first_rest))

    # 写入输出文件，重新编号ID
    print(f"写入输出文件: {output_file}")
    with open(output_file, 'w') as fout:
        # 可选：写入简单注释
        fout.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for new_id, (x, y, z, r, g, b, rest) in enumerate(output_points, start=1):
            # rest 是一个列表，需要转换为空格分隔的字符串
            rest_str = ' '.join(rest)
            fout.write(f"{new_id} {x:.8f} {y:.8f} {z:.8f} {r} {g} {b} {rest_str}\n")

    print(f"降采样完成：原始点数 {len(data_list)} -> 输出点数 {len(output_points)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="体素降采样点云txt文件（COLMAP points3D格式）")
    parser.add_argument("input", help="输入txt文件路径")
    parser.add_argument("output", help="输出txt文件路径")
    parser.add_argument("--voxel_size", type=float, default=None, help="体素边长（若不指定，自动根据目标点数估算）")
    parser.add_argument("--target_points", type=int, default=15000, help="目标输出点数（仅当voxel_size=None时生效）")
    args = parser.parse_args()

    voxel_downsample_txt(args.input, args.output, args.voxel_size, args.target_points)

#if __name__ == "__main__":
#    # ================== 配置区 ==================
#    INPUT_PLY = r"D:\Coding\VGGT-Long\exps\._cam00_images\2026-03-08-09-36-17\pcd\combined_pcd.ply"   # 替换为 VGGT-Long 生成的 combined_pcd.ply 的实际路径
#    OUTPUT_DIR = r"D:\Coding\VGGT-Long\exps\._cam00_images\2026-03-08-09-36-17\pcd"   # 生成的 COLMAP 目录
#    
#    # 保留点云的比例。如果原点云有 1000 万个点，0.05 代表下采样保留 50 万个点。
#    # 对于导入 3DGS 等管线，建议保持在 20万 - 100万 之间即可。
#    KEEP_RATIO = 0.05 
#    # ============================================
    
#    process_and_export_colmap(INPUT_PLY, OUTPUT_DIR, keep_ratio=KEEP_RATIO)


