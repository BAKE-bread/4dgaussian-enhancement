import numpy as np
import random
import os
import sys
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, lpips_loss, pearson_depth_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import lpips
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
from utils.scene_utils import render_training_image
import gc

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# ==============================================================================
#  Magic numbers / Hyperparameters (explanation and centralization)
# ==============================================================================
# Foreground weight for mask-weighted L1 loss (emphasize foreground regions)
FOREGROUND_WEIGHT = 5.0
# Base L1 loss weight (RGB)
LAMBDA_L1 = 1.0
# Base depth loss weight (Pearson correlation loss)
LAMBDA_DEPTH_L1 = 0.25
# Depth error threshold for HPGate boosting (relative geometric error, normalized)
HPGATE_DEPTH_ERR_THRESHOLD = 0.05
# Default denominator for depth error normalization when max error is small
HPGATE_DEFAULT_DENOM = 0.1
# Epsilon for numerical stability
EPS = 1e-8
# HPGate boosting strength alpha (linear boost factor = 1 + alpha * error_norm)
HPGATE_ALPHA = 1.0

# Depth loss schedule ratios (as fraction of total iterations)
DEPTH_LOSS_DECAY_START_FRAC = 0.2   # start decaying after 20% of iterations
DEPTH_LOSS_DECAY_END_FRAC = 0.5     # reach final factor at 50% of iterations
DEPTH_LOSS_FINAL_FACTOR = 0.2       # final multiplier for depth loss after decay

# Pruning / densification constants
PRUNE_SIZE_THRESHOLD = 20           # size threshold for pruning (pixels)
DENSIFICATION_GROW_NUM = 5          # number of neighbors for densification grow (both params)
HPGATE_ENABLE_FRAC = 0.9            # enable HPGate only until 90% of densify_until_iter
MAX_GAUSSIANS_COARSE = 360_000      # max Gaussians during coarse stage (hard limit)
PRUNE_LOWER_BOUND = 240_000         # start pruning only when Gaussians exceed this count

# Numerical thresholds for depth statistics
MIN_DEPTH_STD = 1e-4                # clamp depth standard deviation to avoid division by zero
MIN_VALID_DEPTH_PIXELS = 10         # minimum number of valid depth pixels to compute error map
MAX_RENDER_PROJ_W = 1e-7            # small constant to avoid division by zero in projection

# ==============================================================================
#  Helper functions (unchanged logic, only formatting and comments)
# ==============================================================================

def pearson_depth_loss(pred, gt, mask=None, eps=EPS):
    """
    Compute Pearson correlation loss on log-transformed depth maps.
    This version overwrites the imported one from utils.loss_utils.
    """
    pred = pred.view(-1)
    gt = gt.view(-1)
    if mask is not None:
        mask_valid = (mask > 0.5).view(-1)
        if mask_valid.sum() < 5:   # need at least 5 valid pixels to compute correlation
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred = pred[mask_valid]
        gt = gt[mask_valid]

    pred = torch.log(pred + eps)
    gt = torch.log(gt + eps)

    if pred.shape[0] < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pred_mean = pred.mean()
    gt_mean = gt.mean()
    pred_centered = pred - pred_mean
    gt_centered = gt - gt_mean

    pred_var = (pred_centered ** 2).sum()
    gt_var = (gt_centered ** 2).sum()

    if pred_var < eps or gt_var < eps:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    covariance = (pred_centered * gt_centered).sum()
    r = covariance / (torch.sqrt(pred_var) * torch.sqrt(gt_var) + eps)
    r = torch.clamp(r, -1.0, 1.0)

    return 1.0 - r


def get_points_in_mask_and_depth_error(viewpoint_cam, gaussians, visible_indices, gt_mask, depth_error_map):
    """
    Sample mask and depth error values at projected locations of visible Gaussians.
    Used by HPGate to boost gradients of points that lie in foreground and have high depth error.
    """
    if len(visible_indices) == 0:
        return None, None

    xyz_world = gaussians.get_xyz[visible_indices]
    full_proj = viewpoint_cam.full_proj_transform
    if not full_proj.is_cuda:
        full_proj = full_proj.to(xyz_world.device)

    ones = torch.ones((xyz_world.shape[0], 1), device=xyz_world.device)
    xyz_hom = torch.cat([xyz_world, ones], dim=1)

    p_hom = xyz_hom @ full_proj
    p_w = 1.0 / (p_hom[:, 3] + MAX_RENDER_PROJ_W)
    p_ndc = p_hom[:, :3] * p_w.unsqueeze(1)

    sample_grid = p_ndc[:, :2].view(1, 1, -1, 2)

    if gt_mask.dim() == 2:
        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
    elif gt_mask.dim() == 3:
        if gt_mask.shape[0] == 3:
            gt_mask = gt_mask[0:1, :, :]
        gt_mask = gt_mask.unsqueeze(0)

    if depth_error_map.dim() == 2:
        depth_error_map = depth_error_map.unsqueeze(0).unsqueeze(0)
    elif depth_error_map.dim() == 3:
        depth_error_map = depth_error_map.unsqueeze(0)

    combined_map = torch.cat([gt_mask.float(), depth_error_map], dim=1)

    sampled_values = F.grid_sample(combined_map, sample_grid, mode='nearest', align_corners=True)
    sampled_values = sampled_values.view(2, -1)

    mask_sampled = sampled_values[0]
    depth_err_sampled = sampled_values[1]

    return mask_sampled, depth_err_sampled


def check_data_validity(mask, depth, cam_name):
    """
    Validate mask and depth tensors: check for zeros, NaNs, etc.
    Returns (is_valid_mask, is_valid_depth, message_list)
    """
    msg = []
    is_valid_mask = False
    is_valid_depth = False

    if mask is not None:
        if mask.max() <= 0:
            msg.append(f"⚠️ [Data Warning] Cam {cam_name}: Mask is all ZEROS!")
        elif torch.isnan(mask).any():
            msg.append(f"❌ [Data Error] Cam {cam_name}: Mask contains NaNs!")
        else:
            is_valid_mask = True

    if depth is not None:
        if depth.max() <= 0:
            msg.append(f"⚠️ [Data Warning] Cam {cam_name}: Depth is all ZEROS!")
        elif torch.isnan(depth).any():
            msg.append(f"❌ [Data Error] Cam {cam_name}: Depth contains NaNs!")
        else:
            is_valid_depth = True

    return is_valid_mask, is_valid_depth, msg


# ==============================================================================
#  Main training function
# ==============================================================================

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer):
    first_iter = 0
    lpips_model = lpips.LPIPS(net="alex").cuda()

    gaussians.training_setup(opt)
    if checkpoint:
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            return
        if stage in checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter

    progress_bar = tqdm(range(first_iter, final_iter), desc=f"Training {stage} progress")
    first_iter += 1

    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack and not opt.dataloader:
        viewpoint_stack = list(range(len(train_cams)))
        temp_stack = viewpoint_stack.copy()

    batch_size = opt.batch_size

    loader = None
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, sampler=sampler,
                                                num_workers=0, collate_fn=list, persistent_workers=False,
                                                pin_memory=False)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, shuffle=True,
                                                num_workers=0, collate_fn=list, persistent_workers=False,
                                                pin_memory=False)
            random_loader = True
        loader = iter(viewpoint_stack_loader)

    # Loss weights (will be used as multipliers)
    foreground_weight = FOREGROUND_WEIGHT
    lambda_l1 = LAMBDA_L1
    lambda_depth_l1 = LAMBDA_DEPTH_L1

    for iteration in range(first_iter, final_iter + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    viewpoint_index = (iteration) % len(video_cams)
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer,
                                       stage=stage, cam_type=scene.dataset_type)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        viewpoint_cams = []
        if opt.dataloader:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size, shuffle=True,
                                                        num_workers=0, collate_fn=list, persistent_workers=False,
                                                        pin_memory=False)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)
                viewpoint_cams = next(loader)
        else:
            idx = 0
            while idx < batch_size:
                if not viewpoint_stack:
                    viewpoint_stack = temp_stack.copy()
                viewpoint_cam_idx = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
                viewpoint_cam = train_cams[viewpoint_cam_idx]
                viewpoint_cams.append(viewpoint_cam)
                idx += 1

        if (iteration - 1) == debug_from:
            pipe.debug = True

        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        depth_error_maps = []   # per-view depth error maps (for densification)
        gt_masks_list = []

        # Batch-wise loss storage
        gt_masks_for_loss = []
        depth_losses = []

        log_mask_l1 = 0.0
        log_depth_l1 = 0.0
        log_ssim = 0.0
        log_lpips = 0.0
        log_tv = 0.0

        # Optional: early sanity check on Gaussians
        with torch.no_grad():
            if torch.isnan(gaussians._xyz).any() or torch.isinf(gaussians._xyz).any():
                if torch.isnan(gaussians._xyz).all():
                    raise RuntimeError("All Gaussians are NaN! Training failed.")

        debug_this_iter = (iteration <= first_iter + 4)

        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            pred_depth = render_pkg["depth"]

            if scene.dataset_type != "PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
                gt_mask = viewpoint_cam.mask.cuda() if hasattr(viewpoint_cam, 'mask') and viewpoint_cam.mask is not None else None
                gt_depth = viewpoint_cam.depth.cuda() if hasattr(viewpoint_cam, 'depth') and viewpoint_cam.depth is not None else None
            else:
                gt_image = viewpoint_cam['image'].cuda()
                gt_mask = None
                gt_depth = None

            gt_image = gt_image.cuda(non_blocking=True)
            if gt_mask is not None:
                gt_mask = gt_mask.cuda(non_blocking=True)
            if gt_depth is not None:
                gt_depth = gt_depth.cuda(non_blocking=True)

            # === 新增调试打印 (保留原样，仅格式化注释) ===
            # if debug_this_iter:
            #     cam_name = viewpoint_cam.image_name if hasattr(viewpoint_cam, 'image_name') else 'unknown'
            #     print(f"[train-gate.py] Iter {iteration}, Cam: {cam_name}")
            #     print(f"[train-gate.py]   - gt_mask: {type(gt_mask)} | shape={gt_mask.shape if gt_mask is not None else None} | "
            #           f"has_nan={torch.isnan(gt_mask).any().item() if gt_mask is not None else 'N/A'} | "
            #           f"max={gt_mask.max().item() if gt_mask is not None else 'N/A'}")
            #     print(f"[train-gate.py]   - gt_depth: {type(gt_depth)} | shape={gt_depth.shape if gt_depth is not None else None} | "
            #           f"has_nan={torch.isnan(gt_depth).any().item() if gt_depth is not None else 'N/A'} | "
            #           f"max={gt_depth.max().item() if gt_depth is not None else 'N/A'}")
            #     if hasattr(viewpoint_cam, 'mask_path'):
            #         print(f"[train-gate.py]   - mask_path: {viewpoint_cam.mask_path}")
            #     if hasattr(viewpoint_cam, 'depth_path'):
            #         print(f"[train-gate.py]   - depth_path: {viewpoint_cam.depth_path}")

            if iteration % 1000 == 0:
                is_mask_ok, is_depth_ok, msgs = check_data_validity(gt_mask, gt_depth, viewpoint_cam.image_name)
                for msg in msgs:
                    print(msg)

            images.append(image.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

            # --- Prepare depth error map for densification (HPGate) ---
            d_err = torch.zeros((1, 1, gt_image.shape[1], gt_image.shape[2]), device="cuda")

            if pred_depth is not None and gt_depth is not None:
                # Convert Depth Anything disparity to pseudo depth (inverse)
                gt_disp_safe = torch.clamp(gt_depth, min=EPS)
                gt_pseudo_depth = 1.0 / gt_disp_safe

                # Compute depth loss using Pearson correlation (scale invariant)
                if gt_mask is not None:
                    depth_diff = pearson_depth_loss(pred_depth, gt_pseudo_depth, mask=gt_mask)
                else:
                    depth_diff = pearson_depth_loss(pred_depth, gt_pseudo_depth, mask=None)
                depth_losses.append(depth_diff.mean())

                # Compute scale‑invariant error map for densification
                if gt_mask is not None:
                    mask_valid = (gt_mask > 0.5)
                else:
                    mask_valid = torch.ones_like(gt_depth, dtype=torch.bool)

                if mask_valid.sum() > MIN_VALID_DEPTH_PIXELS:
                    p_valid = pred_depth[mask_valid]
                    g_valid = gt_pseudo_depth[mask_valid]

                    p_mean = p_valid.mean()
                    p_std = torch.clamp(p_valid.std(), min=MIN_DEPTH_STD)
                    g_mean = g_valid.mean()
                    g_std = torch.clamp(g_valid.std(), min=MIN_DEPTH_STD)

                    p_norm_map = (pred_depth - p_mean) / p_std
                    g_norm_map = (gt_pseudo_depth - g_mean) / g_std

                    err_map = torch.abs(p_norm_map - g_norm_map)

                    if gt_mask is not None:
                        err_map = err_map * gt_mask

                    d_err = err_map.unsqueeze(0)   # [1, 1, H, W]

            depth_error_maps.append(d_err)
            gt_masks_list.append(gt_mask)

            if gt_mask is not None:
                m_loss = gt_mask.clone()
                if m_loss.dim() == 2:
                    m_loss = m_loss.unsqueeze(0)
                elif m_loss.dim() == 3 and m_loss.shape[0] > 1:
                    m_loss = m_loss[0:1, :, :]
                gt_masks_for_loss.append(m_loss.unsqueeze(0))   # [1, 1, H, W]

        # --- Adaptive depth loss weight schedule ---
        current_progress = (iteration - first_iter) / (final_iter - first_iter)
        if current_progress < DEPTH_LOSS_DECAY_START_FRAC:
            effective_lambda_depth = lambda_depth_l1
        elif current_progress < DEPTH_LOSS_DECAY_END_FRAC:
            # linear decay from 1.0 to DEPTH_LOSS_FINAL_FACTOR
            decay_range = DEPTH_LOSS_DECAY_END_FRAC - DEPTH_LOSS_DECAY_START_FRAC
            decay_ratio = (current_progress - DEPTH_LOSS_DECAY_START_FRAC) / decay_range
            effective_lambda_depth = lambda_depth_l1 * (1.0 - (1.0 - DEPTH_LOSS_FINAL_FACTOR) * decay_ratio)
        else:
            effective_lambda_depth = lambda_depth_l1 * DEPTH_LOSS_FINAL_FACTOR

        image_tensor = torch.cat(images, 0)
        gt_image_tensor = torch.cat(gt_images, 0)
        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)

        loss = 0.0

        # --- RGB loss with optional foreground weighting ---
        if len(gt_masks_for_loss) == len(viewpoint_cams) and all(m is not None for m in gt_masks_for_loss):
            mask_tensor = torch.cat(gt_masks_for_loss, 0)
            pixel_weights = 1.0 + (foreground_weight - 1.0) * mask_tensor
            # Force RGB channels (first three) to avoid alpha channel mismatch
            l1_map = torch.abs(image_tensor - gt_image_tensor[:, :3, :, :])
            Ll1 = (l1_map * pixel_weights).mean()
        else:
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
            foreground_weight = -1   # mark that mask was not used

        ssim_val = ssim(image_tensor, gt_image_tensor[:, :3, :, :])
        Lssim = 1.0 - ssim_val

        if opt.lambda_lpips > 0:
            Llpips = lpips_loss(image_tensor, gt_image_tensor[:, :3, :, :], lpips_model)
        else:
            Llpips = torch.tensor(0.0).cuda()

        log_mask_l1 = lambda_l1 * Ll1.item()
        log_ssim = opt.lambda_dssim * Lssim.item()
        log_lpips = opt.lambda_lpips * Llpips.item()
        loss += lambda_l1 * Ll1 + opt.lambda_dssim * Lssim + opt.lambda_lpips * Llpips

        # --- Depth loss (averaged over batch) ---
        if len(depth_losses) > 0 and effective_lambda_depth > 0:
            avg_depth_loss = torch.stack(depth_losses).mean()
            log_depth_l1 = effective_lambda_depth * avg_depth_loss.item()
            loss += effective_lambda_depth * avg_depth_loss

        # --- Temporal smoothness (4D GS) ---
        if hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight,
                                                   hyper.l1_time_planes,
                                                   hyper.plane_tv_weight)
            log_tv = tv_loss.item()
            if stage == "fine":
                loss += tv_loss

        loss.backward()

        # Optional gradient clipping or NaN check before optimizer step
        #   with torch.no_grad():
        #       has_nan_grad = False
        #       for group in gaussians.optimizer.param_groups:
        #           for param in group['params']:
        #               if param.grad is not None:
        #                   if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
        #                       has_nan_grad = True
        #                       break
        #           if has_nan_grad:
        #               break
        #       if has_nan_grad:
        #           print(f"[WARNING] Iter {iteration}: NaN gradients detected! Skipping optimizer step.")
        #           gaussians.optimizer.zero_grad(set_to_none=True)
        #           continue
        #       # for group in gaussians.optimizer.param_groups:
        #       #     torch.nn.utils.clip_grad_norm_(group['params'], max_norm=1.0)

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            psnr_ = psnr(image_tensor, gt_image_tensor[:, :3, :, :]).mean().double()
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]

            if iteration % 1000 == 0 or (iteration % 100 == 1 and iteration < 1000):
                print(f"\n[ITER {iteration}] {stage.upper()} Stage - Loss Components (Weighted Values):")
                print(f"  - Mask-weighted L1 Loss: {log_mask_l1:.6f} (λ_fr={foreground_weight})")
                print(f"  - SSIM Loss: {log_ssim:.6f}")
                print(f"  - LPIPS Loss: {log_lpips:.6f}")
                print(f"  - Depth L1 Loss: {log_depth_l1:.6f}")
                print(f"  - 4D Smoothness Loss: {log_tv:.6f}")
                print(f"  - Total Loss: {loss.item():.6f}")
                print(f"  - PSNR: {psnr_:.2f}")
                print(f"  - Active Points: {total_point}")
                print("-" * 50)

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "PSNR": f"{psnr_:.2f}",
                    "Pts": f"{total_point}",
                    "VRAM": f"{torch.cuda.memory_allocated() / 1024 / 1024:.0f}MB"
                })
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, stage)

            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 0) \
                        or (iteration < 3000 and iteration % 50 == 0) \
                        or (iteration < 60000 and iteration % 100 == 0):
                    render_training_image(scene, gaussians, [test_cams[iteration % len(test_cams)]],
                                          render, pipe, background, stage + "test", iteration,
                                          timer.get_elapsed_time(), scene.dataset_type)
                    render_training_image(scene, gaussians, [train_cams[iteration % len(train_cams)]],
                                          render, pipe, background, stage + "train", iteration,
                                          timer.get_elapsed_time(), scene.dataset_type)

            timer.start()

            # ==================== Densification Logic (with HPGate) ====================
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                final_visibility_for_stats = visibility_filter.clone()
                enable_hpgate = (iteration < opt.densify_until_iter * HPGATE_ENABLE_FRAC)

                viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor_list[0])
                view_counts = torch.zeros(viewspace_point_tensor_grad.shape[0], device="cuda")

                batch_size_current = len(viewpoint_cams)

                for idx in range(batch_size_current):
                    view_grad = viewspace_point_tensor_list[idx].grad
                    if view_grad is None:
                        view_grad = torch.zeros_like(viewspace_point_tensor_list[0])
                    else:
                        view_grad = view_grad.clone()

                    # Compensate gradient magnitude for batch accumulation
                    view_grad *= batch_size_current

                    gt_mask = gt_masks_list[idx]
                    depth_err = depth_error_maps[idx]
                    visibility_filter_view = visibility_filter_list[idx].squeeze(0)

                    view_counts[visibility_filter_view] += 1

                    # HPGate: boost gradients for points that are in foreground and have significant depth error
                    if enable_hpgate and gt_mask is not None and depth_err is not None and visibility_filter_view.any():
                        visible_indices_view = torch.where(visibility_filter_view)[0]

                        mask_sampled, depth_err_sampled = get_points_in_mask_and_depth_error(
                            viewpoint_cams[idx],
                            gaussians,
                            visible_indices_view,
                            gt_mask,
                            depth_err
                        )

                        if mask_sampled is not None:
                            is_foreground = mask_sampled > 0.5
                            depth_err_norm = depth_err_sampled.clone()
                            max_err = depth_err_norm.max()
                            if max_err > HPGATE_DEPTH_ERR_THRESHOLD:
                                depth_err_norm = depth_err_norm / max_err
                            else:
                                depth_err_norm = depth_err_norm / HPGATE_DEFAULT_DENOM

                            boost_factor = 1.0 + HPGATE_ALPHA * depth_err_norm

                            boost_indices = visible_indices_view[is_foreground]
                            boost_vals = boost_factor[is_foreground]
                            view_grad[boost_indices] *= boost_vals.unsqueeze(1)

                    viewspace_point_tensor_grad += view_grad

                # Average gradients by number of views each point was visible
                valid_counts = view_counts > 0
                viewspace_point_tensor_grad[valid_counts] /= view_counts[valid_counts].unsqueeze(1)

                gaussians.add_densification_stats(viewspace_point_tensor_grad, final_visibility_for_stats)

                # Determine thresholds based on stage
                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    # Linearly decay thresholds during fine stage
                    opacity_threshold = (opt.opacity_threshold_fine_init -
                                         iteration * (opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) /
                                         (opt.densify_until_iter))
                    densify_threshold = (opt.densify_grad_threshold_fine_init -
                                         iteration * (opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) /
                                         (opt.densify_until_iter))

                if (iteration > opt.densify_from_iter and
                    iteration % opt.densification_interval == 0 and
                    gaussians.get_xyz.shape[0] < MAX_GAUSSIANS_COARSE):
                    size_threshold = PRUNE_SIZE_THRESHOLD if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent,
                                      size_threshold, DENSIFICATION_GROW_NUM, DENSIFICATION_GROW_NUM,
                                      scene.model_path, iteration, stage)

                if (iteration > opt.pruning_from_iter and
                    iteration % opt.pruning_interval == 0 and
                    gaussians.get_xyz.shape[0] > PRUNE_LOWER_BOUND and
                    iteration < opt.densify_until_iter * HPGATE_ENABLE_FRAC):
                    size_threshold = PRUNE_SIZE_THRESHOLD if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                if (iteration % opt.densification_interval == 0 and
                    gaussians.get_xyz.shape[0] < MAX_GAUSSIANS_COARSE and
                    opt.add_point):
                    gaussians.grow(DENSIFICATION_GROW_NUM, DENSIFICATION_GROW_NUM,
                                   scene.model_path, iteration, stage)

                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + f"_{stage}_" + str(iteration) + ".pth")

        if iteration % 50 == 0:
            torch.cuda.empty_cache()
        if iteration % 100 == 0:
            gc.collect()


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, expname):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "coarse", tb_writer, opt.coarse_iterations, timer)
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations, timer)


def prepare_output_and_logger(expname):
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                    scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': [scene.getTestCameras()[idx % len(scene.getTestCameras())]
                                         for idx in range(10, 5000, 299)]},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                          for idx in range(10, 5000, 299)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, stage=stage,
                                                   cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/" + config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                                 image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/" + config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                     gt_image[None], global_step=iteration)
                    except Exception:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(stage + "/" + config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage + "/" + config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate',
                                 scene.gaussians._deformation_table.sum() / scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram",
                                    scene.gaussians._deformation_accum.mean(dim=-1) / 100, iteration, max_bins=500)

        torch.cuda.empty_cache()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 7000, 10000, 14000, 20000, 25000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 14000, 20000, 25000, 30000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from, args.expname)

    print("\nTraining complete.")