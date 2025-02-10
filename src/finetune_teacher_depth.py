import argparse
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pytorch_warmup as warmup
from cityscapesD import Cityscapes
from loss import MidasLoss, MSGLoss, HDNLoss, scale_and_shift_mae
from metrics import evaluate_depth
import matplotlib.pyplot as plt
from PIL import Image
from torch.amp import autocast, GradScaler

import sys
path= "/home/avalocal/thesis23/KD/ml-depth-pro/src"
sys.path.append(path)
import depth_pro

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Teacher Model for Depth Estimation')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='/media/avalocal/T7/pardis/pardis/perception_system/datasets/cityscapes', help='Path to dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Path to save checkpoints')
    parser.add_argument('--experiment', type=int, help='Experiment number')
    parser.add_argument('--results_dir', type=str, default='./results', help='Path to save results')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    return parser.parse_args()

def train(model, train_loader, criterion, optimizer, scheduler, device, epoch, args):
    
    model.train()
    total_loss = 0
    grad_loss = MSGLoss()
    scaler = GradScaler()
    H, W = 1536, 1536
    f_px= 2262.52 #focal length in pixels

    for i, batch in enumerate(tqdm(train_loader, disable=(dist.get_rank() != 0))):
        '''images, gt, depths = batch  # Use actual depth labels
        images, depths = images.to(device, non_blocking=True), depths.to(device, non_blocking=True)
        images = nn.functional.interpolate(images, size=(1536, 1536), mode='bilinear', align_corners=False)
        '''
        images, _, gt_depth= batch #labels and depths are not used
        # img , pseudo files[depth meter] , gt dispatiry to depth[depth meter]
        images = images.to(device)
        gt_depth = gt_depth.to(device)

        valid_pixels = gt_depth > 0  # Mask for valid depth values (non-zero)
        min_depth = torch.min(gt_depth[valid_pixels])  # Minimum valid depth
        max_depth = torch.max(gt_depth[valid_pixels])  # Maximum valid depth
        depth_normalized = torch.zeros_like(gt_depth)  # Initialize with zeros
        depth_normalized[valid_pixels] = (gt_depth[valid_pixels] - min_depth) / (max_depth - min_depth)  # Normalize depth values [0-1]

        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16):
            canonical_inverse_depth, fov_deg= model(images)

            inverse_depth = canonical_inverse_depth * (W / f_px)
            inverse_depth = nn.functional.interpolate(
                inverse_depth, size=(H, W), mode="bilinear", align_corners=False
            )
            pred = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=80)
            pred = pred.squeeze(1) #B, 1, H, W -> B, H, W
            pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))  # Normalize depth values [0-1]
            
            
            loss_midas = criterion(pred, depth_normalized)
            loss_grad = grad_loss(pred, depth_normalized)
            loss = 3.0 * loss_midas + 2.0 * loss_grad
        
        # print("------------?loss", loss)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        if i % 500 == 0 and dist.get_rank() == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}] Batch [{i}/{len(train_loader)}] Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device, epoch, args):
    model.eval()
    total_loss = 0
    grad_loss = MSGLoss()
    metric_dic = {metric: 0.0 for metric in metric_name}

    with torch.no_grad():
        for batch in tqdm(val_loader, disable=(dist.get_rank() != 0)):
            images, _, gt_depth = batch
            images, gt_depth = images.to(device, non_blocking=True), gt_depth.to(device, non_blocking=True)

            valid_pixels = gt_depth > 0  # Mask for valid depth values (non-zero)
            min_depth = torch.min(gt_depth[valid_pixels])  # Minimum valid depth
            max_depth = torch.max(gt_depth[valid_pixels])  # Maximum valid depth

            # depth_normalized = np.zeros_like(gt_depth)  # Initialize with zeros
            depth_normalized = torch.zeros_like(gt_depth)  # Initialize with zeros
            depth_normalized[valid_pixels] = (gt_depth[valid_pixels] - min_depth) / (max_depth - min_depth)  # Normalize depth values
            # print(f"Min depth: {min_depth}, Max depth: {max_depth}")

            canonical_inverse_depth, fov_deg = model(images)
            inverse_depth = canonical_inverse_depth * (1536 / 2262.52)
            inverse_depth = nn.functional.interpolate(
                inverse_depth, size=(1536, 1536), mode="bilinear", align_corners=False
            )
            predictions = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=80)
            predictions = predictions.squeeze(1)  # B, 1, H, W -> B, H, W
            predictions = (predictions - torch.min(predictions)) / (torch.max(predictions) - torch.min(predictions))  # Normalize depth values [0-1]
            
            loss_ssi = criterion(predictions, depth_normalized)
            loss_grad = grad_loss(predictions, depth_normalized)
            loss = 3.0 * loss_ssi + 2.0 * loss_grad
            total_loss += loss.item()

            aligned_predictions = scale_and_shift_mae(predictions, gt_depth, gt_depth > 0)
            metrics = evaluate_depth(aligned_predictions, gt_depth)
            for metric in metric_name:
                metric_dic[metric] += metrics[metric]

    avg_loss = total_loss / len(val_loader)
    for metric in metric_name:
        metric_dic[metric] /= len(val_loader)

    return avg_loss, metric_dic

def main():
    args = parse_args()
    dist.init_process_group(backend="nccl")  # Initialize DDP
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    train_dataset = Cityscapes(args.data_dir, split='train')
    val_dataset = Cityscapes(args.data_dir, split='val')

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=4, pin_memory=True)

    if dist.get_rank() == 0:
        print(f'Training on {len(train_dataset)} samples')
        print(f'Validating on {len(val_dataset)} samples')

    # Load model and preprocessing transform
    model, _ = depth_pro.create_model_and_transforms()
    model = model.half().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-8)
    criterion = MidasLoss()

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # Ensure proper shuffling
        train_loss = train(model, train_loader, criterion, optimizer, scheduler, device, epoch, args)
        val_loss, metrics = validate(model, val_loader, criterion, device, epoch, args)

        if dist.get_rank() == 0:
            print(f'Epoch {epoch+1}/{args.epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}')
            print(f'Validation Metrics: {metrics}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(args.checkpoint_dir, 'best_model.pth'))

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
